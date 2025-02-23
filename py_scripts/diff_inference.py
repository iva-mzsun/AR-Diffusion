import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist

import os
import sys
import cv2
import time
import copy
import math
import yaml
import wandb
import shutil
import decord
import logging
import argparse
import datetime
import subprocess
import numpy as np
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs
from accelerate import PartialState

from PIL import Image
from einops import repeat, rearrange
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from src.utils.util import instantiate_from_config, tensor2img, load_state_dict


logger = get_logger(__name__)

@torch.no_grad()
def log_validation(args, model, valid_dataloader, accelerator, weight_dtype, epoch, global_step):
    model.eval()
    pair_imgs = []
    logs_dict = dict({})
    
    if dist.is_available() and PartialState().num_processes != 1:
        try:
            one_sample = model.module.sample
        except:
            one_sample = model.sample
        progress_bar = tqdm(
            range(0, len(valid_dataloader)),
            desc=f"Validation Steps of rank {dist.get_rank()}"
        )
        video_suffix = f"_{dist.get_rank():02d}"
        set_seed(args.seed + (dist.get_rank() + 1) * 111111)
    else:
        one_sample = model.sample
        progress_bar = tqdm(
            range(0, len(valid_dataloader)),
            desc="Validation Steps",
        )
        video_suffix = ""
        set_seed(args.seed)
    
    for step, batch in enumerate(valid_dataloader):
        # renormalize frames
        def convert_tensor(img):
            img = img.detach().cpu().float()
            x_samples = torch.clamp(img, min=0.0, max=1.0)
            x_samples = 255. * rearrange(x_samples.numpy(), 
                                         'b t c h w -> b t h w c')
            return x_samples.astype(np.uint8)
        
        # print(f"Note initialize xT method: {args.initialize_xT}")
        if args.initialize_xT is None or args.initialize_xT == '':
            batch_size = args.batch_size
            device = batch['frames'].device
            shape = (batch_size, args.num_frames, 32, 4)
            x_T = torch.randn(shape, device=device)

        elif args.initialize_xT == "same_xT_between_frame":
            batch_size = args.batch_size
            device = batch['frames'].device
            shape = (batch_size, 32, 4)
            x_T = torch.randn(shape, device=device)
            x_T = repeat(x_T, 'b l c -> b t l c', t=args.num_frames)
        
        else:
            raise NotImplementedError

        # stime = time.time()
        with accelerator.accumulate(model):
            sample_out = one_sample(batch, x_T=x_T, sample=False,
                                    total_num_frames=args.num_frames,
                                    ardiff_step=args.ardiff_step,
                                    unconditional_guidance_scale=args.ucgs,
                                    use_timestep_shift=args.use_timestep_shift,
                                    window_shift=args.window_shift,
                                    cut_off_value=args.cut_off_value,
                                    verbose=args.verbose)
        x = sample_out['xs']
        x_pred = sample_out['xs_pred']
        # print(f"{(time.time() - stime) / x.shape[0]: .2f} seconds to generate one video")
        x = convert_tensor(x)
        x_pred = convert_tensor(x_pred)

        def save_vid(vid_path, x):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(vid_path, fourcc, 25.0, x[0].shape[:2])
            for t in range(x.shape[0]):
                vw.write(cv2.cvtColor(x[t], cv2.COLOR_RGB2BGR))
            vw.release()

        # save gt and recon frames for calculate metrics
        for b in range(x.shape[0]):
            vid_name = f"s{step:04d}_b{b:02d}_{args.seed}" + video_suffix
            vid_gt = os.path.join(args.tmp_dir_save_gt, f"video{vid_name}.mp4")
            # vid_rec = os.path.join(args.tmp_dir_save_rec, f"video{vid_name}.mp4")
            vid_pred = os.path.join(args.tmp_dir_save_pred, f"video{vid_name}.mp4")
            save_vid(vid_gt, x[b])
            save_vid(vid_pred, x_pred[b])
            
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # vw_gt = cv2.VideoWriter(vid_gt, fourcc, 25.0, x[0,0].shape[:2])
            # # vw_rec = cv2.VideoWriter(vid_rec, fourcc, 25.0, x_rec[0,0].shape[:2])
            # vw_pred = cv2.VideoWriter(vid_pred, fourcc, 25.0, x_pred[0,0].shape[:2])
            # for t in range(x.shape[1]):
            #     vw_gt.write(cv2.cvtColor(x[b, t], cv2.COLOR_RGB2BGR))
            #     # vw_rec.write(cv2.cvtColor(x_rec[b, t], cv2.COLOR_RGB2BGR))
            #     vw_pred.write(cv2.cvtColor(x_pred[b, t], cv2.COLOR_RGB2BGR))
            # vw_gt.release(); vw_pred.release()
            # # vw_rec.release(); 

        progress_bar.update(1)
        if len(os.listdir(args.tmp_dir_save_pred)) >= args.sample_num:
            logger.info(f"Have generated {len(os.listdir(args.tmp_dir_save_pred))} video samples!")
            break
    
    return None

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--ignore_module_key", type=str, nargs='*', default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=47)

    # inference config
    parser.add_argument("--initialize_xT", type=str, default=None)
    parser.add_argument("--ucgs", type=float, default=1.0)
    parser.add_argument("--cosine_beta_schedule_s", type=float, default=0.008)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--ardiff_step", type=int, default=0)
    parser.add_argument("--scheduling_matrix", type=str, default=None)
    parser.add_argument("--sampling_timesteps", type=int, default=None)
    parser.add_argument("--ddim_timestep_schedule", type=str, default=None)
    
    # parameters for dataset
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sample_num", type=int, default=2048)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # parameters for sampler
    parser.add_argument("--use_timestep_shift", action="store_true")
    parser.add_argument("--window_shift", type=int, default=None)
    parser.add_argument("--cut_off_value", type=float, default=None)

    # experiment log settings
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="tmp_dir")
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--resume_skip_optimizer", action="store_true")
    parser.add_argument("--resume_skip_discriminator",  action="store_true")
    parser.add_argument("--exp_root", type=str, default="experiments")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])

    args = parser.parse_args()
    return args

def main():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    args = get_parser()

    # set save directories
    assert args.resume and os.path.exists(args.resume)
    if args.resume.find('checkpoint-') >= 0: # args.resume=/experiment/exp/ckpt_dir/checkpoint-$step
        exp_dir = os.path.dirname(os.path.dirname(args.resume))
    elif args.resume.endswith('ckpt_dir'): # args.resume=/experiment/exp/ckpt_dir
        exp_dir = os.path.dirname(args.resume)
    else: # args.resume=/experiment/exp
        exp_dir = args.resume
    
    args.exp_dir = exp_dir
    args.log_dir = os.path.join(exp_dir, "logs")
    args.img_dir = os.path.join(exp_dir, "images")
    args.cfg_dir = os.path.join(exp_dir, "configs")
    args.ckpt_dir = os.path.join(exp_dir, "ckpt_dir")
    
    accelerator_project_config = ProjectConfiguration(project_dir=os.path.basename(exp_dir), 
                                                      logging_dir=args.log_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Initalize logging: Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # load model
    model_config = OmegaConf.load(args.model_config.strip())
    try:
        # diffusion forcing
        model_config.model.params.scheduling_matrix = args.scheduling_matrix or \
            model_config.model.params.scheduling_matrix
        model_config.model.params.diffusion_cfg.sampling_timesteps = args.sampling_timesteps or \
            model_config.model.params.diffusion_cfg.sampling_timesteps
        model_config.model.params.diffusion_cfg.ddim_timestep_schedule = args.ddim_timestep_schedule or \
            model_config.model.params.diffusion_cfg.get('ddim_timestep_schedule', "linspace")
        if model_config.model.params.diffusion_cfg.beta_schedule == "cosine":
            model_config.model.params.diffusion_cfg.schedule_fn_kwargs = dict(s=args.cosine_beta_schedule_s)
        logger.info(f"Scheduling_matrix: {model_config.model.params.scheduling_matrix}")
        logger.info(f"Sampling_timesteps: {model_config.model.params.diffusion_cfg.sampling_timesteps}")
        logger.info(f"DDIM_timestep_schedule: {model_config.model.params.diffusion_cfg.ddim_timestep_schedule}")
    except:
        # for fullseq
        pass

    logger.info(f"Loading model: {model_config.model.target}")
    model = instantiate_from_config(model_config.model)
    model = model.to(accelerator.device)
    for p in model.parameters():
        p.requires_grad = False

    # model.load_state_dict(torch.load("/mnt/bn/lq-mzsun-arnold/codes/video_token/1d-tokenizer/tokenizer_titok_l32.bin", map_location="cpu"))
    # model.to(accelerator.device)

    # load data
    data_config = OmegaConf.load(args.data_config.strip())
    logger.info(f"Loading valid data: {data_config.valid_data.target}")
    if data_config.valid_data.params.clip_nframe != args.num_frames:
        print(f"Note! The number of frames is changed from {data_config.valid_data.params.clip_nframe} to {args.num_frames}")
        data_config.valid_data.params.clip_nframe = args.num_frames
    valid_dataset = instantiate_from_config(data_config.valid_data)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    model, valid_dataloader = accelerator.prepare(model, valid_dataloader)
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    
    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(valid_dataset)}")
    logger.info(f"  Num batches each epoch = {len(valid_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    
    # Potentially load in the weights and states from a previous save
    if args.resume.find('checkpoint-') >= 0: 
        resume_ckpt = args.resume
    else:
        # Resume from the most recent checkpoint
        dirs = os.listdir(args.ckpt_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint-")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        resume_ckpt = os.path.join(args.ckpt_dir, dirs[-1]) if len(dirs) > 0 else None
        global_step = int(resume_ckpt.split("-")[-1])

    assert os.path.exists(resume_ckpt)
    logger.info(f"Resuming from checkpoint {resume_ckpt}")
    accelerator.load_state(resume_ckpt, strict=False) # rotary_emb.freqs may appear in missing keys

    # create dirs for save gt and recon frames
    if args.suffix is not None and args.suffix != "":
        args.save_dir = args.save_dir + f"_{args.suffix}"
    args.tmp_dir_save_gt = os.path.join(exp_dir, args.save_dir, 'test_gt')
    # args.tmp_dir_save_rec = os.path.join(exp_dir, args.save_dir, 'test_rec')
    args.tmp_dir_save_pred = os.path.join(exp_dir, args.save_dir, 'test_pred')
    os.makedirs(args.tmp_dir_save_gt, exist_ok=True)
    # os.makedirs(args.tmp_dir_save_rec, exist_ok=True)
    os.makedirs(args.tmp_dir_save_pred, exist_ok=True)

    # inference
    model.eval()
    logger.info(f"GT video save path: {args.tmp_dir_save_gt}")
    # logger.info(f"REC video save path: {args.tmp_dir_save_rec}")
    logger.info(f"PRED video save path: {args.tmp_dir_save_pred}")
    log_validation(args, model, valid_dataloader, accelerator, weight_dtype, -1, -1)

    os.makedirs(args.log_dir, exist_ok=True)
    with open(f"{args.log_dir}/log_val_loss_{os.path.basename(resume_ckpt)}.log", "w") as f:
        f.write(f"Finish validation\n")


if __name__ == "__main__":
    main()
