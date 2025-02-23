import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist

import os
import sys
import cv2
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
        progress_bar = tqdm(
            range(0, len(valid_dataloader)),
            desc=f"Validation Steps of rank {dist.get_rank()}"
        )
        video_suffix = f"_{dist.get_rank():02d}"
        set_seed(args.seed + dist.get_rank() * 111)
    else:
        progress_bar = tqdm(
            range(0, len(valid_dataloader)),
            desc="Validation Steps",
        )
        video_suffix = ""
        set_seed(args.seed)
    
    for step, batch in enumerate(valid_dataloader):
        video_name = batch['video_name']
        with accelerator.accumulate(model):
            x = batch["frames"].to(dtype=weight_dtype)
            # x_rec, vq_loss = model(x.to(accelerator.device))
            try:
                ret = model(x.to(accelerator.device))
            except:
                ret = model(x.to(accelerator.device), sample_posterior=False)
            
            if isinstance(ret, tuple):
                x_rec = ret[0]
            else: 
                x_rec = ret

        # renormalize frames
        def convert_tensor(img):
            img = img.detach().cpu().float()
            x_samples = torch.clamp(img, min=0.0, max=1.0)
            x_samples = 255. * rearrange(x_samples.numpy(), 'b t c h w -> b t h w c')
            return x_samples.astype(np.uint8)
        x = convert_tensor(x)
        x_rec = convert_tensor(x_rec)

        def save_vid(vid_path, x):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(vid_path, fourcc, 25.0, x[0].shape[:2])
            for t in range(x.shape[0]):
                vw.write(cv2.cvtColor(x[t], cv2.COLOR_RGB2BGR))
            vw.release()

        # save gt and recon frames for calculate metrics
        for b in range(x.shape[0]):
            vid_id = b+step*x.shape[0]
            if args.save_mode == 'frame':
                vid_gt = os.path.join(args.tmp_dir_save_gt, f"{step:04d}_{b:02d}_{video_name[b]}_{video_suffix}")
                vid_rec = os.path.join(args.tmp_dir_save_rec, f"{step:04d}_{b:02d}_{video_name[b]}_{video_suffix}")
                os.makedirs(vid_gt, exist_ok=True)
                os.makedirs(vid_rec, exist_ok=True)
                for t in range(x.shape[1]):
                    frame_name = f"vid{vid_id:04d}_frame{t:04d}"
                    Image.fromarray(x[b, t]).save(os.path.join(vid_gt, f"{frame_name}.jpg"))
                    Image.fromarray(x_rec[b, t]).save(os.path.join(vid_rec, f"{frame_name}.jpg"))
            elif args.save_mode == 'video':
                vid_gt = os.path.join(args.tmp_dir_save_gt, f"{step:04d}_{b:02d}_{video_name[b]}_{video_suffix}.mp4")
                vid_rec = os.path.join(args.tmp_dir_save_rec, f"{step:04d}_{b:02d}_{video_name[b]}_{video_suffix}.mp4")
                save_vid(vid_gt, x[b])
                save_vid(vid_rec, x_rec[b])
            else:
                raise NotImplementedError(f"save_mode={args.save_mode} not supported")
        
        progress_bar.update(1)
    return None

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--save_mode", type=str, default='video') # [frame, video]
    parser.add_argument("--save_dir", type=str, default="tmp_dir")
    parser.add_argument("--ignore_module_key", type=str, nargs='*', default=None)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=23)

    # parameters for dataset
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # experiment log settings
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--resume", type=str, default="")
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
    assert args.resume and os.path.exists(args.resume), f"{args.resume} does not exist!"
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
    set_seed(args.seed)

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
    logger.info(f"Loading model: {model_config.model.target}")
    model = instantiate_from_config(model_config.model)
    model = model.to(accelerator.device)
    
    # model.load_state_dict(torch.load("/mnt/bn/lq-mzsun-arnold/codes/video_token/1d-tokenizer/tokenizer_titok_l32.bin", map_location="cpu"))
    # model.to(accelerator.device)

    # load data
    data_config = OmegaConf.load(args.data_config.strip())
    logger.info(f"Loading valid data: {data_config.valid_data.target}")
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
    accelerator.load_state(resume_ckpt)

    # create dirs for save gt and recon frames
    args.tmp_dir_save_gt = os.path.join(exp_dir, args.save_dir, 'test_gt')
    args.tmp_dir_save_rec = os.path.join(exp_dir, args.save_dir, 'test_rec')
    os.makedirs(args.tmp_dir_save_gt, exist_ok=True)
    os.makedirs(args.tmp_dir_save_rec, exist_ok=True)

    # inference
    model.eval()
    log_validation(args, model, valid_dataloader, accelerator, weight_dtype, -1, -1)
    with open(f"{args.log_dir}/log_val_loss_{os.path.basename(resume_ckpt)}.log", "w") as f:
        f.write(f"Finish validation\n")
    # val_log = log_validation(args, model, valid_dataloader, accelerator, weight_dtype, -1, -1)
    # print(f"validation log: {val_log}")
    # with open(f"{args.log_dir}/log_val_loss_{os.path.basename(resume_ckpt)}.log", "w") as f:
    #     for k, v in val_log.items():
    #         f.write(f"{k}:\t\t{v}\n")

if __name__ == "__main__":
    main()
