import os
import sys
import copy
import math
import yaml
import wandb
import shutil
import logging
import argparse
import datetime
import subprocess
import numpy as np
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs

from src.utils.training_utils import EMAModel
from src.utils.util import instantiate_from_config, tensor2img, load_state_dict

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

logger = get_logger(__name__)

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def run_subprocess(cmd_list):
    try:
        logger.info(f"Conducting {cmd_list}")
        out_bytes = subprocess.check_output(cmd_list)
    except subprocess.CalledProcessError as e:
        out_bytes = e.output       # Output generated before error
        code      = e.returncode   # Return code
        logger.warning(f"Fail to conduct {cmd_list} with error: {code}, {out_bytes}")

@torch.no_grad()
def log_validation(args, model, criterion, valid_dataloader, accelerator, weight_dtype, epoch, global_step):
    model.eval()
    criterion.eval()
    pair_imgs = []
    logs_dict = dict({})

    progress_bar = tqdm(
        range(0, len(valid_dataloader)),
        desc="Validation Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for step, batch in enumerate(valid_dataloader):
        with accelerator.accumulate(model):
            x = batch["frames"].to(dtype=weight_dtype)
            x_rec = model(x.to(accelerator.device))
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            x_rec = rearrange(x_rec, 'b t c h w -> (b t) c h w')
            ae_loss, log_dict = criterion(inputs=x, 
                                          reconstructions=x_rec, 
                                          optimizer_idx=0,
                                          global_step=global_step,
                                          last_layer=model.pixel_decoder.conv_out.weight if args.debug else model.module.pixel_decoder.conv_out.weight,
                                          split="valid")

        progress_bar.update(1)
        logs = {"val loss": ae_loss.detach().item()}
        progress_bar.set_postfix(**logs)
        
        if step < 8:
            pair_imgs.append((x, x_rec))
        
        for k, v in log_dict.items():
            if step == 0:
                logs_dict[k] = [v]
            else:
                logs_dict[k].append(v)
                
    # process x & x_rec to save
    xs, xs_rec = zip(*pair_imgs)
    xs = torch.cat(xs, dim=0)
    xs_rec = torch.cat(xs_rec, dim=0)
    save_x = tensor2img(xs.detach().cpu(), col=args.display_col)
    save_x_rec = tensor2img(xs_rec.detach().cpu(), col=args.display_col)
    try:
        save_x.save(os.path.join(args.img_dir, f"epoch{epoch:03d}_step{global_step:07d}_x.jpg"))
        save_x_rec.save(os.path.join(args.img_dir, f"epoch{epoch:03d}_step{global_step:07d}_x_rec.jpg"))
        if args.upload_to_hdfs:
            hdfs_save_img = os.path.join(args.hdfs_exp_dir, os.path.relpath(args.img_dir, args.exp_dir))
            run_subprocess(['hdfs', 'dfs', '-put', os.path.join(args.img_dir, f"epoch{epoch:03d}_step{global_step:07d}_x.jpg"), hdfs_save_img])
            run_subprocess(['hdfs', 'dfs', '-put', os.path.join(args.img_dir, f"epoch{epoch:03d}_step{global_step:07d}_x_rec.jpg"), hdfs_save_img])
    except Exception as e:
        logger.info(f"save validation img error: {e}")

    # calculate mean loss
    val_logs = dict({})
    for k, v in logs_dict.items():
        val_logs[k] = torch.mean(torch.tensor(v))
    
    model.train()
    return val_logs

@torch.no_grad()
def save_checkpoint(args, accelerator, global_step):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(args.ckpt_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(args.ckpt_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)
                if args.upload_to_hdfs:
                    hdfs_remove_ckpt = os.path.join(args.hdfs_exp_dir, os.path.relpath(removing_checkpoint, args.exp_dir))
                    run_subprocess(['hdfs', 'dfs', '-rm', '-rf', hdfs_remove_ckpt])

    save_path = os.path.join(args.ckpt_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")
    if args.upload_to_hdfs:
        hdfs_save_ckpt = os.path.join(args.hdfs_exp_dir, os.path.relpath(save_path, args.exp_dir))
        run_subprocess(['hdfs', 'dfs', '-put', save_path, hdfs_save_ckpt])

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--trainable_module_key", type=str, nargs='*', default=None)
    parser.add_argument("--discriminator_loss_start_step", type=int, default=-1)
    parser.add_argument("--skip_initial_model", action='store_true')
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=23)
    parser.add_argument("--display_col", type=int, default=8)

    # parameters for dataset
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_power", type=int, default=1)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # experiment log settings
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--resume_skip_optimizer", action="store_true")
    parser.add_argument("--resume_skip_discriminator",  action="store_true")
    parser.add_argument("--exp_root", type=str, default="experiments")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=2)
    parser.add_argument("--suffix", type=str, default="", help="suffix for logdir")
    parser.add_argument("--upload_to_hdfs", type=bool, default=False)
    parser.add_argument("--hdfs_exp_root", type=str, 
                        default="hdfs://haruna/home/byte_labcv_gan/common/sunmingzhen.triz/va_link/codes/3d_tokenizer/experiments")

    # parameters for validation
    parser.add_argument("--validation_steps", type=int, default=None)

    # parameters for training
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")

    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clip_norm", type=float, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # parameters for optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    
    args = parser.parse_args()

    assert args.num_train_epochs or args.max_train_steps, f"Require either `num_train_epochs` or `max_train_steps`"

    # set save directories
    if args.resume:
        assert os.path.exists(args.resume)
        if args.resume.find('checkpoint-') >= 0: # args.resume=/experiment/exp/ckpt_dir/checkpoint-$step
            exp_dir = os.path.dirname(os.path.dirname(args.resume))
        elif args.resume.endswith('ckpt_dir'): # args.resume=/experiment/exp/ckpt_dir
            exp_dir = os.path.dirname(args.resume)
        else: # args.resume=/experiment/exp
            exp_dir = args.resume
    else:
        print("Using data config {}".format(args.data_config))
        print("Using model config {}".format(args.model_config))
        data_cfg_name = os.path.splitext(os.path.split(args.data_config)[-1])[0]
        model_cfg_name = os.path.splitext(os.path.split(args.model_config)[-1])[0]
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        exp_name = now + "_" + model_cfg_name + "_" + data_cfg_name + args.suffix
        exp_dir = os.path.join(args.exp_root, exp_name)

    args.exp_dir = exp_dir
    args.log_dir = os.path.join(exp_dir, "logs")
    args.img_dir = os.path.join(exp_dir, "images")
    args.cfg_dir = os.path.join(exp_dir, "configs")
    args.ckpt_dir = os.path.join(exp_dir, "ckpt_dir")
    assert args.batch_size % args.display_col == 0
    if args.upload_to_hdfs:
        # Note: Put exp dir to hdfs when we are ready to start training
        args.hdfs_exp_dir = os.path.join(args.hdfs_exp_root, os.path.basename(args.exp_dir))
    return args

def main():
    args = get_parser()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator_project_config = ProjectConfiguration(project_dir=os.path.basename(args.exp_dir), 
                                                      logging_dir=args.log_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    if accelerator.is_main_process:
        os.makedirs(args.exp_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.img_dir, exist_ok=True)
        os.makedirs(args.cfg_dir, exist_ok=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)
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
    def initialize_model(model, model_config):
        model_state_dict = torch.load("tokenizer_titok_l32.bin", map_location="cpu")
        origin_spatial_latent_num = model_state_dict['latent_tokens'].shape[0]
        target_spatial_latent_num = model_config.model.params.get("num_latent_tokens_spatial", None) or \
            model_config.model.params.get("num_latent_tokens", None)
        if origin_spatial_latent_num < target_spatial_latent_num: # resize latent tokens
            logger.warning(f"The number of spatial latent tokens is increased from {origin_spatial_latent_num} to {target_spatial_latent_num}")
            model_state_dict['latent_tokens'] = torch.cat([model_state_dict['latent_tokens'],
                                                            torch.zeros(target_spatial_latent_num - origin_spatial_latent_num, model_state_dict['latent_tokens'].shape[1])], dim=0)
            model_state_dict['encoder.latent_token_positional_embedding'] = torch.cat([model_state_dict['encoder.latent_token_positional_embedding'],
                                                            torch.zeros(target_spatial_latent_num - origin_spatial_latent_num, model_state_dict['encoder.latent_token_positional_embedding'].shape[1])], dim=0)
            model_state_dict['decoder.latent_token_positional_embedding'] = torch.cat([model_state_dict['decoder.latent_token_positional_embedding'],
                                                            torch.zeros(target_spatial_latent_num - origin_spatial_latent_num, model_state_dict['decoder.latent_token_positional_embedding'].shape[1])], dim=0)
        elif origin_spatial_latent_num > target_spatial_latent_num:
            logger.warning(f"The number of spatial latent tokens is decreased from {origin_spatial_latent_num} to {target_spatial_latent_num}")
            model_state_dict['latent_tokens'] = model_state_dict['latent_tokens'][:target_spatial_latent_num]
            model_state_dict['encoder.latent_token_positional_embedding'] = model_state_dict['encoder.latent_token_positional_embedding'][:target_spatial_latent_num]
            model_state_dict['decoder.latent_token_positional_embedding'] = model_state_dict['decoder.latent_token_positional_embedding'][:target_spatial_latent_num]
        model.load_state_dict(model_state_dict, strict=False)
        return model
    if not args.skip_initial_model:
        model = initialize_model(model, model_config).to(accelerator.device)
    if accelerator.is_main_process:
        OmegaConf.save(model_config, os.path.join(args.cfg_dir, "model_config.yaml"))
    
    # load criterion
    from src.vae_model.modules.losses import LPIPSWithDiscriminator_woKL
    criterion = LPIPSWithDiscriminator_woKL(disc_start=args.discriminator_loss_start_step)

    # initialize optimizer
    if args.trainable_module_key is None or args.trainable_module_key == []:
        optim_params = model.parameters()
        logger.info(f"Train all model parameters!")
    else:
        optim_params = []
        for n, p in model.named_parameters():
            if_trainable = False
            p.requires_grad=False
            for k in args.trainable_module_key:
                if k in n:
                    if_trainable = True
                    p.requires_grad = True
                    optim_params.append(p)
                    logger.info(f"Add trainable param: {n}")
                    break
    
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # load data
    data_config = OmegaConf.load(args.data_config.strip())
    logger.info(f"Loading train data: {data_config.train_data.target}")
    logger.info(f"Loading valid data: {data_config.valid_data.target}")
    train_dataset = instantiate_from_config(data_config.train_data)
    train_dataloader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.dataloader_num_workers,
    )
    valid_dataset = instantiate_from_config(data_config.valid_data)
    valid_dataloader = DataLoaderX(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.dataloader_num_workers,
    )
    if accelerator.is_main_process:
        OmegaConf.save(data_config, os.path.join(args.cfg_dir, "data_config.yaml"))

    # Initialize lr scheduler
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    logger.info(f"Prepare learning rate scheduler: {args.lr_scheduler}")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    model, criterion, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, criterion, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("3d tokenizer", config=tracker_config,
                                  init_kwargs={"wandb": {"name": os.path.basename(args.exp_dir)}, "group": "diff"})

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Gradient clip norm value = {args.gradient_clip_norm}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if accelerator.is_main_process:
        yaml.dump(args.__dict__, open(os.path.join(args.cfg_dir, "args_config.yaml"), 'w')) # save args config
    if args.upload_to_hdfs:
        # Note: Now the hdfs_exp_dir is valid
        run_subprocess(['hdfs', 'dfs', '-put', args.exp_dir, args.hdfs_exp_dir])

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume:
        if args.resume.find('checkpoint-') >= 0: 
            resume_ckpt = args.resume
        else:
            # Resume from the most recent checkpoint
            dirs = os.listdir(args.ckpt_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_ckpt = os.path.join(args.ckpt_dir, dirs[-1]) if len(dirs) > 0 else None

        if resume_ckpt is None:
            logger.warning(f"No checkpoint found in {args.resume}. Starting a new training run.")
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {resume_ckpt}")
            accelerator.load_state(resume_ckpt,
                                   resume_skip_optimizer=args.resume_skip_optimizer)
            global_step = int(resume_ckpt.split("-")[-1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    elif args.pretrained_ckpt:
        assert os.path.exists(args.pretrained_ckpt), f"{args.pretrained_ckpt} does not exists!"
        accelerator.load_state(args.pretrained_ckpt,
                               resume_skip_optimizer=args.resume_skip_optimizer,
                               strict=False)
        initial_global_step = 0
    else:
        initial_global_step = 0

    # Prepare EMA model
    if args.use_ema:
        logger.info(f"Using EMA model!")
        model_for_ema = copy.deepcopy(model)
        ema_model = EMAModel(model_for_ema.parameters(), 
                             model_config=model_config,
                             decay=args.ema_decay, foreach=args.foreach_ema,)
        if args.offload_ema:
            ema_model.pin_memory()
        else:
            ema_model.to(accelerator.device)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            
            # Skip steps until we reach the resumed step
            if global_step < initial_global_step:
                initial_global_step -= 1
                continue

            with accelerator.accumulate(model):
                model.train()
                criterion.train()
                x = batch["frames"].to(dtype=weight_dtype)

                # train generator
                x_rec = model(x)
                x = rearrange(x, 'b t c h w -> (b t) c h w')
                x_rec = rearrange(x_rec, 'b t c h w -> (b t) c h w')
                ae_loss, log_dict = criterion(inputs=x, 
                                              reconstructions=x_rec, 
                                              optimizer_idx=0,
                                              global_step=global_step,
                                              last_layer=model.pixel_decoder.conv_out.weight if args.debug else model.module.pixel_decoder.conv_out.weight,
                                              split="train")

                accelerator.backward(ae_loss)

                if args.gradient_clip_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                logs = {"ae_loss": ae_loss.detach().item()}

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    if args.offload_ema:
                        ema_model.to(device=accelerator.device, non_blocking=True)
                    ema_model.step(model.parameters())
                    if args.offload_ema:
                        ema_model.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.use_ema:
                            ema_model.store(model.parameters())
                            ema_model.copy_to(model.parameters())
                        save_checkpoint(args, accelerator, global_step)
                        if args.use_ema:
                            ema_model.restore(model.parameters())
                        
                if global_step % args.validation_steps == 0:
                    if args.use_ema:
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                    val_log = log_validation(args, model, criterion, valid_dataloader, accelerator, weight_dtype, epoch, global_step)
                    log_dict.update(val_log)
                    if args.use_ema:
                        ema_model.restore(model.parameters())

            logs = {"p_loss": log_dict['train/p_loss'].detach().item(),
                    "rec_loss": log_dict['train/rec_loss'].detach().item(),
                    "loss": ae_loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            log_dict.update({"lr": lr_scheduler.get_last_lr()[0]})
            accelerator.log(log_dict, step=global_step)
            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if global_step % args.checkpointing_steps == 0:
                if args.use_ema:
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                save_checkpoint(args, accelerator, global_step)
                if args.use_ema:
                    ema_model.restore(model.parameters())
                
            if global_step % args.validation_steps == 0:
                if args.use_ema:
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                val_log = log_validation(args, model, criterion, valid_dataloader, accelerator, weight_dtype, epoch, global_step)
                if args.use_ema:
                    ema_model.restore(model.parameters())
                accelerator.log(val_log, step=global_step)

if __name__ == "__main__":
    main()

