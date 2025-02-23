import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
# from src.ardiff.modules.ddim import DDIMSampler
# from src.ardiff.modules.multiddim import DDIMSampler
from src.ardiff.modules.fifoddim import DDIMSampler
from src.ardiff.modules.diffusion import LatentDiffusion
from src.ardiff.modules.timestep_sampler import Incremental_Timesteps

from src.utils.util import instantiate_from_config
from accelerate.logging import get_logger
logger = get_logger(__name__)

class BaseModel(nn.Module):
    def __init__(self,
                 # params for model
                 model_cfg, diffusion_cfg,
                 # params for inputs
                 in_channels, frame_resolution, 
                 num_frames, n_token_per_frame,
                 n_channel_per_token, 
                 # params for vae
                 vae_cfg_file, vae_pth_path, skip_vae=False,
                 # Other params
                 noise_level = None,
                 scale_factor = None,
                 norm_preprocess = False,
                 condition_mode = "uncond",
                 *args, **kwargs):
        super().__init__()

        # params for inputs
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.noise_level = noise_level
        self.frame_resolution = frame_resolution
        self.n_token_per_frame = n_token_per_frame
        self.n_channel_per_token = n_channel_per_token
        self.norm_preprocess = norm_preprocess
        self.train_step_sampler = None
        self.condition_mode = condition_mode

        # diffusion_models settings
        global instantiate_from_config
        model = instantiate_from_config(model_cfg)
        self.diffusion_model = LatentDiffusion(model, **diffusion_cfg.params)
        self.timesteps = self.diffusion_model.num_timesteps

        # Dataset Condition: Initialize learnable embedding for current dataset
        if self.condition_mode.lower() == "uncond":
            self.context_emb = nn.Parameter(torch.randn(77, 768) * 0.1)
        
        elif self.condition_mode.lower() == "class_cond":
            num_class = kwargs.pop('num_class', None)
            if not num_class:
                raise RuntimeError(f"None valid num_class")
            self.class_emb = nn.Parameter(torch.randn(num_class + 1, 77, 768) * 0.1)
        
        else:
            raise NotImplementedError

        # video auto encoder
        self.skip_vae = skip_vae
        self.vae_cfg_file = vae_cfg_file
        self.vae_pth_path = vae_pth_path
        logger.info(f"NOTE!!! skip vae: {self.skip_vae}")
        if not skip_vae:
            import safetensors
            from omegaconf import OmegaConf
            from src.utils.util import instantiate_from_config
            vae_config = OmegaConf.load(self.vae_cfg_file.strip())
            self.vae = instantiate_from_config(vae_config.model)
            state_dict = safetensors.torch.load_file(self.vae_pth_path, device='cpu')
            self.vae.load_state_dict(state_dict, strict=True)
            for n, p in self.vae.named_parameters():
                p.requires_grad = False

    def _preprocess_batch(self, batch):
        xs = batch['frames'] # (B, T, D, H, W)
        batch_size, n_frames = xs.shape[:2]
        masks = torch.ones(n_frames, batch_size).to(xs.device)
        
        # Condition features, [B L C]
        if self.condition_mode.lower() == "uncond":
            c = repeat(self.context_emb, 'l c -> b l c', b=batch_size)
        
        elif self.condition_mode.lower() == "class_cond":
            class_ids = batch['class_id']
            class_embs = []
            for cid in class_ids:
                if torch.randn(1) < 0.2:
                    cid = 0
                class_embs.append(self.class_emb[cid].unsqueeze(0))
            c = torch.cat(class_embs, dim=0)
        
        else:
            raise NotImplementedError

        # Frame-wise features, [B T L C]
        with torch.no_grad():
            if self.skip_vae:
                zs = xs
            else:
                zs = self.vae.encode(xs)
            if self.scale_factor:
                zs = zs * self.scale_factor
            
        zs = rearrange(zs, 'b t c h w -> b t (h w) c')
        if self.norm_preprocess:
            zs = F.normalize(zs, dim=-1)
        
        return zs, dict(c_crossattn=[c])

    def forward(self, batch, sample=True, **kwargs):
        zs, cond = self._preprocess_batch(batch)
        noise_levels = self._generate_noise_levels(zs)
        zs_pred, loss = self.diffusion_model.p_losses(zs, t=noise_levels, cond=cond)
        return {'loss': loss.mean(), }

    @torch.no_grad()
    def _postprocess(self, zs, verbose=False):
        zs = rearrange(zs, 'b t (h w) c -> b t c h w', h=1)
        if self.scale_factor:
            zs = zs / self.scale_factor
        if self.skip_vae:
            xs = zs
        else:
            xs = self.vae.decode(zs)
        return xs

    @torch.no_grad()
    def sample(self, batch, x_T=None, sample=True, ardiff_step=0,
               use_ddim=True, ddim_steps=50, ddim_eta=0.0,
               verbose=False, unconditional_guidance_scale=3.0, 
               total_num_frames=None, **kwargs):
        log = dict()
        log['xs'] = batch['frames']
        zs, cond = self._preprocess_batch(batch)
        log['xs_rec'] = self._postprocess(zs)
        device = zs.device
        batch_size, _ = zs.shape[:2]

        # obtain inputs & conditions
        assert use_ddim == (ddim_steps > 0)
        # obtain conditions
        if self.condition_mode.lower() == "uncond":
            context = repeat(self.context_emb, 'l c -> b l c', b=batch_size)
            uc_context = torch.zeros_like(context)
        
        elif self.condition_mode.lower() == "class_cond":
            class_ids = batch['class_id']
            class_embs = []
            uc_class_embs = []
            for cid in class_ids:
                class_embs.append(self.class_emb[cid].unsqueeze(0))
                uc_class_embs.append(self.class_emb[0].unsqueeze(0))
            context = torch.cat(class_embs, dim=0)
            uc_context = torch.cat(uc_class_embs, dim=0)
        
        else:
            raise NotImplementedError

        new_cond = {"c_crossattn": [context]}
        uc_cond = {"c_crossattn": [uc_context]}
        ddim_kwargs = {"use_ddim": use_ddim,
                       "ddim_eta": ddim_eta,
                       "ddim_steps": ddim_steps}
        
        if sample or unconditional_guidance_scale == 1.0:
            zs_woucgs, _ = self.sample_log(cond=new_cond, batch_size=batch_size, 
                                           x_T=x_T, total_num_frames=total_num_frames,
                                           ardiff_step=ardiff_step, device=device, verbose=verbose, **ddim_kwargs)
            samples = self._postprocess(zs_woucgs)
        
        # sampling with unconditional guidance scale
        elif unconditional_guidance_scale >= 0.0:
            zs_ucgs, _ = self.sample_log(cond=new_cond, batch_size=batch_size, 
                                         x_T=x_T, total_num_frames=total_num_frames,
                                         unconditional_conditioning=uc_cond,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         ardiff_step=ardiff_step, device=device, verbose=verbose, **ddim_kwargs)
            samples = self._postprocess(zs_ucgs)
        
        log["xs_pred"] = samples
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, total_num_frames, 
                   use_ddim, ddim_steps, x_T, ardiff_step, device, verbose=False, **kwargs):
        ddim_sampler = DDIMSampler(self.diffusion_model)
        shape = (batch_size, total_num_frames, self.n_token_per_frame, self.n_channel_per_token)
        x_T = torch.randn(shape, device=device) if x_T is None else x_T
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, self.num_frames, shape,
                                                     cond, x_T=x_T, ardiff_step=ardiff_step, 
                                                     verbose=verbose, **kwargs)
        return samples, intermediates

    def _generate_noise_levels(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        batch_size, num_frames, *_ = xs.shape
        if self.noise_level == "full_sequence":
            timesteps = torch.randint(0, self.timesteps, (batch_size,), device=xs.device)
            timesteps = timesteps.unsqueeze(1).expand((batch_size, num_frames, ))
        elif self.noise_level == "diff_forcing":
            timesteps = torch.randint(0, self.timesteps, (batch_size, num_frames, ), device=xs.device)
        elif self.noise_level == "autoregressive_diffusion":
            if self.train_step_sampler == None:
                self.train_step_sampler = Incremental_Timesteps(self.num_frames, self.timesteps)
            timesteps = torch.zeros((batch_size, num_frames, ))
            for b in range(batch_size):
                cur_timesteps = self.train_step_sampler.sample_step_sequence()
                timesteps[b, :] = torch.tensor(cur_timesteps)
        
        elif self.noise_level == "ar_diff_midsamplet":
            if self.train_step_sampler == None:
                self.train_step_sampler = Incremental_Timesteps(self.num_frames, self.timesteps)
            timesteps = torch.zeros((batch_size, num_frames, ))
            for b in range(batch_size):
                cur_timesteps = self.train_step_sampler.sample_stepseq_from_mid()
                timesteps[b, :] = torch.tensor(cur_timesteps)
        
        else:
            raise ValueError(f"No supported {self.noise_level}")

        return timesteps.long().to(xs.device)
