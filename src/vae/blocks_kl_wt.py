"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

from einops import rearrange, repeat
from collections import OrderedDict
from functools import partial
import torch.utils.benchmark as benchmark
# from torch.backends.cuda import sdp_kernel, SDPBackend

def zero_module(module):
	"""
	Zero out the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().zero_()
	return module

# Helpful arg mapper
# backend_map = {
#     SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
#     SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
#     SDPBackend.EFFICIENT_ATTENTION: {
#         "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
# }

device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
if device_properties.major >= 8 and device_properties.minor == 0:
    print(
        "A100 GPU detected, using flash attention if input tensor is on cuda"
    )
    CUDA_BACKENDS = [
        SDPBackend.FLASH_ATTENTION, # flash attention currently does not support attention mask.
        SDPBackend.EFFICIENT_ATTENTION,
    ]
else:
    print(
        "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
    )
    CUDA_BACKENDS = [
        SDPBackend.MATH,
        SDPBackend.EFFICIENT_ATTENTION,
    ]

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor,
            attn_mask: bool
    ):
        with sdpa_kernel(backends=CUDA_BACKENDS):
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: torch.bool = None,
    ):  
        attn_output = checkpoint(self.attention, self.ln_1(x), attn_mask)
        # attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + checkpoint(self.mlp, self.ln_2(x))
            # x = x + self.mlp(self.ln_2(x))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(d_model)
        self.ln_v = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)

    def attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: bool
    ):
        with sdpa_kernel(backends=CUDA_BACKENDS):
            return self.attn(q, k, v, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: torch.bool = None,
    ):  
        attn_output = checkpoint(self.attention, self.ln_q(q), self.ln_k(k), self.ln_v(v), attn_mask)
        q = q + attn_output
        return q


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class TiTokEncoder(nn.Module):
    def __init__(self, image_size, patch_size, model_size, num_latent_tokens, token_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size

        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size * 2, kernel_size=1, bias=True)

    def forward(self, pixel_values, latent_tokens):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding[:x.shape[1], :].to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        latent_tokens = x[:, 1+self.grid_size**2:]
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        latent_tokens = latent_tokens.reshape(batch_size, self.width, self.num_latent_tokens, 1)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size * 2, 1, self.num_latent_tokens)
        return latent_tokens
    

class TiTokDecoder(nn.Module):
    def __init__(self, num_frame, image_size, patch_size, model_size, num_latent_tokens, token_size):
        super().__init__()
        self.num_frame = num_frame
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)

        # Define causal mask for fine and coarse temporal attn
        mask_cond = torch.arange(num_frame)
        fine_attn_mask = mask_cond > mask_cond.view(num_frame, 1) # 2D mask
        coarse_attn_mask = repeat(fine_attn_mask, 't1 t2 -> t1 n1 (t2 n2)', 
                                  n1=self.num_latent_tokens + self.grid_size**2, 
                                  n2=self.num_latent_tokens) # 3D mask, need to extend the first dim with batch
        self.fine_attn_mask = fine_attn_mask
        self.coarse_attn_mask = coarse_attn_mask

        # Define temporal and origin transformer
        self.transformer = nn.ModuleList()
        self.fine_temporal_blocks = nn.ModuleList()
        self.coarse_temporal_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            fine_temporal_attn = ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=-1.0)
            self.fine_temporal_blocks.append(zero_module(fine_temporal_attn))
            coarse_temporal_attn = CrossAttentionBlock(self.width, self.num_heads)
            self.coarse_temporal_blocks.append(zero_module(coarse_temporal_attn))
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        self.ffn = nn.Sequential(
            nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
            nn.Tanh(),
            nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
        )
        self.conv_out = nn.Identity()
    
    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape
        B = N // self.num_frame
        dtype = z_quantized.dtype
        device = z_quantized.device

        # Embed latent tokens
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        z_tokens = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        z_tokens = self.decoder_embed(z_tokens)
        batchsize, seq_len, _ = z_tokens.shape
        z_tokens = z_tokens + self.latent_token_positional_embedding[:seq_len]

        # Initialize and embed pixel tokens
        pixel_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(dtype)
        class_tokens = _expand_token(self.class_embedding, pixel_tokens.shape[0]).to(dtype)
        pixel_tokens = torch.cat([class_tokens, pixel_tokens], dim=1)
        pixel_tokens = pixel_tokens + self.positional_embedding[None, :pixel_tokens.shape[1], :].to(dtype)
        
        # Initialize attn masks
        fine_attn_mask = self.fine_attn_mask.to(device)
        coarse_attn_mask = repeat(self.coarse_attn_mask.to(device),
                                  't n_latent_and_pixel txn_latents -> (b t n_head) n_latent_and_pixel txn_latents', 
                                  b=B, n_head=self.num_heads)

        # Preproject all tokens
        x = torch.cat([pixel_tokens, z_tokens], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(self.num_layers):
            class_tokens, pixel_tokens, latent_tokens = x[:1], x[1:1+self.grid_size**2], x[1+self.grid_size**2:]

            # fine temporal correlations
            pixel_tokens = rearrange(pixel_tokens, 'n (b t) d -> t (b n) d', b=B)
            pixel_tokens = self.fine_temporal_blocks[i](pixel_tokens, fine_attn_mask)
            pixel_tokens = rearrange(pixel_tokens, 't (b n) d -> n (b t) d', b=B)
            # coarse temporal correlations
            latent_and_pixel = torch.cat([latent_tokens, pixel_tokens], dim=0) # (n_latent + n_pixel), (b t), d
            txlatent_tokens = rearrange(latent_tokens, 'n (b t) d -> (t n) b d', b=B)
            txlatent_tokens = repeat(txlatent_tokens, 'txn b d -> txn (b t1) d', t1=self.num_frame) # (t x n_latent), (b t), d
            latent_and_pixel = self.coarse_temporal_blocks[i](q=latent_and_pixel, k=txlatent_tokens, 
                                                              v=txlatent_tokens, attn_mask=coarse_attn_mask)
            latent_tokens = latent_and_pixel[:latent_tokens.shape[0]]
            pixel_tokens = latent_and_pixel[-self.grid_size ** 2:]
            
            # Non temporal correlated transformer
            x = torch.cat([class_tokens, pixel_tokens, latent_tokens], dim=0)
            x = self.transformer[i](x)

        # Post project
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x