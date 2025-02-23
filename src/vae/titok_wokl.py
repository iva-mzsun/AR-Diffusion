"""This file contains the model definition of TiTok.

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
"""

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf

from .blocks import TiTokEncoder, TiTokDecoder
from .quantizer import VectorQuantizer
from .maskgit_vqgan import Decoder as Pixel_Decoder
from .maskgit_vqgan import VectorQuantizer as Pixel_Quantizer

from .modules.distributions.distributions import DiagonalGaussianDistribution

class TiTok_woKL(nn.Module):
    def __init__(self, 
                image_size = 256,
                codebook_size = 4096,
                token_size = 12,
                use_l2_norm = True,
                commitment_cost = 0.25,
                # vit arch
                vit_enc_model_size = "large",
                vit_dec_model_size = "large",
                vit_enc_patch_size = 16,
                vit_dec_patch_size = 16,
                num_latent_tokens = 32,
                lpips_loss_weight = 0.1,
                use_l1_loss = False,
    ):
        super().__init__()
        self.use_l1_loss = use_l1_loss
        self.use_l2_norm = use_l2_norm

        self.encoder = TiTokEncoder(image_size, vit_enc_patch_size, vit_enc_model_size, 
                                    num_latent_tokens, token_size)
        self.decoder = TiTokDecoder(image_size, vit_dec_patch_size, vit_dec_model_size, 
                                    num_latent_tokens, token_size)
        
        self.num_latent_tokens = num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)
        self.pixel_quantize_conv = nn.Conv2d(1024, 256, padding=0, stride=1, kernel_size=1, bias=True)
        self.pixel_quantize_conv.bias.data.zero_()
        self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
             "num_resolutions": 5,
             "dropout": 0.0,
             "hidden_channels": 128,
             "num_channels": 3,
             "num_res_blocks": 2,
             "resolution": 256,
             "z_channels": 256}))

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x, sample_posterior=False):
        B, T = x.shape[:2]
        x = rearrange(x, 'b t ... -> (b t) ...')
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=1)
        z = rearrange(z, '(b t) ... -> b t ...', b=B, t=T)
        return z
    
    def decode(self, z):
        B, T =  z.shape[:2]
        z = rearrange(z, 'b t ... -> (b t) ...')
        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=1)
        decoded_latent = self.decoder(z)
        latent = self.pixel_quantize_conv(decoded_latent.softmax(1))
        decoded = self.pixel_decoder(latent)
        decoded = rearrange(decoded, '(b t) ... -> b t ...', b=B, t=T)
        return decoded

    def forward(self, x, return_loss=True):
        # x: [b t c h w]
        b = x.shape[0]
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        # encode
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        
        # decode
        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=1)
        decoded_latent = self.decoder(z)
        latent = self.pixel_quantize_conv(decoded_latent.softmax(1))
        x_rec = self.pixel_decoder(latent)

        x_rec = rearrange(x_rec, '(b t) c h w -> b t c h w', b=b)
        return x_rec
