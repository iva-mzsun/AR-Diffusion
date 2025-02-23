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

class TiTok(nn.Module):
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

        self.encoder = TiTokEncoder(image_size, vit_enc_patch_size, vit_enc_model_size, 
                                    num_latent_tokens, token_size)
        self.decoder = TiTokDecoder(image_size, vit_dec_patch_size, vit_dec_model_size, 
                                    num_latent_tokens, token_size)
        
        self.num_latent_tokens = num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(
            codebook_size=codebook_size,
            token_size=token_size,
            commitment_cost=commitment_cost,
            use_l2_norm=use_l2_norm)
        
        self.pixel_quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
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

    def encode(self, x):
        B, T = x.shape[:2]
        x = rearrange(x, 'b t ... -> (b t) ...')
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_quantized, result_dict = self.quantize(z)
        z_quantized = rearrange(z_quantized, '(b t) ... -> b t ...', b=B, t=T)
        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        B, T =  z_quantized.shape[:2]
        z_quantized = rearrange(z_quantized, 'b t ... -> (b t) ...')
        decoded_latent = self.decoder(z_quantized)
        quantized_states = torch.einsum(
            'nchw,cd->ndhw', decoded_latent.softmax(1),
            self.pixel_quantize.embedding.weight)
        decoded = self.pixel_decoder(quantized_states)
        decoded = rearrange(decoded, '(b t) ... -> b t ...', b=B, t=T)
        return decoded
    
    def forward(self, x, return_loss=True):
        # x: [b t c h w]
        b = x.shape[0]
        # x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        z_quantized, result_dict = self.encode(x)
        x_rec = self.decode(z_quantized)

        vq_total_loss = result_dict["quantizer_loss"]
        # x_rec = rearrange(x_rec, '(b t) c h w -> b t c h w', b=b)
        return x_rec, vq_total_loss
        