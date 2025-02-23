import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqperceptual import *  # TODO: taming dependency yes/no?


class VQLPIPS(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        print(f"VQLPIPS running without loss.")
        
    def forward(self, codebook_loss, inputs, reconstructions, 
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            nll_loss = rec_loss * self.pixel_weight + self.perceptual_weight * p_loss
        else:
            nll_loss = rec_loss * self.pixel_weight
            p_loss = torch.tensor([0.0])

        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # generator update
        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/p_loss".format(split): p_loss.detach().mean(),
                }
        return loss, log
