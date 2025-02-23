import torch
import torch.nn as nn

from einops import repeat, rearrange

class Discriminator(nn.Module):
    def __init__(self, 
                num_channels=3, 
                hidden_size=64,
                d_loss_type="bce",
                ):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # input is (num_channels) x 256 x 256
            nn.Conv2d(num_channels, hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size) x 128 x 128
            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size * 2) x 64 x 64
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size * 4) x 32 x 32
            nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size * 8) x 16 x 16
            nn.Conv2d(hidden_size * 8, hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*8) x 8 x 8
            nn.Conv2d(hidden_size * 8, hidden_size * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*16) x 4 x 4
            nn.Conv2d(hidden_size * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if len(x.shape) == 5:
            # b t c h w
            x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        logits = self.model(x)

        return logits