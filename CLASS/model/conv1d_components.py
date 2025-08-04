import math
import torch
import torch.nn as nn

class LearnablePosEmb(nn.Module):
    def __init__(self, dim, max_steps):
        super().__init__()
        self.embedding = nn.Embedding(max_steps, dim)

    def forward(self, x):
        # Assuming x is a tensor of shape (batch_size,), representing step indices
        return self.embedding(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, device="cuda"):
        super().__init__()
        self.dim = dim
        self.device = device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

    def forward(self, x):
        emb = x[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> SwapAxes --> GroupNorm --> SwapAxes --> Mish
    """
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(
            inp_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.activation = nn.Mish()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1, 256, 16))
    o = cb(x)
