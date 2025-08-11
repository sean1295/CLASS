from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from .conv1d_components import (
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    SinusoidalPosEmb,
)

logger = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond=None):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        if cond is not None:
            embed = self.cond_encoder(cond)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class Unet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        pred_horizon,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        dropout_ratio=0.00,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.pred_horizon = pred_horizon
        self.learned_embed = nn.Parameter(torch.randn(1, pred_horizon, input_dim))

        cond_dim = global_cond_dim

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        self.down_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                Downsample1d(dim_out) if i < len(in_out) - 1 else nn.Identity(),
            ])
            for i, (dim_in, dim_out) in enumerate(in_out)
        ])

        self.up_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                Upsample1d(dim_in) if i < len(in_out) - 1 else nn.Identity(),
            ])
            for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:]))
        ])

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.dropout = nn.Dropout(p=dropout_ratio)
        print(
            "Unet1D (BC) parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(self, global_cond: torch.Tensor):
        """
        global_cond: (B, global_cond_dim) or (B, 1, global_cond_dim)
        returns: (B, action_dim, pred_horizon)
        """
        B = global_cond.shape[0]
    
        if global_cond.ndim == 3:
            global_cond = global_cond.flatten(start_dim=1)  # (B, global_cond_dim)
    
        cond = global_cond  # no transformation
    
        sample = self.learned_embed.expand(B, -1, -1)  # (B, T, input_dim)
        x = einops.rearrange(sample, "b t c -> b c t")  # <-- CORRECT
    
        h = []
        for res1, res2, down in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            h.append(x)
            x = down(x)
    
        for mid in self.mid_modules:
            x = mid(x, cond)
    
        for res1, res2, up in self.up_modules:
            x = torch.cat([x, h.pop()], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = up(x)
    
        x = self.final_conv(x)
        return x.permute(0, 2, 1)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        dropout_ratio=0.05,
    ):
        super().__init__()

        dsed = diffusion_step_embed_dim
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        # self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.dropout = nn.Dropout(p=dropout_ratio)
        print(
            "number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor,
        **kwargs
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        if len(global_cond.shape) == 3:
            global_cond = global_cond.flatten(start_dim=1)

        global_feature = torch.cat([global_feature, global_cond], axis=-1)
        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x