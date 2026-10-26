"""Modules of the shared-diffusion-prior pipeline.

* ``Time_aware_refinementS`` -- Diffusion Prior Alignment (DPA, Sec. 3.2):
  refines the one-step ``z0`` prediction of the denoising network and injects
  the noise-level embedding estimated by ANE.
* ``Timeaware_PriorTransfer`` -- Diffusion Prior Transfer (DPT, Sec. 3.4):
  transfers the main-view prior to the auxiliary view with Look-Back Attention
  (LBA, Eq. 12).
* ``DDIMSolver``-style time-step table + boundary-condition scalings used by the
  two-step sampler.
"""

import numpy as np
import torch
import torch.nn as nn

from compressor.layer import (
    RRDB,
    ResidualBottleneckBlock,
    conv3x3,
    make_layer,
)
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
)


def ddim_timesteps(num_timesteps=1000, ddim_steps=50):
    """Equally spaced DDIM time steps, e.g. [19, 39, ..., 999] for 1000/50."""
    step_ratio = num_timesteps // ddim_steps
    return (np.arange(1, ddim_steps + 1) * step_ratio).round().astype(np.int64) - 1


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data ** 2 / ((timestep / 0.1) ** 2 + sigma_data ** 2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out


def append_dims(x, target_dims):
    """Append trailing dimensions until ``x`` has ``target_dims`` dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


class AttentionAlignment(nn.Module):
    """Symmetric cross-attention between the two compressed views (DPA)."""

    def __init__(self, channels):
        super().__init__()
        self.rb1 = ResidualBottleneckBlock(channels)
        self.rb2 = ResidualBottleneckBlock(channels)
        self.attn = EfficientAttention(
            key_in_channels=channels,
            query_in_channels=channels,
            key_channels=channels // 8,
            head_count=2,
            value_channels=channels // 4,
        )
        self.refine = nn.Sequential(
            conv3x3(2 * channels, channels), ResidualBottleneckBlock(channels)
        )

    def forward(self, x, c):
        identity_left, identity_right = x, c
        x_left, conf = self.rb1(x), self.rb2(c)
        A_right_to_left = self.attn(x_left, conf)
        compact_left = identity_left + self.refine(
            torch.cat((A_right_to_left, x_left), dim=1)
        )
        return compact_left


class LBAAlignment(nn.Module):
    """Look-Back Attention alignment used by DPT."""

    def __init__(self, channels):
        super().__init__()
        self.rb1 = ResidualBottleneckBlock(channels)
        self.rb2 = ResidualBottleneckBlock(channels)
        self.attn = LookBackattention(
            key_in_channels=channels,
            query_in_channels=channels,
            key_channels=channels // 8,
            head_count=2,
            value_channels=channels // 4,
        )
        self.refine = nn.Sequential(
            conv3x3(2 * channels, channels), ResidualBottleneckBlock(channels)
        )

    def forward(self, x, x_side, c):
        identity_left, identity_right = x, c
        x_left, x_side, conf = self.rb1(x), self.rb2(x_side), self.rb2(c)
        A_right_to_left = self.attn(x_left, x_side, conf)
        compact_left = identity_left + self.refine(
            torch.cat((A_right_to_left, x_left), dim=1)
        )
        return compact_left


class EfficientAttention(nn.Module):
    """Linear-complexity attention: keys/values from one tensor, queries from another.

    The caller decides which latent plays which role; SDiD feeds the *compressed*
    main view as keys so that the attention map stays reliable at very low bitrates.
    """

    def __init__(
        self,
        key_in_channels,
        query_in_channels,
        key_channels=32,
        head_count=8,
        value_channels=64,
    ):
        super().__init__()
        self.in_channels = query_in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(key_in_channels, key_channels, 1)
        self.queries = nn.Conv2d(query_in_channels, key_channels, 1)
        self.values = nn.Conv2d(key_in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, query_in_channels, 1)

    def forward(self, target, input):
        n, _, h, w = input.size()
        keys = self.keys(input).reshape((n, self.key_channels, h * w))
        queries = self.queries(target).reshape(n, self.key_channels, h * w)
        values = self.values(input).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = torch.softmax(
                keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2
            )
            query = torch.softmax(
                queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1
            )
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(
                n, head_value_channels, h, w
            )
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        return self.reprojection(aggregated_values)


class LookBackattention(nn.Module):
    """Look-Back Attention (Eq. 12): Q from the aux view, K from the compressed
    main view, V from the main-view prior."""

    def __init__(
        self,
        key_in_channels,
        query_in_channels,
        key_channels=32,
        head_count=8,
        value_channels=64,
    ):
        super().__init__()
        self.in_channels = query_in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(key_in_channels, key_channels, 1)
        self.queries = nn.Conv2d(query_in_channels, key_channels, 1)
        self.values = nn.Conv2d(key_in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, query_in_channels, 1)

    def forward(self, target1, target2, input):
        n, _, h, w = input.size()
        keys = self.keys(target2).reshape((n, self.key_channels, h * w))
        queries = self.queries(target1).reshape(n, self.key_channels, h * w)
        values = self.values(input).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = torch.softmax(
                keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2
            )
            query = torch.softmax(
                queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1
            )
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(
                n, head_value_channels, h, w
            )
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        return self.reprojection(aggregated_values)


class Time_aware_refinementS(nn.Module):
    """DPA refinement A_phi: (z0_hat, c_main, t) -> refined z0."""

    def __init__(self, in_ch, out_ch, model_channels, num_grow_ch=32):
        super().__init__()
        self.model_channels = model_channels
        self.time_embed = nn.Sequential(
            linear(model_channels, model_channels * 4),
            nn.SiLU(),
            linear(model_channels * 4, model_channels * 4),
        )
        self.in_layer = nn.Sequential(
            conv_nd(2, 4, 32, 3, padding=1),
            normalization(32),
            nn.SiLU(),
        )
        self.IN_1 = nn.Sequential(
            conv_nd(2, 4, 32, 3, padding=1),
            normalization(32),
            nn.SiLU(),
        )
        self.IN_2 = nn.Sequential(
            conv_nd(2, 4, 32, 3, padding=1),
            normalization(32),
            nn.SiLU(),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            linear(model_channels * 4, in_ch),
        )
        self.attn = AttentionAlignment(32)
        self.mid_layer = make_layer(RRDB, 12, num_feat=64, num_grow_ch=num_grow_ch)
        self.out_layer = nn.Sequential(
            conv_nd(2, 64, 64, 3, padding=1),
            normalization(64),
            nn.SiLU(),
            conv_nd(2, 64, out_ch, 3, padding=1),
        )

    def forward(self, h, c_c, t, c_side=None):
        if c_side is None:
            c_side = c_c
        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        t_emb = self.emb_layer(self.time_embed(t_emb)).type(h.dtype)
        while len(t_emb.shape) < len(h.shape):
            t_emb = t_emb[..., None]
        c = self.attn(self.IN_1(c_c), self.IN_2(c_side))
        h = h + t_emb
        h = self.in_layer(h)
        h = torch.cat([h, c], dim=1)
        h = self.out_layer(self.mid_layer(h))
        return h + c_c


class Timeaware_PriorTransfer(nn.Module):
    """DPT: transfers the main-view prior ``prior`` to the auxiliary view."""

    def __init__(self, in_ch, out_ch, model_channels, num_grow_ch=32):
        super().__init__()
        self.model_channels = model_channels
        self.time_embed = nn.Sequential(
            linear(model_channels, model_channels * 4),
            nn.SiLU(),
            linear(model_channels * 4, model_channels * 4),
        )
        self.in_layer = nn.Sequential(
            conv_nd(2, 4, 32, 3, padding=1),
            normalization(32),
            nn.SiLU(),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            linear(model_channels * 4, in_ch),
        )
        self.attn = LBAAlignment(32)
        self.IN_1 = nn.Sequential(
            conv_nd(2, 4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.IN_2 = nn.Sequential(
            conv_nd(2, 4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mid_layer = make_layer(RRDB, 18, num_feat=68, num_grow_ch=num_grow_ch)
        self.out_layer = nn.Sequential(
            conv_nd(2, 68, 64, 3, padding=1),
            normalization(64),
            nn.SiLU(),
            conv_nd(2, 64, out_ch, 3, padding=1),
        )

    def forward(self, input, c_side, prior, t):
        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        t_emb = self.emb_layer(self.time_embed(t_emb)).type(prior.dtype)
        while len(t_emb.shape) < len(prior.shape):
            t_emb = t_emb[..., None]
        c = self.attn(self.IN_1(c_side[0]), self.IN_2(c_side[1]), self.IN_2(prior))
        prior = prior + t_emb
        prior = self.in_layer(prior)
        h = torch.cat([prior, c, input], dim=1)
        h = self.out_layer(self.mid_layer(h))
        return h + c_side[0]
