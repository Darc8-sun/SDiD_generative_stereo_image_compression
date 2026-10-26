"""Codebook-based latent compressor with a checkerboard (CKBD) entropy model.

Architecture follows RDEIC's latent compressor; the entropy model structure
follows ELIC.  Both views share this module (distributed coding: the encoders
are independent, only the decoder side shares priors).

The module is inference-only: ``forward`` returns the quantised reconstruction
``x_hat`` together with per-symbol likelihoods, which is all that is needed to
measure bpp.  Real entropy coding (``compress``/``decompress``) is intentionally
not implemented here.
"""

import os

import torch
import torch.nn as nn
import numpy as np

from compressai.entropy_models import GaussianConditional
from compressai.models.base import CompressionModel
from compressai.ops import quantize_ste

from compressor.ckbd import ckbd_anchor, ckbd_merge, ckbd_nonanchor, ckbd_split
from compressor.compression_modules import (
    ChannelContextEX,
    Decoder,
    Encoder,
    EntropyParametersEX,
    HyperDecoder,
    HyperEncoder,
    VectorQuantiser,
)
from compressor.layer import Quantconv

QUANT_FILES = ("quant.pth", "post_quant.pth")


class LatentCompressor(CompressionModel):
    def __init__(
        self,
        K=8,
        N=256,
        M=256,
        num_slice=10,
        channel_c=(8, 8, 8, 8, 16, 16, 32, 32, 64, 64),
        cb_size=8192,
        quant_z_channels=4,
        quant_embed_dim=4,
        quant_ckpt=None,
    ):
        super().__init__()
        channel_c = list(channel_c)
        assert len(channel_c) == num_slice, (
            f"`channel_c` has {len(channel_c)} entries but `num_slice` is {num_slice}"
        )

        self.slice_num = num_slice
        self.slice_ch = channel_c

        self.encoder = Encoder(K, M)
        self.hyper_enc = HyperEncoder(N, M)
        self.hyper_dec = HyperDecoder(N, M)
        self.decoder = Decoder(M)
        self.out = nn.Conv2d(M, K, 3, 1, 1)

        self.local_context = nn.ModuleList(
            nn.Conv2d(
                in_channels=channel_c[i],
                out_channels=channel_c[i] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for i in range(len(channel_c))
        )

        self.channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(channel_c[:i]), out_dim=channel_c[i] * 2)
            if i
            else None
            for i in range(num_slice)
        )

        # Use channel_ctx and hyper_params
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + channel_c[i] * 2, out_dim=channel_c[i] * 2)
            if i
            else EntropyParametersEX(in_dim=M * 2, out_dim=channel_c[i] * 2)
            for i in range(num_slice)
        )

        # Entropy parameters for non-anchors: spatial ctx + channel ctx + hyper params
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(
                in_dim=M * 2 + channel_c[i] * 4, out_dim=channel_c[i] * 2
            )
            if i
            else EntropyParametersEX(
                in_dim=M * 2 + channel_c[i] * 2, out_dim=channel_c[i] * 2
            )
            for i in range(num_slice)
        )

        self.codebook_size = cb_size
        self.quantize = VectorQuantiser(self.codebook_size, N, contras_loss=True)
        self.gaussian_conditional = GaussianConditional(None)

        self.quant = Quantconv(quant_z_channels, quant_embed_dim).eval()
        for param in self.quant.parameters():
            param.requires_grad = False
        #: True once ``quant.pth``/``post_quant.pth`` have been restored, i.e.
        #: when a checkpoint without ``compress_model.quant.*`` may keep them.
        self.quant_loaded = False
        if quant_ckpt is not None:
            self._load_quant(quant_ckpt)

    def _load_quant(self, quant_ckpt):
        """Load the two 1x1 projections of the quantiser stage from disk.

        They are otherwise restored from the main checkpoint
        (``compress_model.quant.*``), so this is only needed if the checkpoint
        does not carry them.
        """
        state = {}
        for name in QUANT_FILES:
            path = quant_ckpt
            if os.path.isdir(quant_ckpt):
                path = os.path.join(quant_ckpt, name)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Quantiser weights not found: {path}. Either pass a directory "
                    f"containing {QUANT_FILES} or omit `quant_ckpt` and rely on the "
                    f"main checkpoint."
                )
            state.update(torch.load(path, map_location="cpu"))
        self.quant.load_state_dict(state)
        self.quant_loaded = True

    def forward(self, x):
        y = self.encoder(x)
        z = self.hyper_enc(y)
        z_q, emb_loss, _ = self.quantize(z)

        hyper_params = self.hyper_dec(z_q)

        y_slices = [
            y[:, sum(self.slice_ch[:i]): sum(self.slice_ch[: (i + 1)]), ...]
            for i in range(len(self.slice_ch))
        ]
        y_hat_slices = []
        y_likelihoods = []

        num_bits = (
            np.log2(self.codebook_size) * z_q.shape[-2] * z_q.shape[-1] * z_q.shape[0]
        )
        z_likelihoods = torch.tensor(num_bits, device=z_q.device)

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)

            if idx == 0:
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](
                    torch.cat([channel_ctx, hyper_params], dim=1)
                )

            scales_anchor, means_anchor = params_anchor.chunk(2, 1)
            scales_anchor = ckbd_anchor(scales_anchor)
            means_anchor = ckbd_anchor(means_anchor)
            slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor

            # Non-anchor symbols are conditioned on the reconstructed anchor
            # (local context) and, from the second slice on, on the channel context.
            local_ctx = self.local_context[idx](slice_anchor)
            if idx == 0:
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, hyper_params], dim=1)
                )
            else:
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, channel_ctx, hyper_params], dim=1)
                )

            scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
            scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
            means_nonanchor = ckbd_nonanchor(means_nonanchor)

            scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
            means_slice = ckbd_merge(means_anchor, means_nonanchor)

            _, y_slice_likelihoods = self.gaussian_conditional(
                y_slice, scales_slice, means_slice
            )

            slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
            y_hat_slices.append(slice_anchor + slice_nonanchor)
            y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)

        guide_hint = self.decoder(y_hat)
        output = self.out(guide_hint)

        # 1x1 projections that map the residual latent into the diffusion
        # latent space (the "quantised" latent consumed by the sampler).
        post = self.quant.encode(output)

        return {
            "y_hat": y_hat,
            "x_hat": post.mode(),
            "likelihoods": {"y": y_likelihoods},
            "codebook_loss": emb_loss,
            "z_bits": z_likelihoods,
        }

    def load_state_dict(self, state_dict, strict=False):
        result = super().load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            raise RuntimeError(
                "Latent compressor weights are incomplete "
                f"({len(result.missing_keys)} missing keys, e.g. "
                f"{result.missing_keys[:3]}). The checkpoint does not match "
                "this module; refusing to run with partially random weights."
            )
        return result
