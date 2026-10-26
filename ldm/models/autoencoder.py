"""KL-regularised autoencoder (the "latent encoder/decoder" E(.) of the paper).

Trimmed from the Latent-Diffusion reference implementation
(https://github.com/CompVis/latent-diffusion, MIT license).

In SDiD the VAE is frozen and shared by every view.  ``encode(..., return_fea=True)``
additionally returns the multi-scale features that feed the latent compressor.
"""

import torch.nn as nn

from ldm.modules.diffusionmodules.model import Decoder, Encoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config


class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig, lossconfig, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x, return_fea=False):
        if return_fea:
            h, fea_list = self.encoder(x, return_fea)
        else:
            h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior if not return_fea else (posterior, h, fea_list)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior
