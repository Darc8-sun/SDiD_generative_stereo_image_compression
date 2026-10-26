"""SDiD model: shared diffusion prior for distributed stereo image compression.

Two views are encoded independently by the shared latent compressor.  At the
decoder side the main view is reconstructed by a controllable diffusion prior
(DPA + ANE) and the auxiliary view is produced by transferring that prior
(DPT + Look-Back Attention).  See `SDiD_ICML26.pdf` for details.
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from compressor.entropy_model import LatentCompressor
from ldm.models.latent_diffusion import LatentDiffusion
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import extract_into_tensor, timestep_embedding
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
from sdid.prior_modules import (
    Time_aware_refinementS,
    Timeaware_PriorTransfer,
    append_dims,
    ddim_timesteps,
    scalings_for_boundary_conditions,
)
from sdid.sampler import StereoSampler

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
COMPRESSOR_CONFIG = OmegaConf.load(CONFIG_DIR / "latent_compressor.yaml")

#: Number of DDIM sub-steps used to discretise the ANE search space (Sec. 3.3).
ANE_NUM_BINS = 50


def configure_quant_ckpt(path):
    """Point the latent compressor at a directory holding quant.pth/post_quant.pth.

    Only needed when the main checkpoint does not carry ``compress_model.quant.*``.
    Must be called before the model is instantiated.
    """
    COMPRESSOR_CONFIG["compress_base"]["quant_ckpt"] = path


class ControlledUnetModel(UNetModel):
    """UNet whose decoder blocks receive ControlNet residuals."""

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        ori=None,
        **kwargs,
    ):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)
        if control is not None:
            h += control.pop()

        for module in self.output_blocks:
            hs_ = hs.pop()
            if only_mid_control or control is None:
                h = torch.cat([h, hs_], dim=1)
            else:
                h = torch.cat([h, hs_ + control.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        return self.out(h)


class ControlEncoder(UNetModel):
    """The encoding half of the UNet: returns the multi-level control residuals."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self._modules["output_blocks"]
        del self._modules["out"]

    def forward(self, x, timesteps=None, context=None, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        hs.append(h)
        return hs


class SDiD(LatentDiffusion):
    """Distributed stereo image compression with a shared diffusion prior."""

    def __init__(
        self,
        control_stage_config,
        control_key="hint",
        only_mid_control=False,
        samplesteps=2,
        progressive=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Diffusion prior alignment (DPA) and diffusion prior transfer (DPT).
        self.tar = Time_aware_refinementS(in_ch=4, out_ch=4, model_channels=320).eval()
        self.aux_view_trans = Timeaware_PriorTransfer(
            in_ch=4, out_ch=4, model_channels=320
        ).eval()
        self.control_model = instantiate_from_config(control_stage_config)

        self.sample_steps = samplesteps
        self.progressive = progressive
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        # ANE: the SNR table restricted to the 50-step DDIM grid, sorted so that
        # `bucketize` returns the closest achievable time step.
        ddim_t = ddim_timesteps(self.num_timesteps, ANE_NUM_BINS)
        self.snr_select, self.snr_index = self.SNR_diff[ddim_t].sort()

        self.sampler = StereoSampler(self, ddim_samplestep=ANE_NUM_BINS)

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.compress_model = LatentCompressor(**COMPRESSOR_CONFIG["compress_base"])
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x, return_fea=True)

    def get_first_stage_encoding(self, encoder_posterior, sample=True):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample() if sample else encoder_posterior.mode()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    def Get_index_fromSNR(self, pred, org):
        """Adaptive noise-level estimation: infer the time step of a compressed latent.

        ``pred`` is the compressed main-view latent, ``org`` the lossless one.
        Their SNR is bucketed into the DDPM forward SNR table (Eq. 9-11).
        """
        snr_d = torch.square(org).sum([1, 2, 3]) / torch.square(pred - org).sum([1, 2, 3])
        index_n = torch.bucketize(snr_d.detach().cpu(), self.snr_select)
        index_n = torch.where(index_n > ANE_NUM_BINS - 1, ANE_NUM_BINS - 1, index_n)
        del snr_d
        return self.snr_index[index_n].numpy()

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond["c_crossattn"], 1)
        if cond["c_concat"] is None:
            return diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                only_mid_control=self.only_mid_control,
            )
        control = self.control_model(
            x=cond["c_concat"][0], timesteps=t, context=cond_txt
        )
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        return diffusion_model(
            x=x_noisy,
            control=control,
            timesteps=t,
            context=cond_txt,
            only_mid_control=self.only_mid_control,
        )

    def predict_x0(self, x_t, t, eps):
        x0_hat = (
            x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps
        ) / extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)
        return x0_hat

    def build_cm_function(self, x_noisy, x_pred, t):
        """Boundary-condition mixing of the current sample and the x0 prediction."""
        c_skip, c_out = scalings_for_boundary_conditions(t)
        c_skip, c_out = [append_dims(x, x_noisy.ndim) for x in [c_skip, c_out]]
        return c_skip * x_noisy + c_out * x_pred

    @torch.no_grad()
    def sample_latent(self, cond, steps, txt, start_t=50):
        b, c, h, w = cond["c_concat"][0].shape
        shape = (b, self.channels, h, w)
        return self.sampler.sample_latent(
            steps,
            shape,
            cond["c_concat"][0],
            cond["aux_c"],
            cond["c_crossattn"][0],
            start_t_ind=start_t,
            progressive=self.progressive,
        )

    @torch.no_grad()
    def sample_log(self, cond, steps, txt, start_t=50):
        """Run the sampler once and decode both views.

        The original research code called ``sample_latent`` twice in a single
        expression to fetch ``[0]`` and ``[1]``, doubling the decoding cost for
        identical results.
        """
        latents = self.sample_latent(cond, steps, txt, start_t)
        return [
            (self.decode_first_stage(latents[0]) + 1) / 2,
            (self.decode_first_stage(latents[1]) + 1) / 2,
        ]
