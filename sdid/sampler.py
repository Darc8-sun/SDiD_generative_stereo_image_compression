"""Two-step stereo sampler.

Implements the sampling loop used by SDiD: ANE provides the start time step
(see ``SDiD.Get_index_fromSNR``), the first view is denoised with the
diffusion prior + DPA, and the second view is produced by DPT from the same
trajectory.  Both views are returned in one pass.

``space_timesteps`` follows the IDDPM / guided-diffusion schedule (Apache-2.0,
OpenAI guided-diffusion).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ldm.modules.diffusionmodules.util import make_beta_schedule


def space_timesteps(num_timesteps, section_counts):
    """Create a list of timesteps to use from an original diffusion process.

    Given the number of timesteps we want to take from equally-sized portions of
    the original process, returns the set of step indices to keep.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def _extract_into_tensor(arr, t, x_shape):
    """Gather per-sample values from a 1-D array, then broadcast over ``x_shape``.

    ``t`` carries the index into the DDIM sub-step table; it is widened to 2-D
    only when ``arr`` itself is multi-dimensional (matching the batched schedule
    layout).
    """
    if arr.dim() - 1:
        t = t.unsqueeze(dim=1)
    b, *_ = t.shape
    out = arr.gather(0, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class StereoSampler:
    """Runs the aligned prior for both views in a single trajectory."""

    def __init__(self, model, schedule: str = "linear", ddim_samplestep: int = 50):
        self.model = model
        self.original_num_steps = model.num_timesteps
        self.schedule = schedule
        self.ddim_samplestep = ddim_samplestep

    def make_schedule(self, num_steps: int, start_t_ind: int = 50) -> None:
        betas = make_beta_schedule(
            self.schedule,
            self.original_num_steps,
            linear_start=self.model.linear_start,
            linear_end=self.model.linear_end,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        step_ratio = self.original_num_steps // self.ddim_samplestep
        self.ddim_timesteps = (
            np.arange(1, self.ddim_samplestep + 1) * step_ratio
        ).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alphas_cumprod[self.ddim_timesteps]
        self.ddim_timesteps_prev = np.asarray([0] + self.ddim_timesteps[:-1].tolist())

        start_t_ind = int(start_t_ind)
        # `start_t_ind` is the ANE estimate; `num_steps` sub-steps are placed
        # between it and t=0 with the IDDPM spacing.
        indices = torch.tensor(
            sorted(list(space_timesteps(start_t_ind, str(num_steps))))
        )
        self.timesteps = self.ddim_timesteps[indices.numpy()]
        self.ind = indices
        self.ddim_alpha_cumprods_prev = torch.from_numpy(
            np.asarray(alphas_cumprod[np.asarray([0] + self.timesteps[:-1].tolist())])
        )

    def ddim_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha_cumprod_prev = _extract_into_tensor(
            self.ddim_alpha_cumprods_prev.to(x_start.device), t, x_start.shape
        )
        if t[0] == 0:
            return x_start
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * noise
        return (alpha_cumprod_prev.sqrt() * x_start + dir_xt).float()

    def p_sample(
        self,
        x: List[torch.Tensor],
        cond: Dict[str, torch.Tensor],
        t: torch.Tensor,
        index: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """One denoising step: prior + DPA for the main view, DPT for the aux view."""
        model_output = self.model.apply_model(x[0], t, cond)
        e_t = self.model.predict_x0(x[0], t, model_output)
        e_t = self.model.tar(e_t, cond["decompress_result"][0], t)
        e_t = self.model.build_cm_function(x[0], e_t, t)
        e_t_aux = self.model.aux_view_trans(
            x[1], [cond["aux_c"], cond["decompress_result"][0]], e_t, t
        )
        pred = self.ddim_sample(e_t, index, model_output)
        return [pred, e_t_aux], e_t

    @torch.no_grad()
    def sample_latent(
        self,
        steps: int,
        shape: Tuple[int, ...],
        cond_img: torch.Tensor,
        aux_img: torch.Tensor,
        ctx: torch.Tensor,
        start_t_ind: int = 50,
        x_T: Optional[torch.Tensor] = None,
        progressive: bool = False,
    ) -> List[torch.Tensor]:
        """Denoise both views.

        Args:
            cond_img: compressed main-view latent (also the ANE input).
            aux_img: compressed auxiliary-view latent, used as DPT's side info.
            ctx: text conditioning, shape (B, 77, width).
            start_t_ind: time step chosen by the adaptive noise-level estimator.
        """
        start_t_ind = int(np.asarray(start_t_ind).reshape(-1)[0])
        start_t_ind = max(start_t_ind, steps)
        self.make_schedule(num_steps=steps, start_t_ind=start_t_ind)

        device = next(self.model.parameters()).device
        if start_t_ind == 50:
            # No ANE estimate available: fall back to a Gaussian initialisation.
            noise = torch.randn(shape, device=device) if x_T is None else x_T
            img = [noise, aux_img]
        else:
            img = [cond_img, aux_img]

        time_range = np.flipud(self.timesteps)
        total_steps = len(self.timesteps)

        cond = {
            "c_concat": [cond_img],
            "c_crossattn": [ctx],
            "decompress_result": [cond_img],
            "aux_c": aux_img,
        }

        for i, step in enumerate(time_range):
            ts = torch.full((shape[0],), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)
            img, e_t = self.p_sample(img, cond, ts, index=index)
            if progressive:
                cond["c_concat"] = [e_t]

        return img
