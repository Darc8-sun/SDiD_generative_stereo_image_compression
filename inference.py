"""Evaluate SDiD on a stereo image dataset.

Example:
    python inference.py \
        --ckpt /path/to/sdid.ckpt \
        --data_root /path/to/Instereo2K \
        --save_dir results/instereo2k

Pass ``--sd_ckpt`` to read the frozen Stable Diffusion weights
(``first_stage_model.*`` / ``model.diffusion_model.*``) from a separate file
instead of ``--ckpt``.
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from ldm.util import instantiate_from_config
from sdid.color_fix import adaptive_instance_normalization
from sdid.dataset import PROMPT, StereoImageDataset
from sdid.model import CONFIG_DIR, configure_quant_ckpt
from utils.checkpoint import (
    load_checkpoint_file,
    load_split_state_dict,
    load_state_dict,
)
from utils.meters import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="SDiD stereo image compression")
    parser.add_argument(
        "--ckpt",
        required=True,
        help="path to the SDiD checkpoint with the trained weights "
        "(control_model, aux_view_trans, compress_model, tar)",
    )
    parser.add_argument(
        "--sd_ckpt",
        default='./ckpt/sd_21.ckpt',
        help="optional path to the frozen Stable Diffusion weights "
        "(first_stage_model.* / model.diffusion_model.*, e.g. sd-v2-1.ckpt); "
        "when omitted, both groups are read from --ckpt",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="dataset root containing <split>/left and <split>/right",
    )
    parser.add_argument("--split", default="test", help="dataset split (default: test)")
    parser.add_argument(
        "--save_dir", default="results", help="where reconstructions are written"
    )
    parser.add_argument(
        "--clip_pretrained",
        default="./ckpt/clip/open_clip_pytorch_model.bin",
        help="OpenCLIP ViT-H-14 weights: HF tag or path to a local .bin "
        "(default: value from configs/sdid_inference.yaml)",
    )
    parser.add_argument(
        "--quant_ckpt",
        default='./ckpt/',
        help="directory with quant.pth/post_quant.pth, only if the checkpoint "
        "does not carry compress_model.quant.*",
    )
    parser.add_argument("--steps", type=int, default=None, help="diffusion steps")
    parser.add_argument(
        "--header_bits",
        type=float,
        default=4,
        help="side-information cost of transmitting the ANE time step, in bits",
    )
    parser.add_argument("--prompt", default=PROMPT, help="text conditioning")
    parser.add_argument("--device", default="cuda:0", help="torch device, e.g. cpu")
    parser.add_argument(
        "--max_images", type=int, default=None, help="evaluate only the first N pairs"
    )
    parser.add_argument(
        "--no_save_images", action="store_true", help="skip writing PNGs"
    )
    return parser.parse_args()


def tensor_to_uint8(tensor):
    image = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def compute_bpp(compress_dict, num_pixels):
    return sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in compress_dict["likelihoods"].values()
    ).item()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU (this will be slow).")
        args.device = "cpu"
    device = torch.device(args.device)

    config = OmegaConf.load(CONFIG_DIR / "sdid_inference.yaml")
    if args.clip_pretrained is not None:
        config["params"]["cond_stage_config"]["params"]["version"] = args.clip_pretrained
    if args.quant_ckpt is not None:
        configure_quant_ckpt(args.quant_ckpt)

    model = instantiate_from_config(config)
    ckpt = load_checkpoint_file(args.ckpt)
    if args.sd_ckpt is None:
        load_state_dict(model, ckpt)
    else:
        load_split_state_dict(model, load_checkpoint_file(args.sd_ckpt), ckpt)
    model.to(device)
    model.freeze()

    steps = args.steps if args.steps is not None else model.sample_steps
    dataset = StereoImageDataset(args.data_root, split=args.split, prompt=args.prompt)

    from DISTS_pytorch import DISTS
    from lpips import LPIPS

    lpips_fn = LPIPS(net="alex").to(device).eval()
    dists_fn = DISTS().to(device).eval()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    meters = {key: AverageMeter() for key in
              ("bpp", "bpp_aux", "psnr", "psnr_aux", "lpips", "lpips_aux", "dists", "dists_aux")}
    records = []

    num_images = len(dataset) if args.max_images is None else min(len(dataset), args.max_images)
    with torch.no_grad():
        for index in range(num_images):
            batch = dataset[index]
            left, right = batch["img"][0].unsqueeze(0).to(device), batch["img"][1].unsqueeze(0).to(device)
            num_pixels = left.size(2) * left.size(3)

            # --- encoding end: both views are compressed independently ---
            posterior, fea_out, _ = model.first_stage_model.encode(left, return_fea=True)
            compress_dict = model.compress_model(fea_out)
            z = model.get_first_stage_encoding(posterior, sample=False)
            x_hat = model.get_first_stage_encoding(compress_dict["x_hat"])

            _, fea_out_aux, _ = model.first_stage_model.encode(right, return_fea=True)
            compress_dict_aux = model.compress_model(fea_out_aux)
            x_hat_aux = model.get_first_stage_encoding(compress_dict_aux["x_hat"])

            # --- decoding end: ANE chooses the start step, DPA/DPT reconstruct ---
            ctx = model.get_learned_conditioning(batch["prompt"])
            start_t = model.Get_index_fromSNR(x_hat, z)
            preds = model.sample_log(
                cond={"c_concat": [x_hat], "c_crossattn": [ctx], "aux_c": x_hat_aux},
                steps=steps,
                txt=batch["prompt"],
                start_t=start_t,
            )
            preds[0], n_bit_0 = adaptive_instance_normalization(preds[0], (left + 1) / 2, bits=8)
            preds[1], n_bit_1 = adaptive_instance_normalization(preds[1], (right + 1) / 2, bits=8)

            bpp = compute_bpp(compress_dict, num_pixels) +(n_bit_0+args.header_bits)/num_pixels
            bpp_aux = compute_bpp(compress_dict_aux, num_pixels) + (n_bit_1+args.header_bits)/num_pixels


            rec_left, rec_right = tensor_to_uint8(preds[0]), tensor_to_uint8(preds[1])
            ref_left, ref_right = tensor_to_uint8((left + 1) / 2), tensor_to_uint8((right + 1) / 2)

            psnr = compare_psnr(ref_left, rec_left, data_range=255)
            psnr_aux = compare_psnr(ref_right, rec_right, data_range=255)
            lpips_value = lpips_fn(left, preds[0] * 2 - 1).item()
            lpips_aux = lpips_fn(right, preds[1] * 2 - 1).item()
            dists_value = dists_fn((left + 1) / 2, preds[0]).item()
            dists_aux = dists_fn((right + 1) / 2, preds[1]).item()

            meters["bpp"].update(bpp)
            meters["bpp_aux"].update(bpp_aux)
            meters["psnr"].update(psnr)
            meters["psnr_aux"].update(psnr_aux)
            meters["lpips"].update(lpips_value)
            meters["lpips_aux"].update(lpips_aux)
            meters["dists"].update(dists_value)
            meters["dists_aux"].update(dists_aux)
            records.append(
                {
                    "name": batch["name"],
                    "bpp": bpp,
                    "bpp_aux": bpp_aux,
                    "psnr": psnr,
                    "psnr_aux": psnr_aux,
                    "lpips": lpips_value,
                    "lpips_aux": lpips_aux,
                    "dists": dists_value,
                    "dists_aux": dists_aux,
                }
            )

            if not args.no_save_images:
                plt.imsave(os.path.join(save_dir, f"{batch['name']}_left.png"), rec_left)
                plt.imsave(os.path.join(save_dir, f"{batch['name']}_right.png"), rec_right)

            print(
                f"[{index + 1}/{num_images}] {batch['name']}: "
                f"bpp {bpp:.4f}/{bpp_aux:.4f}  "
                f"psnr {psnr:.2f}/{psnr_aux:.2f}  "
                f"lpips {lpips_value:.3f}/{lpips_aux:.3f}  "
                f"dists {dists_value:.3f}/{dists_aux:.3f}"
            )

    summary = {key: meter.avg for key, meter in meters.items()}
    summary["num_pairs"] = num_images
    print("\n=== mean over {} stereo pairs ===".format(num_images))
    print(
        "bpp {bpp:.4f} | psnr {psnr:.2f} | lpips {lpips:.4f} | dists {dists:.4f}".format(**summary)
    )
    print(
        "aux: bpp {bpp_aux:.4f} | psnr {psnr_aux:.2f} | lpips {lpips_aux:.4f} | "
        "dists {dists_aux:.4f}".format(**summary)
    )

    with open(os.path.join(save_dir, "metrics.json"), "w") as handle:
        json.dump({"summary": summary, "per_image": records}, handle, indent=2)
    print(f"\nResults written to {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()
