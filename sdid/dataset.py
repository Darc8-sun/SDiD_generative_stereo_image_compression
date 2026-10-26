"""Dataset for stereo image compression benchmarks.

Expects the directory layout::

    <root>/<split>/left/*.png
    <root>/<split>/right/*.png

e.g. InStereo2K:  ``data/Instereo2K/test/left/0001.png`` ...
     Cityscapes:  ``data/cityscapes/val/left/*.png`` (left/right view folders)

Both views are normalised to ``[-1, 1]``.  Images are centre-cropped to a
multiple of 64 because the VAE downsamples by 2^4 and the latent compressor
splits the 8-channel latent into 2x2 checkerboard groups.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROMPT = "high-resolution, best quality, and extremely detail"


class MinimalCrop:
    """Centre-crop to the largest size divisible by ``min_div``."""

    def __init__(self, min_div=16):
        self.min_div = min_div

    def __call__(self, image):
        w, h = image.size
        h_new = h - (h % self.min_div)
        w_new = w - (w % self.min_div)
        if h_new == 0 and w_new == 0:
            return image
        h_diff, w_diff = h - h_new, w - w_new
        top = int(h_diff / 2)
        bottom = h_diff - top
        left = int(w_diff / 2)
        right = w_diff - left
        return image.crop((left, top, w - right, h - bottom))


class StereoImageDataset(Dataset):
    """Paired left/right images, returned as a 3-channel tensor pair."""

    def __init__(self, root, split="test", min_div=64, prompt=PROMPT):
        super().__init__()
        self.path = Path(root)
        self.split = split
        self.prompt = prompt
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 6, std=[0.5] * 6),
            ]
        )
        self.crop = MinimalCrop(min_div=min_div)
        self.left_image_list, self.right_image_list = self.get_files()
        print(
            f"Loaded {len(self.left_image_list)} stereo pairs from "
            f"{self.path / split}."
        )

    def __len__(self):
        return len(self.left_image_list)

    def __getitem__(self, index):
        images = [
            Image.open(self.left_image_list[index]).convert("RGB"),
            Image.open(self.right_image_list[index]).convert("RGB"),
        ]
        images = [self.crop(image) for image in images]
        if images[0].size != images[1].size:
            raise ValueError(
                f"Left/right views of {self.left_image_list[index]} have different "
                f"sizes ({images[0].size} vs {images[1].size}) after cropping."
            )
        frames = np.concatenate([np.asarray(image) for image in images], axis=-1)
        left, right = torch.chunk(self.transform(frames), 2)
        return {
            "img": (left, right),
            "prompt": self.prompt,
            "name": os.path.splitext(os.path.basename(self.left_image_list[index]))[0],
        }

    def get_files(self):
        split_dir = self.path / self.split
        left_dir, right_dir = split_dir / "left", split_dir / "right"
        if not left_dir.is_dir() or not right_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {left_dir} and {right_dir} to exist. The dataset root "
                "should contain `<split>/left` and `<split>/right` sub-folders."
            )
        names = sorted(
            f for f in os.listdir(left_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        left = [str(left_dir / n) for n in names]
        right = [str(right_dir / n) for n in names]
        missing = [n for n in names if not (right_dir / n).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} right-view images are missing, e.g. {missing[0]}"
            )
        return left, right
