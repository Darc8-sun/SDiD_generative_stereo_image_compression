"""Adaptive instance normalisation used as the final colour correction.

Borrowed from StableSR (https://github.com/pkuliyi2015/sd-webui-stablesr,
colorfix.py by Li Yi).  After denoising, statistics of the reconstruction are
aligned to the decoded compressed image so that luminance/chroma drift does not
show up in the perceptual metrics.
"""

from torch import Tensor
import torch

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Mean and std over the spatial dimensions of a 4D tensor."""
    size = feat.size()
    assert len(size) == 4, "The input feature should be 4D tensor."
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std




class StraightThroughRound(torch.autograd.Function):
    """Straight-Through Estimator (STE) for rounding, keeps gradient flow during training."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_tensor(x: Tensor, num_bits: int = 8,
                    min_val: float = None, max_val: float = None,
                    symmetric: bool = False):
    """Uniform quantization with configurable precision.

    Quantizes a float tensor into discrete levels defined by `num_bits`,
    then dequantizes back to float for downstream use. Gradient is preserved
    via Straight-Through Estimator (STE) during training.

    Args:
        x (Tensor): Input tensor, e.g. style_mean or style_std of shape (B, C, 1, 1).
        num_bits (int): Quantization precision in bits. Number of levels = 2^num_bits.
            Default: 8.
        min_val (float | None): Lower bound of quantization range. If None, auto-computed
            from x.min(). Default: None.
        max_val (float | None): Upper bound of quantization range. If None, auto-computed
            from x.max(). Default: None.
        symmetric (bool): If True, use symmetric range [-R, R] where R = max(|min|, |max|).
            Suitable for signed values like style_mean. Default: False.

    Returns:
        dict:
            - 'x_q'     (Tensor): Dequantized float tensor (same shape as x).
            - 'x_int'   (Tensor): Quantized integer codes (same shape as x, dtype long).
            - 'scale'   (float):  Quantization step size.
            - 'min_val' (float):  Effective lower bound used.
            - 'max_val' (float):  Effective upper bound used.
            - 'num_bits' (int):   Bits used.
    """
    num_levels = 2 ** num_bits

    # --- Determine quantization range ---
    if min_val is None:
        min_val = float(x.min().item())
    if max_val is None:
        max_val = float(x.max().item())

    if symmetric:
        abs_max = max(abs(min_val), abs(max_val))
        min_val, max_val = -abs_max, abs_max

    # Guard against degenerate range
    if max_val == min_val:
        max_val = min_val + 1e-6

    scale = (max_val - min_val) / (num_levels - 1)

    # --- Quantize (STE: forward = round, backward = identity) ---
    x_clamped = x.clamp(min_val, max_val)
    x_scaled  = (x_clamped - min_val) / scale
    x_int     = StraightThroughRound.apply(x_scaled).long()   # integer codes [0, num_levels-1]
    x_q       = x_int.float() * scale + min_val                # dequantize

    return {
        'x_q':      x_q,
        'x_int':    x_int,
        'scale':    scale,
        'min_val':  min_val,
        'max_val':  max_val,
        'num_bits': num_bits,
    }


def quantize_style_params(style_mean: Tensor, style_std: Tensor,
                          num_bits_mean: int = 8, num_bits_std: int = 8,
                          mean_range: tuple = None, std_range: tuple = None):
    """Convenience wrapper to quantize both style_mean and style_std together.

    style_mean is signed  → symmetric quantization by default.
    style_std  is positive → asymmetric (unsigned) quantization by default.

    Args:
        style_mean (Tensor): Shape (B, C, 1, 1), channel-wise mean.
        style_std  (Tensor): Shape (B, C, 1, 1), channel-wise std.
        num_bits_mean (int): Bits for style_mean. Default: 8.
        num_bits_std  (int): Bits for style_std.  Default: 8.
        mean_range (tuple | None): (min_val, max_val) for style_mean. None = auto.
        std_range  (tuple | None): (min_val, max_val) for style_std.  None = auto.

    Returns:
        dict:
            - 'mean' : quantize_tensor result dict for style_mean
            - 'std'  : quantize_tensor result dict for style_std
            - 'total_bits' (int): total number of quantized bits for both tensors
    """
    mean_min = mean_range[0] if mean_range else None
    mean_max = mean_range[1] if mean_range else None
    std_min  = std_range[0]  if std_range  else 0.0   # std is always >= 0
    std_max  = std_range[1]  if std_range  else None

    q_mean = quantize_tensor(style_mean, num_bits=num_bits_mean,
                             min_val=mean_min, max_val=mean_max,
                             symmetric=True)

    q_std  = quantize_tensor(style_std, num_bits=num_bits_std,
                             min_val=std_min, max_val=std_max,
                             symmetric=False)

    total_bits = style_mean.numel() * num_bits_mean + style_std.numel() * num_bits_std

    return {
        'mean':       q_mean,
        'std':        q_std,
        'total_bits': total_bits,
    }






def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor,bits=None):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)

    if bits is not None:
        style = quantize_style_params(style_mean, style_std, num_bits_mean=bits, num_bits_std=bits)
        style_mean, style_std = style['mean']['x_q'], style['std']['x_q']
        bitsforstyle = style['total_bits']
    else:
        bitsforstyle = None
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size), bitsforstyle