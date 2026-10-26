"""Checkpoint loading with explicit failure instead of silent partial loads.

SDiD weights can live in one file or in two:

* the frozen Stable Diffusion part -- ``first_stage_model.*`` (VAE) and
  ``model.diffusion_model.*`` (UNet);
* the trained SDiD part -- ``control_model.*``, ``tar.*``,
  ``aux_view_trans.*`` and ``compress_model.*``.

``load_state_dict`` keeps the single-file behaviour, ``load_split_state_dict``
restores the two groups from separate files.

The original research code called ``load_state_dict(..., strict=False)`` three
times in a row, which happily produces noise images when the checkpoint and the
model disagree.  Every key group that matters for SDiD is verified here, and a
missing group raises instead of degrading quietly.
"""

from collections import OrderedDict

import torch

# Frozen Stable Diffusion 2.1 weights, taken verbatim from the pretrained prior.
FROZEN_PREFIXES = (
    "first_stage_model.",
    "model.diffusion_model.",
)

# Weights trained by SDiD: control encoder, DPA, DPT and latent compressor.
TRAINABLE_PREFIXES = (
    "control_model.",
    "tar.",
    "aux_view_trans.",
    "compress_model.",
)

# Key groups of a usable single-file SDiD checkpoint.
REQUIRED_PREFIXES = FROZEN_PREFIXES + TRAINABLE_PREFIXES

COMPRESS_PREFIX = "compress_model."
QUANT_PREFIX = "quant."


def load_checkpoint_file(path, map_location="cpu"):
    """``torch.load`` that also works on checkpoints holding Python objects.

    torch >= 2.6 defaults ``weights_only=True``, which rejects Lightning
    checkpoints that store hyper-parameters next to the tensors.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13 has no `weights_only`
        return torch.load(path, map_location=map_location)


def _strip_dataparallel_prefix(state_dict):
    if len(state_dict) and all(k.startswith("module.") for k in state_dict):
        return OrderedDict((k[len("module."):], v) for k, v in state_dict.items())
    return state_dict


def _unwrap_state_dict(state_dict):
    """Unwrap Lightning's ``state_dict`` entry and DataParallel prefixes."""
    return _strip_dataparallel_prefix(state_dict.get("state_dict", state_dict))


def _keep_quant_from_module(model, compress_state_dict, verbose, label):
    """Fill in ``compress_model.quant.*`` when the checkpoint does not carry it.

    ``--quant_ckpt`` restores ``quant.pth`` / ``post_quant.pth`` while the model
    is built, so the values already sitting in the module are the ones to keep.
    Without that fallback ``LatentCompressor.load_state_dict`` would report the
    projections as missing, i.e. the split checkpoint could not be loaded at all.
    """
    expected = [
        key
        for key in model.compress_model.state_dict()
        if key.startswith(QUANT_PREFIX)
    ]
    missing = [key for key in expected if key not in compress_state_dict]
    if not missing:
        return
    if not getattr(model.compress_model, "quant_loaded", False):
        raise RuntimeError(
            f"The {label} has no `{COMPRESS_PREFIX}{QUANT_PREFIX}*` tensors "
            f"({len(missing)} keys) and no quant.pth/post_quant.pth was loaded "
            "either. Pass `--quant_ckpt` (or call `configure_quant_ckpt`) so that "
            "the quantiser projections are restored."
        )
    current = model.compress_model.quant.state_dict()
    for key in missing:
        compress_state_dict[key] = current[key[len(QUANT_PREFIX):]]
    if verbose:
        print(
            f"[checkpoint] {len(missing)} `{COMPRESS_PREFIX}{QUANT_PREFIX}*` "
            "tensors are not in the checkpoint; keeping the weights loaded from "
            "quant.pth/post_quant.pth."
        )


def _restore(model, state_dict, prefixes, label, verbose=True):
    """Load the tensors owned by ``prefixes`` from ``state_dict`` into ``model``.

    Tensors outside ``prefixes`` are ignored, so one file can carry more than
    SDiD needs (a vanilla SD 2.1 checkpoint, for instance, also stores
    ``cond_stage_model.*`` and the training state).
    """
    subset = OrderedDict(
        (key, value) for key, value in state_dict.items() if key.startswith(prefixes)
    )
    if not subset:
        groups = ", ".join(f"`{prefix}*`" for prefix in prefixes)
        raise RuntimeError(
            f"The {label} contains none of {groups}. This does not look like the "
            "file SDiD expects."
        )

    # The compressor has its own strict loader, so it is restored separately.
    compress_state_dict = OrderedDict()
    for key in [key for key in subset if key.startswith(COMPRESS_PREFIX)]:
        compress_state_dict[key[len(COMPRESS_PREFIX):]] = subset.pop(key)

    if compress_state_dict:
        _keep_quant_from_module(model, compress_state_dict, verbose, label)
        # Raises if the compressor is only partially restored (see
        # LatentCompressor.load_state_dict).
        model.compress_model.load_state_dict(compress_state_dict)
    elif COMPRESS_PREFIX in prefixes:
        raise RuntimeError(
            f"No `{COMPRESS_PREFIX}*` weights found in the {label}. SDiD needs the "
            "latent compressor to be restored together with the diffusion prior."
        )

    provided = set(subset) | {
        COMPRESS_PREFIX + key for key in compress_state_dict
    }

    result = model.load_state_dict(subset, strict=False)

    for prefix in prefixes:
        if not any(key.startswith(prefix) for key in provided):
            raise RuntimeError(
                f"The {label} has no `{prefix}*` tensors. This does not look like "
                "an SDiD checkpoint."
            )
        # `provided` also covers the compressor keys, which were loaded above.
        missing = [
            key
            for key in result.missing_keys
            if key.startswith(prefix) and key not in provided
        ]
        if missing:
            preview = ", ".join(missing[:3])
            raise RuntimeError(
                f"{len(missing)} `{prefix}*` tensors of the model were not restored "
                f"from the {label} (e.g. {preview}); the checkpoint layout does not "
                "match the model."
            )

    if verbose:
        print(
            f"[checkpoint] {label}: restored {len(provided)} tensors "
            f"(unexpected {len(result.unexpected_keys)})."
        )
        if result.unexpected_keys:
            preview = ", ".join(result.unexpected_keys[:5])
            print(f"[checkpoint] unexpected keys in the {label} (first 5): {preview}")

    return result


def load_state_dict(model, state_dict, verbose=True):
    """Load a single-file SDiD checkpoint (frozen and trained groups together).

    Args:
        model: an ``sdid.model.SDiD`` instance (already instantiated, with
            ``model.compress_model`` attached).
        state_dict: the raw object returned by ``torch.load``; the Lightning
            ``state_dict`` wrapper is unwrapped transparently.
    """
    return _restore(
        model,
        _unwrap_state_dict(state_dict),
        REQUIRED_PREFIXES,
        label="checkpoint",
        verbose=verbose,
    )


def load_split_state_dict(model, frozen_state_dict, trainable_state_dict, verbose=True):
    """Load SDiD from two checkpoints (frozen prior + trained modules).

    Args:
        model: an ``sdid.model.SDiD`` instance (already instantiated, with
            ``model.compress_model`` attached).
        frozen_state_dict: raw object returned by ``torch.load`` for the frozen
            Stable Diffusion weights; only ``first_stage_model.*`` and
            ``model.diffusion_model.*`` are read, so a vanilla SD 2.1
            checkpoint works as well.
        trainable_state_dict: raw object returned by ``torch.load`` for the
            trained SDiD weights (``control_model.*``, ``tar.*``,
            ``aux_view_trans.*`` and ``compress_model.*``).
    """
    frozen = _restore(
        model,
        _unwrap_state_dict(frozen_state_dict),
        FROZEN_PREFIXES,
        label="frozen SD checkpoint",
        verbose=verbose,
    )
    trainable = _restore(
        model,
        _unwrap_state_dict(trainable_state_dict),
        TRAINABLE_PREFIXES,
        label="trained SDiD checkpoint",
        verbose=verbose,
    )
    return frozen, trainable
