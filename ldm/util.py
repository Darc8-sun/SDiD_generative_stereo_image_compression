"""Small helpers shared across the model code.

Trimmed from the Latent-Diffusion reference implementation
(https://github.com/CompVis/latent-diffusion, MIT license).
"""

import importlib

__conditioning_keys__ = {
    "concat": "c_concat",
    "crossattn": "c_crossattn",
    "adm": "y",
}


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        if config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module
