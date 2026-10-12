"""Shared selection for optional FP32 structure-network kernels."""

import importlib.util
import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def supported_runtime():
    if importlib.util.find_spec("triton") is None:
        return False
    try:
        import triton
        version = tuple(int(part) for part in triton.__version__.split(".")[:2])
        return version >= (3, 3)
    except (ImportError, AttributeError, ValueError):
        return False


def structure_backend():
    backend = os.environ.get("EM3DFOLD_STRUCTURE_BACKEND", "auto").lower()
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("EM3DFOLD_STRUCTURE_BACKEND must be auto, torch or triton")
    return backend


def use_triton(module, tensor):
    return (
        module.inference_backend != "torch" and not module.training
        and not torch.is_grad_enabled() and not torch.is_autocast_enabled() and tensor.is_cuda
        and tensor.dtype == torch.float32 and torch.version.cuda is not None
        and supported_runtime()
        and torch.cuda.get_device_capability(tensor.device)[0] >= 8
    )
