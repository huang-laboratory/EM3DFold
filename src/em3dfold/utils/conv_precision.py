"""Optional local FP16 autocast for structure-network Conv3d inference."""

import os

import torch
from torch import nn


def conv_fp16_requested():
    return os.environ.get("EM3DFOLD_CONV_FP16", "0").lower() in ("1", "true", "yes", "on")


class InferenceConv3d(nn.Conv3d):
    """Keep FP32 parameters and return FP32 after a local FP16 convolution."""

    conv_fp16 = True

    def forward(self, x):
        if (self.conv_fp16 and not self.training and not torch.is_grad_enabled()
                and not torch.is_autocast_enabled() and x.is_cuda
                and x.dtype == torch.float32 and self.weight.dtype == torch.float32):
            with torch.autocast("cuda", dtype=torch.float16):
                result = super().forward(x)
            return result.float()
        return super().forward(x)


def enable_structure_conv_fp16(module):
    """Replace only the supplied structure encoder's Conv3d layers.

    Parameters retain their identities and checkpoint keys. The module remains
    pickleable for spawned workers; CPU, training and autograd use native Conv3d.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, InferenceConv3d):
            child.conv_fp16 = True
        elif isinstance(child, nn.Conv3d):
            replacement = InferenceConv3d(
                child.in_channels, child.out_channels, child.kernel_size,
                stride=child.stride, padding=child.padding, dilation=child.dilation,
                groups=child.groups, bias=child.bias is not None,
                padding_mode=child.padding_mode, device=child.weight.device,
                dtype=child.weight.dtype,
            )
            replacement.weight = child.weight
            replacement.bias = child.bias
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            enable_structure_conv_fp16(child)
