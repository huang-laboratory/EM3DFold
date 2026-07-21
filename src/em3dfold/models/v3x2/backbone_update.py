import math
import torch
import torch.nn as nn

from em3dfold.utils.affine_utils import (
    affine_composition,
    quaternion_to_matrix,
    get_affine,
)

class BackboneUpdate(nn.Module):
    def __init__(self, d: int=256):
        super().__init__()
        self.backbone_fc = nn.Linear(d, 6)
        self.shift = nn.Parameter(torch.tensor(1.5, dtype=torch.float))
        self.eps = 1e-6
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.normal_(self.backbone_fc.weight, std=0.02)
        self.backbone_fc.bias.data = torch.Tensor([0, 0, 0, 0, 0, 0])

    def forward(self, x, affines):
        y = self.backbone_fc(x) # (..., 6)

        y = torch.cat(
            [
                torch.sqrt(torch.square(self.shift) + self.eps) * torch.ones(size=y.shape[:-1] + (1,)).float().to(y.device),
                y,
            ],
            dim=-1,
        )

        # To rotmats and trans
        quats = y[..., :4]
        trans = y[..., 4:]
        rotmats = quaternion_to_matrix(quats)
        new_affines = affine_composition(
            affines,
            get_affine(rotmats, trans),
        ) # (..., 3, 4)

        return new_affines

