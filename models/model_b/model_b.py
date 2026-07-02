"""Model B — 3D Gaussian geometry -> per-Gaussian feature (in Model A's shared space).

NOT built yet. Kept strictly separate from Model A. Planned architecture (from the design):
a point/Gaussian encoder (PTv3 / PartField-style) over the local Gaussian neighborhood —
relative neighbor positions, relative covariance orientations (anisotropy), neighborhood
opacity/color statistics, multi-scale — trained Sonata/Concerto-style (self-distillation +
cross-modal alignment to frozen Model A) with Utopia robustness tricks.

    gaussians {means, scales, rots, opacity, ...}  ->  features [N, feat_dim]
"""
import torch.nn as nn


class ModelB(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, gaussians):
        raise NotImplementedError("Model B architecture not built yet (see module docstring).")
