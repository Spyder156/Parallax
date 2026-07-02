"""Model A — DINO-injected FPN: image -> sharp, full-resolution feature map.

Frozen DINOv2 supplies semantics at the coarse (/16) stage; the conv encoder + FPN supply
high-frequency detail and upsample back to full input resolution. Output is the shared
feature space that Model B's per-Gaussian features are matched against at inference.

    image [B,3,H,W]  ->  features [B, feat_dim, H, W]

Trainable: conv encoder, dino_proj, FPN, head.  Frozen: DINOv2.
"""
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder
from .dino_backbone import DinoBackbone
from .fpn import FPN


class ModelA(nn.Module):
    def __init__(self, feat_dim=64, fpn_dim=256, chs=(64, 128, 256),
                 dino="facebook/dinov2-base"):
        super().__init__()
        self.encoder = ImageEncoder(chs)
        self.dino = DinoBackbone(dino)
        self.dino_proj = nn.Conv2d(self.dino.hidden, chs[2], 1)      # 768 -> c4 channels
        self.fpn = FPN(self.encoder.out_channels, fpn_dim)
        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, feat_dim, 1),
        )
        self.feat_dim = feat_dim

    def forward(self, image, normalize=True):
        H, W = image.shape[-2:]
        c = self.encoder(image)                                     # sharp multi-scale
        d = self.dino(image)                                        # semantics /14
        d = F.interpolate(d, size=c["c4"].shape[-2:], mode="bilinear", align_corners=False)
        c["c4"] = c["c4"] + self.dino_proj(d)                       # inject DINO at /16
        p = self.fpn(c)
        f = self.head(p["p2"])                                      # /4
        f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)  # full res
        if normalize:
            f = F.normalize(f, dim=1)
        return f
