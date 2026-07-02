"""FPN top-down pathway — fuses the multi-scale encoder features and carries the coarse
(DINO-injected) semantics down to the finest scale before the full-res head.
"""
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels: dict, out_ch=256):
        super().__init__()
        self.lat = nn.ModuleDict({k: nn.Conv2d(c, out_ch, 1) for k, c in in_channels.items()})
        self.smooth = nn.ModuleDict({k: nn.Conv2d(out_ch, out_ch, 3, 1, 1) for k in in_channels})

    def forward(self, feats):
        p4 = self.lat["c4"](feats["c4"])
        p3 = self.lat["c3"](feats["c3"]) + F.interpolate(p4, size=feats["c3"].shape[-2:],
                                                         mode="nearest")
        p2 = self.lat["c2"](feats["c2"]) + F.interpolate(p3, size=feats["c2"].shape[-2:],
                                                         mode="nearest")
        return {"p2": self.smooth["c2"](p2), "p3": self.smooth["c3"](p3),
                "p4": self.smooth["c4"](p4)}
