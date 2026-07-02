"""Frozen DINOv2 backbone — injects multiview-ish semantic features at the coarse stage.

Kept frozen (per the design: A's semantics come from DINO; the trainable part is the FPN
that sharpens + upsamples). Robust to CLS/register tokens by taking the last gh*gw tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DinoBackbone(nn.Module):
    """image [B,3,H,W] (imagenet-normalized) -> dense patch features [B, hidden, gh, gw]."""

    def __init__(self, name="facebook/dinov2-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(name).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.patch = self.model.config.patch_size
        self.hidden = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, x):
        B, _, H, W = x.shape
        gh, gw = H // self.patch, W // self.patch
        xr = F.interpolate(x, size=(gh * self.patch, gw * self.patch),
                           mode="bilinear", align_corners=False)
        tok = self.model(pixel_values=xr).last_hidden_state       # [B, 1(+reg)+gh*gw, hidden]
        tok = tok[:, -gh * gw:, :]                                # drop CLS/registers
        return tok.transpose(1, 2).reshape(B, self.hidden, gh, gw)
