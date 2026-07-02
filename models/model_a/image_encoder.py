"""Lightweight convolutional image encoder — supplies sharp, high-frequency detail at
multiple scales. DINO provides semantics; this provides the crisp edges the FPN upsamples.
"""
import torch.nn as nn


def conv_bn_relu(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ImageEncoder(nn.Module):
    """image [B,3,H,W] -> {c2:/4, c3:/8, c4:/16} feature maps."""

    def __init__(self, chs=(64, 128, 256)):
        super().__init__()
        c0, c1, c2 = chs
        self.stem = nn.Sequential(conv_bn_relu(3, c0, 2), conv_bn_relu(c0, c0, 2))   # /4
        self.stage3 = nn.Sequential(conv_bn_relu(c0, c1, 2), conv_bn_relu(c1, c1))   # /8
        self.stage4 = nn.Sequential(conv_bn_relu(c1, c2, 2), conv_bn_relu(c2, c2))   # /16
        self.out_channels = {"c2": c0, "c3": c1, "c4": c2}

    def forward(self, x):
        c2 = self.stem(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return {"c2": c2, "c3": c3, "c4": c4}
