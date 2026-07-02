"""Flavor (a) with UNTRAINED Model A instead of raw DINOv2.

Same bias test as bicycle_energy_a.py, but the target feature model is our random-weight
DINO-injected FPN (full-res 64-d output) rather than blocky DINOv2. Tests whether the
full-resolution FPN structure changes the landscape even before any training.

Run:
    docker run --rm --gpus all -v "$PWD:/workspace" \
      -v "$PWD/data/hf_cache:/root/.cache/huggingface" parallax:base \
      python /workspace/scripts/bicycle_energy_a_modelA.py
"""
import sys, os
sys.path.insert(0, "/workspace")          # for `models`
import numpy as np
import torch
import cv2

from bike_common import (DEVICE, F, load_gaussians, robust_bbox, load_cameras, render_feat,
                         render_rgb, orthonormal_tangents, select_flat_patch, cams_seeing,
                         save_pca_rgb, bake, run_landscape)
from models.model_a import ModelA

PLY = "/workspace/data/ckpt/bicycle_lichtfeld/splat_30000.ply"
SPARSE = "/workspace/data/mipnerf360/bicycle/sparse_4/0"
IMG_DIR = "/workspace/data/mipnerf360/bicycle/images_4"
OUT = "/workspace/outputs/bicycle/landscape_a_modelA"
DOWNSCALE = 2
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


@torch.no_grad()
def modelA_maps(model, cams):
    """Model A features per camera, full-res (F,H,W) — no PCA needed (A already emits F)."""
    targets = []
    for c in cams:
        img = cv2.imread(os.path.join(IMG_DIR, c["name"]))[..., ::-1]
        img = cv2.resize(img, (c["W"], c["H"]), interpolation=cv2.INTER_AREA)
        x = torch.tensor(img.copy(), device=DEVICE).float().permute(2, 0, 1)[None] / 255.0
        targets.append(model((x - MEAN) / STD)[0])          # (F,H,W)
    return targets


def main():
    os.makedirs(OUT, exist_ok=True)
    means, scales, rots, opac = load_gaussians(PLY)
    bmin, bmax = robust_bbox(means)
    cams = load_cameras(SPARSE, downscale=DOWNSCALE)
    nn, centroid, radius, normal = select_flat_patch(means, bmin, bmax)
    t1, t2 = orthonormal_tangents(normal)
    view_cams = [cams[i] for i in cams_seeing(cams, centroid)]

    model = ModelA(feat_dim=F).to(DEVICE).eval()             # RANDOM weights
    print("[modelA] untrained, feat_dim", F)
    targets = modelA_maps(model, view_cams)
    baked = bake(means, view_cams, targets)

    b0 = save_pca_rgb(targets[0], f"{OUT}/A_target_view0.png")
    save_pca_rgb(render_feat(view_cams[0], means, scales, rots, opac, baked),
                 f"{OUT}/baked_render_view0.png", b0)
    base = torch.full((means.shape[0], 3), 0.5, device=DEVICE); base[nn] = torch.tensor([1., 0, 0], device=DEVICE)
    hl = render_rgb(view_cams[0], means, scales, rots, opac, base)
    cv2.imwrite(f"{OUT}/patch_highlight.png",
                (hl.detach().clamp(0, 1).permute(1, 2, 0)[..., [2, 1, 0]].cpu().numpy() * 255).astype(np.uint8))
    print(f"[viz] {OUT}/patch_highlight.png")

    run_landscape(means, scales, rots, opac, nn, normal, t1, t2, radius,
                  view_cams, targets, baked, OUT, "Bicycle (a): untrained Model A target")
    print("[done] flavor (a) with untrained Model A complete.")


if __name__ == "__main__":
    main()
