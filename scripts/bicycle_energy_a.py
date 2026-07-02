"""Flavor (a): the REAL Crux-1 bias test on bicycle, with DINOv2 targets.

Model A = frozen DINOv2 dense features on the real images (PCA->F). Oracle B = those
features multiview-fused (baked) onto the true geometry = the BEST possible A-aligned B.
Keep the baked payload FIXED, perturb a flat patch, and measure the landscape of
|| rasterize(baked B ; geom) - A_target ||^2.

Unlike (b) the target is INDEPENDENT (real DINOv2, not our own render), so the minimum
is NOT at truth by construction: **argmin bias is the kill metric**. If matching real
image features bottoms out off the true surface, the mechanism is unsound on off-the-shelf
parts.

Run:
    docker run --rm --gpus all -v "$PWD:/workspace" \
      -v "$PWD/data/hf_cache:/root/.cache/huggingface" parallax:base \
      python /workspace/scripts/bicycle_energy_a.py
"""
import os
import numpy as np
import torch
import cv2
from transformers import AutoModel

from bike_common import (DEVICE, F, load_gaussians, robust_bbox, load_cameras, render_feat,
                         render_rgb, project, orthonormal_tangents, select_flat_patch,
                         cams_seeing, save_pca_rgb, curvature, save_curves, save_2d)

PLY = "/workspace/data/ckpt/bicycle_lichtfeld/splat_30000.ply"
SPARSE = "/workspace/data/mipnerf360/bicycle/sparse_4/0"
IMG_DIR = "/workspace/data/mipnerf360/bicycle/images_4"
OUT = "/workspace/outputs/bicycle/landscape_a"
DOWNSCALE = 2
PATCH = 14
DINO_W, DINO_H = 616, 406                 # 44x29 patches
N_SWEEP = 31
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


@torch.no_grad()
def dino_maps(model, cams):
    """DINOv2 patch features per camera, PCA->F (shared basis), upsampled to render res."""
    raw = []
    for c in cams:
        img = cv2.imread(os.path.join(IMG_DIR, c["name"]))[..., ::-1]
        img = cv2.resize(img, (DINO_W, DINO_H), interpolation=cv2.INTER_AREA)
        t = torch.tensor(img.copy(), device=DEVICE).float().permute(2, 0, 1)[None] / 255.0
        feat = model(pixel_values=(t - MEAN) / STD).last_hidden_state[:, 1:, :]
        raw.append(feat.reshape(DINO_H // PATCH, DINO_W // PATCH, -1))   # (gh,gw,768)
    allX = torch.cat([f.reshape(-1, f.shape[-1]) for f in raw], 0)
    mu = allX.mean(0, keepdims=True)
    _, _, Vt = torch.linalg.svd(allX - mu, full_matrices=False)
    basis = Vt[:F].T                                                     # (768,F)
    targets = []
    for f, c in zip(raw, cams):
        red = ((f.reshape(-1, f.shape[-1]) - mu) @ basis).reshape(f.shape[0], f.shape[1], F)
        red = red.permute(2, 0, 1)[None]                                # (1,F,gh,gw)
        up = torch.nn.functional.interpolate(red, size=(c["H"], c["W"]), mode="bilinear",
                                             align_corners=False)[0]     # (F,H,W)
        targets.append(up)
    return targets


def bake(means, cams, targets):
    """multiview-fuse A_target onto each Gaussian by sampling at its projection."""
    acc = torch.zeros(means.shape[0], F, device=DEVICE)
    cnt = torch.zeros(means.shape[0], 1, device=DEVICE)
    for c, tgt in zip(cams, targets):
        px, valid = project(means, c)
        gx = px[:, 0] / c["W"] * 2 - 1
        gy = px[:, 1] / c["H"] * 2 - 1
        grid = torch.stack([gx, gy], 1)[None, :, None, :]              # (1,N,1,2)
        samp = torch.nn.functional.grid_sample(tgt[None], grid, align_corners=False)
        samp = samp[0, :, :, 0].T                                      # (N,F)
        acc += samp * valid[:, None]
        cnt += valid[:, None]
    feats = acc / cnt.clamp(min=1)
    print(f"[bake] {(cnt.squeeze() > 0).float().mean().item()*100:.1f}% of gaussians seen")
    return feats[:, None, :]


def main():
    os.makedirs(OUT, exist_ok=True)
    means, scales, rots, opac = load_gaussians(PLY)
    bmin, bmax = robust_bbox(means)
    cams = load_cameras(SPARSE, downscale=DOWNSCALE)
    nn, centroid, radius, normal = select_flat_patch(means, bmin, bmax)
    t1, t2 = orthonormal_tangents(normal)
    view_cams = [cams[i] for i in cams_seeing(cams, centroid)]

    model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(DEVICE)
    targets = dino_maps(model, view_cams)
    baked = bake(means, view_cams, targets)

    # baking sanity: A_target vs render(baked) at truth, same view, shared basis
    b0 = save_pca_rgb(targets[0], f"{OUT}/A_target_view0.png")
    save_pca_rgb(render_feat(view_cams[0], means, scales, rots, opac, baked),
                 f"{OUT}/baked_render_view0.png", b0)
    # patch highlight
    base = torch.full((means.shape[0], 3), 0.5, device=DEVICE); base[nn] = torch.tensor([1., 0, 0], device=DEVICE)
    hl = render_rgb(view_cams[0], means, scales, rots, opac, base)
    cv2.imwrite(f"{OUT}/patch_highlight.png",
                (hl.detach().clamp(0, 1).permute(1, 2, 0)[..., [2, 1, 0]].cpu().numpy() * 255).astype(np.uint8))
    print(f"[viz] {OUT}/patch_highlight.png")

    def loss_at(offset):
        m = means.clone(); m[nn] = m[nn] + offset       # features FIXED (baked)
        tot = 0.0
        for c, g in zip(view_cams, targets):
            tot = tot + ((render_feat(c, m, scales, rots, opac, baked) - g) ** 2).mean()
        return (tot / len(view_cams)).item()

    R = 2.5 * radius
    deltas = np.linspace(-R, R, N_SWEEP)
    curves = {}
    for name, ax in {"normal": normal, "inplane_1": t1, "inplane_2": t2}.items():
        ls = [loss_at(float(d) * ax) for d in deltas]
        curves[name] = ls
        kappa, amin = curvature(deltas, ls, R)
        peak = max(ls) + 1e-12
        bias = amin / R
        alias = float(np.array(ls)[np.abs(deltas) > 0.6 * R].min()) / peak
        print(f"[{name:9s}] curvature={kappa:.4e}  argmin={amin:+.3f} (bias {bias:+.2f}R)  "
              f"alias={alias:5.3f}")
    save_curves(deltas, curves, f"{OUT}/landscape_1d.png", "Bicycle (a): DINOv2-target landscape")

    G = 17
    dd = np.linspace(-R, R, G)
    Z = np.array([[loss_at(float(dny) * normal + float(dnx) * t1) for dnx in dd] for dny in dd])
    save_2d(dd, Z, f"{OUT}/landscape_2d.png", "Bicycle (a): DINOv2 landscape (+ = truth)")
    np.savez(f"{OUT}/curves.npz", deltas=deltas, R=R, radius=radius, Z2d=Z, dd=dd, **curves)
    print("[done] flavor (a): DINOv2-target energy landscape complete.")


if __name__ == "__main__":
    main()
