"""Stage 0 kill-test: does the N-dim feature rasterizer differentiate on sm_120?

Builds a tiny scene of random Gaussians, rasterizes a random F-dim feature payload,
backprops a dummy loss, and finite-difference-checks the gradient w.r.t. a few
Gaussian means. Pass = autograd grad matches finite-diff within tolerance.

Run inside the Parallax container:
    python parallax/scripts/test_rasterizer.py
"""
import math
import os
import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

DEVICE = "cuda"
F = 64  # must match NUM_SEMANTIC_CHANNELS the image was compiled with


def look_at(cam_pos):
    """Trivial world->camera: camera at cam_pos looking down -Z at origin, up +Y."""
    z = -cam_pos / cam_pos.norm()
    up = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    x = torch.cross(up, z); x = x / x.norm()
    y = torch.cross(z, x)
    R = torch.stack([x, y, z], dim=1)  # columns are basis
    Rt = torch.eye(4, device=DEVICE)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = -R.T @ cam_pos
    return Rt.T  # rasterizer wants column-major view transform


def make_raster(W=128, H=128, fov=1.0):
    cam_pos = torch.tensor([0.0, 0.0, 4.0], device=DEVICE)
    view = look_at(cam_pos)
    tanfov = math.tan(fov * 0.5)
    znear, zfar = 0.01, 100.0
    P = torch.zeros(4, 4, device=DEVICE)
    P[0, 0] = 1.0 / tanfov
    P[1, 1] = 1.0 / tanfov
    P[2, 2] = zfar / (zfar - znear)
    P[3, 2] = -(zfar * znear) / (zfar - znear)
    P[2, 3] = 1.0
    full = view @ P
    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=tanfov, tanfovy=tanfov,
        bg=torch.zeros(3, device=DEVICE),
        scale_modifier=1.0,
        viewmatrix=view, projmatrix=full,
        sh_degree=0, campos=cam_pos,
        prefiltered=False, debug=False,
    )
    return GaussianRasterizer(settings)


def render(means, feats, raster, N, ret_aux=False):
    scales = torch.full((N, 3), 0.15, device=DEVICE)
    rots = torch.zeros(N, 4, device=DEVICE); rots[:, 0] = 1.0
    opac = torch.full((N, 1), 0.9, device=DEVICE)
    colors = torch.zeros(N, 3, device=DEVICE)  # RGB unused here
    screenspace = torch.zeros_like(means, requires_grad=True)
    try:
        screenspace.retain_grad()
    except Exception:
        pass
    out = raster(
        means3D=means, means2D=screenspace, opacities=opac,
        shs=None, colors_precomp=colors, scales=scales, rotations=rots,
        semantic_feature=feats,
    )
    # API returns (rendered_image, feature_map, radii, depth) in feature-3dgs
    feat_map = out[1]
    if ret_aux:
        return feat_map, screenspace
    return feat_map


def save_pca_rgb(feat_map, path):
    """feat_map: (F,H,W) tensor -> PCA top-3 channels -> RGB PNG for visual inspection."""
    import cv2
    F, H, W = feat_map.shape
    X = feat_map.detach().reshape(F, -1).T.float().cpu().numpy()  # (HW, F)
    Xc = X - X.mean(0, keepdims=True)
    # top-3 principal components via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    rgb = Xc @ Vt[:3].T            # (HW, 3)
    rgb = rgb.reshape(H, W, 3)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (rgb[..., ::-1] * 255).astype(np.uint8))  # RGB->BGR
    print(f"[viz] saved feature-map PCA-RGB -> {path}")


def save_gray(img, path):
    """img: (H,W) tensor -> normalized grayscale PNG."""
    import cv2
    a = img.detach().float().cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (a * 255).astype(np.uint8))
    print(f"[viz] saved per-pixel feature energy -> {path}")


def save_points_grad(means, grad, path):
    """Scatter Gaussian xy colored by |grad| so the user can see where geometry gradient lands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = means.cpu().numpy(); gm = grad.norm(dim=1).cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(m[:, 0], m[:, 1], c=gm, cmap="viridis", s=14)
    plt.colorbar(label="|dL/dmean3D|"); plt.title("geometry gradient magnitude")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
    print(f"[viz] saved geometry-gradient scatter -> {path}")


def main():
    torch.manual_seed(0)
    N = 200
    raster = make_raster()
    means = (torch.rand(N, 3, device=DEVICE) - 0.5) * 2.0
    feats = torch.rand(N, 1, F, device=DEVICE)  # feature-3dgs expects [N, 1, F]

    means.requires_grad_(True)
    feats.requires_grad_(True)
    feat_map, screen = render(means, feats, raster, N, ret_aux=True)
    target = torch.zeros_like(feat_map)
    loss = ((feat_map - target) ** 2).mean()
    loss.backward()
    g = means.grad.clone()
    sg = None if screen.grad is None else screen.grad.norm().item()
    fg = None if feats.grad is None else feats.grad.norm().item()
    print(f"[ok] forward+backward ran. loss={loss.item():.6f}  feat_map={tuple(feat_map.shape)}")
    print(f"[grad] means3D={g.norm().item():.4e}  means2D/screen={sg}  features={fg}")

    # ALWAYS save visualizations for the user to inspect, regardless of pass/fail.
    save_pca_rgb(feat_map, "/workspace/outputs/stage0_feature_map_pca.png")
    save_gray(((feat_map ** 2).mean(0)), "/workspace/outputs/stage0_perpixel_loss.png")
    if g.norm() > 0:
        save_points_grad(means.detach(), g, "/workspace/outputs/stage0_means_grad.png")

    if feat_map.abs().sum() == 0:
        print("FAIL — nothing rendered (empty image). See saved viz.")
        return
    if g.norm() == 0:
        print("STATUS — forward renders, but means3D grad is zero. "
              "Viz saved to parallax/outputs/. Awaiting your read before I touch anything.")
        return

    # Finite-difference check on a few Gaussians with nonzero grad.
    eps = 1e-3
    idx = g.abs().sum(1).topk(5).indices
    max_rel = 0.0
    for i in idx.tolist():
        for d in range(3):
            m1 = means.detach().clone(); m1[i, d] += eps
            m2 = means.detach().clone(); m2[i, d] -= eps
            with torch.no_grad():
                l1 = ((render(m1, feats, raster, N) - target) ** 2).mean()
                l2 = ((render(m2, feats, raster, N) - target) ** 2).mean()
            fd = (l1 - l2) / (2 * eps)
            ana = g[i, d]
            denom = max(abs(fd.item()), abs(ana.item()), 1e-6)
            rel = abs(fd.item() - ana.item()) / denom
            max_rel = max(max_rel, rel)
    print(f"[fd] max relative error (autograd vs finite-diff): {max_rel:.3e}")
    print("PASS" if max_rel < 0.1 else "FAIL — gradient mismatch, investigate")


if __name__ == "__main__":
    main()
