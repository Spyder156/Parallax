"""Flavor (b): does the rasterizer turn geometry error into usable loss? (pure machinery)

Synthetic scene with ANALYTIC ground truth (plane floor + sphere + box). An isotropic,
injective ORACLE B assigns each Gaussian a feature = equal-frequency low-freq Fourier of
its normalized xyz (same frequency set on x/y/z, so NO faked anisotropy -- the
normal-vs-in-plane comparison is then purely the renderer's doing).

Protocol: render multiview feature-GT from the true scene. Pick a patch on the flat floor
(normal = +y, in-plane = x,z). Perturb ONLY that patch along each axis, re-run oracle B,
re-render all views, accumulate MSE vs GT -> loss(delta) per axis.

What this measures (target is self-consistent, so min-at-truth is by construction -- NOT
the test): per-axis basin CURVATURE (is any needed direction flat / signal-starved?), the
normal/in-plane curvature RATIO (renderer anisotropy), and a wide-sweep ALIAS scan (does
loss return to ~0 at a WRONG geometry -> compositing many-to-one). Real bias test is (a).

Run in container:
    docker run --rm --gpus all -v "$PWD:/workspace" parallax:base \
        python /workspace/scripts/test_b_machinery.py
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
F = 64                # must match NUM_SEMANTIC_CHANNELS the image was compiled with
OUT = "/workspace/outputs"
W = H = 160
N_VIEWS = 8
CAM_RADIUS = 4.0
CAM_ELEV_DEG = 20.0


# --------------------------------------------------------------------------------------
# Camera (reused convention from test_rasterizer.py: look at origin, scene centered there)
# --------------------------------------------------------------------------------------
def look_at(cam_pos):
    z = -cam_pos / cam_pos.norm()
    up = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    if abs(torch.dot(z, up)) > 0.99:
        up = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
    x = torch.cross(up, z, dim=0); x = x / x.norm()
    y = torch.cross(z, x, dim=0)
    R = torch.stack([x, y, z], dim=1)
    Rt = torch.eye(4, device=DEVICE)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = -R.T @ cam_pos
    return Rt.T


def make_raster(cam_pos, fov=0.9):
    view = look_at(cam_pos)
    tanfov = math.tan(fov * 0.5)
    znear, zfar = 0.01, 100.0
    P = torch.zeros(4, 4, device=DEVICE)
    P[0, 0] = 1.0 / tanfov; P[1, 1] = 1.0 / tanfov
    P[2, 2] = zfar / (zfar - znear); P[3, 2] = -(zfar * znear) / (zfar - znear)
    P[2, 3] = 1.0
    full = view @ P
    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tanfov, tanfovy=tanfov,
        bg=torch.zeros(3, device=DEVICE), scale_modifier=1.0,
        viewmatrix=view, projmatrix=full, sh_degree=0, campos=cam_pos,
        prefiltered=False, debug=False,
    )
    return GaussianRasterizer(settings)


def camera_rig():
    rigs = []
    elev = math.radians(CAM_ELEV_DEG)
    for i in range(N_VIEWS):
        az = 2 * math.pi * i / N_VIEWS
        cam = torch.tensor([
            CAM_RADIUS * math.cos(elev) * math.cos(az),
            CAM_RADIUS * math.sin(elev),
            CAM_RADIUS * math.cos(elev) * math.sin(az),
        ], device=DEVICE)
        rigs.append(make_raster(cam))
    return rigs


# --------------------------------------------------------------------------------------
# Synthetic scene: floor plane + sphere + box, with analytic surfaces
# --------------------------------------------------------------------------------------
def build_scene():
    pts, labels = [], []   # label: 0=floor 1=sphere 2=box

    # floor at y=-1.0, x,z in [-1.5,1.5]
    g = 60
    xs = np.linspace(-1.5, 1.5, g)
    zs = np.linspace(-1.5, 1.5, g)
    fx, fz = np.meshgrid(xs, zs)
    floor = np.stack([fx.ravel(), np.full(fx.size, -1.0), fz.ravel()], 1)
    pts.append(floor); labels.append(np.zeros(len(floor)))

    # sphere center (-0.7,-0.4,0.0) r=0.5  (fibonacci sampling)
    ns = 2000
    ii = np.arange(ns) + 0.5
    phi = np.arccos(1 - 2 * ii / ns)
    theta = np.pi * (1 + 5 ** 0.5) * ii
    sph = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)
    sph = sph * 0.5 + np.array([-0.7, -0.4, 0.0])
    pts.append(sph); labels.append(np.ones(len(sph)))

    # box center (0.7,-0.5,0) half 0.4: sample 6 faces
    nb = 22
    u = np.linspace(-0.4, 0.4, nb)
    a, b = np.meshgrid(u, u)
    a, b = a.ravel(), b.ravel()
    o = np.full_like(a, 0.4)
    faces = [
        np.stack([a, b, o], 1), np.stack([a, b, -o], 1),
        np.stack([a, o, b], 1), np.stack([a, -o, b], 1),
        np.stack([o, a, b], 1), np.stack([-o, a, b], 1),
    ]
    box = np.concatenate(faces, 0) + np.array([0.7, -0.5, 0.0])
    pts.append(box); labels.append(np.full(len(box), 2))

    means = np.concatenate(pts, 0).astype(np.float32)
    labels = np.concatenate(labels, 0).astype(np.int64)
    return (torch.tensor(means, device=DEVICE), torch.tensor(labels, device=DEVICE))


# --------------------------------------------------------------------------------------
# Oracle B: equal-frequency low-freq Fourier of normalized xyz (isotropic, injective)
# --------------------------------------------------------------------------------------
_BBOX = None
def oracle_b(means):
    """means (N,3) -> feats (N,1,F). Same freq set on x/y/z so the mapping is isotropic."""
    global _BBOX
    if _BBOX is None:
        lo = means.min(0).values; hi = means.max(0).values
        _BBOX = (lo, hi)
    lo, hi = _BBOX
    p = (means - lo) / (hi - lo + 1e-8) * 2 - 1        # normalize to [-1,1]^3
    nfreq = F // 6                                      # 3 axes * (sin,cos)
    freqs = torch.arange(1, nfreq + 1, device=DEVICE, dtype=torch.float32) * 0.5  # low freq
    feats = []
    for f in freqs:                                     # SAME freqs for every axis
        feats.append(torch.sin(math.pi * f * p))
        feats.append(torch.cos(math.pi * f * p))
    feat = torch.cat(feats, 1)                          # (N, 6*nfreq)
    if feat.shape[1] < F:
        feat = torch.cat([feat, torch.zeros(feat.shape[0], F - feat.shape[1], device=DEVICE)], 1)
    return feat[:, None, :]                             # (N,1,F)


# --------------------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------------------
def render(means, feats, raster):
    N = means.shape[0]
    scales = torch.full((N, 3), 0.04, device=DEVICE)
    rots = torch.zeros(N, 4, device=DEVICE); rots[:, 0] = 1.0
    opac = torch.full((N, 1), 0.95, device=DEVICE)
    colors = torch.zeros(N, 3, device=DEVICE)
    screen = torch.zeros_like(means)
    out = raster(means3D=means, means2D=screen, opacities=opac, shs=None,
                 colors_precomp=colors, scales=scales, rotations=rots,
                 semantic_feature=feats)
    return out[1]   # feature_map (F,H,W)


def render_all(means, feats, rig):
    return [render(means, feats, r) for r in rig]


def loss_vs_gt(means, gt_maps, rig):
    feats = oracle_b(means)
    tot = 0.0
    for r, gt in zip(rig, gt_maps):
        fm = render(means, feats, r)
        tot = tot + ((fm - gt) ** 2).mean()
    return (tot / len(rig)).item()


# --------------------------------------------------------------------------------------
# Viz helpers
# --------------------------------------------------------------------------------------
def save_pca_rgb(feat_map, path):
    import cv2
    Fc, Hh, Ww = feat_map.shape
    X = feat_map.detach().reshape(Fc, -1).T.float().cpu().numpy()
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    rgb = (Xc @ Vt[:3].T).reshape(Hh, Ww, 3)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (rgb[..., ::-1] * 255).astype(np.uint8))
    print(f"[viz] {path}")


def save_ply(means, feats, path):
    """point cloud colored by feature PCA-RGB, so the user can open it in any viewer."""
    X = feats[:, 0, :].detach().float().cpu().numpy()
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    rgb = Xc @ Vt[:3].T
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    m = means.detach().cpu().numpy(); c = (rgb * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(m)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for (x, y, z), (r, gg, bb) in zip(m, c):
            fh.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {gg} {bb}\n")
    print(f"[viz] {path}")


def curvature(deltas, losses):
    """parabola fit on the central window -> 2a (second derivative); also argmin."""
    d = np.array(deltas); l = np.array(losses)
    c = np.abs(d) < 0.15
    a, _, _ = np.polyfit(d[c], l[c], 2)
    amin = d[np.argmin(l)]
    return 2 * a, amin


# --------------------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    means, labels = build_scene()
    rig = camera_rig()
    feats_true = oracle_b(means)
    print(f"[scene] {means.shape[0]} gaussians  (floor/sphere/box), {N_VIEWS} views")

    # GT renders
    gt_maps = render_all(means, feats_true, rig)
    save_pca_rgb(gt_maps[0], f"{OUT}/b_gt_view0.png")
    save_pca_rgb(gt_maps[2], f"{OUT}/b_gt_view2.png")
    save_ply(means, feats_true, f"{OUT}/b_scene.ply")

    # floor patch: Gaussians within r=0.5 of (0,-1,0) on the floor
    floor = labels == 0
    on_floor = means[floor]
    dist = ((on_floor[:, [0, 2]]) ** 2).sum(1).sqrt()
    patch_local = dist < 0.5
    patch_idx = torch.where(floor)[0][patch_local]
    print(f"[patch] floor patch = {patch_idx.numel()} gaussians (normal=+y, in-plane=x,z)")

    axes = {
        "normal_y":   torch.tensor([0.0, 1.0, 0.0], device=DEVICE),
        "inplane_x":  torch.tensor([1.0, 0.0, 0.0], device=DEVICE),
        "inplane_z":  torch.tensor([0.0, 0.0, 1.0], device=DEVICE),
    }
    deltas = np.linspace(-0.5, 0.5, 41)

    curves = {}
    for name, ax in axes.items():
        ls = []
        for dlt in deltas:
            m = means.clone()
            m[patch_idx] = m[patch_idx] + float(dlt) * ax
            ls.append(loss_vs_gt(m, gt_maps, rig))
        curves[name] = ls
        kappa, amin = curvature(deltas, ls)
        peak = max(ls)
        near = ls[np.argmin(np.abs(deltas - 0.1))]            # loss at delta=+0.1
        # alias scan: lowest loss outside the central basin
        far = np.array(ls)[np.abs(deltas) > 0.25]
        alias = far.min() / (peak + 1e-12)
        flat = near / (peak + 1e-12)
        print(f"[{name:9s}] curvature={kappa:9.4f}  argmin={amin:+.3f}  "
              f"flatness(L@0.1/peak)={flat:5.3f}  alias(min_far/peak)={alias:5.3f}")

    # 1-D curves plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for name, ls in curves.items():
        plt.plot(deltas, ls, marker=".", label=name)
    plt.axvline(0, color="k", lw=0.6, ls="--")
    plt.xlabel("patch displacement (world units)"); plt.ylabel("feature loss (MSE vs GT)")
    plt.title("Flavor (b): energy landscape, floor patch")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/b_landscape_1d.png", dpi=120); plt.close()
    print(f"[viz] {OUT}/b_landscape_1d.png")

    # 2-D landscape: normal_y x inplane_x
    G = 21
    dd = np.linspace(-0.5, 0.5, G)
    Z = np.zeros((G, G))
    ny = axes["normal_y"]; nx = axes["inplane_x"]
    for i, dy in enumerate(dd):
        for j, dx in enumerate(dd):
            m = means.clone()
            m[patch_idx] = m[patch_idx] + float(dy) * ny + float(dx) * nx
            Z[i, j] = loss_vs_gt(m, gt_maps, rig)
    plt.figure(figsize=(5.2, 4.3))
    plt.imshow(Z, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], aspect="auto", cmap="viridis")
    plt.colorbar(label="feature loss"); plt.plot(0, 0, "r+", ms=12)
    plt.xlabel("in-plane x"); plt.ylabel("normal y")
    plt.title("Flavor (b): 2-D landscape (+ = truth)")
    plt.tight_layout(); plt.savefig(f"{OUT}/b_landscape_2d.png", dpi=120); plt.close()
    print(f"[viz] {OUT}/b_landscape_2d.png")

    np.savez(f"{OUT}/b_curves.npz", deltas=deltas, **curves, Z2d=Z, dd=dd)
    print("[done] flavor (b) machinery test complete.")


if __name__ == "__main__":
    main()
