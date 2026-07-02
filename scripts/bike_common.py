"""Shared infrastructure for the bicycle energy-landscape tests (flavors b and a).

Loading (ply + COLMAP cameras), feature/RGB rasterization, flat-patch selection via
local PCA, camera visibility, and landscape viz. Kept separate so the (b) oracle test
and the (a) DINOv2 test share one correct implementation.
"""
import math, os, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/External/feature-3dgs")
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

DEVICE = "cuda"
F = 64


def load_gaussians(path):
    from plyfile import PlyData
    p = PlyData.read(path)["vertex"]
    g = lambda *k: np.stack([p[x] for x in k], 1).astype(np.float32)
    t = lambda a: torch.tensor(a, device=DEVICE)
    means = t(g("x", "y", "z"))
    scales = torch.exp(t(g("scale_0", "scale_1", "scale_2")))
    rots = torch.nn.functional.normalize(t(g("rot_0", "rot_1", "rot_2", "rot_3")), dim=1)
    opac = torch.sigmoid(t(np.asarray(p["opacity"], np.float32)[:, None]))
    print(f"[ply] {means.shape[0]} gaussians")
    return means, scales, rots, opac


def robust_bbox(means, lo=1.0, hi=99.0):
    q = torch.tensor([lo / 100, hi / 100], device=DEVICE)
    b = torch.quantile(means.float(), q, dim=0)
    return b[0], b[1]


def load_cameras(sparse, downscale=1):
    extr = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    cams = []
    for im in extr.values():
        cam = intr[im.camera_id]
        W, H = cam.width // downscale, cam.height // downscale
        fx, fy = (cam.params[0], cam.params[1]) if cam.model in ("PINHOLE", "OPENCV") \
            else (cam.params[0], cam.params[0])
        FoVx, FoVy = focal2fov(fx, cam.width), focal2fov(fy, cam.height)
        R = np.transpose(qvec2rotmat(im.qvec)); T = np.array(im.tvec)
        w2v = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(DEVICE)
        proj = getProjectionMatrix(0.01, 100.0, FoVx, FoVy).transpose(0, 1).to(DEVICE)
        full = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        cams.append(dict(name=im.name, W=W, H=H, FoVx=FoVx, FoVy=FoVy,
                         w2v=w2v, full=full, center=w2v.inverse()[3, :3]))
    cams.sort(key=lambda c: c["name"])
    return cams


def raster_of(cam):
    return GaussianRasterizer(GaussianRasterizationSettings(
        image_height=cam["H"], image_width=cam["W"],
        tanfovx=math.tan(cam["FoVx"] * 0.5), tanfovy=math.tan(cam["FoVy"] * 0.5),
        bg=torch.zeros(3, device=DEVICE), scale_modifier=1.0,
        viewmatrix=cam["w2v"], projmatrix=cam["full"], sh_degree=0,
        campos=cam["center"], prefiltered=False, debug=False))


def render_feat(cam, means, scales, rots, opac, feats):
    out = raster_of(cam)(means3D=means, means2D=torch.zeros_like(means), opacities=opac,
                         shs=None, colors_precomp=torch.zeros(means.shape[0], 3, device=DEVICE),
                         scales=scales, rotations=rots, semantic_feature=feats)
    return out[1]


def render_rgb(cam, means, scales, rots, opac, rgb):
    out = raster_of(cam)(means3D=means, means2D=torch.zeros_like(means), opacities=opac,
                         shs=None, colors_precomp=rgb, scales=scales, rotations=rots,
                         semantic_feature=torch.zeros(means.shape[0], 1, F, device=DEVICE))
    return out[0]


def project(means, cam):
    """world means (N,3) -> pixel coords (N,2) in [W,H], and validity mask (in front & in frame)."""
    ones = torch.ones(means.shape[0], 1, device=DEVICE)
    mh = torch.cat([means, ones], 1)
    clip = mh @ cam["full"]
    w = clip[:, 3:4]
    ndc = clip[:, :3] / (w + 1e-8)
    vz = (mh @ cam["w2v"])[:, 2]
    px = (ndc[:, 0] * 0.5 + 0.5) * cam["W"]
    py = (ndc[:, 1] * 0.5 + 0.5) * cam["H"]        # match rasterizer: ndc_y=-1 -> top row
    valid = (vz > 0) & (ndc[:, 0].abs() < 1) & (ndc[:, 1].abs() < 1)
    return torch.stack([px, py], 1), valid


def orthonormal_tangents(n):
    a = torch.tensor([1.0, 0, 0], device=DEVICE)
    if abs(torch.dot(a, n)) > 0.9:
        a = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    t1 = torch.nn.functional.normalize(torch.cross(a, n, dim=0), dim=0)
    t2 = torch.cross(n, t1, dim=0)
    return t1, t2


def select_flat_patch(means, bmin, bmax, k=3000, n_cand=200):
    """seed whose k-NN neighborhood is most PLANAR (local-PCA); normal = smallest evec."""
    core = ((means >= bmin) & (means <= bmax)).all(1)
    core_idx = torch.where(core)[0]
    torch.manual_seed(0)
    cand = core_idx[torch.randint(0, core_idx.numel(), (n_cand,), device=DEVICE)]
    best = None
    for s in cand.tolist():
        d = ((means - means[s]) ** 2).sum(1)
        nn = torch.topk(d, k, largest=False).indices
        evals, evecs = torch.linalg.eigh(torch.cov(means[nn].T))
        planarity = ((evals[1] - evals[0]) / (evals[2] + 1e-9)).item()
        if best is None or planarity > best[0]:
            best = (planarity, nn, evecs[:, 0], evals)
    planarity, nn, normal, evals = best
    centroid = means[nn].mean(0)
    radius = ((means[nn] - centroid) ** 2).sum(1).sqrt().max().item()
    normal = torch.nn.functional.normalize(normal, dim=0)
    print(f"[patch] {k} gaussians | planarity {planarity:.3f} | radius {radius:.3f} | "
          f"eigs {evals.cpu().numpy()} | normal {normal.cpu().numpy()}")
    return nn, centroid, radius, normal


def cams_seeing(cams, centroid, k=8):
    ch = torch.cat([centroid, torch.ones(1, device=DEVICE)])
    good = []
    for i, c in enumerate(cams):
        clip = ch @ c["full"]
        if clip[3] <= 0:
            continue
        ndc = clip[:3] / clip[3]
        if (ch @ c["w2v"])[2] > 0 and ndc[:2].abs().max() < 0.95:
            good.append(i)
    if len(good) > k:
        good = good[:: max(1, len(good) // k)][:k]
    print(f"[cams] {len(good)} of {len(cams)} see the patch; using {good}")
    return good


def save_pca_rgb(feat_map, path, basis=None):
    import cv2
    Fc, H, W = feat_map.shape
    X = feat_map.detach().reshape(Fc, -1).T.float().cpu().numpy()
    Xc = X - X.mean(0, keepdims=True)
    if basis is None:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False); basis = Vt[:3].T
    rgb = (Xc @ basis).reshape(H, W, 3)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (rgb[..., ::-1] * 255).astype(np.uint8))
    print(f"[viz] {path}")
    return basis


def curvature(deltas, losses, R):
    d, l = np.array(deltas), np.array(losses)
    c = np.abs(d) < 0.3 * R
    return 2 * np.polyfit(d[c], l[c], 2)[0], d[np.argmin(l)]


def save_curves(deltas, curves, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for name, ls in curves.items():
        plt.plot(deltas, ls, marker=".", label=name)
    plt.axvline(0, color="k", lw=0.6, ls="--")
    plt.xlabel("patch displacement (world units)"); plt.ylabel("feature loss (MSE vs target)")
    plt.title(title); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True); plt.savefig(path, dpi=120); plt.close()
    print(f"[viz] {path}")


def save_2d(dd, Z, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5.2, 4.3))
    plt.imshow(Z, origin="lower", extent=[dd[0], dd[-1], dd[0], dd[-1]], aspect="auto", cmap="viridis")
    plt.colorbar(label="feature loss"); plt.plot(0, 0, "r+", ms=12)
    plt.xlabel("in-plane"); plt.ylabel("normal"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(); print(f"[viz] {path}")
