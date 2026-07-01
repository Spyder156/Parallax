"""Energy-landscape test on the real bicycle scene (Crux 1), flavor (b).

Isotropic injective ORACLE B = equal-frequency low-freq Fourier of robust-normalized
xyz (same freq set on x/y/z -> any normal-vs-in-plane asymmetry is the RENDERER's, not
faked feature anisotropy). Real Gaussians (true scales/rots/opacity = surfel-aware),
real COLMAP cameras.

Self-consistent target => min-at-truth is by construction (NOT the test). What (b)
measures: per-axis basin CURVATURE (is a needed direction flat?), normal/in-plane
curvature RATIO (renderer anisotropy), and a wide ALIAS scan (does loss return to ~0 at
WRONG geometry?). Real bias test is (a), with DINOv2 targets.

Run:
    docker run --rm --gpus all -v "$PWD:/workspace" parallax:base \
        python /workspace/scripts/bicycle_energy.py
"""
import math, os, sys
import numpy as np
import torch
from plyfile import PlyData

sys.path.insert(0, "/workspace/External/feature-3dgs")
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

DEVICE = "cuda"
F = 64
PLY = "/workspace/data/ckpt/bicycle_lichtfeld/splat_30000.ply"
SPARSE = "/workspace/data/mipnerf360/bicycle/sparse_4/0"
OUT = "/workspace/outputs"
DOWNSCALE = 2
K_CAMS = 8
PATCH_K = 3000
N_SWEEP = 31


def load_gaussians(path):
    p = PlyData.read(path)["vertex"]
    g = lambda *k: np.stack([p[x] for x in k], 1).astype(np.float32)
    t = lambda a: torch.tensor(a, device=DEVICE)
    means = t(g("x", "y", "z"))
    normals = torch.nn.functional.normalize(t(g("nx", "ny", "nz")), dim=1)
    scales = torch.exp(t(g("scale_0", "scale_1", "scale_2")))
    rots = torch.nn.functional.normalize(t(g("rot_0", "rot_1", "rot_2", "rot_3")), dim=1)
    opac = torch.sigmoid(t(np.asarray(p["opacity"], np.float32)[:, None]))
    print(f"[ply] {means.shape[0]} gaussians")
    return means, scales, rots, opac, normals


def robust_bbox(means, lo=1.0, hi=99.0):
    q = torch.tensor([lo / 100, hi / 100], device=DEVICE)
    b = torch.quantile(means.float(), q, dim=0)
    return b[0], b[1]


def oracle_b(means, bmin, bmax):
    p = ((means - bmin) / (bmax - bmin + 1e-8) * 2 - 1).clamp(-1, 1)
    nfreq = F // 6
    freqs = torch.arange(1, nfreq + 1, device=DEVICE, dtype=torch.float32) * 0.5
    feats = []
    for f in freqs:
        feats.append(torch.sin(math.pi * f * p)); feats.append(torch.cos(math.pi * f * p))
    feat = torch.cat(feats, 1)
    if feat.shape[1] < F:
        feat = torch.cat([feat, torch.zeros(feat.shape[0], F - feat.shape[1], device=DEVICE)], 1)
    return feat[:, None, :]


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


def orthonormal_tangents(n):
    a = torch.tensor([1.0, 0, 0], device=DEVICE)
    if abs(torch.dot(a, n)) > 0.9:
        a = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    t1 = torch.nn.functional.normalize(torch.cross(a, n, dim=0), dim=0)
    t2 = torch.cross(n, t1, dim=0)
    return t1, t2


def select_flat_patch(means, bmin, bmax, k=PATCH_K, n_cand=200):
    """seed whose k-NN neighborhood is most PLANAR (local-PCA), since ply normals are 0.
    normal = smallest-eigenvector of the neighborhood covariance."""
    core = ((means >= bmin) & (means <= bmax)).all(1)
    core_idx = torch.where(core)[0]
    torch.manual_seed(0)
    cand = core_idx[torch.randint(0, core_idx.numel(), (n_cand,), device=DEVICE)]
    best = None
    for s in cand.tolist():
        d = ((means - means[s]) ** 2).sum(1)
        nn = torch.topk(d, k, largest=False).indices
        C = torch.cov(means[nn].T)
        evals, evecs = torch.linalg.eigh(C)              # ascending
        planarity = ((evals[1] - evals[0]) / (evals[2] + 1e-9)).item()  # 1=plane,0=blob
        if best is None or planarity > best[0]:
            best = (planarity, nn, evecs[:, 0], evals)
    planarity, nn, normal, evals = best
    centroid = means[nn].mean(0)
    radius = ((means[nn] - centroid) ** 2).sum(1).sqrt().max().item()
    normal = torch.nn.functional.normalize(normal, dim=0)
    print(f"[patch] {k} gaussians | planarity {planarity:.3f} | radius {radius:.3f} | "
          f"eigs {evals.cpu().numpy()} | normal {normal.cpu().numpy()}")
    return nn, centroid, radius, normal


def cams_seeing(cams, centroid, k=K_CAMS):
    ch = torch.cat([centroid, torch.ones(1, device=DEVICE)])
    good = []
    for i, c in enumerate(cams):
        clip = ch @ c["full"]
        if clip[3] <= 0:
            continue
        ndc = clip[:3] / clip[3]
        vz = (ch @ c["w2v"])[2]
        if vz > 0 and ndc[:2].abs().max() < 0.95:
            good.append(i)
    # spread evenly
    if len(good) > k:
        good = good[:: max(1, len(good) // k)][:k]
    print(f"[cams] {len(good)} of {len(cams)} see the patch; using {good}")
    return good


def save_curves(deltas, curves, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for name, ls in curves.items():
        plt.plot(deltas, ls, marker=".", label=name)
    plt.axvline(0, color="k", lw=0.6, ls="--")
    plt.xlabel("patch displacement (world units)"); plt.ylabel("feature loss (MSE vs GT)")
    plt.title("Bicycle (b): energy landscape @ flat patch"); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True); plt.savefig(path, dpi=120); plt.close()
    print(f"[viz] {path}")


def save_2d(dd, Z, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5.2, 4.3))
    plt.imshow(Z, origin="lower", extent=[dd[0], dd[-1], dd[0], dd[-1]], aspect="auto", cmap="viridis")
    plt.colorbar(label="feature loss"); plt.plot(0, 0, "r+", ms=12)
    plt.xlabel("in-plane"); plt.ylabel("normal"); plt.title("Bicycle (b): 2-D landscape (+ = truth)")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(); print(f"[viz] {path}")


def curvature(deltas, losses, R):
    d, l = np.array(deltas), np.array(losses)
    c = np.abs(d) < 0.3 * R
    a = np.polyfit(d[c], l[c], 2)[0]
    return 2 * a, d[np.argmin(l)]


def main():
    means, scales, rots, opac, normals = load_gaussians(PLY)
    bmin, bmax = robust_bbox(means)
    feats = oracle_b(means, bmin, bmax)
    cams = load_cameras(SPARSE, downscale=DOWNSCALE)

    nn, centroid, radius, normal = select_flat_patch(means, bmin, bmax)
    t1, t2 = orthonormal_tangents(normal)
    view_ids = cams_seeing(cams, centroid)
    view_cams = [cams[i] for i in view_ids]

    # highlight render: gray scene, red patch, from the first seeing-camera
    base = torch.full((means.shape[0], 3), 0.5, device=DEVICE)
    base[nn] = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
    hl = render_rgb(view_cams[0], means, scales, rots, opac, base)
    import cv2
    cv2.imwrite(f"{OUT}/bike_patch_highlight.png",
                (hl.detach().clamp(0, 1).permute(1, 2, 0)[..., [2, 1, 0]].cpu().numpy() * 255).astype(np.uint8))
    print(f"[viz] {OUT}/bike_patch_highlight.png")

    # GT feature maps
    gt = [render_feat(c, means, scales, rots, opac, feats).detach() for c in view_cams]

    def loss_at(offset):
        m = means.clone(); m[nn] = m[nn] + offset
        fe = oracle_b(m, bmin, bmax)
        tot = 0.0
        for c, g in zip(view_cams, gt):
            tot = tot + ((render_feat(c, m, scales, rots, opac, fe) - g) ** 2).mean()
        return (tot / len(view_cams)).item()

    R = 2.5 * radius
    deltas = np.linspace(-R, R, N_SWEEP)
    axes = {"normal": normal, "inplane_1": t1, "inplane_2": t2}
    curves = {}
    for name, ax in axes.items():
        ls = [loss_at(float(d) * ax) for d in deltas]
        curves[name] = ls
        kappa, amin = curvature(deltas, ls, R)
        peak = max(ls) + 1e-12
        near = ls[int(np.argmin(np.abs(deltas - 0.3 * R)))]
        alias = float(np.array(ls)[np.abs(deltas) > 0.6 * R].min()) / peak
        print(f"[{name:9s}] curvature={kappa:.4e}  argmin={amin:+.3f}  "
              f"flat(L@0.3R/peak)={near/peak:5.3f}  alias(min_far/peak)={alias:5.3f}")
    save_curves(deltas, curves, f"{OUT}/bike_landscape_1d.png")

    # 2-D normal x in-plane_1
    G = 17
    dd = np.linspace(-R, R, G)
    Z = np.zeros((G, G))
    for i, dny in enumerate(dd):
        for j, dnx in enumerate(dd):
            Z[i, j] = loss_at(float(dny) * normal + float(dnx) * t1)
    save_2d(dd, Z, f"{OUT}/bike_landscape_2d.png")
    np.savez(f"{OUT}/bike_curves.npz", deltas=deltas, R=R, radius=radius,
             patch_normal=normal.cpu().numpy(), Z2d=Z, dd=dd, **curves)
    print("[done] STAGE 3: bicycle energy landscape complete.")


if __name__ == "__main__":
    main()
