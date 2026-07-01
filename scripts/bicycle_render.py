"""Step 1 sanity: load the trained bicycle .ply + real COLMAP cameras, render RGB.

If these renders look like the bicycle scene from the viewer, the ply loader + pose
conventions are correct and we can trust the downstream feature/energy-landscape code.

Run in the parallax container:
    docker run --rm --gpus all -v "$PWD:/workspace" parallax:base \
        python /workspace/scripts/bicycle_render.py
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
C0 = 0.28209479177387814  # SH band-0 constant


def load_gaussians(path):
    p = PlyData.read(path)["vertex"]
    xyz = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float32)
    normals = np.stack([p["nx"], p["ny"], p["nz"]], 1).astype(np.float32)
    fdc = np.stack([p["f_dc_0"], p["f_dc_1"], p["f_dc_2"]], 1).astype(np.float32)
    opacity = np.asarray(p["opacity"], np.float32)[:, None]
    scale = np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], 1).astype(np.float32)
    rot = np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], 1).astype(np.float32)
    t = lambda a: torch.tensor(a, device=DEVICE)
    means = t(xyz)
    scales = torch.exp(t(scale))                       # raw log-scale -> scale
    rots = torch.nn.functional.normalize(t(rot), dim=1)
    opac = torch.sigmoid(t(opacity))
    rgb = torch.clamp(C0 * t(fdc) + 0.5, 0.0, 1.0)     # SH deg-0 albedo
    print(f"[ply] {means.shape[0]} gaussians | xyz range "
          f"{xyz.min(0)} .. {xyz.max(0)}")
    return means, scales, rots, opac, rgb, t(normals)


def load_cameras(sparse):
    extr = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    cams = []
    for im in extr.values():
        cam = intr[im.camera_id]
        W, H = cam.width, cam.height
        if cam.model in ("PINHOLE", "OPENCV"):
            fx, fy = cam.params[0], cam.params[1]
        else:  # SIMPLE_PINHOLE / SIMPLE_RADIAL
            fx = fy = cam.params[0]
        FoVx, FoVy = focal2fov(fx, W), focal2fov(fy, H)
        R = np.transpose(qvec2rotmat(im.qvec))
        T = np.array(im.tvec)
        w2v = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(DEVICE)
        proj = getProjectionMatrix(0.01, 100.0, FoVx, FoVy).transpose(0, 1).to(DEVICE)
        full = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        center = w2v.inverse()[3, :3]
        cams.append(dict(name=im.name, W=W, H=H, FoVx=FoVx, FoVy=FoVy,
                         w2v=w2v, full=full, center=center))
    cams.sort(key=lambda c: c["name"])
    print(f"[cam] {len(cams)} cameras | first {cams[0]['W']}x{cams[0]['H']} "
          f"FoV {math.degrees(cams[0]['FoVx']):.1f}x{math.degrees(cams[0]['FoVy']):.1f}")
    return cams


def make_raster(cam):
    return GaussianRasterizer(GaussianRasterizationSettings(
        image_height=cam["H"], image_width=cam["W"],
        tanfovx=math.tan(cam["FoVx"] * 0.5), tanfovy=math.tan(cam["FoVy"] * 0.5),
        bg=torch.zeros(3, device=DEVICE), scale_modifier=1.0,
        viewmatrix=cam["w2v"], projmatrix=cam["full"], sh_degree=0,
        campos=cam["center"], prefiltered=False, debug=False))


def render_rgb(cam, means, scales, rots, opac, rgb):
    raster = make_raster(cam)
    feats = torch.zeros(means.shape[0], 1, F, device=DEVICE)  # unused payload
    out = raster(means3D=means, means2D=torch.zeros_like(means), opacities=opac,
                 shs=None, colors_precomp=rgb, scales=scales, rotations=rots,
                 semantic_feature=feats)
    return out[0]  # (3,H,W)


def save_png(img_chw, path):
    import cv2
    a = img_chw.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (a[..., ::-1] * 255).astype(np.uint8))
    print(f"[viz] {path}")


def main():
    means, scales, rots, opac, rgb, normals = load_gaussians(PLY)
    cams = load_cameras(SPARSE)
    idxs = [0, len(cams) // 2, len(cams) - 1]
    for k in idxs:
        img = render_rgb(cams[k], means, scales, rots, opac, rgb)
        save_png(img, f"{OUT}/bike_rgb_view{k}_{cams[k]['name'].split('.')[0]}.png")
    print("[done] sanity RGB renders written. Compare against the LichtFeld viewer.")


if __name__ == "__main__":
    main()
