# Parallax

**Feature-based 3D Gaussian Splatting** — reconstruct geometry from **deep features
instead of RGB color**, so reconstruction is robust to lighting changes, exposure shifts,
and capture jitter.

> Learned keypoint methods (SuperPoint et al.) replaced raw-RGB matching with deep
> features years ago. 3DGS never did. Parallax is that swap for 3DGS.

---

## The one-paragraph version

Standard 3DGS matches *rendered RGB* to *captured RGB*, so it breaks when lighting
changes between photos. Parallax replaces the per-Gaussian color payload with a 16-dim
**feature** that describes geometry, not appearance — a table corner has the same feature
however it's lit. Two frozen pretrained models supply that feature space: **Model A**
(image → multiview-consistent, lighting-invariant dense feature map) and **Model B**
(Gaussian geometry → per-Gaussian feature). At inference we optimize the Gaussian cloud
so B's rasterized features match A's image features — identical to RGB-3DGS with the
color channel swapped out. Output: accurate geometry, no RGB. Color, if wanted, is a
separate pass on top.

---

## How it works

### Two models (pretrained once, frozen at scene time)

- **Model A — 2D feature model.** DINO-injected FPN. Image → sharp, full-resolution,
  multiview-consistent, **lighting-invariant** feature map. (This invariance is the whole
  robustness claim and is *our* contribution — see Discussion.)
- **Model B — 3D feature model.** Gaussian cloud geometry → 16-dim feature per Gaussian.
  Trained Concerto-style (Sonata self-distillation + cross-modal alignment to A) with
  Utopia robustness tricks so it survives the messy clouds optimization produces.

A and B share one feature space, so a physical point gets the same feature from either.

### Building a scene at inference

```
frozen A(images)            → target feature maps   (fixed for the run)
random Gaussians  ─┐
                   ├─ frozen B(geom) → per-Gaussian features
                   │        ↓ rasterize (same as RGB, different payload)
                   │   rendered feature maps
                   └──────── loss = || rendered − A_target ||  → move Gaussians → repeat
```

The optimizer only minimizes feature loss. All geometric understanding lives in A, B,
and their alignment.

---

## Why features, not RGB

| | RGB-3DGS | Parallax (feature-3DGS) |
|---|---|---|
| Payload | per-Gaussian color (free param) | per-Gaussian feature = `B(geometry)` |
| Supervision | captured RGB | A's multiview-consistent features |
| Lighting change | breaks (colors mismatch) | invariant (features describe geometry) |
| Output | photorealistic render | accurate geometry (color optional later) |

The feature is **not** a free parameter — it is a function of geometry. That's what
makes the feature loss constrain geometry (and is also the project's central risk; see
[INITIAL_DISCUSSION.md](INITIAL_DISCUSSION.md), Crux 1).

---

## Status

Early design / pre-implementation. See:

- **[INITIAL_DISCUSSION.md](INITIAL_DISCUSSION.md)** — the idea, the open problems
  (cruxes), the design decisions, and how the Sonata/Concerto/Utopia line reshapes them.
- **[ROADMAP.md](ROADMAP.md)** — staged, test-gated implementation plan (including a
  fast off-the-shelf de-risk path before any training).

---

## Repository layout

```
External/
├── README.md                 # this file
├── INITIAL_DISCUSSION.md     # design rationale + cruxes + decisions + literature
├── ROADMAP.md                # staged implementation plan
└── feature-3dgs/             # base repo: CUDA rasterizer for N-dim features + depth
    └── submodules/diff-gaussian-rasterization-feature/
```

We build on **[Feature-3DGS](feature-3dgs/)** (Zhou et al., CVPR 2024) for the
N-dim feature + depth rasterizer.

---

## Related work

- **Feature-3DGS** (CVPR 2024) — N-dim feature rasterizer (base code).
- **FiT3D** (ECCV 2024) — 3D-aware fine-tuning for multiview-consistent 2D features (Model A).
- **Sonata** (CVPR 2025) — point SSL; the "geometric shortcut" problem & fix (Model B core).
- **Concerto** (NeurIPS 2025) — joint 2D–3D SSL; cross-modal prediction = our A↔B alignment.
- **Utopia** (2026) — one point encoder across density/scale/modality shifts = our
  inference-robustness recipe for B.
- **Distilled Feature Fields / LERF / N3F** — feature distillation into 3D fields.
