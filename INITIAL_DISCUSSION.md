# Parallax — Initial Discussion

This document captures the design discussion that seeded **Parallax**. It records the
idea, the open problems we found ("cruxes"), the decisions we made, and how the
Sonata/Concerto/Utopia line of work reshapes the plan.

---

## 1. The Idea

Build a version of 3D Gaussian Splatting (3DGS) that trains on **deep features instead
of RGB color**, so reconstruction is robust to lighting changes, exposure shifts, and
video jitter during capture.

**Why:** Standard 3DGS matches *rendered RGB* to *captured RGB*. If lighting changes
between photos, the colors don't match and the geometry breaks. Learned keypoint
methods (SuperPoint etc.) abandoned raw RGB for deep features long ago; 3DGS has no
equivalent. We want that equivalent.

**Core swap:** Replace the per-Gaussian color payload with a 16-dim **feature** that
describes *geometry, not appearance*. A table corner gets the same feature regardless
of how it's lit → lighting stops mattering.

### Two models (pretrained once, frozen at scene time)

- **Model A — 2D feature model.** DINO-injected FPN. Image → sharp, full-resolution
  feature map. Trained to be **multiview-consistent and lighting-invariant** (same
  physical point → same feature across all views/lighting), which is what makes it useful.
- **Model B — 3D feature model.** Gaussian cloud geometry → a 16-dim feature per Gaussian,
  computed from its position/neighborhood.

A and B share **one feature space**: a physical point gets the same feature from either.

### Inference (RGB-3DGS with the color channel swapped for features)

1. Run frozen **A** on all posed images → target feature maps (fixed for the whole run).
2. Start with random Gaussians.
3. Run frozen **B** on the Gaussians → each gets a 16-dim feature from its geometry.
4. Rasterize those features to each image (same rasterizer as RGB, different payload).
5. Compare rendered features to A's targets → loss.
6. Move Gaussians, re-run B to refresh features, rasterize, compare. Repeat.

The Gaussian optimizer is "dumb" — it just minimizes feature loss the way RGB-3DGS
minimizes color loss. All geometric understanding lives in A, B, and their alignment.

**Output:** a geometrically-accurate Gaussian scene. No RGB rendering by design. Color
can be added later as a separate pass.

---

## 2. The Three Cruxes (problems we found)

### Crux 1 — B's injectivity is the whole ballgame

In RGB-3DGS, color is a **free** per-Gaussian parameter, decoupled from position. The
only way to get the right color at the right pixel is to *physically place a Gaussian
there* → photometric loss constrains geometry.

In Parallax the feature is **not free** — it is `B(geometry)`. The loss is:

```
L(geom) = Σ_views || rasterize( B(geom) ; geom ) − A_target ||²
```

Geometry feeds the loss through two paths: B's output, and the splat/alpha-compositing
weights. **Danger:** if B is *not injective* (many geometries → same feature), the
optimizer satisfies the feature loss with *wrong* geometry. The method only works if
"B's features match A ⟺ geometry is correct." That biconditional is an assumption, and
nothing automatically enforces it.

This turns out to be **exactly the "geometric shortcut"** that Sonata identifies (§5).

### Crux 2 — collapse, and the ambiguous "randomize the features" plan

An agreement-only objective (`B's render == A's image`) is *perfectly minimized by
constant features everywhere* — useless, but trivially lighting-invariant. This is the
**collapse** failure common to all self-supervised feature learning; every SSL method
has an explicit anti-collapse mechanism (DINO: centering+sharpening; VICReg: variance;
contrastive: negatives).

The original "randomize the features" plan was ill-posed (random per-scene codes make B
unlearnable across scenes). **Resolution:** drop randomization. Use **SSL
self-distillation** for B; distinctness is enforced by the anti-collapse term, not random
labels. (Superseded again by the Concerto recipe — see §5/§6.)

### Crux 3 — cold start / supervision strength

Original worry: features are smoother/lower-frequency than RGB → weaker early gradients,
no free-color slack. **Conceded:** the FPN gives a pixel-sharp full-res map, so resolution
isn't the issue. Only residual concern is **textureless regions** (near-constant features)
— but **RGB has the same weakness there**, so feature-GS is no worse and plausibly stronger
where structure exists, plus lighting-invariant. Claim dropped.

---

## 3. Key Decisions

### Decision 1 — Train B Concerto-style; A frozen *(updated, see §5)*

Original plan was "train B SSL-only, then regress A onto B." That is **dominated by the
Concerto recipe**: train B with **Sonata self-distillation + cross-modal prediction toward
a frozen A**. This bakes the A↔B alignment in during B's training (shared space = A's
space, which is what inference targets) and collapses the old two-stage alignment into one
asymmetric training (A frozen, B trained). See §5/§6.

### Decision 2 — B trained Sonata-style; watch the "geometric shortcut"

Sonata's headline finding *is* Crux 1: naive geometry-only training collapses onto trivial
cues (height/normal). Its recipe avoids that. B's validation must show **cross-scene
correspondence** (same structure → same code in a *different* scene), the direct test that
B avoided the shortcut.

### Decision 3 — Data reuse across stages

Converged ("perfect") GS scenes are frozen training data, reused per stage:

| Stage | Reads from converged scenes |
|---|---|
| **B training** | Geometry + posed RGB + poses. Self-distillation on geometry; cross-modal term predicts frozen A's patch features at the points' pixels. |
| **A** | A is frozen (off-the-shelf or separately trained for lighting-invariance). |

No circularity: scenes are frozen training data; inference builds brand-new scenes.

### Decision 4 — Close the clean/dirty domain gap for B *(upgraded by Utopia, see §5)*

B trains on **clean** geometry but at inference sees **garbage, varying-density** clouds.
Utopia's recipe is exactly this problem: **modality blinding** (drop scale/opacity/normal
per-sample & per-point), **granularity rescale**, **RoPE on aligned coords**, plus
scale/density jitter. Adopt these so B stays meaningful on the clouds optimization produces.

### Caveat — "perfect geometry for free" is optimistic

Converged RGB-GS geometry has floaters and over-reconstruction (it optimizes view
synthesis, not metric geometry). Prefer a **cleaner geometry source** (2DGS/GOF with
depth-normal regularization, or scanned meshes).

---

## 4. Open Questions

1. **B's inputs:** positions only, or + scale/opacity/normal? (Treat extras as Utopia-style
   droppable modalities.)
2. **Geometry source for training:** sloppy RGB-GS vs cleaner 2DGS/meshes (leaning cleaner).
3. **Feature dim:** start at 16; revisit if discriminability is insufficient.
4. **Is DINOv2 lighting-invariant enough** to use off-the-shelf as A for the de-risk path,
   or do we need a custom FiT3D-style A from day one?

---

## 5. Literature Integration — Sonata → Concerto → Utopia

These three papers (Pointcept group: HKU + Meta) are a single lineage and map onto
Parallax almost directly. They solve **Model B and the A↔B alignment**; they do **not**
solve our two novel pieces (lighting-invariant A; using B inside a reconstruction loop).

### Sonata (CVPR 2025) — B's anti-shortcut recipe
- Identifies the **geometric shortcut** (= our Crux 1): 3D SSL collapses to height/normal
  because point coords feed operators directly (unlike images, where info lives in features).
- Fix: **encoder-only self-distillation at coarse scales** (obscure spatial info), **EMA
  teacher** (asymmetric), **Sinkhorn-Knopp centering** (anti-collapse), **masked-point
  jitter**, **progressive scheduler**. Backbone **PTv3**. Local + global + masked views,
  matched by original spatial distance.

### Concerto (NeurIPS 2025) — our A↔B alignment, already built
- = Sonata self-distillation **+ cross-modal joint-embedding prediction**: the 3D encoder
  **predicts a frozen image model's (DINOv2) patch features**, using camera params to map
  points→pixels, **cosine loss**. (For each image patch, mean the features of points falling
  in it; match to the patch's image feature.)
- That is exactly Parallax's inference objective — "B's feature at a Gaussian = A's feature
  at the corresponding pixel" — **baked into training**. Image side stays frozen.
- Naive 2D+3D feature *concatenation* already beats either alone; Concerto's *joint*
  training beats the concatenation → cross-modal synergy is real.

### Utopia (2026) — B's inference-robustness recipe
- Goal: one encoder across **density, scale, and modality** shifts — *exactly* the condition
  B faces during GS optimization (densify/prune churns density; random init is garbage).
- Three fixes to steal: **Causal Modality Blinding** (drop color/normal per-sample &
  per-point → don't over-rely on auxiliary channels), **Perceptual Granularity Rescale**
  (normalize observing scale), **RoPE on granularity-aligned coords** (attention keys off
  *local relative* geometry, not absolute coords → less shortcut, density-robust).
- Our primitives are **Gaussians, not points** → Utopia's "coords + optional modalities"
  framework takes scale/opacity/rotation as droppable modalities natively.

### The reframe in one line
> **A = frozen lighting-invariant image model. Train B Concerto-style (Sonata
> self-distillation + cross-modal prediction toward A) with Utopia robustness tricks
> (modality blinding, scale/density jitter, RoPE) so B survives garbage inference clouds.**

---

## 6. What the Papers Do NOT Solve (our novelty + risk)

1. **A lighting-invariant Model A.** All three use *frozen DINOv2*, which is not
   lighting-invariant nor strongly multiview-consistent. Making A lighting-invariant
   (FiT3D-style) is *our* contribution and the entire robustness thesis.
2. **Using B inside a reconstruction optimization loop.** They assume geometry is *given*
   and produce features. Parallax inverts this — optimize geometry so features agree. This
   is the genuinely novel (and risky) part. **The energy-landscape test (Roadmap Stage 3)
   stays THE make-or-break.**
3. **New tension to watch:** semantic *invariance* (good for matching) fights positional
   *sensitivity* (good for optimization gradients). B's features must vary informatively
   with geometric error, or there is nothing for the optimizer to descend.

### Fast de-risk path (days, no training)
Concerto/Utopia checkpoints are public (Pointcept). Test the **whole thesis with zero
training**: A = frozen DINOv2 (or FiT3D ckpt), B = frozen pretrained Concerto/Utopia
encoder → run the energy-landscape test immediately. If feature-agreement bottoms out at
true geometry, the mechanism works and we invest in a custom lighting-invariant A; if not,
we learn what's missing for almost no cost. (Roadmap **Stage 0.5**.)

---

## 7. Related Work to Track

- **Feature-3DGS** (Zhou et al., CVPR 2024) — N-dim feature rasterizer; our starting code.
- **FiT3D** (Yue et al., ECCV 2024) — 3D-aware fine-tuning of 2D features for multiview
  consistency; essentially Model A's training recipe.
- **Sonata** (CVPR 2025) — point SSL; geometric-shortcut problem & fix → Model B core.
- **Concerto** (NeurIPS 2025) — joint 2D–3D SSL; cross-modal prediction → A↔B alignment.
- **Utopia** (2026) — one encoder across density/scale/modality shifts → B robustness.
- **Distilled Feature Fields / LERF / N3F** — feature distillation into 3D fields.
