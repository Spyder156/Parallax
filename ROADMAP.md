# Parallax — Implementation Roadmap

Staged, **test-gated** plan. Each stage has a concrete deliverable, a **kill-test** (a
pass/fail gate that must hold before moving on), and required visualizations. We do not
advance past a failing kill-test — we pause and discuss.

**Principle:** the riskiest assumptions (Crux 1 = does feature-agreement imply correct
geometry?) get tested *as early and as cheaply as possible*. We front-load that test onto
**off-the-shelf** components (Stage 0.5) before training anything.

Visualization is mandatory at every stage (Rerun / PCA-to-RGB / live CLI). The user
reviews visualizations; their read is ground truth.

---

## Stage 0 — Infrastructure & data

**Goal:** clean repo, working N-dim feature rasterizer, curated converged scenes.

- [ ] Dockerfile pinned (Blackwell/sm_120, cu128); build the `feature-3dgs` CUDA rasterizer
      (`External/feature-3dgs/submodules/diff-gaussian-rasterization-feature`); confirm
      forward+backward on a toy N-dim payload.
- [ ] Pick geometry source (Open Q2): cleaner **2DGS/GOF** scenes or scanned meshes
      preferred over sloppy RGB-GS. Curate ~5–10 scenes.
- [ ] Data loader exposing per scene: Gaussian cloud (centers, scale, opacity, rotation,
      normal), posed RGB images, camera intrinsics/extrinsics.
- [ ] Pull public **Concerto / Utopia** (Pointcept) and a **DINOv2 / FiT3D** checkpoint.

**Kill-test:** rasterize a random 16-dim payload on a known cloud; finite-difference check
that gradients flow to positions/opacity.

**Viz:** PCA-to-RGB of the rasterized random feature map; gradient-magnitude overlay.

---

## Stage 0.5 — Fast de-risk with off-the-shelf parts (NO training)

**Goal:** test the core mechanism (Crux 1) before building anything, using frozen public
checkpoints. This is the highest-value/lowest-cost experiment in the project.

- [ ] A = frozen **DINOv2** (or FiT3D ckpt); B = frozen pretrained **Concerto/Utopia** encoder.
- [ ] On one scene with known geometry, run the **energy-landscape test** (same as Stage 3):
      perturb Gaussians off truth, evaluate `L(geom) = Σ_views ||rasterize(B(geom)) − A_target||²`.

**Kill-test:**
1. `L` is minimized at (near) true geometry.
2. Gradient points back toward truth over a meaningful basin.

**Viz:** loss-vs-perturbation 1-D/2-D slices; gradient quiver toward truth.

> **Decision gate.** Pass → the mechanism works on off-the-shelf parts; proceed to invest
> in custom A/B. Fail → diagnose *why* (B not discriminative? DINOv2 not invariant enough?
> alignment mismatch?) before training. Either way we learn the key thing cheaply.

---

## Stage 1 — Model B (geometry → feature), Concerto-style

**Goal:** a frozen B that emits **discriminative, cross-scene-consistent, A-aligned**
per-Gaussian features and **survives messy inference clouds**.

- [ ] Backbone: **PTv3** (Sonata) → **RoPE-PTv3** (Utopia) for density/scale robustness.
- [ ] **Intra-modal:** Sonata self-distillation — EMA teacher, coarse-scale SSL,
      Sinkhorn-Knopp centering, masked-point jitter, progressive scheduler.
- [ ] **Cross-modal (the alignment):** Concerto objective — predict frozen A's patch
      features at the points' projected pixels (camera params), cosine loss. Balance
      intra:cross ≈ 1:1 (Concerto's tuned ratio).
- [ ] **Inference-robustness (Utopia / Decision 4):** Causal Modality Blinding on
      Gaussian attributes (scale/opacity/rotation/normal), granularity rescale, scale &
      density jitter, injected floaters / dropped points.
- [ ] **Inputs (Open Q1):** start positions-only; ablate adding Gaussian attributes as
      droppable modalities.

**Kill-test (the Crux 1/2 gate — most important training test):**
1. **No collapse:** feature variance / rank across points is high.
2. **Cross-scene correspondence:** same structure in a *different* scene → nearby code
      (kNN-feature correspondence accuracy).
3. **No geometric shortcut:** codes *not* linearly predictable from normal/height alone.
4. **A-alignment:** B's code at a point ≈ A's feature at its pixel (cosine), on held-out scenes.

**Viz:** PCA-to-RGB of per-Gaussian features (consistent colors on matching structures
across scenes); cross-scene nearest-neighbor match lines in Rerun.

> If this fails, **stop** and fix B before anything downstream.

---

## Stage 2 — Model A (image → feature), lighting-invariant

**Goal:** a frozen A that maps RGB → the shared feature space, multiview-consistently and
**lighting-invariantly**. (Stage 0.5 tells us whether off-the-shelf DINOv2/FiT3D suffices
or we must train this.)

- [ ] A = DINO-injected FPN (DINO at one encoder stage, FPN decoder to full res).
- [ ] Train/fine-tune FiT3D-style for multiview consistency; add **explicit lighting/
      exposure/white-balance augmentation** for invariance (our contribution, not in the
      Pointcept papers).
- [ ] Keep A and B in one space (either B aligned to frozen A per Stage 1, or a short joint
      refinement).

**Kill-test:**
1. **Multiview consistency:** same 3D point in two views → matching A-features.
2. **Lighting invariance (headline claim):** synthetic exposure/color/lighting
      perturbation moves A's features far less than it moves RGB.
3. **A↔B agreement** on held-out scenes.

**Viz:** side-by-side PCA-to-RGB of {A(image), B-rasterized}; lighting-sweep showing RGB
changing wildly while A's features stay put.

---

## Stage 3 — Energy-landscape test on the trained models

**Goal:** re-run Stage 0.5's test on the *custom-trained* A & B — directly verify Crux 1
on the real models, still with no optimization loop.

- [ ] One scene, known geometry, frozen custom A & B. Perturb Gaussians; evaluate `L(geom)`.

**Kill-test:**
1. **Minimum at truth.**
2. **Useful gradient** toward truth over a meaningful basin.
3. **Basin width** characterized → sets init-quality expectations for Stage 4.
4. **Positional sensitivity** (the §6 tension): `L` varies informatively with small
      geometric error, not flat near truth.

**Viz:** loss-landscape slices; gradient quiver; basin-width plot.

> The make-or-break scientific test. If `L` doesn't bottom out at true geometry, the method
> is unsound as specified — **pause and rethink B** before writing the optimizer.

---

## Stage 4 — Inference optimizer (the actual reconstruction)

**Goal:** reconstruct a scene from posed images, RGB-3DGS-style but on features.

- [ ] Pipeline: A(images) → targets; init Gaussians; loop { B(geom) → rasterize → feature
      loss → step → refresh B }.
- [ ] Port densification/pruning; **re-tune thresholds** (positional gradients now come
      from feature loss). Note B must stay valid as density churns (Utopia robustness).
- [ ] Init ablation: random vs SfM/COLMAP points (informed by Stage 3 basin width).

**Kill-test:**
1. **Convergence:** feature loss decreases stably; geometry stabilizes.
2. **Geometry accuracy vs GT:** Chamfer / depth / normal error, **competitive with
      RGB-3DGS under clean capture**.

**Viz:** live Rerun of the cloud evolving; rendered-vs-target feature maps over iterations;
geometry-error heatmaps vs GT.

---

## Stage 5 — The payoff experiment (robustness under bad capture)

**Goal:** prove the thesis — Parallax recovers better geometry than RGB-GS when capture is bad.

- [ ] Controlled perturbations on the *input images only*: lighting change, exposure shift,
      white-balance, video jitter / motion blur.
- [ ] Run RGB-3DGS and Parallax on identical perturbed inputs.
- [ ] Compare **geometry accuracy** (Chamfer/depth/normal vs GT) across perturbation strength.

**Success criterion:** as perturbation increases, RGB-3DGS geometry degrades sharply while
Parallax stays flat (or degrades far less). That curve is the paper figure.

**Viz:** geometry-error vs perturbation-strength curves (both methods); side-by-side
reconstructed clouds at high perturbation.

---

## Stage 6 — Ablations & write-up

- [ ] B inputs (pos-only vs +attributes); feature dim (16 vs 32/64).
- [ ] Anti-collapse mechanism; Utopia robustness tricks on/off; RoPE vs not.
- [ ] Geometry-source quality (2DGS vs sloppy RGB-GS) effect on final accuracy.
- [ ] Off-the-shelf vs custom A (does lighting-invariant A actually move Stage 5?).
- [ ] Init strategy (random vs SfM).

---

## Test-gate summary

| Stage | The one question it answers | Gate |
|---|---|---|
| 0 | Does the N-dim rasterizer differentiate? | grads flow |
| **0.5** | **Does the mechanism work on off-the-shelf parts?** | `L` minimized at truth (frozen ckpts) |
| 1 | Is B discriminative, shortcut-free, A-aligned, robust? | cross-scene corr. + alignment ✔ |
| 2 | Does A match B and ignore lighting? | consistency + invariance ✔ |
| 3 | **Does feature-agreement imply correct geometry (custom)?** | `L` minimized at truth |
| 4 | Can we actually reconstruct? | accuracy ≈ RGB-GS (clean) |
| 5 | **Are we robust where RGB-GS fails?** | flat error curve under perturbation |

Stages **0.5 / 1 / 3** are the scientific make-or-break gates (Crux 1/2). Stage **5** is
the thesis.
