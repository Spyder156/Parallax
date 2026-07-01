# NO_B.md — Parked idea: does A's feature field even carry geometric signal? (no B, no training)

**Status:** parked. A cheap sanity-measurement to run *before or alongside* committing to B.
Uses **only Model A** (a feature extractor on the real images) — **no B, no training, no
Concerto, no Blackwell builds.**

---

## The question it answers

Parallax optimizes geometry by matching rendered features to A's image features. That can
only work if **A's feature field actually varies where we need geometry to be constrained.**
If A's features are near-constant across a region, the feature loss is flat there → that
region is unconstrained, no matter how good B is. So before trusting the mechanism, measure
the *target side* directly: **where does A's feature field carry signal, and where is it
flat?**

This is independent of B: it characterizes the supervision A can provide. If the target
field is signal-starved in places geometry must be pinned, that's a problem A's training
must fix — not something a clever B can rescue.

> **Do NOT assume the field is smooth.** A is an FPN going H×W → H×W at full resolution and
> the FPN is *trained* — there is no a-priori reason its output is lower-frequency than RGB.
> Whether it is smooth, sharp, or sharper-than-RGB is exactly what this measures. (Earlier
> drafts wrongly asserted "features are smoother" — that was an unfounded premise.)

---

## The measurement

Run A over the posed `room` images and probe the feature map's spatial structure on
**known surface types** (use the converged geometry / depth to label regions):

1. **Tangent along a flat, textured wall** — slide along the surface.
   - Measure feature **decorrelation rate** (spatial autocorrelation length) and local
     **variance**. Near-constant features here → tangentially unconstrained (signal-starved).
2. **Across an edge / depth discontinuity** — step across a geometric boundary.
   - Features should change **sharply**. This is where the method *must* have signal.
3. **Normal / depth direction** — compare the same physical point across views at correct
   vs wrong depth (multiview consistency).
   - Features should be view-consistent at correct depth and diverge with depth error.
4. **Curved / low-relief region** — does the feature field **track curvature** even with no
   texture? If yes, features carry geometric signal RGB does **not**.

Summary statistics: spatial **autocorrelation length** and **effective rank** of the
feature map across each surface type; feature variance on a known-flat patch vs a known-
curved patch.

---

## How to read the outcome

- **Sharp at edges / depth-steps, tracks curvature** → the field has signal where geometry
  matters. Method is plausible; flat textureless interiors are fine (in-plane drift on a
  flat surface doesn't change the reconstructed surface — see the main energy-landscape test).
- **Flat even across depth steps / fails to track curvature** → A's features just relocate
  RGB's texture-dependence into N-D without adding geometric constraint. Real problem —
  A's training (multiview consistency / geometric awareness) must be fixed first.

---

## Relation to the main line

This is the **target-side** half of the picture. The **source/landscape-side** question
("does perturbing geometry actually move the loss, unbiased, toward truth?") is the main
energy-landscape test (normal vs in-plane perturbation; basin curvature + argmin bias).
Both are A-only / oracle-B and answerable **before committing to B's architecture.**
