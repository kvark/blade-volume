# Inverse rendering: photographs in, a scene that can be lit out

Status: working end to end, and not yet good. Numbers live in
`benchmarks/inverse_rendering.toml`; this document is the shape of the thing
and the reasoning behind it.

## The goal

Given a sequence of images and their poses (COLMAP), recover

- the **light** the scene was under,
- the **surface geometry**,
- the **material** of every surface,

such that rendering the result with Blade reproduces the original images.

## Why the stated goal cannot be the only measurement

Reproducing the input images is achievable, exactly, by a reconstruction that
has recovered nothing. Give every surface element its own free albedo and the
renderer's diffuse term `albedo * E(normal)` can be solved pointwise: pick any
light, set `albedo = observed / E(normal)`, and every photograph comes back
perfect. What that produces is the illumination painted onto the surfaces — an
asset that looks right from the captured poses and falls apart the moment the
light moves, which is the entire point of having materials.

So the pipeline reports two families of number and never mixes them:

| | what it says | where it comes from |
| --- | --- | --- |
| **re-rendering** | how close the scene is to the images it was built from | any capture |
| **decomposition** | whether material and light are separately anything | only a capture whose answer is known |

`reconstruct` produces the first. `inverse_truth` produces the second, against
a scene built for the purpose.

## The four stages

    capture ──▶ surface ──▶ decompose ──▶ score
    images       discs      material +     render it back
    + poses                 light          and compare

`blade-volume-train/src/inverse/`, one module each. The fourth shares no code
with the first three, so it has no opportunity to agree with the solver by
construction: it goes through the same `RelightTracer` a viewer uses.

### capture

Reads COLMAP poses and rectifies each image onto the calibrated pinhole plane.
Everything downstream is in **linear** radiance — fitting a physical material
against display-encoded values recovers a material wrong by a power of 2.2,
which looks like a plausible albedo and is not one.

### surface

Two sources, and the choice matters more than anything else in the pipeline.

**Sparse points.** COLMAP's triangulated points, with normals from the local
covariance and a side chosen by the cameras that can see them. Free, needs no
training, and covers only what COLMAP could triangulate — 44 % of the frame on
bonsai.

**A trained foam.** Trace every training view through this repo's own
reconstruction and take where each ray was absorbed. This covers the frame,
and getting a usable surface out of it took three corrections:

1. **Where along the ray.** A density field trained on photographs has long
   thin tails. The *mean* absorption depth sits out in them, and two views of
   one wall agree about the wall and disagree about their tails — so a cloud
   fused from means came out dozens of sheets thick, with only 13 % of surfels
   ever the frontmost thing in any view. The **mode** — the single segment that
   absorbed the most — is the position two views actually agree on. It took
   that figure to 75 %.
2. **Whether there is a surface at all.** A room's field is genuinely foggy:
   100 % of rays reach full absorption. Mean, median and mode all return a
   confident position for a ray that passed through haze and met nothing. The
   weight carried by that strongest segment is the only available signal that
   separates a wall from dust, and rays below a threshold contribute nothing.
3. **Discs that meet.** At 0.75 of the merge cell they cannot tile it, and
   every pixel comes out part background. This reads as a uniformly dark
   render rather than as visible holes, which is why it survived a round of
   looking at pictures.

### decompose

Alternating least squares. Given the light, each material's albedo is a ratio
of sums; given the albedo, the light is a multiplicative update that cannot
step past zero and so stays a light without a constraint solver.

**Sharing is what makes it a decomposition.** The number of materials is the
knob between re-rendering and recovering: one per surfel fits best and
recovers nothing, while a few hundred shared across many normals leave the
light as the only remaining explanation for how shading varies. The knob is
exposed rather than tuned out of sight, because the sweep across it is the
measurement.

Two gauges have no evidence behind them at all and are therefore **assumed and
stated**:

- **overall scale** — doubling albedo and halving the light renders the same
  image. Anchored by "the most reflective thing in the scene is about 0.8".
- **per-channel white balance** — a red wall under white light and a white
  wall under red light are the same photograph. The light is assumed neutral
  on average, which is wrong for a tungsten-lit room in exactly the way you
  would expect, and hands the colour to the materials.

What survives both gauges — the *ratio* between two materials' albedos, and
the *shape* of the light — is what the measurement is about.

### visibility, and why the bounce is not optional

Without shadowing, a patch of floor in a sphere's shadow has one way to be
dark: dark paint. Adding shadowing was expected to fix that. It made
everything worse — the light error went from 46 % to 159 %.

The reason is that a patch the model calls fully shadowed is not black in any
real photograph, because whatever shadows it also lights it. Modelling the
occlusion without modelling the light that fills it leaves the fit with a
large unexplained brightness and nowhere to put it except the sky, which then
comes back wrong in shape and not merely in scale.

One bounce, gathered from the same shadow maps — they already know *which*
surfel is in the way, so they know what is lighting the shadow — takes the
light error to 35 % and the albedo error to 23 %. Shadowing and the bounce are
one feature; neither is worth having alone. That matches what the forward
renderer already says about itself, where `diffuse_samples` buys both together
because each ray either reaches the sky or meets something.

## Where it stands

On a scene whose answer is known, the split works: albedo recovered to 23 %
and the light's shape to 35 %, against a forward model that is itself 27 % away
from the photographs because it has no term for the second bounce or for
gloss.

On bonsai it does not yet reach the goal: 14 dB on the views the geometry was
built from and 10 dB on those it was not. That gap is the finding. A
view-independent shading model should not care which views it is scored on, so
a four-decibel train/test difference is not the material or the light — it is
the geometry, which is a per-view point cloud rather than a surface the views
agree about.

## What the numbers are limited by, in order

1. **Geometry.** A surfel cloud fused from the depth maps of a 21 dB radiance
   field is a scatter. Everything downstream inherits it, and shadowing —
   which is a clear win on good geometry — is a 3 dB regression on this one,
   because occlusion computed against the wrong surface is worse than none.
2. **The radiance field itself**, at 21.7 dB train / 20.0 test, is the ceiling
   the geometry is extracted from.
3. **The shading model**: Lambertian, no specular term, one bounce. Worth
   about 27 % on the truth scene even with everything else exact.

The order matters: improving 3 before 1 would be measuring the wrong thing.
