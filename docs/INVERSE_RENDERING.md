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
and getting a usable surface out of it took four corrections:

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
4. **Views that agree.** A merge cell supported by one camera is still one
   camera's density tail, not a surface. Requiring two distinct training views
   removes 48 % of the fused surfels while retaining 98 % of those the material
   fit actually observed. Three views begins to remove real coverage, so two is
   the selected boundary. Within a surviving cell, the peak absorption also
   weights the center and normal: a sharply localized mode is better evidence
   than one that barely cleared the haze threshold. That adds another 0.06 dB
   held out without changing coverage.

### decompose

Alternating least squares. Given the light, each material's albedo is a ratio
of sums; given the albedo, the light is a multiplicative update that cannot
step past zero and so stays a light without a constraint solver.

**One material per surfel**, and this was got wrong once. Sharing materials
across surfels was originally presented as what turns the fit into a
decomposition. It is not: per-surfel albedo was already the best albedo *and*
the best re-rendering, and sharing only bought a light of better shape. The
claim was made by reading one column of three.

What sharing was really covering for is that most of a room's surfels are never
seen by any view — seventy per cent of bonsai's — and still get drawn, because
the renderer averages every disc within a depth band. A shared material gave
those a fitted albedo by accident. Copying the nearest *measured* surfel's
material does the same thing on local evidence rather than a scene-wide prior,
and it is worth 3.7 dB. Sharing remains available and measured; it is not the
default.

### beyond albedo: roughness and metalness

A Lambertian surface looks the same from every direction, so averaging a
surfel's observations across views loses nothing — which is why the first
version did it, and why that had to be undone. A lobe is the opposite: gloss
and reflectance are recoverable *only* from how one surfel changes between
views. Observations are now kept per view.

The two material hypotheses are solved separately rather than as one free fit:

- a **dielectric** reflects about 4 % at normal incidence whatever colour it is,
  and carries its colour in the diffuse term;
- a **metal** has no diffuse term at all, and carries its colour in the
  reflectance.

Left free they are close to collinear — a bright lobe and a bright albedo both
make a surfel brighter — and the solver returns a half-metal that is neither.
Solved separately each is one unknown per channel, and metalness becomes a
decision with a residual behind it.

Neither is believed without a margin. **A rough dielectric puts two to nine per
cent of its brightness in the lobe**, measured on the truth scene, and at that
level every hypothesis fits equally well. Without a margin about a third of a
matte floor came back as metal, each of those surfels losing its albedo
entirely — which cost far more than the metals it was trying to find were
worth. With one, those surfaces keep the default roughness, which is the honest
answer: the data does not say.

The lobe is occluded the same way the diffuse term is. Leaving it unshadowed is
not a small inconsistency — a metal sphere standing on a floor reflects that
floor over much of its surface, and a fit that thinks it is reflecting open sky
over-predicts the lobe, rejects the metal hypothesis, and puts the metal's
colour in the wrong channel. Occluding both is worth sixteen points of albedo
error.

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

On a scene whose answer is known, the albedo/light split works: albedo
recovered to 23 % and the light's shape to 35 %, against a forward model that is
itself 27 % away from the photographs because it has no term for the second
bounce.

**Roughness and metalness are not recovered**, and the two reasons are
different. Dielectric roughness is not a solver failure — the parameter moves
two to nine per cent of the signal and the fit correctly declines to guess it.
The metal is a genuine open item: the solver recovers a gold sphere's
reflectance and roughness exactly when it is given the real light and the
renderer's own shading, which is asserted as a unit test, and still fails in the
studio scene, where the photographs carry a bounce the lobe path does not model
and the orbit sees each surfel from one elevation.

On bonsai it does not yet reach the goal. The fresh Smooth-L1 foam plus
two-view depth consensus and confidence weighting reaches 14.89 dB on the views
used for extraction and 14.03 dB on held-out views. That supersedes the earlier
14.71/10.25 result: the four-decibel generalisation failure is gone, but the
absolute reconstruction is still visibly a coarse set of fused layers.
Consensus removes a large
unsupported foreground sheet in the held-out renders; it does not recover the
thin structure, normals, or material boundaries that the density field never
made into one shared surface.

The observation pass now has an opt-in overlap diagnostic. It rules out one
tempting local fix. Only 2.9 % of Bonsai samples share their *centre pixel*, but
every sampled pixel is covered by several projected disc footprints: 42.5 on
average and as many as 114 under the observer's circular support approximation.
Overlap itself is correct — the known-truth studio surface also overlaps at
98.7 % of its samples — but that scene averages 3.3 supports rather than 42.5.
Dividing each observation's confidence by its share of the projected coverage
does not deblend the image: it loses 0.04 dB train and 0.02 dB held out, changes
no known-truth albedo error, and is rejected. The excess depth has to be removed
or moved as geometry; a scalar weight cannot turn dozens of layers into one
surface.

Four further local shortcuts fail the same gate. Per-ray distortion loss makes
the source field worse before it makes the fused geometry better. Sparse COLMAP
anchors either fragment fusion or leak points triangulated from held-out views.
Training-only photometric normal offsets improve Bonsai but regress Room, and
local PCA plane normals do the reverse. The full measurements are recorded in
`benchmarks/inverse_rendering.toml`; none remains as a command-line option.
Together with the responsibility result, these controls say that normal or
weight corrections around independently fused depths are not the missing
algorithm. Positions must be coupled by one multi-view objective.

Tighter fusion statistics do not change that conclusion. Raising the existing
normal-agreement cutoff improves Bonsai's average by pruning difficult cells,
while its worst held-out view falls by as much as 0.55 dB and Room eventually
regresses. Weighting square-on samples more strongly merely moves error between
splits, and rejecting cells by their within-voxel RMS spread removes coverage
without separating layers. All three experimental controls were removed.

The first differentiable final-surface attempt also fails a stronger test. It
used the production disc intersection, rim coverage, first-surface blend and
fixed visibility candidates, differentiating one normal offset per surfel. Its
own loss fell, and both real scenes moved by roughly one hundredth of a decibel,
but a known surface displaced by 0.05 recovered at most 0.00070 while the same
optimizer moved an exact surface by 0.00271. Fixed appearance and coverage
gradients confined to disc rims let it rearrange overlap instead of recovering
depth, so the entire stage was removed. A useful shared objective needs a
radiance model that remains tied to observations as geometry moves, plus a
synthetic displacement test as a mandatory gate.

Depth extraction no longer rebuilds camera ray constants for every pixel or an
operating-system worker pool for every view. It traces all training cameras in
one scoped pool and retains the one-view API for small callers. On five
alternating runs of the selected Bonsai protocol this reduces median wall time
from 4.77 to 4.49 seconds (1.06x throughput) while producing the same 37,217
surfels and identical train, held-out, and worst-view scores. Median maximum RSS
rises by 12.9 MiB; the complete benchmark cgroup stayed below 164 MiB of charged
memory with no swap, pressure, or OOM event.

The workers now claim cameras dynamically instead of receiving contiguous
blocks. Camera costs vary with how many foam cells their rays traverse, so the
static split left cores idle behind its slowest block on the larger Room field.
Five alternating runs reduce Room's median end-to-end time from 8.44 to 7.94
seconds (1.063x) with unchanged RSS and quality. Bonsai is deliberately kept as
the neutral control: 4.53 to 4.50 seconds, inside run noise. Maps are sorted
back into camera order before fusion, and the batch-versus-single-view test
asserts exact depth, opacity, and peak results.

The reconstruction tracer now has an opacity-only evaluation mode and a
compile-time-specialized unweighted RadFoam neighbour loop. Opacity-only avoids
SH evaluation that cannot affect extracted depth; specialization removes the
radius loads, squared distance, and radical-plane division that plain RadFoam
does not need. PowerFoam keeps the weighted equations and is covered by the
existing CPU/GPU power-cell tests. Five alternating runs on top of dynamic
scheduling reduce Bonsai's median from 4.50 to 4.07 seconds (1.106x) and Room's
from 7.97 to 7.28 seconds (1.095x). Surfel counts and every reported quality
score are unchanged; both A/B cgroups used zero swap and recorded no OOM event.

An independent extraction sweep then found that the factor-4/2.5-disc recipe
was over-dense. Fusing at five pixel footprints and rendering discs at 1.7
merge cells removes redundant layers while reducing world-space support. On
Bonsai this raises train/held-out PSNR from 14.34/13.55 to 14.89/14.03 dB and
cuts 37,217 surfels to 28,312. On Room it raises 14.40/14.08 to 14.59/14.14 dB
and cuts 38,099 to 29,662. Worst-view PSNR improves on both scenes; coverage
falls by at most 0.8 percentage points, and the score already includes that
loss. Radius 1.6 through 1.8 is a stable aggregate improvement, with 1.7 the
first value that also passes every tail gate. These are now the library and CLI
defaults. Durable scenes and comparison images are under
`target/audit-runs/inverse-coarse-consensus-{bonsai,room}-local/`.

## What the numbers are limited by, in order

1. **Geometry.** Voxel-averaging per-view depth modes is an initializer, not a
   final surface. Two-view consensus halves the cloud and improves both open-sky
   and shadowed scores, yet nearly three quarters of the retained surfels are
   still never directly observed by the material fit, and the overlap
   diagnostic finds an order of magnitude more projected supports than on the
   truth surface. The next step is to optimize one shared oriented particle
   cloud against all views, without introducing a polygonal intermediate.
   Observation weighting is now measured and is not a substitute for that
   optimization.
2. **The radiance field itself.** The selected fresh Smooth-L1 foam reaches
   24.94 dB over all 37 every-eighth views at 128². Its density modes are still
   a much weaker geometry signal than its rendered colour, so a better colour
   score alone is not proof of a better surface. A matched 200K→229K capacity
   gate makes that concrete: +0.02 dB all-view radiance at step 14,000 becomes
   -0.32 dB held-out and -1.10 dB worst-view after depth-mode extraction.
3. **The shading model**: one bounce, and a lobe that only pays where the lobe
   is a meaningful share of the brightness. Worth about 27 % on the truth scene
   even with everything else exact.

The order matters: improving 3 before 1 would be measuring the wrong thing.
