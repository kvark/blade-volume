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

`reconstruct --brightest-albedo` exposes the scale anchor for a calibrated
capture or a scene containing a reference patch. Without such information the
default remains 0.8 and the output reports an assumption, not a recovered
absolute reflectance.

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
used for extraction and 14.03 dB on held-out views. The selected multi-view
surface refinement raises those to 14.97/14.07 dB and improves the worst
held-out view from 11.48 to 11.67 dB. That supersedes the earlier 14.71/10.25
result: the four-decibel generalisation failure is gone, but the absolute
reconstruction is still visibly a coarse set of fused layers. Consensus
removes a large unsupported foreground sheet in the held-out renders; it does
not recover the thin structure, normals, or material boundaries that the
density field never made into one shared surface.

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

The selected refinement meets that gate without introducing another geometry
representation. It searches nine positions over half a surfel radius in each
normal direction, reprojects a 3×3 world-space tangent patch into four to eight
training photographs, and minimizes robust pairwise normalized patch error.
Candidate-specific source-foam depth rejects occluded matches, and a local
median keeps nearby, similarly oriented moves coherent. In the synthetic gate,
all 49 particles recover a known 0.05-unit displacement to below `1e-6`, while
an exact surface moves no particles. On Bonsai it scores 5,445 of 28,316
surfels, moves 3,398, and raises train/held-out PSNR from 14.89/14.03 to
14.97/14.07 dB; worst held out improves 11.48→11.67 dB, while worst training
falls 11.69→11.58 dB. On Room it scores 5,407 of 29,658, moves 3,255, preserves
14.59 dB training PSNR, and raises held out 14.14→14.17 dB with the 13.08 dB
tail and both coverage values unchanged. The same-binary pass adds 0.11 seconds
on Bonsai and 0.20 seconds on Room. This is a real generalization improvement,
not completion: only 18–19% of surfels have enough visible texture to score and
11–12% move.

The first follow-ups narrow the next problem. Before testing texture, only
7,786 Bonsai and 9,504 Room surfels have four front-facing, in-frame,
source-visible views; 5,445 and 5,407 respectively then pass the texture floor.
Halving that floor scores about 1,100–1,200 more particles but loses 0.02 dB on
Bonsai, doubling it also loses 0.03 dB, and choosing textured views before the
facing-ranked cap loses 0.02 dB on Room. A second sweep gains 0.03 dB on Room
but loses Bonsai average quality and 0.1 percentage point of coverage, even
with a stricter acceptance threshold. Lowering geometric support from four to
three photographs nearly doubles the scored set and improves Bonsai held-out
PSNR by 0.06 dB, but loses 0.01 dB mean and 0.03 dB worst held-out on Room and
0.04 dB on the synthetic held-light tail; stronger acceptance gates do not
recover that transfer. All five changes were removed. The next objective needs
better geometric visibility or a different cue for unsupported particles, not
a looser patch threshold, fewer views, or another identical local pass.

An analytic textured sphere now makes that boundary a test rather than an
inference. With curvature and self-occlusion, all 25 particles that retain four
valid views recover an exact 0.0375 radial displacement; the other 24 account
for all remaining error. Starting from the exact sphere scores all 49 and moves
none. Propagating a median correction through nearby, similarly oriented
particles fills most of the synthetic gap, but the 75%-agreement form loses
0.01 dB held-out Bonsai and an all-neighbor-unanimous form loses the same
amount. Room is neutral for the broader form. Propagation was removed:
unsupported geometry needs evidence, not just a smoother version of its
neighbors' decision.

Using robust multi-view source-depth residual as a fallback does cover every
geometrically supported particle, but it is still not independent evidence:
those are the same foam depths that initialized fusion. It leaves Bonsai's
rounded scores unchanged and raises Room train/held-out averages by 0.02/0.01
dB, while its worst held-out view falls 13.08→13.07 dB. Raising the fallback's
acceptance threshold changes only twelve Room moves and not the tail. This path
was also removed; source depth remains an occlusion check for the photometric
objective.

Refitting normals only within coherently moved neighborhoods does not repair
that limitation either. A 25% blend toward the local least-variance direction
raises Bonsai held-out PSNR 14.07→14.11 dB, but drops Room's worst held-out view
13.08→13.06 dB and costs Bonsai 0.1 percentage point of coverage. Reducing the
blend to 10% removes the average gain and drops both Bonsai tails by 0.02 dB.
The code was removed. Even after photometric motion, the fused neighborhood is
not a reliable enough surface estimate to replace the depth-derived normal.

Scoring now keeps one GPU relight tracer alive across the training and held-out
splits instead of rebuilding the same acceleration structure and prefiltered
environment twice. Five complete Room runs fall from a 3.25-second median to
3.18 seconds (1.022× throughput); median user CPU falls 3.36→3.06 seconds and
peak charged memory is unchanged. The serialized scene is byte-identical and
all Bonsai and Room scores are unchanged.

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

Depth extraction now uses that same production walk on the GPU. Its dedicated
full-precision target returns mode depth, alpha, and peak weight without
evaluating SH colour; the ordinary viewer target remains half precision. A
half-float depth prototype failed the Bonsai held-out gate and was rejected.
Across five runs, full-precision GPU extraction reduces the complete selected
Bonsai command from 3.73 to 2.45 seconds (1.52x) and Room from 6.93 to 3.19
seconds (2.17x). Bonsai remains 14.89/14.03 dB with its worst held-out view
improving 11.47→11.48; every reported Room score is unchanged. The CPU oracle
remains available through `--cpu-depth`. Both A/B cgroups use zero swap and
record no memory event.

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
The latest GPU-produced equivalents are under
`target/audit-runs/gpu-depth-selected-{bonsai,room}-local/`.
The selected refined scenes and held-out comparison images are under
`target/audit-runs/multi-view-refined-{bonsai,room}-local/`.

An optional final pass now optimizes the surface against the renderer that
will actually consume it. `--render-refine N` chooses up to `N` observed
Gaussians in deterministic cloud order, tests a fixed quarter-radius
displacement on both sides of each current normal, and accepts only a reduction
in joint sRGB error across every training photograph. The fitted material and
capture light remain fixed, so geometry cannot win by immediately repainting
itself; held cameras never enter the objective. It adds no polygonal
intermediate and no shader variant.

A 3,000-particle real-scene gate improves Bonsai held-view mean/worst by
0.02/0.03 dB and Room by 0.07/0.05 dB. A complete Bonsai pass improves the
held mean by 0.18 dB, although its worst view loses 0.01 dB and coverage loses
0.1 percentage point. A complete Room pass improves mean/worst by 0.16/0.13 dB
and gains 0.1 coverage point. The option is therefore a validated research/
final-quality path rather than a default. On synthetic data it is more
decisive: a full pass improves held-light mean/worst by 0.36/0.36 dB and
position RMSE by 0.0018 while leaving the still-poor normal error unchanged.
The current quarter-radius choice is the conservative point of a five-step
synthetic sweep; larger steps buy more image score but perturb truth geometry
less consistently.

The implementation keeps the relight pipeline, environment, materials and
shared primitive BLAS alive while updating only surfel data and the TLAS.
That makes a 300-particle synthetic pass 9.4 times faster than rebuilding the
tracer for every candidate, with byte-identical output. Training cameras are
also recorded into one command submission and one contiguous readback per
candidate, and the scalar loss is evaluated directly from that mapped buffer.
This adds another 7.2% synthetic throughput (3.966 to 3.698 seconds) and
5.4--5.7% on the two real scenes, again with byte-identical scene files. When
shadow rays are requested, every candidate replays the same deterministic
sample sequence so noise cannot decide whether a coordinate is accepted.

`--render-refine-radii` adds two support candidates, 80% and 120% of each
current Gaussian radius, after its position decision. Twenty percent is the
selected point of a 5/10/20% five-cloud sweep. Relative to position-only at
3,000 particles, it raises Bonsai held mean/worst by another 0.04/0.03 dB and
Room by 0.21/0.22 dB; coverage is restored or increased. This doubles exact
coordinate cost, so it remains a separate final-quality switch. A four-way
normal search was rejected: even 10-degree candidates improve five synthetic
held-light means by only 0.03--0.05 dB and truth normal RMSE by roughly 0.1
degree while tripling position-only cost.

`--render-refine-rounds 64` is the faster full-cloud alternative. Every round
assigns a deterministic ±2.5% radius-normalized displacement to every observed
Gaussian, renders both signs through the same retained production tracer, and
keeps the better sign only when joint training-camera error plus a 0.001
anchor prior falls. Thus two renderer evaluations update thousands of scalar
coordinates without a new shader or differentiable surrogate. It can be
followed by `--render-refine N` when both options are supplied.

On five synthetic clouds, 64 rounds beat the 300-coordinate pass on every
held-light mean and worst view in 0.83--0.86 rather than 3.53--3.67 seconds.
Truth position RMSE is 0.0007--0.0016 world units worse, which is why this
remains an explicit speed/quality tradeoff. On Bonsai it matches the coordinate
held result at 14.52/11.71 dB and improves training PSNR to 15.90 dB in 10.0
seconds rather than 45.7. On Room it is mean-neutral and +0.01 dB worst versus
the control in 11.0 seconds; the exact 300-coordinate pass remains stronger at
14.14/13.20 dB but costs 50.7 seconds.

Tail-aware and approximate-camera shortcuts were also rejected. Weighting the
worst training camera does not transfer to the worst unseen pose. Rendering
only the cameras that observed a particle cuts the probe time by 43%, but
regresses three of five strict held-view gates; a final all-camera acceptance
render restores quality and erases the speedup. A matched ten-pair Blade
experiment also finds no measurable gain from reusing the host-side TLAS
instance buffer (5.406 seconds in both arms, byte-identical output). The
synchronized TLAS rebuild is the remaining exact-coordinate bottleneck.

Support radii cannot simply join the paired position schedule. Joint and
separate radius perturbations improve every fixed synthetic cloud and Room,
but a 64-round phase loses 0.11 dB on the worst Bonsai view and 0.2 coverage
points. Eight and sixteen radius rounds avoid that tail loss but become neutral
or negative on Room, while forbidding shrinkage accepts no useful changes.
That prototype is removed; `--render-refine-radii` continues to apply only to
the exact coordinate pass.

The production relight tracer now preserves accumulated Gaussian coverage in
its output alpha. This replaces scoring's black/white background pair with one
black-background render: RGB supplies PSNR and alpha supplies coverage. A GPU
oracle checks the coverage compositor against the CPU implementation, and the
existing fixed-cloud score keeps the same 53.6% coverage while score dispatch
and readback count is halved. Presentation continues to ignore alpha.

A follow-up foreground-MSE experiment did not justify extending the optimizer.
At weight 0.05, a 64-round position-plus-radius pass improved three of five
synthetic held-light means and tails, was mixed on one, and regressed the fifth.
Since ordinary COLMAP captures also have no foreground mask, both the loss
control and the batched-radius prototype were removed.

## What the numbers are limited by, in order

1. **Geometry.** Voxel-averaging per-view depth modes is an initializer, not a
   final surface. Two-view consensus halves the cloud, and the first selected
   shared-view plane sweep improves held-out quality without changing coverage.
   It can act only on the 18–19% of particles that have four visible textured
   patches, however; geometric/source-depth visibility rejects most particles
   before texture is tested, and loosening the texture floor is measured and
   rejected. Nearly three quarters of the retained surfels are also never
   directly observed by the material fit, and the overlap diagnostic finds an
   order of magnitude more projected supports than on the truth surface. The
   analytic curved/occlusion fixture now covers this boundary. The next step is
   cloud-only refinement that gains reliable evidence for thin or weakly
   textured geometry and updates normals or support radii when positions move;
   local offset propagation is measured and is not that evidence.
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
