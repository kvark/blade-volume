# Relightable Reconstruction Roadmap

## Goal

Recover one cloud-only scene from posed photographs that supports both novel
views and novel lighting. The durable asset contains point-sampled geometry,
surface orientation, material parameters, and lighting/transport data. It does
not contain or fall back to polygonal geometry.

## Where we are

- Novel-view rendering under the captured light works.
- Controlled multi-light captures prove the material and relighting path end
  to end.
- Training-mask filtering gives a clean, reproducible gain on a fresh
  OpenIllumination object, but the excluded-light result remains too uniform
  where the photograph contains directional self-shadowing.
- Increasing opacity or radius globally, splitting a broad Gaussian by its
  missing-ray owner, and validating an owner-depth patch all improve mask
  recall but lose known-light or held-light image quality. These experiments
  show that the remaining surface cannot be invented from a current
  particle's footprint: it needs independent cross-view correspondence.
- Independent all-foreground correspondences recover a recognizable outline,
  but their depth layers are not accurate enough to initialize or replace the
  production cloud. Appearance-based filtering and normal-guided point motion
  improve known-light renders while regressing unseen lights.
- Dense calibrated-light normals are coherent at object scale, but integrating
  them away from verified track depths fails the independent-camera precision
  gate. Normals are surface evidence, not a substitute for correspondence.
- A uniform Gaussian sheet remains image-stable under deterministic 2× point
  resampling, both before and after the staged support fit. Real 5,000-point
  failure is therefore not a generic opacity-density law; it comes from adding
  mutually inconsistent surface layers.
- The final PBR Gaussian now supports the existing coupled sampled-visibility
  and one-bounce renderer. On the selected fresh object this improves every
  held-light mean/tail at four and eight samples, although it cannot repair the
  wrong surface beneath those shadows.
- Observation-backed additive groups can improve the scalar point surface, but
  the PBR Gaussian fit either perturbs the established cloud or suppresses the
  additions. Restoring the exact established Gaussian prefix leaves only four
  useful-opacity additions and no measurable Gaussian image gain.
- A fixed-cloud scalar-alpha distillation probe confirms a real representation
  gap, but reducing it is not a valid reconstruction objective by itself.
  Foreground-balanced fitting raises Gaussian recall and scalar-teacher PSNR
  while lowering held-photo tails and precision. The scalar renderer is a
  useful geometric control, not ground-truth opacity for the volumetric
  compositor.

The current bottleneck is therefore **cross-view surface ownership first,
non-diffuse material recovery second, radiometry and unknown-light estimation
third**. The selected representation-aware diffuse transfer and coupled
transport renderer remove two implementation gaps, but material, visibility,
and illumination can still compensate for wrong or missing geometry.
Optimizing all of them together now would make the decomposition less
identifiable rather than more accurate. The scalar surfel asset remains a
valid cloud-only geometry control; no polygonal fallback is needed.

## Execution plan

The rows are ordered by reconstruction dependency. Independent renderer work
may land early, but no later parameter family may supervise an unaccepted
surface.

| Phase | Status | Next action | Decision gate |
| --- | --- | --- | --- |
| Capture integrity | selected | Keep held cameras physically absent from dense reconstruction; canonicalize masks on import | Rebuilding a training asset cannot read an excluded pose or light |
| Dense support | selected and independently validated | Keep the training-mask soft visual hull before spatial downsampling; do not use it as evidence that Gaussian transfer or relighting is solved | Held-camera scalar foreground mean/worst and precision improve on another object; report the small recall trade and mixed Gaussian/light results |
| Missing support | diagnostics only | Keep verified tracks and normal/depth sweeps out of production until a complete-render held-view gain | Recall and covered quality rise without a precision or PSNR regression |
| Representation scale | final-compositor diffuse transfer selected and automatic for large individual tables; topology replacement, additive support, and scalar-alpha distillation rejected | Keep the one-proposal transfer narrow; keep small shared palettes behind `--render-refine-materials` and on their bounded joint solver | Preserve the exact same-cloud gains on three material classes while geometry, visibility, and non-diffuse properties remain unchanged |
| Light transport | renderer selected | Use coupled sampled visibility and one bounded bounce: four samples for iteration, eight for selected output | Both excluded-light mean/tail improve with no coverage regression or GPU fault |
| Radiometry | blocked on surface | Measure one scalar exposure residual per capture; fit it only if the residual is view-wide rather than directional | A constrained gain transfers to held lights and does not enter albedo |
| Materials and capture layout | later | Add spatially shared roughness only after diffuse transfer; then accept sparse `(camera, light, exposure)` observations | Two held lights and a second object improve; otherwise keep the feature off |

The selected dense-support prerequisite remains the training-mask hull. On the
`obj_31_painted_toy` split, filtering 68,278 of 1,254,170 one-view fused samples
before allocating the 2,500-point budget raises observed surfels from 1,887 to
2,385. Known-light held-camera foreground mean/worst improves
`16.46/15.38→17.54/16.87` dB and precision `78.9%→93.1%`; recall changes
`98.4%→96.8%`. Excluded patterns 005/006 also improve by `0.32/0.24` and
`0.13/0.08` dB foreground mean/worst, respectively, but pattern 006 still
loses badly to black because its reference is directionally self-shadowed and
the reconstruction is nearly uniformly lit.

A 5,000-point filtered scalar selection control reaches `18.15/17.52` dB
held-camera foreground with 97.7% recall and 93.5% precision, showing that
denser surface support is useful. The complete held-light run is mixed:
pattern 005 moves `15.32/14.06→15.47/14.01` dB and pattern 006 moves
`13.61/12.06→13.68/12.11` dB. It is therefore not selected. The Gaussian fit
retains only 1,390 of its 5,000 inputs; disabling pruning leaves a median
opacity of 0.019, which says the optimizer does not want most added layers.
Raising per-pixel candidates from 64 to 128, scoring at twice the image width,
and expanding survivor support do not reverse that decision: support expansion
raises recall but loses precision and foreground PSNR. A 3,500-point control is
also worse for Gaussian PBR. Nearest-view PatchMatch source generation,
equal-weight radius-mask loss, and skipping the joint known-light material fit
were isolated and removed after negative controls.

The pose-only path now emits that nearest-camera graph directly from calibrated
training poses, fuses COLMAP depth/normal maps in Rust, keeps each point's
camera/pixel/depth/normal/confidence observations together, and can persist the
raw candidate set. Masks are reapplied on every replay. It does not use a fake
sparse track or any polygonal
intermediate. The implementation is useful and remains opt-in, but it does not
yet change the selected reconstruction: its fixed scale gate is mixed, as
recorded below.

The same hull filter now passes its intended independent-object scalar gate on
`obj_29_fabric_toy`. COLMAP receives only 38 construction poses; ten official
test poses are absent from its byte-identical nearest-12 graph. Native Rust
fusion produces 213,642 groups from 1,511,977 observations. The hull removes
5,772 groups before the fixed 2,500-point selection and raises observed surfels
from 2,289 to 2,378. Two filtered runs reproduce scalar held-camera scores
within 0.01 dB:

| Fabric-toy surface | Scalar known-light test | Scalar recall / precision | Gaussian known-light test | Gaussian recall / precision |
| --- | --- | --- | --- | --- |
| Unfiltered | `27.27/25.26; 18.22/16.00` | `97.0/91.1%` | `26.85/26.02; 18.04/16.71` | `95.8/90.0%` |
| Training-mask hull | `27.58/25.70; 18.27/16.14` | `96.8/93.2%` | `26.73–26.81/25.73–25.82; 18.10–18.14/16.77` | `96.0–96.1/88.9–89.2%` |

Values are whole-frame mean/worst; foreground mean/worst. This validates the
surface cleanup for the scalar cloud: every scalar image metric and precision
improves, at a 0.2-point recall cost. It does not validate the complete
relightable Gaussian. Gaussian foreground improves, but whole-frame quality and
precision regress. Under excluded pattern 006 the filtered scalar improves
from `23.77/22.81;14.01/12.72` to
`23.99/22.97;14.12/12.75`; under pattern 005 whole-frame mean rises 0.07 dB
while foreground falls 0.09/0.06 dB. Both excluded-light outputs remain well
below trivial foreground baselines. The hull therefore stays selected as an
upstream geometric cleanup, while representation transfer and relighting remain
separate open gates.

## Grouped dense-fusion gate

`import_openillumination` now writes `patch-match-train.cfg` with the nearest
12 calibrated training cameras per reference. On the fresh painted-toy
capture, the Rust output is byte-identical to the independently generated
configuration (SHA-256
`96e835239a90b1b92097dfc60f07338a60b0116ab895137d3ef8d1cd01102e88`).
Held cameras never occur in that file. Native fusion through the graph produces
163,693 depth-consistent groups from 1,108,042 observations; the training-mask
soft visual hull leaves 151,969 groups and 1,072,403 observations. The 41.2 MB
raw-fusion cache binds the ordered 33-image construction split and fusion
thresholds, while deliberately reapplying the current masks on every replay.
It has SHA-256
`4b875b0be0c77aab16c1558c9905d0896dbd598521bbdc3f7defcfb35694ab5d`.

The adjacent one-view control and two replays from that exact cache give:

| Candidate | Scalar test whole | Scalar test foreground | Scalar recall / precision | Gaussian test whole | Gaussian test foreground | Gaussian recall / precision |
| --- | --- | --- | --- | --- | --- | --- |
| one-view control, 2,500 | 28.08 / 26.35 | 18.39 / 17.29 | 97.2% / 94.2% | 25.97 / 24.67 | 16.74 / 15.65 | 92.5% / 88.6% |
| grouped, 2,500 replay 1 | 28.00 / 26.20 | 18.36 / 17.39 | 97.3% / 93.8% | 26.02 / 24.84 | 16.73 / 15.91 | 92.2% / 89.9% |
| grouped, 2,500 replay 2 | 27.99 / 26.20 | 18.35 / 17.38 | 97.3% / 93.8% | 26.02 / 24.78 | 16.90 / 15.96 | 92.1% / 89.2% |
| grouped, 5,000 | 28.34 / 26.65 | 18.55 / 17.53 | 97.3% / 94.7% | 25.18 / 23.85 | 17.02 / 15.77 | 95.2% / 80.3% |

Values are mean/worst dB. At 2,500 points the scalar whole-frame and foreground
means regress slightly, while the Gaussian foreground mean straddles the
control across repeats; at 5,000 points the scalar improves while Gaussian
precision loses 8.3 points and whole-frame quality falls sharply. Neither
passes the contract, so the official cameras and
excluded patterns 005/006 remain closed and the default stays on the selected
one-view input. Ranking by confidence before view count, multiplying confidence
by view count, and giving each camera one vote in the fused representative were
all tested from the same cache and removed after worse internal results.

The fixed runs peak at 0.76--1.46 GB cgroup memory with zero swap, OOM, or GPU
fault. A later unclean host reboot left one Cargo output as a correctly sized
but zero-filled sparse file; it was detected before execution. A clean build in
an isolated target produced a valid x86-64 ELF and subsequent cgrouped GPU runs
completed normally. This is build-artifact corruption across a reboot, not an
accepted reconstruction result.

## Latest surface attribution

The fresh-object split now rules out seven tempting shortcuts:

- Matching every textured foreground pixel across 26 construction cameras
  yields 1,827 accepted tracks; 1,711 pass the independent selection-camera
  visual hull and 1,654 form coherent shared-view patches. Their held-view
  footprint is recognizable, but direct initialization falls to 53.5% recall
  and 62.7% precision. The ordinary all-camera hull leaves only 882 points and
  63.0% recall. The generic missing-alpha diagnostic remains useful; an
  all-foreground production mode does not.
- Filtering 20,000 dense hypotheses by agreement of aligned-light response
  improves known-light foreground to `17.63/17.00` dB, but pattern 005's worst
  view falls to 13.95 dB and pattern 006 falls to `13.58/11.99` dB. Response
  similarity is not a depth or visibility certificate.
- Moving all points a bounded distance towards planes implied by photometric
  normals raises observed support to 2,424 and known-light foreground to
  `17.70/16.90` dB, but held patterns fall to `15.21/13.75` and
  `13.36/11.90` dB. Photometric normals cannot move geometry without an
  independent depth anchor.
- Anchoring local tangent-plane integration at verified tracks produces 1,450
  cross-camera candidates, but the three components reach only 84.6–89.8%
  selection foreground precision and are all rejected. Tight conditioning
  leaves only two accepted points and 0.0% validation missing recall; a 3×3
  normal consensus is no better. Dense normals remain valid evidence once a
  depth is independently confirmed, but the unused estimator, rejected depth
  propagation, and all tuning hooks were removed.
- A track-local multi-view plane sweep fits 8,552 neighboring depths and
  cross-validates 2,155 of them. One 495-point component passes the footprint
  screen at 5.3% validation missing recall, 67.6% missing precision, and 99.3%
  foreground precision. At 0.25 opacity it improves selection recall/precision
  but loses 0.07/0.06 dB foreground mean/worst; at 0.05 opacity it is nearly
  neutral but still negative. A five-light shared patch material does not
  rescue it, and an anchor-scale descriptor threshold rejects every component.
  The sweep, append, and fitting code were removed.
- Using cast-shadow visibility in the five-light material solve is correct on
  an exact occluder, but the reconstructed visibility makes every real score
  worse. Reducing shadow bias from three radii to one changes openness
  `46%→35%` and is nearly neutral, with mixed held-light tails. Transport must
  follow an accepted surface update rather than supervise the current one.
- Treating the 5,000-point failure as a generic density-normalization problem
  is contradicted by the point-only sheet oracle. Its 9×9 and 17×17
  initializations have central mean alpha `0.3714/0.3799`, integrated mean
  alpha `0.0312/0.0276`, peak alpha `0.4144/0.4054`, and mean/max central error
  `0.0167/0.0338`. After 90 staged updates, whole-image mean alpha is
  `0.0550/0.0555` with `0.0050` mean image error. Normalizing the clean case
  would conceal the real input-layer problem.
- COLMAP's paired visibility stream confirms that the selected `fused-min1`
  input is not multi-view support: all 1,254,170 points have exactly one source
  image. A `min5` control has 24,117 points with 8.1 sources on average, so the
  parser and the input are behaving as specified. Equalizing source images in
  each output voxel and requiring two sources was tested and removed. At 2,500
  points its scalar result loses 0.10/0.13 dB foreground mean/worst, and its
  Gaussian result loses mean quality and 1.4 points of precision. At 5,000
  points it still leaves only 1,332 useful Gaussian particles. Source count is
  provenance, not evidence that samples belong to the same surface layer.
- Reprojected front-most counts do not fix ownership either. Requiring eight
  cameras raises the internal scalar foreground result from `18.38/17.32` to
  `18.58/17.70` dB, but the Gaussian retains only 726 of 2,499 particles and
  recall falls to 37.0%. At twelve cameras recall is 43.7% with a scalar
  regression. Both the flag and implementation were removed.
- The pose-only OpenIllumination model has no sparse tracks, so COLMAP's fusion
  overlap graph is empty and `min1` cannot traverse to another image. An
  external all-camera overlap hint makes the unchanged depth/reprojection/
  normal checks produce 121,346 genuine `min2` points (3.7 source images on
  average) or 64,025 `min3` points (5.2 on average). Both are mixed on the
  official ten-camera/two-light gate: they improve several foreground tails
  but lose whole-frame means and about two points of precision. They are not
  selected. The durable fix is an explicit capture-stage overlap graph, not a
  synthetic geometry track.
- The PBR support guard previously accepted a fit when exactly one quarter of
  its particles survived. Fusion and responsibility controls retain 27–40%
  yet render only 36–39% recall, so the permissive boundary moves to one half.
  Established successful fits retain more than 80%; the selected cloud was
  already below the old boundary and is unchanged. This removes an arbitrary
  quality cliff without selecting a new reconstruction input.

The ownership decision now sits upstream of final downsampling: the explicit
graph, native fusion, intact observation groups, mask filtering, deterministic
cache, and group-preserving initialization are implemented. The fixed gate says
that view count and geometric confidence are still insufficient to choose a
useful Gaussian mixture.

### Rejected static-light responsibility gate

An exact leave-one-out complete-render score was tested as a selector over a
fixed 5,000-group proposal pool. Early controls exposed two invalid comparisons:
recomputing the output grid changed the spatial partition, and subsetting the
proposal Gaussian retained radii estimated at 5,000-point spacing. The corrected
experiment nests the selected 2,500 groups inside the proposal pool, ranks
replacements only within the original raw-cloud 2,500-cell grid, and rebuilds
positions, radii, orientation, and PBR support from the chosen groups.

The following internal results use 33 construction and five test cameras; the
official cameras and excluded lighting patterns 005/006 remain unopened. Values
are whole-frame mean/worst, foreground mean/worst, then recall/precision:

| Selection | Scalar test | Gaussian test | Gaussian recall/precision |
| --- | --- | --- | --- |
| Zero-score rebuild control | `28.00/26.20; 18.36/17.38` | `26.10/24.94; 16.87/16.00` | `92.1/89.6%` |
| Static responsibility, repeat 1 | `28.03/26.27; 18.38/17.40` | `26.03/24.87; 16.63/15.99` | `92.3/90.1%` |
| Static responsibility, repeat 2 | `27.98/26.24; 18.35/17.39` | `26.18/25.13; 16.79/15.86` | `92.1/89.6%` |

The zero-score control reproduces the direct 2,500-point checkpoint within run
variance, validating the corrected rebuild. Static-light responsibility is
neutral for the scalar cloud but lowers Gaussian foreground mean and/or tail in
both repeats, so the selector, CLI surface, and scoring code were removed. Low
static opacity is descriptive, not evidence that a proposal is unnecessary
under a different light after normals, material, visibility, and overlap have
exchanged responsibility.

### Rejected calibrated scalar-PBR ownership gate

The follow-up evaluated one nested alternate per fixed raw-cloud cell after
normal and material fitting. One antithetic pair of complete scalar PBR renders
across all five calibrated lights supplied a local error difference for every
projected point; a third complete render accepted the mixed proposal only when
it lowered the full objective. Both runs considered 1,128 cell-local
alternatives and selected 69. The selected intact depth groups were written to
a cache, then replayed through a fresh surface, material, support, and Gaussian
fit rather than reusing proposal radii or appearance.

| Selection | Scalar test | Scalar recall/precision | Gaussian test | Gaussian recall/precision |
| --- | --- | --- | --- | --- |
| Zero-score rebuild control | `28.00/26.20; 18.36/17.38` | `97.3/93.8%` | `26.10/24.94; 16.87/16.00` | `92.1/89.6%` |
| Scalar PBR ownership, repeat 1 | `28.05/26.28; 18.41/17.44` | `97.5/93.7%` | `26.08/24.55; 16.84/15.79` | `92.5/89.4%` |
| Scalar PBR ownership, repeat 2 | `28.03/26.24; 18.41/17.43` | `97.5/93.7%` | `26.17/24.91; 16.81/16.03` | `91.9/90.3%` |

The calibrated scalar objective falls from `0.0031346→0.0031269` and
`0.0031364→0.0031276`, and scalar fresh-camera quality improves slightly. It
does not transfer consistently to volumetric PBR support: repeat 1 loses both
Gaussian foreground metrics and repeat 2 trades foreground mean and recall for
tail and precision. The selector, grouped-index API, environment hook, and test
were removed. All runs stayed below 0.9 GiB scoped memory with no swap, OOM, or
GPU fault.

### Rejected Gaussian-PBR fixed-cell ownership gate

The same one-alternate cells were then scored in the learned PBR Gaussian
compositor after calibrated normal, material, opacity, covariance, and center
fitting. A temporary count-preserving Gaussian upload updated its existing
buffers and TLAS; no shader or render variant was added. The physical-GPU
oracle selected a known-correct center and rejected a known-wrong one.

On the real capture, 1,093 of the 1,128 local alternatives correspond to an
observed fitted particle. Neither the all-alternate nor localized complete
multi-light proposal improves the baseline in either fit: zero alternatives are
selected, at unchanged objectives `0.0045062` and `0.0044915`. With no changed
group set there is nothing to rebuild or score on fresh cameras. The updater,
selector, grouped-index API, environment hook, and test were removed. Both runs
stay below 0.9 GiB scoped memory with zero swap, OOM, or GPU fault.

This closes the fixed-cell replacement branch: confidence already chose the
best useful group in those cells, and selecting another existing layer does not
recover missing support. The bounded additive diagnostic below tests missing
support without replacing that established ownership. The official cameras and
excluded lights remain closed. No polygonal intermediate or fallback is
introduced.

### Rejected observation-backed additive-support gate

The additive diagnostic starts with the exact selected 2,500-group prefix and
a nested 5,000-cell proposal set. It renders final Gaussian alpha on the 33
construction cameras, then considers an alternate only in cameras that are
named by that group's retained fusion observations. This visibility boundary
matters: merely counting every camera where a point projects selected occluded
depth layers and was discarded as an invalid control. A candidate must
under-cover foreground in at least three and a majority of its actual source
cameras. At most one addition is retained near any established point.

Of 3,604 non-baseline proposals, 99 pass that observation-backed test and 76
remain after spatial separation. They are appended rather than substituted. A
stage-matched checkpoint keeps the original surface and Gaussian prefix exact,
while the suffix learns opacity, covariance, position, orientation, and PBR
appearance through the ordinary pipeline. Values below are internal fresh
cameras: whole-frame mean/worst; foreground mean/worst, then recall/precision.

| Support | Scalar test | Scalar recall/precision | Gaussian test | Gaussian recall/precision |
| --- | --- | --- | --- | --- |
| Exact 2,500 baseline | `28.00/26.19; 18.36/17.37` | `97.3/93.8%` | `26.07/24.99; 16.83/15.93` | `92.3/89.6%` |
| 76 additions, ordinary joint fit | `28.06/26.30; 18.42/17.48` | `97.6/93.8%` | `25.72/24.42; 16.76/15.94` | `93.4/86.4%` |
| 76 additions, exact final Gaussian prefix | `28.02/26.30; 18.38/17.47` | `97.6/93.7%` | `26.07/24.98; 16.83/15.94` | `92.4/89.5%` |
| Fresh rebuild of four learned survivors | `27.99/26.25; 18.34/17.36` | `97.3/93.9%` | `26.07/24.99; 16.83/15.93` | `92.3/89.6%` |

The scalar result says the observation-backed geometry contains a small amount
of useful surface. The ordinary Gaussian fit, however, removes 72 of 76
additions and loses 3.2 points of precision while perturbing the established
mixture. Restoring every established Gaussian field makes the remaining four
additions image-neutral. Selecting those four by learned opacity and rebuilding
from their intact evidence groups does not transfer the scalar gain: Gaussian
metrics reproduce baseline and scalar foreground falls slightly. A scale cap
is not causal—the suffix already shrinks to `0.76×` after the first support fit
and reaches only `1.00×` after the multi-light fit.

This rejects undercoverage alone as the next production selector. More raw
groups are not useful until the PBR Gaussian can reproduce the better scalar
surface without changing that surface's ownership. The next gate therefore
holds one cloud, material table, cameras, and lights fixed and trains only the
Gaussian representation against scalar-rendered alpha/radiance. It must improve
fresh-camera scalar-to-Gaussian parity without losing photo quality or mask
precision before any additive candidates return. All selector, checkpoint,
prefix-isolation, survivor-cache, grouped-index, and scale-cap code was removed.
The scoped GPU runs peak below 0.80 GiB, with zero swap, OOM, or GPU fault.

### Rejected scalar-alpha distillation gate

The follow-up freezes one reconstructed 2,500-point surface, its material
table, cameras, lights, Gaussian centers, rotations, and durable appearance.
The scalar production renderer supplies an in-memory RGB/alpha teacher at the
same 128-pixel resolution and eight-sample transport setting. The untouched
PBR Gaussian is scored against both photographs and that teacher in the same
process, eliminating reconstruction and renderer-startup variation.

The fixed cloud has a measurable representation gap. On the five fresh
cameras, the scalar surface scores `28.00/26.20;18.36/17.39` dB and
`97.3/93.8%` recall/precision. The untouched Gaussian scores
`26.06/24.84;16.86/15.99` dB and `92.5/89.3%` against photographs, while its
scalar-teacher score is `28.05/26.75;19.21/17.95` dB and `91.4/91.5%`. Across
all 38 cameras, 65,000--66,000 pixels have scalar alpha more than 0.01 above
the Gaussian and roughly 40,000--42,000 differ in the other direction.

The valid continuation uses the pointwise maximum of established Gaussian and
scalar alpha, so it can fill scalar coverage but cannot erase established
support. Neutral SH-0 colour is premultiplied consistently with alpha, and only
opacity plus the existing three anisotropic scales learn. An ordinary uniform
sampler is nearly neutral. A deterministic sampler then draws half of every
batch from the target support and half from the full frame, using the same
graph, operations, candidates, and optimizer:

| Matched candidate | Photo test | Photo recall / precision | Scalar-teacher test | Teacher recall / precision |
| --- | --- | --- | --- | --- |
| Untouched control, 25-step run | `26.08/24.88;16.89/16.03` | `92.4/89.2%` | `28.03/26.71;19.21/17.89` | `91.3/91.4%` |
| 25 balanced updates | `26.07/24.85;16.89/15.98` | `92.7/89.1%` | `28.04/26.72;19.24/17.88` | `91.6/91.4%` |
| Untouched control, 100-step run | `26.09/24.80;16.87/16.01` | `92.3/89.5%` | `28.02/26.58;19.15/17.87` | `91.2/91.7%` |
| 100 balanced updates | `26.04/24.68;16.87/15.91` | `93.5/89.1%` | `28.07/26.62;19.26/17.89` | `92.5/91.5%` |

Values are whole-frame mean/worst; foreground mean/worst. The longer fit
lowers its sampled objective `0.1321805→0.1187916` and clearly improves
teacher means and recall, yet loses photo whole-frame mean/tail, foreground
tail, and precision. Earlier symmetric targets were invalid controls: the
mostly black frame drove opacity toward zero, and a black RGB target was
inconsistent with the existing neutral-grey SH convention. They are excluded
from the decision rather than presented as negative evidence.

This closes scalar alpha as a standalone distillation target. The two
renderers deliberately use different response and same-sheet composition
semantics, so closer alpha is not necessarily closer photographic radiance.
Future transfer must match the final Gaussian compositor's colour and
visibility response, and must beat the untouched photo control rather than
only its scalar teacher. The sampler, teacher, environment hook, matched clone,
and scoring code are all removed; no API, shader, graph operation, model field,
or dependency remains. The cgrouped runs peak at 0.76 GiB with zero swap, OOM,
validation error, Xid, or GPU fault. Ignored logs and artifacts remain under
`target/audit-runs/openillumination/gaussian-parity-*`.

## Selected final-compositor diffuse transfer

An RGBA error-budget diagnostic on the fixed fabric-toy Gaussian locates
70.9% of held-camera RGB error on correctly covered foreground and only 1.5%
on missing foreground; false-positive support contributes another 18.7%.
The rendered covered foreground is roughly twice as bright as the photograph.
This changes the next action from adding support to correcting the material
table in the renderer that actually consumes it.

The existing `--render-refine-materials` path could solve at most 32 shared
materials jointly. With 2,500 one-per-surfel materials it instead attempted
thousands of full renders and was not practical. Large individual-mode tables
now receive one bounded proposal automatically after Gaussian geometry,
opacity, covariance, normals, and materials are final:

1. render the final cloud with its current and zero diffuse albedo using the
   same deterministic transport samples;
2. use its zero-to-current secant response to fit one scalar gain on covered
   construction-camera foreground only;
3. apply half of the fitted correction toward identity, protecting unseen-view
   tails from the full training optimum;
4. retain it only when one exact complete construction render lowers the
   ordinary objective.

Small shared palettes keep their existing joint solve and coordinate polish.
Large scalar-surface passes stay unchanged, and exact identity-mapped tables
skip meaningless material reassignment. There is no new option, shader,
graph operation, binding, model field, format, or dependency.

| Exact matched Gaussian | Known-light held camera | Excluded 005 | Excluded 006 | Recall / precision |
| --- | ---: | ---: | ---: | ---: |
| Fabric toy, before | `26.73–26.81/25.73–25.82; 18.10–18.14/16.77` | `23.55–23.59/22.74–22.81; 13.97–14.01/12.99–13.02` | `23.72–23.75/22.21; 14.08–14.10/12.76–12.79` | `96.0–96.1/88.9–89.2%` |
| Fabric toy, gain `0.631581` | `27.54/26.47; 18.76/17.60` | `24.35/23.24; 14.80/13.39` | `25.04/22.71; 15.43/13.25` | `96.1/89.1%` |
| Painted toy, matched before | `26.38/25.60; 16.56/15.77` | `24.98/24.47; 14.87/14.07` | `23.45/22.87; 13.19/12.38` | `93.3/89.5%` |
| Painted toy, gain `0.765983` | `26.91/25.98; 17.03/16.18` | `25.63/25.15; 15.52/14.58` | `24.32/23.24; 14.07/12.97` | `93.3/89.5%` |
| Fresh metal sculpture, exact before | `25.57/23.33; 16.39/14.36` | `23.78/22.16; 13.33/12.75` | `24.02/22.56; 13.55/12.73` | `87.2/74.5%` |
| Fresh metal sculpture, gain `0.616812` | `26.46/24.04; 17.05/14.80` | `24.77/23.27; 14.33/13.26` | `25.11/23.09; 14.64/13.34` | `87.2/74.5%` |

Values are whole-frame mean/worst; foreground mean/worst. All three objects
improve every image metric under the known light and both excluded lights;
matched geometry metrics are identical. The full unregularized painted-toy optimum was
rejected because it lost 0.05 dB on one known-light whole-frame tail. The
conservative transfer still does not beat the trivial excluded-light
foreground baselines, so it closes a Gaussian representation mismatch—not the
real relighting gate. The metal sculpture was predeclared before its images or
scores were inspected. Its exact control removes the accepted gain from the
same persisted cloud and scores both variants in one process: every known and
excluded-light mean/tail improves with identical recall and precision. This
passes the automatic-enablement gate, but its low precision and badly missing
thin parts make improved surface ownership and non-diffuse material recovery
the next quality problem.

## Milestones

### 1. Recover missing surface tracks

Build explicit tracks from foreground pixels the current cloud does not cover:

1. Encode four aligned known-light images as the existing gain-invariant
   photometric response.
2. Match compact local response descriptors along calibrated epipolar lines.
3. Require mutual matches, useful parallax, and agreement in at least three
   training cameras.
4. Triangulate one world point from the independent camera rays and reject it
   by reprojection error, foreground masks, descriptor cost, and scene bounds.
5. Write a diagnostic point cloud and track statistics before changing a
   production model.

The current cloud may constrain the depth interval, but its particle ownership
must not define a correspondence.

**Gate:** synthetic triangulation tests are sub-pixel; real tracks form
contiguous object patches rather than rays or isolated dust; no held camera or
held light contributes to track construction.

### 2. Turn accepted tracks into cloud support

Merge coherent neighboring tracks into oriented point samples, estimate local
spacing and normals, and feed them through the existing Gaussian support,
material, and refinement paths. Keep old points in place until a complete
render proves that a new patch is useful.

**Gate:** on withheld training cameras, foreground recall and covered-pixel
quality improve without reducing precision or known-light mean/worst PSNR.
Then require the same direction on both excluded OpenIllumination patterns.

### 3. Calibrate captures before fitting more light parameters

Fit one exposure scalar, or a tightly constrained RGB gain when a gray-card
measurement justifies it, per capture. Do not add per-point/per-light nuisance
appearance. Record exposure, ISO, white balance, and measured emitter power in
new captures.

**Gate:** calibration improves held-light residuals consistently and does not
paint lighting into the recovered diffuse albedo.

### 4. Alternate geometry and finite-light visibility

The renderer now evaluates arbitrary environments with coupled sampled
visibility and one bounded bounce for both scalar surfels and the final PBR
Gaussian. Once support is contiguous, use that same estimator while
alternating small fitting stages instead of freezing a transport factor:

1. recompute direct visibility from the current point cloud;
2. fit diffuse material and the known/low-dimensional light;
3. update geometry and normals from the residual;
4. repeat, then add a bounded indirect term.

Start unknown illumination with low-order spherical harmonics. Add a small
positive set of sharper lobes only after the diffuse gate passes.

**Gate:** shadows move in the correct direction under excluded lights and the
model improves over black and capture-light-copy baselines on foreground
mean/worst PSNR, not only on training re-renders.

### 5. Add material complexity only when identified

Fit diffuse albedo and normals first. Cluster roughness spatially rather than
giving every point a free value. Fit reflectance only where several light/view
combinations observe a stable specular response.

**Gate:** each additional parameter family improves two held lights and a
second object. Otherwise it remains opt-in or is removed.

### 6. Generalize the capture layout

After the aligned multi-light gate passes, replace the directory-alignment
assumption with sparse `(image, camera, light, exposure)` observations. This is
the route from a fixed rig to a moving phone with cycled or strobed lights.

## Evaluation contract

Every selected result records:

- exact training, validation, held-camera, and held-light splits;
- whole-frame and foreground mean/worst PSNR;
- foreground precision, recall, and covered-pixel quality;
- reference/render images from the same persisted asset;
- point count, wall time, peak cgroup memory, swap/OOM counters, and GPU faults;
- black, average-image, and capture-light-copy baselines where applicable.

No held observation may choose geometry, thresholds, iterations, or a
checkpoint. A change is selected only after an adjacent unchanged control and
a deterministic repeat agree within the measured training variance.

## Immediate implementation

- [x] Add a small Rust module for calibrated ray triangulation, epipolar
  candidate search, mutual descriptor matching, and track validation.
- [x] Cover exact synthetic points, weak-parallax rejection, mismatched
  descriptors, masks, and held-view exclusion with CPU tests.
- [x] Add an opt-in diagnostic to `reconstruct` that writes proposed tracks;
  do not mutate the fitted surface yet.
- [x] Run it on the fixed OpenIllumination training split and inspect the cloud
  plus per-stage rejection counts.
- [x] Group tracks by shared-view adjacency, replay them twice from one fixed
  base, and screen every component through complete selection and validation
  renders.
- [x] Open the official camera and excluded-light gate only after the repeats;
  reject and remove the append when its tiny internal gain does not transfer.
- [x] Preserve the verified surfels and add deterministic half-radius samples
  on their two-nearest local cloud edges; repeat the track-only validation.
- [x] Run continuous shared-view patches and an all-foreground track
  initializer on the fresh object; reject both when their recognizable
  outlines fail complete-render precision and held-light gates.
- [x] Recover a dense photometric normal map independently in each aligned
  training view, with confidence from directional-light conditioning.
- [x] Test local tangent-plane depth integration from verified tracks; reject
  and remove it when all useful components fail selection-camera precision.
- [x] Extend verified tracks with a bounded local multi-view depth sweep;
  remove it when the dense component passes footprint but fails complete
  renders at both useful and conservative opacity.
- [x] Add a synthetic Gaussian-sheet density-invariance oracle and record peak
  alpha, integrated alpha, covered pixels, and image error at 1×/2× sampling.
- [x] Repeat staged support fitting at 1×/2× sheet density; reject generic
  normalization because the fitted images already agree within 0.005 mean
  alpha error.
- [x] Isolate the real 5,000-point collapse with candidate-count, image-scale,
  no-prune, and support-expansion controls; retain 2,500 points because the
  extra real samples are inconsistent layers rather than clean resampling.
- [x] Enable coupled sampled visibility and one bounce for PBR Gaussians, and
  gate four/eight samples on known and two excluded lights.
- [x] Parse and validate COLMAP's paired dense source-view provenance; record
  that the selected `min1` cloud contains exactly one source per point.
- [x] Test and reject equal source weighting plus a minimum-source voxel rule;
  it does not establish cross-view depth agreement and loses the adjacent gate.
- [x] Map source-image indices through the dense workspace and test/reject a
  front-most projection-count filter at adjacent thresholds.
- [x] Isolate the empty-overlap failure in pose-only COLMAP fusion and test
  genuine two/three-view fused controls on the full official scoreboard.
- [x] Produce an explicit overlap graph without synthetic geometry and retain
  grouped fusion observations through Gaussian support initialization.
- [x] Repeat the fixed 2,500/5,000-point gate from one persisted candidate set;
  require both scalar and Gaussian PBR to improve before changing the default.
- [x] Build and repeat the fixed 5,000-group static-light responsibility
  selector; reject and remove it because rebuilt PBR Gaussian foreground
  quality loses to its zero-score control in both repeats.
- [x] Attribute support with calibrated multi-light scalar PBR renders inside
  fixed raw-cloud cells; reject and remove it because its small scalar gain does
  not transfer to rebuilt Gaussian foreground quality in two repeats.
- [x] Attribute fixed-cell alternatives in the learned PBR Gaussian compositor;
  reject and remove it because neither repeat selects one useful alternative.
- [x] Add bounded grouped candidates only where several actual fusion source
  cameras agree the fitted Gaussian under-covers foreground; reject and remove
  the selector because 72 of 76 are suppressed and the four fresh-rebuild
  survivors do not improve Gaussian images.
- [x] Measure fixed-cloud scalar-to-Gaussian alpha/radiance parity and test
  uniform plus foreground-balanced alpha continuations; reject scalar alpha as
  a standalone target when stronger parity loses photo tails and precision.
- [x] Validate the selected training-mask surface filter on another object,
  retaining the scalar surfel output as the cloud-only quality control; record
  that Gaussian whole-frame/precision and one excluded light remain mixed.
- [x] Diagnose the final Gaussian compositor's colour/visibility residual on a
  fixed surface; retain one conservative construction-only diffuse transfer
  that improves matched known and excluded-light tails on two real objects.
- [x] Validate the conservative large-table transfer on the predeclared fresh
  metal sculpture with an exact same-cloud control; make only large
  individual-mode tables automatic after every image metric improves.
- [ ] Improve thin/disconnected surface ownership on the fresh metal sculpture
  without trading its 87.2% recall against the current 74.5% precision.

The calibrated-light estimator fits per-pixel diffuse albedo analytically,
searches world-space orientation, and reports both normalized residual and the
loss margin to a direction at least 15 degrees away. Six real maps have median
confidence `0.14–0.43` and median normalized residual `0.12–0.19`; synthetic
directional lights recover a known normal above 0.995 cosine agreement. The
standalone implementation is not retained without a depth-confirmed consumer.
The rejected integration and plane-sweep runs used only the ordinary 16/4/4
construction, selection, and validation cameras. They never opened the ten
official cameras or patterns 005/006. Ignored artifacts are under
`target/audit-runs/openillumination/lighting-pattern-audit/depth-patches-v1/`
through `depth-patches-v7/`; scoped runs peaked below 671 MB with zero swap,
OOM, throttling, or GPU fault.

The sampled Gaussian transport gate uses the persisted selected 2,500-point
asset, not a retrained checkpoint. Against analytic direct lighting, eight
samples move known-light whole/foreground mean-worst
`25.64/24.14, 16.03/14.43→26.42/25.65, 16.57/15.84` dB. Excluded pattern 005
moves `24.40/23.31, 14.36/13.14→24.98/24.44, 14.82/14.08` dB; pattern 006
moves `22.98/21.87, 12.73/11.73→23.54/22.88, 13.24/12.40` dB. Four samples
already improve every mean and tail at 16.7–20.7 ms per view; eight take
26.2–33.9 ms. Mask recall/precision stays `92.9%/90.1%`. A first shader that
restarted a ray query to sort and de-duplicate every Gaussian proxy hit was
removed after its extreme traversal cost triggered NVIDIA Xid 109. The retained
path culls each closed proxy's exit faces and applies its opacity once; the full
gate peaked at 1.13 GB with zero swap, OOM, pressure, throttle, Xid, or GPU
fault.

The leakage-free diagnostic uses 16 matching, four selection, and four
validation cameras from the ordinary training set; the ten official test
cameras and patterns 005/006 remain untouched. The first pass accepted 219
tracks with 3.8 observations, 0.342-pixel reprojection error, and 50-degree
parallax on average. A selection-only de-duplication screen now also requires
a track to land in foreground missed by the current Gaussian in at least 75%
of the selection cameras. A representative run retains 81 tracks; on the four
validation cameras their footprints have 98.8% foreground precision, 62.7%
missing-only precision, and 4.3% recall of the missing foreground.

A direct append was tested but not retained. Five-light photometric normals
and materials plus conservative 0.25 opacity move selection recall
`65.2%→66.0%` and precision `92.3%→92.4%`, while foreground mean stays at
16.14 dB to displayed precision and the worst view falls `15.39→15.38` dB.
That violates the zero-regression gate, so neither the append API nor the
candidate asset is shipped, and no official test camera or excluded light was
loaded to tune it. The next implementation should turn neighboring tracks into
one coherent oriented patch and optimize its support through complete renders,
not treat every correspondence as an independent fixed-radius disc. Ignored
diagnostics and telemetry are under
`target/audit-runs/openillumination/lighting-pattern-audit/missing-tracks-v9/`
and `/tmp/blade-volume-missing-tracks-v9-gpu.log`; peak scoped memory is 700 MB
with zero swap, OOM, throttling, or GPU fault.

The next diagnostic now reuses the ordinary point-cloud surface estimator with
four-point neighborhoods. It rejects isolated tracks, takes orientation from
local covariance, and caps each support radius by twice its directly observed
pixel footprint so sparse neighborhoods cannot create giant discs. An adjacent
repeat retains 73 of 79 selected tracks and reaches 5.1% validation missing
recall, 66.2% missing-only precision, and 99.0% foreground precision. This is
the first coherent patch primitive, not yet a production merge. Artifacts are
under `target/audit-runs/openillumination/lighting-pattern-audit/missing-tracks-v11r/`.

The complete-render probe explains the remaining blocker. Appending the local
patch after five-light normal/material fitting improves selection recall
`65.05%→66.16%` and precision `92.11%→92.20%`, but foreground mean/worst fall
`16.347/15.682→16.323/15.666` dB. Fitting the radiance left after subtracting
the established cloud also fails (`16.184/15.788→16.166/15.733` dB), because
one assumed behind-layer transmission cannot represent exact per-ray depth
ordering. Both implementations were removed. The next production attempt must
render the combined cloud end to end while freezing the established prefix and
optimizing only new patch opacity, support, normals, and material. It should
reuse the existing Gaussian graph through parameter partitioning rather than
add a shader or a second renderer.

That parameter-partitioning probe was implemented with existing split,
concatenate, and stop-gradient nodes; a GPU test proved the prefix stayed
bitwise fixed. The real selection gate still failed after 625 suffix-only
updates: recall rose `65.14%→65.97%`, but precision fell
`91.52%→91.26%` and foreground mean/worst fell
`16.151/15.565→16.128/15.549` dB. The graph mode and merge plumbing were
removed. The problem is therefore earlier than optimizer isolation: selected
points still mix useful and harmful surface fragments. Next, form connected
patches using shared-view image adjacency, fit each patch on matching cameras,
and select whole patches by complete selection-camera renders. Only a patch
that passes independently may enter a joint refinement.

That pre-optimization discriminator is now implemented. Tracks belong to the
same patch only when their observations are within four pixels in at least two
shared cameras; components smaller than five tracks are discarded. Normals and
support are estimated inside each component, and a selection-only alpha render
keeps a component only above 75% missing-region precision and 98% foreground
precision. One run produced a 90-point component whose track-only validation
render reached 7.0% missing recall, 73.2% missing precision, and 98.2%
foreground precision. A later run passed the complete-render selection and
validation gates by small margins, but an immediately adjacent repeat from a
newly trained base Gaussian lost 0.013 dB mean and 0.009 dB worst selection
PSNR. No official test camera or excluded light was opened, and all automatic
merge/fitting code was removed. The diagnostic patch builder and footprint
screen remain because they reject isolated harmful tracks without altering the
trained asset.

The next experiment must persist one base PBR Gaussian, derive the missing mask
and patches from that exact file, and replay patch fitting twice without
retraining the prefix. Only a component that passes both identical-base runs
may advance to excluded cameras and lights. This separates patch stability from
the known run-to-run variation of upstream GPU reconstruction.

The fixed-input boundary is now available as `--missing-tracks-base`. It is
mutually exclusive with training a new `--pbr-gaussian-output`, and the
diagnostic also refuses fewer than six training cameras so matching and
selection cannot silently share every view. Two complete OpenIllumination
replays against the persisted v19 Gaussian produced byte-identical PLY files
and identical diagnostic PNGs. Both retained the same 45 points and reported
3.3% validation missing recall, 59.1% missing precision, and 98.9% foreground
precision. The full runs peaked at 689 MB and 665 MB with no swap, OOM,
throttling, or GPU fault. This clears reproducibility for track discovery and
patch construction only.

The fixed-base complete-render replay is now resolved. A conservative
0.25-opacity component reuses the nearest established material and the common
five-light normal refinement. Two adjacent runs made the same decision and
emitted byte-identical 20-point candidates. Selection foreground mean/worst
improved `16.318/15.723→16.326/15.737` dB and validation improved
`16.561/16.218→16.569/16.242` dB; recall and precision also increased in both
splits. The exact candidate then failed the official gate: known-light held
camera mean moved `23.86→23.85` dB, and excluded patterns 005/006 contained
mixed one-hundredth-decibel foreground and whole-frame changes. A variant that
sampled radiance at exact track observations did not change that conclusion.
The append, candidate CLI, and model-merge helper were therefore removed; no
shader, Meganeura operation, dependency, or persisted field was added.

This changes the immediate algorithmic target. Twenty isolated samples can
slightly fill a silhouette without producing a visible surface or a reliable
material estimate. The next implementation should interpolate a continuous
local point patch from the verified multi-view component, optimize its tangent
support and opacity through the existing complete renderer, and share material
parameters spatially across that patch. It must use a fresh held split or
second object for its final decision because the current official split has now
served its one evaluation. Only after that surface gate passes do finite-light
visibility, bounded indirect transport, and clustered roughness become
identifiable enough to pursue.

The first interpolation step is now in the diagnostic. Each oriented track
surfel stays byte-for-byte unchanged; one half-radius point sample is added at
each unique edge to its two nearest neighbors inside the shared-view component.
This is point-cloud resampling, not a polygonal intermediate. The fixed replay
retains 51 evidence tracks, adds 68 samples, and selection keeps 105 points.
Two runs emit byte-identical PLYs (SHA-256
`197d15fd46248282f97b61434cf0272d938fe565b1862c4bba0ba670391d799b`) and
PNGs. Track-only validation improves missing recall `3.3%→3.8%`, missing
precision `59.1%→59.6%`, and foreground precision `98.9%→99.0%`; peak memory
is 659/672 MB with zero swap, OOM, throttling, or GPU fault. A control that
re-estimated every support radius after interpolation fell to 2.3% recall and
was removed. This clears deterministic local resampling only. It does not
resurrect the rejected append: area coverage and a regularized patch material
still need a fresh complete-render gate.
