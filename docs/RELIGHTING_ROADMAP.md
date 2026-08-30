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
- A second, independently calibrated DiLiGenT-MV gate now excludes both camera
  and distant-light axes. Its scalar Bear cloud reaches 21.16/18.12 dB
  foreground mean/worst on the held/held cross-product, while a same-pixel
  diffuse oracle reaches 33.57 dB. More lights and a denser extraction are
  mixed or worse, independently isolating cross-view geometry correspondence.
- The final diffuse solve now follows the runtime particle blend and solves all
  surfels together. It improves every surface mean/tail on LUCES-MV and
  DiLiGenT-MV without changing geometry; held/held foreground reaches
  25.51/23.28 and 21.16/18.12 dB respectively. Gaussian appearance remains
  backend-specific.
- Complete-render point-light normal refinement improves Owl and Bear means,
  but a fresh Cow reconstruction loses four fitted/held tails by up to 0.05 dB.
  Radius and center variants also trade foreground support for background fit.
  All three are rejected; image-space fitting cannot replace cross-view depth
  evidence.
- A fresh Cow diagnostic matches missing pixels against ordinary foreground
  using robust diffuse albedo recovered from 24 calibrated lights. A 32-surfel
  cloud-only patch subset improves every final fitted/held light/camera mean
  and tail; held/held recall rises by 0.82 points and precision also rises.
  Only the albedo estimator is selected: the multi-pass correspondence search
  is not a production merge, its direct one-pass form loses Cow tails, and the
  exact Bear repeat produces no internally valid patch. A fresh Pot2 repeat
  shows the productive route: seed nine surfels before joint fitting and every
  Gaussian mean plus coverage improves, but two fitted-camera foreground tails
  still lose 0.0054/0.0028 dB. A global continuation checkpoint cannot resolve
  the opacity/tail frontier; only patch-local final-compositor appearance
  remains open.
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
- The first Objects With Lighting gate now holds out both camera and natural
  HDR environment independently. A 25,000-point scalar cloud is recognizable
  and reaches 21.88/20.16 dB foreground mean/worst on nine official pairs, but
  remains grey and texture-poor. A local construction-mask recovery now
  restores 3,664 of the supports suppressed by ordinary Gaussian fitting and
  improves official mean, tail, recall, and precision together. It does not
  recover the missing material detail.

The current bottleneck is therefore **preserving cross-view surface support in
the Gaussian objective first, separating material from unknown illumination
second, and adding non-diffuse capacity third**. The selected
representation-aware diffuse transfer and coupled transport renderer remove two
implementation gaps, but material, visibility, and illumination can still
compensate for wrong or missing geometry. The scalar surfel asset remains a
valid cloud-only geometry control; no polygonal fallback is needed.

## Execution plan

The rows are ordered by reconstruction dependency. Independent renderer work
may land early, but no later parameter family may supervise an unaccepted
surface.

| Phase | Status | Next action | Decision gate |
| --- | --- | --- | --- |
| Capture integrity | selected | Keep held cameras physically absent from dense reconstruction; canonicalize masks on import | Rebuilding a training asset cannot read an excluded pose or light |
| Controlled-light diversity | LUCES-MV and DiLiGenT-MV routes selected | Keep their near- and distant-light models shared with the production renderer; do not depend on OLATverse | Improve whole-frame and foreground mean/worst, recall, and precision on both fixed camera/light splits |
| Dense support | selected and independently validated | Keep the training-mask soft visual hull before spatial downsampling; do not use it as evidence that Gaussian transfer or relighting is solved | Held-camera scalar foreground mean/worst and precision improve on another object; report the small recall trade and mixed Gaussian/light results |
| Missing support | Cow diagnostic passes; Bear and Pot2 production gates reject | Preserve calibrated albedo and seeded geometry; fit only proposed-patch appearance in the final Gaussian compositor before another object gate | Recall, precision, and every complete-render mean/tail rise on two controlled objects |
| Representation scale | final-compositor diffuse transfer selected and automatic for large individual tables; topology replacement, additive support, and scalar-alpha distillation rejected | Keep the one-proposal transfer narrow; keep small shared palettes behind `--render-refine-materials` and on their bounded joint solver | Preserve the exact same-cloud gains on three material classes while geometry, visibility, and non-diffuse properties remain unchanged |
| Light transport | renderer selected | Use coupled sampled visibility and one bounded bounce: four samples for iteration, eight for selected output | Both excluded-light mean/tail improve with no coverage regression or GPU fault |
| Radiometry and transport | attributed; scalar correction and extra samples rejected | Preserve measured light scale; treat the repeated `(light, view)` residual as surface/response evidence, not exposure | A correction repeats across objects and every known-light tail before either excluded light is opened |
| Materials and capture layout | coupled runtime-blend diffuse solve selected on LUCES-MV and DiLiGenT-MV | Keep Gaussian appearance backend-specific; next add capacity only after geometry correspondence improves | Preserve every surface mean/tail on both fixed light/camera splits without changing coverage |
| Unknown natural illumination | local support recovery selected; appearance quality fails | Replace safety restoration with an ownership-aware training objective before changing material or light capacity | Improve over the recovered cloud's official mean/worst PSNR and recall without losing precision on the fixed nine-pair gate |

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

### Explicit broad-specular response diagnostic

The fresh object separates two failures that looked like missing geometry.
Doubling native groups to 5,000 raises Gaussian recall `87.2→90.2%`, but drops
precision `74.8→68.4%` and regresses several whole-frame means/tails. Refusing
two-view fusion groups cuts unseen particles `835→427`, but loses known-light
Gaussian mean/tail, foreground, recall, and precision. At 256 px the independent
photometric matcher accepts 48 selection-valid triangulations; all already lie
under the persisted Gaussian's alpha, so none forms a missing-support patch.

The existing free lobe solver is also rejected: four rounds lower the fitted
residual but collapse the metal sculpture to `18.69/15.38` and `19.53/13.77`
dB whole-frame mean/worst under excluded patterns 005/006. A fixed-cloud bound
instead keeps roughness at one and transfers a conservative quarter of base
colour from diffuse response to `F0`:

| Exact metal-sculpture Gaussian | Known light | Excluded 005 | Excluded 006 |
| --- | ---: | ---: | ---: |
| Diffuse transfer only | `26.46/24.01; 17.08/14.92` | `24.80/23.27; 14.34/13.34` | `25.14/23.11; 14.64/13.28` |
| + 25% broad-specular allocation | `27.25/25.03; 17.56/15.49` | `26.05/24.57; 15.59/14.37` | `26.39/24.31; 15.88/14.46` |

The same exact-cloud allocation improves the five-known-light aggregate and
both excluded lights on the fabric and painted toys. It nevertheless regresses
the painted toy's primary-light held-camera whole/foreground tails by
0.30/0.34 dB. `--render-transfer-specular` therefore exposes one exact-render-
accepted proposal for large rough-dielectric tables, but it is opt-in and is
not physical metalness evidence. The following gate therefore fits a spatially
shared response on construction lights and accepts it independently on each
withheld known light before excluded lights are opened.

### Spatial-lobe and diffuse-margin bounds

The next gate rejects extra material capacity rather than adding it. Four
deterministic Euclidean regions of the fixed Gaussian were allowed only the
same binary 25% response transfer. Choices were fitted on patterns 001--003
and 30 construction cameras, then screened on patterns 004/013 and eight
orbit-spread training cameras before the ten official cameras were opened. On
both the metal sculpture and painted toy, all four regions were selected in
sequence. The spatial model therefore reduces exactly to the global proposal;
it supplies no evidence for four material lobes.

The painted result explains the failed tail. Official view `CB8` changes
`16.481→15.920` dB foreground under pattern 001, but improves to
`19.209`, `17.339`, `18.122`, and `14.454` dB under patterns
002/003/004/013. Activating each spatial region alone also hurts pattern 001,
landing at `16.361`, `16.292`, `16.406`, and `16.407` dB. This is a
view/light interaction rather than one bad surface region. Sharpening the
roughness is not the missing dimension: on the painted object, changing only
the dielectric initializer from 1.0 to 0.75 drops the five-light held-camera
foreground aggregate from `16.70/13.63` to `15.00/10.45` dB and is worse
under both excluded patterns; roughness 0.5 collapses further.

A separate scale diagnostic initially looks stronger. Additional diffuse
darkening improves all foreground means/tails across seven patterns and three
objects. Moving the production interpolation from 50% to 60% of the fitted
global correction, however, improves only 83 of 84 complete fixed-cloud
mean/tail metrics: the painted primary-light whole-frame tail changes
`25.98202→25.94` dB. A fine bound shows even a 1% additional albedo reduction
changes it to `25.97649` dB. The one-line production experiment and its test
fixture were reverted. The exact half correction remains the selected point.

These controls close the immediate spatial-capacity branch. The following
fixed-cloud attribution now separates the common and directional residuals.

### Fixed-cloud transport, radiometry, and response attribution

The sampled renderer is already close to its practical convergence point.
Across the painted, fabric, and metal fixed clouds, moving from eight to 64
diffuse samples costs roughly 6--9× as much per view. Mean foreground quality
changes by only `0.01--0.37` dB, with mixed worst-view changes. Randomized
stratification of the environment and cosine halves, separately and together,
improves several means but regresses known-light tails at the same ray count;
the sampler experiments are reverted. The sweep did expose one count bug: a
one-ray request launched one environment and one cosine ray. The split now
uses ceil/floor halves, so one means one ray and all selected even counts are
unchanged. Four samples remain the iteration point and eight the
selected-output point. More rays do not explain the residual.

A construction-only light-scale sweep is not calibration either. On each
object, 30 cameras select one scalar per known pattern; eight orbit-spread
cameras then remain untouched. The selected scalar improves only
`5/8, 6/8, 5/8, 5/8, 6/8` painted validation cameras for patterns
001/002/003/004/013. Fabric improves `5/8, 8/8, 6/8, 7/8, 8/8`; metal
improves `6/8, 6/8, 6/8, 7/8, 7/8`. Individual camera optima span
`0.375--1.125` under the same emitter group. They repeat by physical
camera/light across objects: for `NF4`, patterns 001--004 prefer approximately
`0.375, 0.375, 0.875--1.0, 0.375` on all three objects. The analytic
zero-sample path retains the same split. A true emitter-power scalar cannot
change with view, and a camera exposure scalar cannot change with light. No
per-light or per-camera nuisance gain is added.

The remaining response is real but not yet safe to persist. A fixed-roughness
two-basis solve fits diffuse albedo and `F0` jointly from patterns 001--003 and
30 cameras, with diagonal regularization toward the current material. Patterns
004/013 and all five patterns on eight other cameras improve every complete
whole/foreground mean and tail for regularization factors 0.25 through 16.
The strongest painted candidate then improves six of seven official pattern
groups, including both excluded lights, but pattern-001 whole-frame tail moves
`25.982→25.888` dB while its foreground tail improves
`16.182→16.487` dB. Even the 16×-regularized candidate changes that whole tail
to `25.954` dB while raising the foreground tail to `16.365` dB. The material
is improving the object and simultaneously brightening inaccurate Gaussian
extent outside its mask. Requiring at least 16 of 30 construction-camera
observations leaves only seven eligible materials and gains at most 0.001 dB;
that is numerically safe but not useful evidence for a production feature.

A 25% blend toward each Gaussian covariance normal already loses construction
quality, so the failure is not repaired by deriving shading orientation from
the ellipsoid. Apart from correcting the one-ray count, no sampler experiment,
operation, option, model field, format, dependency, or response solver is
retained. The next accepted step must improve cross-view ownership/precision
of the existing support. Once false extent no longer converts a better
foreground material into a worse whole frame, the same bounded diffuse/F0
basis is the first material continuation to revisit.

### Mask-only false-extent bound

The next fixed-cloud diagnostic differentiates the exact ordered production
Gaussian alpha composite with respect to each covariance axis. Only shrink
directions whose mask-loss sign repeats across at least one quarter of 30
construction cameras are considered. Shrinking the best 3--52 axes improves
whole-frame quality and precision monotonically, but the smallest prefix loses
one construction foreground tail by about `0.0001` dB. Screening coordinates
individually leaves 13 that improve every construction metric. Their best
eight-axis prefix moves whole-frame mean/tail
`26.844531/26.005165→26.846105/26.005695` dB and precision
`90.8357→90.8517%`, but no individual coordinate passes the separate
known-light/camera gate. Supplying all training masks does not change that
result. Raising opacity to preserve each shrunken axis's typical foreground
response leaves 15 construction-safe coordinates, and all fail the same
validation. Mask gradients identify false extent but cannot establish which
physical response owns an unseen ray.

Filtering the bounded diffuse/F0 solve by actual composited ownership gets
closer. At every pixel, each material accumulates its front-to-back Gaussian
contribution on foreground and background. The definitive bound evaluates
every pixel and requires a material to contribute to foreground in all 30
construction cameras with exactly 100% measured precision. It changes 544 of
2,500 materials at ridge 16. Construction and independent light/camera
validation pass; official means and tails improve in patterns 002, 003, 004,
013, 005, and 006. Pattern 001 also improves its whole and foreground means
and its foreground tail, but its whole-frame tail moves backward by less than
0.001 dB. That is the only failing cell out of 28, and it remains a failure
under the strict held-view rule.

Evaluating PBR Gaussians at their learned tangent-plane intersection instead
of their volumetric maximum is not the missing boundary prior: pattern-013
foreground mean falls `14.72→13.17` dB. The shader experiment is reverted.
Earlier all-axis, tangent-axis, and normal-axis covariance controls already
bound simpler sheet priors. No tracked implementation survives this stage.

This closes mask-only ownership. The next geometry experiment must bring in
calibrated 3D evidence at boundaries: group source-camera depth samples into a
consistent tangent patch, constrain covariance not to cross a depth or mask
discontinuity in those source views, and screen the resulting fixed cloud
before refitting response. It remains a point-cloud operation; it does not
introduce polygonal geometry. Only after one covariance proposal passes
construction, independent cameras/lights, and a second object should the
already validated diffuse/F0 solve be attached and opened on excluded lights.

### Calibrated source-plane bound

The paired COLMAP provenance is now reconstructed exactly for both dense paths.
On the established painted-toy cloud, 1,185,892 retained raw observations form
the 2,500 selected cells. A cell sees a median of 12 source cameras and its
source samples fit a plane with median RMS `0.0505` surfel radii, yet the final
Gaussian centre is a median `1.6419` radii away from that plane. The native
grouped-fusion fabric checkpoint independently measures 9 cameras, `0.0262`
plane RMS, and `1.2243` radii of final centre displacement. Nearest depth alone
and broad plane projection both fail, so the useful signal is specifically a
tight, repeated source plane rather than another global center prior.

A deliberately small post-fit control moves only particles with at least 12
source views and plane RMS at most `0.05` radii by 2.5% of their signed plane
offset. Retaining 12.5% of a construction-camera diffuse polish then improves
every complete official metric cell on the two established checkpoints. For
painted toy, excluded-pattern foreground mean/worst move
`15.5202/14.5827→15.5518/14.6167` and
`14.0725/12.9700→14.1222/12.9995` dB. Fabric moves
`14.7964/13.3860→14.8519/13.4110` and
`15.4255/13.2458→15.5250/13.2719` dB.

That apparent selection does not survive a paired reconstruction. Candidate
and control were dumped from the same fresh optimizer trajectory and scored in
one process. The material polish accepted no changes, and the plane-only
candidate regressed all seven known/excluded light groups. This is an optimizer
basin dependency, not run-to-run scoring noise. The 600-line production
prototype, temporary model dump, and tests are removed; no option, field,
format, shader, operation, or dependency is retained.

The next candidate must apply source-plane evidence during the existing
multi-light geometry stage, not after it. Use a robust signed-distance loss
whose confidence comes only from source-view count and normalized plane RMS;
keep covariance, opacity, and appearance free to co-adapt. First compare paired
controls from identical initial clouds across at least two optimizer basins on
painted and fabric objects. Construction cameras fit the candidate; independent
known-light and camera splits select it. Official cameras and excluded lights
remain unopened until every internal mean and tail is non-regressing in every
replica. If that fails, close depth-centre regularization and move to a
depth-discontinuity covariance bound rather than adding another post-fit
heuristic.

That co-optimization gate is now closed too. A minimal prototype introduced no
loss graph or shader: it distributed the requested correction over the existing
20-update geometry synchronization points, preserving one Adam trajectory while
opacity, normals, later material fitting, and candidate-index rebuilds adapted.
Only 509 of 2,500 painted-toy centers carried the established 12-view/5%-RMS
confidence. Patterns 001--003 on 30 cameras trained each arm; patterns 004/013
and eight disjoint cameras were validation. No official camera or excluded
pattern was loaded.

The 2.5% candidate fails every one of the ten light/split rows in two adjacent
runs. A zero-feedback arm measures the ordinary atomic variance at mostly less
than 0.02 dB, but reducing feedback to 0.5% and 0.1% still produces much larger
losses. At 0.1%, pattern-013 fit-view whole-frame tail falls
`21.0944→20.9162` dB and the pattern-003 camera tail falls
`24.7419→24.6146` dB. The response is neither monotonic nor inside measured
noise: tiny center motion changes discrete Gaussian responsibility. Because the
first object fails the predeclared all-cell gate, the second object and all held
observations stay unopened.

The temporary public API and unit fixture are removed. Ignored runs peak below
0.80 GB in 8 GB zero-swap scopes with no OOM, throttle, Xid, or GPU fault.
Calibrated planes remain diagnostic evidence, but neither post-fit nor
co-optimized center feedback is a stable consumer. The next implementation
keeps centers fixed and asks a one-sided question in each source view: does a
Gaussian's finite response ellipsoid cross a measured depth or foreground-mask
discontinuity? Only the offending covariance direction may shrink, and it must
retain foreground response through a coupled opacity adjustment before any
material solve is revisited.

### Rejected calibrated covariance endpoint bound

That fixed-cloud question is now answered. At the production `0.03` response
cutoff, each Gaussian axis is projected to both finite endpoints in every
paired source camera. A local cell plane predicts the depth at each projected
pixel; an axis is flagged only when the measured COLMAP depth differs by more
than 1%, or the endpoint crosses the source silhouette, while the center itself
remains foreground and depth-consistent. Among 7,500 painted-toy axes, 244
have at least 12 valid cameras and cross a discontinuity in at least half of
them. Of these, 240 have a depth crossing and only 70 a mask crossing, so the
screen is not a disguised repeat of the rejected mask gradient.

Broad candidates sweep 8/12 minimum views, 50/75/100% agreement, 1/2.5/5%
scale shrink, and zero/half/full bounded opacity compensation. None of 54
combinations passes all three construction lights. Screening the 64 strongest
axes individually at 1/2.5% with and without compensation leaves two complete
construction passes. One fails light 004 and one fails the first disjoint
camera. Intersecting the calibrated set with the exact compositor's
construction-mask derivative reduces it to 29 image-owning axes and finds
three construction-safe proposals; all three fail independent light/camera
validation.

The apparent finalists are numerical identities rather than useful capacity.
Axis `1854.1` improves construction cells by roughly `0.00001` dB and loses
light-004 whole/foreground means by about `0.000005` dB. Axis `1980.0` gains at
most `0.00018` dB on construction tails, then loses the first disjoint-camera
foreground tail by `0.000023` dB and another camera tail by `0.000129` dB.
Recall is unchanged to reported precision; precision moves at most 0.0001
points. No production threshold can turn those into a meaningful surface gain.

All candidates keep centers and materials fixed. No official camera or
excluded pattern is loaded. The ignored 8 GB, zero-swap runs peak below 0.77 GB
with no OOM, throttle, Xid, or GPU fault, and no tracked code survives. This
closes independent center and endpoint corrections. The next consumer of the
calibrated evidence must operate on ownership directly: sample only exact
multi-view-consistent fusion observations, render the Gaussian composited
depth on those rays, and add a robust normalized depth residual beside RGB and
mask loss. A held source-camera subset selects its weight. This allows
position, covariance, and opacity to explain one observed surface jointly
without hand-assigning a particle or axis.

### Rejected low-resolution composited-depth supervision

The first ownership graph reused only existing primitives. It exposes the
candidate maximum-response depths and front-to-back weights, then evaluates
three alternatives: relative error of the opacity-normalized mean depth,
front-to-back-weighted relative error for every contributor, and the same
per-contributor loss with ownership weights detached so depth updates positions
only. A scalar input makes the zero and fitted arms compile the identical
graph. There is no new Meganeura operation, shader entry/group, format, model
field, dependency, or runtime path.

The input reduction is deliberately recorded because it bounds the result.
Only cells with at least 12 source cameras and plane RMS no greater than 5% of
surfel radius contribute. Their raw points are projected to the 128×192 RGB
capture; collisions keep the nearest depth. This produces 26,024 target pixels
across 38 views. It is calibrated and multi-view filtered, but it is not an
exact observation batch: many original-resolution samples collapse to one
pixel, and nearest reduction biases the target toward the front layer.

Patterns 001--003 on 30 cameras fit each arm; patterns 004/013 and eight
disjoint cameras validate. The zero arm quantifies the atomic optimizer spread.
Mean-depth weights `0.0001`, `0.001`, and `0.01` all produce mixed tails outside
that spread. Weight `0.01` improves many whole-frame means by `0.1--0.4` dB,
but even a 12.5% blend loses pattern-001 and pattern-004 tails. The
per-contributor loss is more coherent: at weight `0.001`, a 12.5% blend passes
every pattern-001/002 construction and camera cell and the pattern-003
construction cell. It still lowers the pattern-003 disjoint-camera foreground
tail `14.6405→14.6383` dB and loses light-004/013 tails. Detaching weights does
not rescue any blend.

No official camera or excluded pattern is loaded. Runs stay below 0.74 GB in
8 GB zero-swap scopes and report no OOM, throttle, Xid, or GPU fault. The public
depth structs, graph outputs/loss, session inputs, and tests are removed. The
negative decision applies to low-resolution front-reduced targets, not to exact
source rays.

The next implementation must sample original fusion observations directly as
`(camera, pixel, depth, confidence)` depth-only batches. It should record exact
Gaussian candidates for those rays, set RGB/mask loss to zero for that batch,
and retain one optimizer state while alternating ordinary image and depth
updates. No per-particle assignment is supplied. Hold complete source cameras
out when selecting weight and cadence; if exact observations still lose the
same tails, close metric-depth supervision and move to ordinal visibility.

### Rejected exact-source-ray metric depth

The follow-up preserves every accepted fusion observation as its original
calibrated source-camera ray and ray-parameter depth. Reproducing the persisted
7,500-cell model requires the same 38-view soft hull used at construction; the
cell centers then match the stored surfels within `1e-6`. Only the 30 fit
cameras are allowed to emit depth observations, leaving eight complete cameras
out of the loss. The 509 tight cells provide 336,113 exact observations, with
no image resizing, pixel collision reduction, nearest-depth selection, or
particle preassignment.

A temporary tracked graph alternates one 512-ray depth-only batch after every
four ordinary RGB/mask steps in the same Adam session. It rebuilds the complete
Gaussian candidate record for each sampled ray and minimizes the
front-to-back-opacity-weighted per-contributor relative depth error. Candidate
and zero-weight control execute the same graph and the same extra optimizer
steps. The temporary implementation adds no Meganeura operation or shader.

At weight `0.001`, the 0.5 control-to-candidate blend improves all four metrics
for both light-001 rows and both unseen-light-013 rows. Whole-frame and
foreground means generally improve elsewhere, but the held-camera foreground
tail regresses for lights 002, 003, and 004; no tested blend passes every row.
Reducing weight to `0.0001` changes which rows fail rather than approaching a
safe identity: all 0.125/0.25/0.5/1.0 blends still lose at least one
construction or held-camera tail. This repeats the low-resolution result
without its raster-collapse ambiguity. The failure is in metric-depth
supervision, not in loss of exact source provenance.

No official camera or excluded light is opened. The corrected runs peak at
0.73 GB in 8 GB zero-swap scopes and report no OOM, pressure, throttle, Xid, or
GPU fault. The observation API, graph inputs and loss, and alternating session
path are removed; no production code, dependency, format, shader, or result
asset remains.

Metric depth is now closed for this milestone. The next bounded experiment is
an ordinal free-space/visibility objective on the same exact rays: penalize
opacity strictly in front of the measured first surface and require support in
a tolerance band, without pulling every contributor to one noisy metric depth.
Keep ordinary image batches and complete held cameras unchanged. Advance only
if a same-graph zero arm passes every construction, held-camera, and unseen-
light mean and tail before opening a second object.

### Rejected exact-ray ordinal visibility

The ordinal follow-up keeps the same 336,113 original source rays, 30/8 camera
split, 512-ray alternating cadence, exhaustive current-cloud candidates, and
same-graph zero control. It tests three bounded objectives at weight `0.001`:
one-sided free-space suppression for opacity-weighted response more than 1% in
front of the fused first surface, a 0.25-opacity minimum from candidates whose
maximum response is already within a 1% support band, and their sum. Candidates
behind the surface are never pulled to its metric depth.

All three arms fail every 0.125/0.25/0.5/1.0 blend. Free-space suppression is
the clearest rejection: even its 0.125 blend lowers light-001 fit and camera
foreground tails and loses additional 002--004/013 rows. The support-only arm
passes every light-001/002 metric at its 0.125 blend, but lowers foreground
tails for lights 003, 004, and 013. Combining the terms merely combines those
failure modes; broad image means often rise while independently held rays get
worse. The fitted volumetric response therefore cannot treat fused depth as a
universal first-hit boundary, even when the loss is ordinal rather than
metric.

No official camera, excluded light, or second object is opened. All corrected
runs peak below 0.73 GB in 8 GB zero-swap scopes with no OOM, pressure,
throttle, Xid, or GPU fault. The observation wrapper, graph mode and inputs,
alternating steps, and support masks are removed. No code, shader, Meganeura
operation, format, dependency, or result asset is retained.

This closes dense-fusion depth as a training constraint on an already fitted
Gaussian for this milestone. Return to independently verified point support:
the next surface gate should expand a coherent shared-view track component by
cloud-only local resampling, share material parameters across that patch, and
optimize it through complete renders against a fixed persisted prefix. Use a
fresh held split or second object for the decision; do not reopen the exhausted
painted-toy official split.

### Internally selected shared-material point patch

The existing deterministic patch supplies the next bounded proposal without a
new representation. It retains its verified oriented tracks, adds half-radius
point samples at unique local neighbor midpoints, and contains 105 points. An
ignored complete-render harness appends those points to the exact persisted
painted-toy PBR Gaussian. Every addition shares one material: the mean of the
nearest established materials, with diffuse albedo and `F0` scaled together.
There is no per-point appearance, new shader, graph mode, operation, field, or
dependency.

Construction-only screens establish the narrow setting. Opacity `0.025` is a
runtime no-op below the 0.03 response cutoff; `0.1` and `0.25` fail, making
`0.05` the smallest visible choice. At that opacity, shared response scales
`0`, `0.25`, and `0.5` pass patterns 001--003 on four selection cameras;
`0.75` and `1.0` fail light 002. The highest passing scale, `0.5`, then passes
patterns 004/013 on selection cameras and patterns 001--003 on four disjoint
cameras. Across those five hidden rows, every whole-frame and foreground mean
and tail plus recall and precision improves. The exact adjacent repeat is
metric-identical. Improvements are deliberately described as small: roughly
`0.001--0.009` dB and `0.074--0.082` recall points.

This clears the internal construction/validation gate, not production. The
painted-toy official cameras and excluded lights are not reopened because they
already served an earlier sparse-patch evaluation. Fresh 38-camera-only checks
on fabric and metal accept 35 and 19 multi-view tracks respectively, but zero
land in foreground that their persisted Gaussians miss. Their selection
threshold is not weakened to manufacture a candidate, so a second-object gate
cannot yet run.

The tracked tree remains unchanged. The ignored combined asset has SHA-256
`a3be5829c14f9f13b4b23ea341869075c12f4a735aa6cdf50b3cc9192cd0a5ff` and
is under `target/audit-runs/openillumination/lighting-pattern-audit/patch-shared-response-selected/`
with paired control/candidate renders. Repeated 8 GB zero-swap runs peak at
1.0 GB and report no OOM, pressure, throttle, Xid, or GPU fault.

The remaining gate is now a data requirement, not an implementation excuse:
acquire or adapt a multi-light capture whose untouched split contains a
non-empty coherent missing-surface component, replay this fixed recipe once,
and retain production merge plumbing only if every final metric passes.

### Predeclared Hugging Face final gate

Before downloading or viewing another object, select OpenIllumination
`obj_33_fabric_basket` from its published directory name alone. Its opaque,
textured material should support response matching without transparent or
mirror-like confounds, while a basket is a useful thin/concave surface test.
This rationale is fixed before seeing any image or reconstruction result.

Use patterns 001--004 and 013 from the 38 ordinary training cameras to build
one 2,500-point, minimum-two-source Gaussian and its missing-track proposal.
Keep the ten official cameras and patterns 005/006 out of loading, dense
fusion, candidate construction, threshold selection, and replay. If and only
if a non-empty coherent patch passes every existing internal complete-render
mean, tail, recall, and precision gate, persist the exact base and candidate,
replay once, and open the official camera/light cross-product for one final
decision. Do not weaken a threshold or choose a different point budget when
the object produces no proposal.

The basket produces no proposal. The fixed construction-only reconstruction
fuses 318,297 depth groups, retains 2,500, and reaches 96.3% foreground recall
at 93.0% precision. Matching accepts 12 tracks and the visual hull keeps 11,
but the selection cameras place none in foreground missed by the persisted
Gaussian. Patterns 005/006 and the ten official cameras remain unopened. The
base PLY has SHA-256
`1db4937a528940387610a4ce585e40ff4c20e467b0441d8c48ef14274582fef3`.
The fit and diagnostic peak at 1.14 GB and 639 MB respectively, with zero
swap, OOM, throttling, Xid, or GPU fault.

Predeclare `obj_15_flower` as the next construction-only screen before
downloading it. Its thin overlapping petals and leaves are a stronger test of
coherent missing support than the already well-covered basket. Reuse the exact
camera/light isolation, 2,500-point budget, thresholds, and one final-decision
rule above; do not inspect an official camera or excluded light unless the
internal fixed-base proposal is non-empty and passes.

The flower also stops before the final gate. Its 2,500-point Gaussian reaches
91.5% construction-camera foreground recall at 86.9% precision. Of 71 accepted
tracks, 69 pass the visual hull and three land in missing foreground, but they
are isolated: shared-view adjacency forms no five-track patch. The official
cameras and patterns 005/006 remain unopened. The base PLY has SHA-256
`3f10df09d6919e305a11ba0253a9b09f8811af6d1d9701eb99c35acce53c2f09`;
fit and diagnostic peak at 1.04 GB and 731 MB with clean zero-swap telemetry.

Use `obj_12_bag` for one terminal OpenIllumination construction screen,
predeclared before download because a handle and occluded interior are more
likely to form one coherent missing patch than fine flower structure. Apply
the same fixed protocol. If it forms no internally passing patch, stop object
screening rather than selecting a favorable dataset after the fact.

The bag closes that screen without opening held data. Its 2,499-point Gaussian
reaches 88.3% construction-camera foreground recall at 95.1% precision. The
independent matcher accepts 72 tracks; the selection hull keeps 70, and 13 lie
in foreground missed by the fixed Gaussian. Five tracks form one shared-view
component, expanded to an 11-point diagnostic patch, but it covers only 0.9%
of missing foreground and the local footprint gate retains zero points. No
candidate is written, so the official cameras and patterns 005/006 remain
unloaded. The fixed base SHA-256 is
`0aba83f2b30b67b618fd02544d481bb40162af476b3945721894c4acf8a4f571`.
Base fitting and track diagnosis peak at 1.05 GB and 731 MB respectively; dense
reconstruction peaks at 181 MB. Every scope uses zero swap and reports no OOM,
throttling, Xid, or GPU fault.

This is the declared stopping condition. Three name-selected Hugging Face
objects span concave texture, thin overlapping surfaces, and an occluded
handle/interior; none supplies a coherent patch that passes the unchanged
construction-only footprint gate. Do not continue screening OpenIllumination
objects. The next final gate needs a capture or dataset split designed to hide
a contiguous surface region from the base reconstruction while exposing it to
separate proposal cameras under calibrated repeated lighting. This is a data
protocol requirement; weakening track, footprint, or point-budget thresholds
would only select retrospectively favorable noise.

### Dataset availability and next protocol

The 2026-08-29 Hugging Face survey finds one official copy of the requested
datasets: `OpenIllumination/OpenIllumination`, already consumed above. There is
no Hub repository matching MIT Multi-Illumination, FAU Multi-Illuminant,
Flash/Ambient, MILL, LUCES-MV, ReNé, DiLiGenT-MV, or OpenSubstance. MIT, FAU,
Flash/Ambient, and MILL also hold the camera fixed; they can train or test an
appearance prior, but cannot validate a novel-view surface reconstruction.
The Hub-hosted M2AD alternative has 12 angle labels and 10 illumination labels,
but its complete metadata archive contains no camera or light calibration, so
its 15--17 GB category archives are not downloaded.

Do not plan around OLATverse access. LUCES-MV is the active controlled-light
route: its official Drive exposes object archives separately, and each public
calibrated object contains 12 masked views under 15 near-field LEDs plus
camera/light calibration and ground-truth normals/depth/shape. DiLiGenT-MV is
the smaller fallback with 20 views by 96 calibrated lights. Stanford-ORB is the
later distant-HDR cross-check. OpenSubstance remains a request-gated,
multi-terabyte option rather than a milestone. Objects With Lighting is the
compact benchmark for one *unknown* natural environment and is evaluated
separately below.

ReNé is the next autonomous surface screen. Its 50 camera poses and 40 robot
light poses have exactly the aligned multi-view/multi-light product needed by
the existing diagnostic. The public 73.7 GB ZIP supports byte ranges, allowing
one object to be materialized without vendoring the archive. Before listing or
viewing its contents, select `lunch` from the published scene names alone: a
multi-part ordinary object is more likely to expose a contiguous occluded
surface than another thin flower, without choosing a reflective category.

Keep the official blacked-out cameras `{4,8,15}` and lights `{2,21,34}` absent.
From the remaining cameras, reserve `{0,5,10,16,22,27,32,38,44}` as an unopened
final set and use the other 38 for the existing match/selection/validation
partition. Use lights `{0,10,20,30}` for the primary and three aligned response
captures; reserve `{13,33}` unopened. Generate masks only from exact black
background pixels shared by those four construction lights—no learned mask or
threshold sweep. Before reconstruction, compare light distance with the object
radius and stop if the worst directional variation exceeds five degrees; in
that case implement point lights instead of pretending that a nearby source is
an environment. Otherwise reuse the 2,500-point, minimum-two-source base and
every unchanged patch threshold above. Open the nine-camera/two-light final
cross-product only after a non-empty internally passing proposal and its exact
replay.

The ReNé screen stops before reconstruction. Range extraction materializes
only the predeclared `lunch` span (3.27 GiB); all 4,161 archive entries pass
CRC. The first permitted construction photograph has nonzero content over the
complete 1440x1080 frame because the object sits inside a textured calibration
enclosure, so the fixed exact-black mask rule yields no background. The four
selected light translations are only 0.368--0.412 m from the world origin,
confirming that this is also a near-field point-light capture rather than the
distant environment assumed by the current renderer. No mask threshold,
background subtraction, or distant-light approximation is chosen after seeing
the image; no reconstruction is run, and the nine reserved cameras, two
reserved lights, and official blacked-out observations remain unopened.

ReNé still needs deterministic background subtraction against its separately
captured empty enclosure; that empty capture is not present in the public
archive. The calibrated point-light training renderer with spatially varying
direction and falloff is now implemented independently through LUCES-MV, so it
is no longer the ReNé blocker. First run the
existing unknown-environment path on Objects With Lighting, whose official
2.35 GB release already supplies calibrated cameras, object masks, and held
HDR environments. Before listing that archive, define its first gate as the
lexicographically first object directory. Use all input cameras except every
fifth sorted camera for one internal same-environment test, reconstruct only a
2,500-point minimum-two-source cloud, estimate the input environment, and open
the official ground-truth views/environments once after the fixed fit. Ignore
all provided polygonal geometry. This gate tests novel view and novel lighting;
it is separate from the still-blocked coherent missing-patch merge.

### Objects With Lighting result

The lexicographically first object is `antman`. Only its 64 input photographs,
calibrated cameras, and masks enter reconstruction; all polygonal assets are
ignored. Every fifth sorted input camera forms a 12-view internal test, leaving
52 construction cameras. Their explicit nearest-12 PatchMatch graph produces
photometric and geometric depths for all 52 views without loading a held
camera. Fusion contains 413,305 depth groups with 4.7 source views on average;
the fixed 2,500-point minimum-two-source surface retains 1,659 points observed
by construction photographs.

The fixed 2,500-point exact arm reaches 25.01/21.02 dB whole-frame and
18.73/15.32 dB foreground mean/worst on the 12 internal cameras. Its Gaussian
reaches 24.50/18.75 and 18.04/12.88 dB, with only 67.6% recall. Exact serial
point proposals take 44.3 minutes; eight simultaneous refinement rounds reach
within 0.19 dB on the scalar surface and 0.51 dB on the Gaussian in 30.6
seconds. The serial path is therefore an analysis oracle, not a production
default.

Only after that fit was fixed were the nine official camera/environment pairs
opened. The evaluator renders each persisted cloud at the published camera,
reprojects the published HDR environment into Blade's convention, and follows
the benchmark's exposure and per-channel colour alignment on the full native
resolution. Results are foreground mean/worst PSNR followed by mask
recall/precision:

| Fixed output | Official result |
| --- | ---: |
| 2.5k scalar | `20.85/18.84 dB; 98.8/94.3%` |
| 2.5k ordinarily fitted Gaussian | `19.64/17.73 dB; 68.3/96.8%` |
| 25k scalar | `21.88/20.16 dB; 99.6/95.7%` |
| 25k ordinarily fitted Gaussian | `19.28/17.39 dB; 60.8/95.8%` |
| 25k mask-recovered Gaussian | `19.64/18.07 dB; 69.6/96.0%` |
| 25k support-preserving Gaussian diagnostic | `20.44/17.95 dB; 87.1/92.5%` |
| Black | `10.84/8.60 dB` |

Density is useful for the scalar cloud: 25k adds 1.03 dB mean over 2.5k and
visibly improves the silhouette. It does not cure the Gaussian objective. The
two-update initialization diagnostic beats the normally trained 25k Gaussian
by 1.16/0.56 dB mean/worst and 26.3 recall points. Raising the generic support
survival guard from 50% to 80% restores 98.7% recall, but scores only
20.32/18.03 dB and reduces precision to 89.8%; that code was removed. A global
particle-count guard is not the required coverage objective.

A narrower recovery is selected. It activates only when the single-light fit
would retain fewer than three quarters of its evidence-backed inputs, after the
existing half-cloud collapse guard. Each low-opacity point recovers initialized
opacity and covariance only if its center and six two-sigma axis endpoints lie
inside at least 97.5% of construction-mask samples. This restores 3,664 points.
On the 12 internal cameras it moves the ordinary Gaussian from
`25.81/21.44;19.51/15.54 dB;60.3/97.6%` to
`26.24/22.28;19.98/16.42 dB;68.6/97.7%`. On the nine official pairs it improves
19.28/17.39→19.64/18.07 dB, recall 60.8→69.6%, and precision 95.8→96.0%.
Healthy fits above the three-quarter boundary, maskless captures, multilight
continuations, and full restores are exact no-ops.

The independently prepared OWL Apple split is the real-capture no-op control.
Its ordinary fit retains 21,512 of 24,996 particles (86.1%), so recovery changes
nothing. The nine internal held cameras score `23.35/18.70;15.00/9.19 dB` with
81.2% recall and 98.3% precision. No Apple ground-truth relighting pair entered
selection or this control.

Foreground-stratified photo batches do not replace recovery. Sampling 50% of
support updates from construction foreground lowers the held Gaussian to
`25.59/21.56;19.91/15.90 dB;67.8/95.5%`. At 25%, foreground quality and recall
rise to `20.27/16.65 dB;72.8%`, but whole-frame mean falls to 26.17 dB and
precision to 96.7%. Both arms fail the fixed all-metric gate, and the sampler
is removed. A separate field ablation shows why the retained recovery is
small: opacity alone is insufficient, opacity plus initialized scale is
byte-identical to restoring opacity, scale, and rotation. Rotation was never
trained on this path, so the redundant write is removed.

The selected evidence is deliberately narrow: an all-cloud pipeline can
recover a recognizable surface and react to an independent natural environment,
but appearance is grey, soft, and missing high-frequency texture. The local
recovery bounds damage after fitting; the next implementation should make
foreground ownership part of the Gaussian objective so useful supports never
need restoration. It may advance only if mean, tail, recall, and precision
improve over the recovered cloud together. Then constrain material/light
decomposition with shared material structure or calibrated repeated
environments; one unknown input environment cannot by itself identify arbitrary
per-point BRDFs. Batch point proposals before revisiting the exact 44-minute
refinement. ReNé remains behind unavailable empty-enclosure data and honest
background subtraction. The next repeated-light gate is public LUCES-MV; no
result depends on OLATverse access.

### LUCES-MV finite-light route

The point-light capability is implemented independently of the dataset
adapter. A public `PointLight` carries world position, outgoing axis, RGB
intensity, and cosine-power exponent; it supplies inverse-square diffuse
response on the CPU. Analytical material/normal refinement and the Gaussian
support optimizer accept one such light per view. The Gaussian path constructs
the response from existing Meganeura graph operations, with no new operation,
shader group, or shader-entry variant. Exact graph/oracle and moving-rig
end-to-end synthetic tests execute on the RTX 5070.

The official LUCES-MV Owl archive and calibration files have been downloaded
under a 2 GiB cgroup into ignored storage. The 1,893,639,182-byte archive is
SHA-256 `ced6a0fb5a6e8ac4fa447ebfcd965ee4c6a74e20fe61dbef4722a3db1942bc2f`;
the download peaks at 1,467,895,808 bytes with no swap, OOM, pressure, or GPU
fault. A simple pure-Rust fifteen-light diffuse solve confirms the public
calibration before geometry work: views 000/018/060 have mean normal errors
13.69°/16.77°/19.69°, medians 10.01°/13.04°/17.41°, and P90
28.58°/36.39°/34.89° against the provided normals at every fourth masked
pixel. These are calibration sanity results, not reconstruction quality.

The LUCES directory adapter is now complete. It loads RGB16 and masks without
display decoding, parses the release's stored NumPy camera extrinsics without
a ZIP/NPY dependency, transforms each camera-local LED into world coordinates,
and forms fifteen aligned captures. Loading all 180 Owl images at 80×60 gives a
0.329 normalized radiance peak and a 1,138,884,608-byte warm peak in a 2 GiB
cgroup, with zero swap, limit, OOM, or GPU event.

The predeclared Owl split is now live: cameras 000/024/048 and LEDs 03/09/15
are held out. Correcting the dataset-scale far plane and balancing masked ray
batches lifts the 4,096-site held-camera static gate from a background-only
21.56 dB to 33.21 dB. A stock-Rust 16,384-site run needs neither Qhull nor a new
dependency. Extracting at a two-pixel rather than five-pixel merge scale leaves
589 point surfels with 98.4% held-camera mask recall and 91.0% precision under
the source light.

The calibrated diffuse fit loads only the twelve construction LEDs until the
scalar and Gaussian clouds are serialized. The production relight tracer now
selects a finite emitter by changing one uniform in the existing pipeline; no
shader group or shader entry was added. Across the nine camera/light pairs
excluded from fitting, the scalar cloud reaches 34.42/32.16 dB whole-frame and
24.72/22.61 dB foreground mean/worst, with 98.0% recall and 93.3% precision.
The 589-particle Gaussian reaches 34.01/32.66 dB whole-frame and 24.13/22.66 dB
foreground, with 94.9% recall. This clears the finite-light transport and split
gate with complete production renders, not projected-center samples. The
visibly soft ridges make spatial surface detail and correspondence precision
the next target. Ground-truth depth was opened only after that gate, as an
offline diagnosis of rejected surfaces rather than a fitting input. If the
non-commercial licence prevents continued use, apply the same contract to
public DiLiGenT-MV rather than waiting for OLATverse.

That first quality step is now bounded. Higher training resolution improves
the static field but worsens projected depth and calibrated relighting; dense,
matched-scale, fixed-radius, adaptive-radius, subpixel-read, and
construction-light-average controls all trade one metric for another. Strict
four-light and RGB epipolar tracks remain too sparse and depth-inconsistent to
merge. The next proposal therefore has to establish cross-view responsibility
before adding point support. It may use the calibrated light set, but it may
not loosen pair matching, choose from excluded cameras/lights, or add a mesh
fallback.

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
- [x] Reject point-count and source-count thresholds on the metal sculpture;
  confirm at 256 px that selection-valid independent tracks already overlap
  Gaussian alpha rather than supplying a coherent missing patch.
- [x] Bound broad specular capacity on the exact cloud and retain the
  conservative transfer as explicit-only after a painted-toy tail regression.
- [x] Reject four spatial response regions and sharper roughness: the regions
  collapse to the global proposal and the directional tail remains.
- [x] Bound a stronger diffuse transfer; retain the halfway correction because
  even the smallest tested 1% extra step regresses one fixed-cloud
  whole-frame tail.
- [x] Attribute the fixed-cloud residual between per-light radiometry,
  visibility, and bounded indirect transport; reject scalar gains, extra
  samples, and same-cost stratification, and bound the remaining diffuse/F0
  response by the false-support whole-frame tail.
- [x] Reconstruct exact dense source-plane evidence and test a bounded post-fit
  centre correction on two established checkpoints plus a paired fresh basin;
  remove it when the fresh basin regresses every light group.
- [x] Co-optimize calibrated source-plane feedback during multi-light geometry;
  remove it when 2.5%, 0.5%, and 0.1% total corrections all exceed zero-arm
  variance and fail the first object's independent internal gate.
- [x] Keep centers fixed and constrain only covariance support that crosses a
  calibrated source-view depth or silhouette discontinuity, coupled to opacity
  so the established foreground response is preserved.
- [x] Screen calibrated depth/mask endpoint crossings broadly, individually,
  and after exact-compositor ownership ranking; remove the diagnostic when all
  construction survivors remain numerically neutral and fail validation.
- [x] Test opacity-mean, per-contributor, and position-only contributor depth
  losses after low-resolution front reduction; remove them when all weights and
  blends lose independent tails.
- [x] Preserve original source camera/pixel/depth observations in alternating
  depth-only batches with exact ray candidates and held source cameras; reject
  metric depth when two weights and all blends lose independent tails.
- [x] Replace metric depth with exact-ray ordinal free-space and support-band
  objectives; remove all three arms when every blend loses independent tails.
- [x] Expand one verified shared-view component by cloud-only local resampling,
  share one patch material, and pass repeated internal complete-render gates.
- [x] Screen three predeclared Hugging Face OpenIllumination objects with fixed
  construction-only thresholds; stop when none yields a coherent proposal and
  leave every official camera and excluded light unopened.
- [x] Bound LUCES raster resolution, point density, global/adaptive support,
  construction-light averaging, subpixel reads, and strict epipolar tracks;
  remove every arm that loses a complete camera/light metric cell.
- [x] Bound 32/64-way shared appearance and exact finite-light image-space
  material polishing; remove it when scalar gains trade against coverage and
  Gaussian whole-frame transfer.
- [x] Acquire a fresh non-empty patch proposal and repeat the fixed recipe on
  an untouched final split before adding any production merge/API.
- [x] Repeat the exact eight-pass anchored-albedo recipe on Bear; keep the
  merge out of production when its only patch loses validation recall.
- [x] Run an untouched Pot2 reconstruction and seed its selected patch before
  normal/material/Gaussian fitting; reject when two final foreground tails
  still regress despite every mean and coverage metric improving.
- [x] Test construction-only geometry/opacity continuation checkpoints; reject
  them when foreground tail and precision cross at different checkpoints.
- [ ] Fit only proposed-patch appearance in the final Gaussian compositor,
  then repeat without changing correspondence or geometry thresholds before
  adding any merge API.

The fresh DiLiGenT-MV Cow proposal closes the first of those two gates. Twenty-
four construction lights recover a robust per-pixel diffuse-albedo image. For
each of eight matching cameras, one diagnostic pass restricts that camera to
missing pixels and lets the other cameras match measured foreground; tracks
without the selected missing observation are discarded, then the eight result
sets are deduplicated. This multi-pass search produces 137 unique tracks and
seven shared-view patches. The unchanged two-pixel support cap leaves five
foreground-safe patches; complete-render selection plus validation chooses
patches 1/4/5, 32 surfels total, with one shared material per patch.

The official four held cameras and eight held lights are opened only after that
subset is fixed. On the held/held cross-product, whole-frame mean/worst moves
`30.812/28.899→30.827/28.912` dB, foreground
`19.737/17.548→19.754/17.592` dB, recall `84.684%→85.500%`, and precision
`94.501%→94.517%`. Every corresponding metric also improves in the other
three fitted/held camera/light cells. The selected artifact is
`target/audit-runs/diligent-mv/cow/16k/missing-surface-candidate/scene-gaussian.ply`.
The complete diagnostic peaks at 266.5 MiB with no swap, cgroup event, or GPU
fault. Production now contains only the calibrated distant-light albedo
estimator; the multi-pass search, dataset-specific subset selection, and merge
remain outside the library.

The controls explain why. A direct one-pass qualifying matcher finds 67 Cow
tracks and three patches of 10/5/5 tracks. Its internally selected 10-surfel
subset passes construction selection and validation, but the final matrix loses
foreground tails by 0.004–0.009 dB (and one whole-frame tail by 0.0004 dB). The
same fixed one-pass recipe on Bear finds 26 qualifying tracks, 24 inside the
visual hull, and no five-track patch. Alternative retention is real: repeating
the exact eight-pass Cow policy on Bear raises this to 71 unique tracks, 67 in
the visual hull, and one six-track patch. But none of the 67 points projects to
missing support in three of four selection views; the patch covers 0.4% of
selection holes and 0% of validation holes, then changes combined validation
recall by -0.010 points. The official held cameras and lights remain unopened.
The clean 190 MiB run has zero swap, OOM, throttle, or GPU fault. The proposed
source/qualifier API and reconstruction integration are therefore removed
rather than encoding a diagnostic search policy.

Pot2 is a third untouched control extracted from the pinned archive without
released mesh, normal, or depth data. Its fixed 16k route produces 1,870
surface points and 1,850 fitted Gaussians. The exact eight-pass search finds 70
unique tracks and two patches of 6/9 points. The six-point patch has no
selection-hole coverage and is rejected. The nine-point patch improves internal
means and coverage; a selection-only albedo screen fixes `1.5×` before opening
validation, where foreground worst still loses 0.0051 dB. Post-hoc insertion is
therefore rejected.

Seeding those nine point-cloud surfels into the unfitted surface and rerunning
the unchanged calibrated normal/material solve and Gaussian continuation is
substantially better. All four final Gaussian means, recall, and precision
improve; held/held whole/foreground mean moves
`33.2416/23.5994→33.2955/23.6644` dB. Only fitted-light/fitted-camera and
held-light/fitted-camera foreground worst regress, by 0.0054 and 0.0028 dB.
The strict gate rejects the candidate at
`target/audit-runs/diligent-mv/pot2-16k/missing-surface-joint/scene.ply`.
This motivates a tail-aware continuation control rather than a broader matcher
or post-fit merge.

A construction-only continuation checkpoint does not resolve the last trade.
At 87.5% interpolation from the seeded initializer to the trained model, all
four PSNR aggregates and recall beat the unseeded control, but precision is
`95.10%` versus `95.30%`. Restoring full learned opacity raises precision to
`95.31%` but moves foreground worst to `20.0533` dB, below the `20.0594` dB
control. Holding position or normals at the checkpoint does not change that
boundary. The diagnostic instrumentation is removed; the remaining degree of
freedom is patch appearance in the final Gaussian compositor, not a global
geometry/opacity rollback.

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
