# Relighting capture and dataset ladder

The target is one point-cloud scene that can be queried by both camera and
illumination: novel camera, captured light; captured camera, novel light; and
novel camera, novel light. A static light-field score establishes only the
first of those.

## What counts as evidence

A useful end-to-end gate needs all of the following:

- one rigid scene observed from multiple calibrated camera poses;
- independently varied, measured illumination on that same scene;
- linear or exposure-calibrated images, preferably with masks;
- cameras and at least one entire lighting condition excluded from fitting;
- references at the camera/light cross-product that was excluded.

Fixed-camera multi-illumination data is still useful for exposure, white
balance, and flash/ambient experiments, but it cannot measure novel-view
reconstruction.

## Ranked datasets

| Dataset | Camera/light coverage | Best use here | Limitation |
| --- | --- | --- | --- |
| [OpenIllumination](https://oppo-us-research.github.io/OpenIllumination/) | 64 objects, 70 views, 13 multi-LED patterns and 142 OLAT conditions | **First real gate.** It has camera poses, light calibration, masks, official train/test views, CC BY 4.0 data, and selective downloads. | Roughly 900 GB in full. The LEDs are finite-distance emitters, while the current renderer accepts a distant environment map. |
| [ReNé](https://eyecan-ai.github.io/rene/) | 20 objects, 50 views by 40 OLAT conditions | Second object-level cross-check with calibrated camera and light poses. | Also uses local point lights; access and preprocessing are less convenient. |
| [DiLiGenT-MV](https://sites.google.com/site/photometricstereodata/mv) | 5 objects, 20 views by 96 calibrated lights | Smallest serious photometric-geometry gate; good for normals and diffuse material. | Only five object-centric scenes and point-light illumination. |
| [OLATverse](https://vcai.mpi-inf.mpg.de/projects/OLATverse/) | 765 objects, 35 cameras by 331 OLATs, plus environment and gradient lights | Eventual broad material/generalization benchmark. | Huge and registration-gated; inappropriate for the first integration. |
| [Stanford-ORB](https://stanfordorb.github.io/) | 14 objects captured in multiple real environments with HDR environment maps and registered poses | Best match for the current distant-environment renderer and an important in-the-wild relighting check. | It does not provide a dense aligned camera/light grid for each background environment. |
| [DTU robot data](https://roboimagedata.compute.dtu.dk/?page_id=24) | 60 scenes, 119 cameras by 19 LEDs | Larger, more scene-like multi-view/multi-light stress test. | Approximately 730 GB in full and built around local LEDs. |
| [OpenSubstance](https://opensubstance.github.io/) | 187 objects, 270 views and 1,637 lighting conditions | Later high-resolution material and specular benchmark. | Multi-terabyte scale and access by request. |

The datasets below do not satisfy the two-axis gate, but can support isolated
capture research:

| Dataset | What it provides | Use, but not as |
| --- | --- | --- |
| [MIT Multi-Illumination Images in the Wild](https://projects.csail.mit.edu/illumination/) | More than 1,000 fixed-view scenes under 25 illuminations, including raw and HDR data | Illumination priors and exposure tests; not novel-view truth. |
| [FAU Multi-Illuminant](https://www5.cs.fau.de/research/data/multi-illuminant-dataset/) | Fixed cameras and lights with controlled combinations and shutter settings | Color constancy; not reconstruction. |
| [Flash/Ambient](https://yaksoy.github.io/flashambient/) | Aligned mobile flash-only and ambient-only pairs | Capture decomposition and denoising; not a posed light field. |
| [MILL](https://arxiv.org/html/2511.15496v1) | Fixed-camera RAW captures at 11 measured low-light intensities | Low-light robustness; not directional relighting or novel view. |

## First gate

The checked-in selective fetch downloads 22 cameras of one object under four
OLATs, about 24 MB rather than the complete 900 GB corpus. Revisions are pinned
in the script. The Rust importer converts the published camera-to-world
matrices to pose-only COLMAP binaries, canonicalizes the published binary masks
to full `0/255` coverage, writes the official five-camera split, emits an
explicit nearest-12 `patch-match-train.cfg` containing training cameras only,
and creates distant-light `.f32` environments. `sparse/0`
contains every pose for final scoring; `sparse/train` physically excludes the
official test cameras and is the input for geometry initialization or dense
reconstruction:

```bash
etc/fetch_openillumination.sh

cargo run -p blade-volume-train --bin import_openillumination -- \
    --input etc/data/openillumination/OLAT/obj_16_friends_cup \
    --output etc/data/openillumination/prepared/obj_16_friends_cup \
    --light-positions etc/data/openillumination/light_pos.npy \
    --light 000 --light 062 --light 082 --light 092
```

OpenIllumination does not provide a sparse SfM cloud. `camera-lattice` therefore
fills the calibrated focus volume with point-cloud sites. With masks it
deterministically tightens that volume and seeds density and DC colour from a
soft visual hull; sites outside the hull remain low-density but trainable.
Train OLAT 062 while keeping the published five cameras completely out of
initialization and fitting:

```bash
cargo run --release -p blade-volume-train --bin train_colmap -- \
    --sparse etc/data/openillumination/prepared/obj_16_friends_cup/sparse/train \
    --images etc/data/openillumination/OLAT/obj_16_friends_cup/Lights/062/raw_undistorted \
    --masks etc/data/openillumination/prepared/obj_16_friends_cup/masks \
    --output target/audit-runs/openillumination/foam.ply \
    --initialization camera-lattice --max-points 4096 --views 0 \
    --width 128 --height 174 --pixel-batch 1024 --views-per-batch 16 \
    --steps-per-view 200 --max-steps 192 --sh-degree 2
```

Then fit OLATs 062, 082, and 092 and score OLAT 000, which is loaded only after
all fitting finishes:

```bash
cargo run --release -p blade-volume-train --bin reconstruct -- \
    --sparse etc/data/openillumination/prepared/obj_16_friends_cup/sparse/0 \
    --images etc/data/openillumination/OLAT/obj_16_friends_cup/Lights/062/raw_undistorted \
    --masks etc/data/openillumination/prepared/obj_16_friends_cup/masks \
    --test-list etc/data/openillumination/prepared/obj_16_friends_cup/test.txt \
    --width 128 --stride 1 \
    --environment etc/data/openillumination/prepared/obj_16_friends_cup/light-062.f32 \
    --normal-images etc/data/openillumination/OLAT/obj_16_friends_cup/Lights/082/raw_undistorted \
    --normal-environment etc/data/openillumination/prepared/obj_16_friends_cup/light-082.f32 \
    --normal-images etc/data/openillumination/OLAT/obj_16_friends_cup/Lights/092/raw_undistorted \
    --normal-environment etc/data/openillumination/prepared/obj_16_friends_cup/light-092.f32 \
    --held-out-images etc/data/openillumination/OLAT/obj_16_friends_cup/Lights/000/raw_undistorted \
    --held-out-environment etc/data/openillumination/prepared/obj_16_friends_cup/light-000.f32 \
    --foam target/audit-runs/openillumination/foam.ply \
    --gaussian-output target/audit-runs/openillumination/static.ply \
    --pbr-gaussian-output target/audit-runs/openillumination/pbr.ply \
    --gaussian-steps 1500 \
    --dump target/audit-runs/openillumination/images
```

Static light-field comparisons are written under `images/static-light/`.
`relight` and `g-relight` use the official held cameras under OLAT 000; their
comparisons are under `images/held-light/{scalar,gaussian}/`. The command also
prints black and capture-light-copy baselines. No downloaded data or generated
results belong in Git.

Both held-light arguments are repeatable. Every fitted asset can therefore be
scored against several lights which remain unloaded until all fitting and
serialization finish. Multi-light comparisons are separated under
`images/held-light/{0,1,...}/` and printed as `relight-N`/`g-relight-N`:

```bash
    --held-out-images .../Lights/000/raw_undistorted \
    --held-out-environment .../light-000.f32 \
    --held-out-images .../Lights/086/raw_undistorted \
    --held-out-environment .../light-086.f32
```

### Current result: plumbing passes, quality does not

The first current-tree run on `obj_16_friends_cup` establishes an honest
failure baseline:

| Output | Held condition | Whole-frame PSNR | Foreground PSNR | Frame coverage | Mask recall / precision | Baseline comparison |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Static Gaussian | OLAT 062, official held cameras | 31.51 / 28.99 dB | — | — | — | Black: 31.10 / 28.46 dB; a narrow numerical win, visibly still a dark blur. |
| Relightable scalar cloud | unseen OLAT 000, official held cameras | 23.62 / 20.61 dB | 16.56 / 14.30 dB | 14.1% | 73.0% / 55.3% | Loses badly to both foreground baselines. |
| Relightable Gaussian | unseen OLAT 000, official held cameras | 26.12 / 23.89 dB | 18.09 / 15.72 dB | 10.3% | 61.4% / 63.5% | All 638 particles persist after rejecting a degenerate support fit. The cloud is visible but blurry and still loses both baselines. |

The OLAT-000 black and capture-light-copy baselines are respectively
29.69/28.11 and 30.30/28.16 dB over the whole frame, but only 19.93/18.08 and
20.55/19.24 dB on the foreground. Mean/worst foreground PSNR is now reported
directly by `reconstruct`; it measures missing support and wrong shading on the
object while mask precision continues to expose geometry rendered outside it.

The full run, including static and held-light reference/render pairs, is under
`target/audit-runs/openillumination/official-fallback-extent/`; the foam
training run is under `target/audit-runs/openillumination/official-static-v1/`.
Both are ignored research artifacts. Peak cgroup memory was 946 MB for training
and 1.12 GB for the guarded reconstruction including its release build, with no
memory pressure, OOM, socket-throttling, or GPU-fault marker.

The initial PBR support optimizer kept only 22 of 638 particles above the
persistence threshold. The reconstruction now treats retaining less than one
quarter of the established cloud as a failed fit and restores its opacity and
scale before calibrated-light continuation and persistence. The rejected fit
no longer feeds its radii back into the scalar surface. Instead, the fallback
matches the Gaussian's actual 0.03 production response cutoff to the
established finite surfel radius; this raises held-light Gaussian quality by
0.38/0.46 dB mean/worst and foreground recall by 5.3 points. The same guard
runs after the multi-light pass. It does not activate on the established
synthetic control, which retained 2,184 of 2,343 particles; that control remains
at 24.26/23.44 dB mean/worst and 55.1% coverage. This is a safety bound, not a
quality claim: the guarded OpenIllumination render exposes that normals,
materials, and the light model remain wrong.

The generated lights are explicit approximations, not claimed radiometric
calibration. The public `light_pos.npy` gives the 142 directions used directly
by the official photometric-stereo code. The
[paper](https://papers.nips.cc/paper_files/paper/2023/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html)
reports identical LED
intensity, an arbitrary common intensity of five, and a fitted spherical-
Gaussian sharpness of 236.9705. The importer normalizes each direction and
assigns unit normal-incidence irradiance to a 0.08-radian distant lobe, fixing
the otherwise arbitrary material/light scale consistently across OLATs.

Four current-tree controls reject the obvious calibration substitutions:

- treating the raw unit-sphere locations as finite inverse-square point lights
  was 0.05–0.59 dB worse than the distant control on held cameras for each of
  OLATs 000, 062, 082, and 092; the mean gain inferred from training lights was
  2.12 dB worse on OLAT 000 than its diagnostic per-light fit;
- reproducing the published spherical-Gaussian lobe convention and sharpness
  in the complete gate reduced held-light Gaussian quality to 25.75/23.57 dB;
- a joint diffuse-albedo solve over OLATs 062, 082, and 092 reached only
  25.64/23.54 dB under OLAT 000 after subtracting the fixed specular term;
- following the
  [supplementary protocol](https://oppo-us-research.github.io/OpenIllumination/files/supp.pdf)'s
  combined object-plus-support mask
  for fitting reduced object-only held-light quality to 24.60/22.96 dB, with
  64.7% recall and 57.6% precision.

The exact ignored controls are under
`target/audit-runs/openillumination/{point-light-control,sg-lobe-full,multilight-albedo-control,official-com-mask}/`.
They leave the distant lobe, object mask, and primary-light material fit as the
selected baseline. Missing foreground support, shadows, interreflection, and
non-diffuse appearance remain live explanations; tuning against OLAT 000 is
still forbidden.

The next attribution pass rules out global geometry and transport knobs as a
solution to the visible failure:

- raising the source field from 4,096 sites at 128 pixels to 16,384 sites at
  256 pixels produces 3,193 rather than 638 extracted particles, but those
  particles remain a fragmented volume. Held-light Gaussian quality falls to
  25.15/23.12 dB whole-frame and 18.06/15.94 dB foreground;
- a silhouette-only visual hull improves the static held-camera result to
  31.62/29.12 dB, but no consensus threshold preserves foreground mean, tail,
  recall, and precision together. Denser hulls are worse. Clipping or weakly
  filling the learned density inside the hull has the same trade: the best
  mean gains come from rendering less of the object, and a capture-light
  validation camera correctly rejects the conservative candidates;
- widening every observation footprint from 1.7 to 2.5 cells lets 381 rather
  than 359 particles receive evidence and raises this real held-light
  Gaussian to 26.14/24.22 dB whole-frame and 18.70/16.21 dB foreground. The
  identical synthetic gate collapses from 24.26/23.44 to 20.38/19.20 dB;
  narrowing only the final output support does not repair that regression;
- primary lattice position learning, SH-0 geometry, an opaque visual-hull
  seed, and one or two sequential secondary-light continuations each move a
  real mean by a few tenths but regress a tail or mask recall. None passes the
  complete gate;
- using the strongest foam cell as a stable cross-view surface identity also
  fails to resolve the volume. One disc per identity collapses 638 particles
  to 356 and reduces held-light Gaussian quality to 25.68/23.14 dB. Splitting
  identities spatially produces 983 particles and 25.89/24.00 dB; filtering
  ordinary spatial fusion by identity consensus is effectively neutral at
  26.15/23.88 dB, with slightly worse foreground tail and mask recall. The
  extra depth-map channel and fusion path were therefore removed;
- enabling sampled visibility and the existing indirect bounce together on
  the scalar control reaches only 23.27/20.26 dB whole-frame and
  16.10/13.73 dB foreground. Transport cannot repair the surface on which it
  is evaluated.

These ignored controls live under
`target/audit-runs/openillumination/{dense-16k-256,visual-hull,primary-position-4k,geometry-sh0-4k,opaque-hull-sh0,secondary-light-4k,two-secondary-4k,wide-observe-narrow-support,wide-observe-narrow-support-synthetic,shadow-bounce-scalar-16,peak-point-fusion,peak-point-voxel-fusion,peak-point-voxel-min3,peak-point-voxel-min3-r18,peak-point-supported-fusion}/`.
All GPU work ran in the guarded 12 GiB scope. The final default-feature and
all-feature workspace test scopes peaked at 7.10 GiB and 6.17 GiB respectively;
the experimental and test scopes reported no swap, OOM, socket throttling, or
GPU fault.

### Broader-light holdout

Three extra selective OLAT downloads test whether calibrated-light coverage,
rather than the surface, explains the first real failure. The original
training directions 062/082/092 are nearly coplanar (+Z/+X/-X). Adding OLAT
139 supplies +Y evidence; adding 015 supplies a -Y/+Z diagonal. OLAT 000 and
the nearly opposite -Z OLAT 086 are both excluded from fitting and scored from
one persisted asset by the repeatable held-light interface above.

| Training OLATs | Held OLAT | Gaussian whole-frame PSNR | Foreground PSNR | Foreground recall / precision |
| --- | --- | ---: | ---: | ---: |
| 062/082/092 | 000 | 26.12 / 23.89 dB | 18.09 / 15.72 dB | 61.4% / 63.5% |
| +139 | 000 | 26.92 / 24.41 dB | 19.06 / 16.12 dB | 60.5% / 63.7% |
| +139/+015 | 000 | 27.87 / 25.80 dB | 19.68 / 17.47 dB | 59.6% / 64.0% |
| 062/082/092 | 086 | 22.83 / 20.57 dB | 13.49 / 11.46 dB | 61.4% / 63.5% |
| +139/+015 | 086 | 22.59 / 20.86 dB | 13.15 / 10.78 dB | 59.6% / 64.0% |

The extra lights make OLAT 000 a much better interpolation, but they do not
improve the independent OLAT 086 transfer: mean and foreground tail regress.
Fixing the Gaussian continuation to 375 total updates improves OLAT-000 recall
but still regresses OLAT 086, so the extra evidence is not merely being
over-optimized. Adding 086 to form an approximate six-axis training set then
reduces OLAT-000 recall to 47.8%. None of these schedules is selected as a new
model default. The transferable result is the evaluation rule: capture lights
must span 3D, and a relightable claim now requires at least two excluded light
directions rather than the nearest unseen OLAT alone.

### Joint known-light material

The baseline above still estimates each durable diffuse albedo from the
primary OLAT only. A small analytical continuation now requires one
per-particle albedo to explain all calibrated captures. It changes no geometry
representation, shader, graph operation, model field, option, or dependency.
The production command selects it only for one material per particle and at
least five known lights: the three-light control improves OLAT 000 but regresses
OLAT 086, and a four-light control still loses one whole-frame tail.

| Fit | Held OLAT 000 whole / foreground PSNR | Held OLAT 086 whole / foreground PSNR |
| --- | ---: | ---: |
| Three lights, primary-only material | 26.12 / 23.89; 18.09 / 15.72 dB | 22.83 / 20.57; 13.49 / 11.46 dB |
| Three lights, joint material | 27.33 / 25.62; 19.54 / 17.59 dB | 22.78 / 20.47; 13.44 / 11.23 dB |
| Five lights, primary-only material | 27.87 / 25.80; 19.68 / 17.47 dB | 22.59 / 20.86; 13.15 / 10.78 dB |
| **Five lights, joint material** | **28.94 / 27.18; 20.48 / 18.49 dB** | **22.81 / 21.01; 13.39 / 10.87 dB** |

The selected row improves every whole-frame and foreground mean/tail from one
persisted asset. OLAT 000 still loses the black and capture-copy whole-frame
baselines. OLAT 086 beats black and the capture-copy mean, but its worst view
still trails capture-copy by 0.22 dB. This is material-transfer progress, not a
claim that real relighting now passes. The next blocker remains the visibly
fragmented surface and the unmodelled shadows/interreflection that its free
albedos still absorb.

The exact ignored runs are under
`target/audit-runs/openillumination/{four-training-lights,four-training-lights-015,five-training-lights-015,six-axis-training-lights,three-training-lights-held-086,four-training-lights-139-held-086,five-training-lights-held-086,five-training-lights-fixed-budget-held-000,five-training-lights-fixed-budget-held-086,five-training-two-held,joint-material-three-training-two-held,joint-material-four-139-two-held,joint-material-four-015-two-held,joint-material-five-training-two-held,joint-material-selected-five}/`.
All complete runs used the guarded 12 GiB scope; the final two-held run peaked
at 1.1 GiB including its release rebuild, with zero swap, OOM, or GPU fault.

### Broad-light dense geometry

The decisive dense test uses lighting pattern 013 from the Friends-cup pattern
capture as a broad stereo source while keeping all five official OLAT test
cameras out of matching, fusion, alignment, and fitting. The published pattern
poses do not fuse at two-view consistency. Joint feature matching is healthy:
177 verified cross-capture pairs provide 8,820 matches. A calibrated joint SfM
model registers 12 OLAT training images and 24 pattern images, and alignment
through only the OLAT training cameras leaves 0.0058 scene-unit mean center
error. Strict five-view stereo fusion then produces 24,036 oriented points.

That apparently successful cloud is not registered accurately enough for the
OLAT images. The joint rotations remain 3.98 degrees from the published OLAT
orientations on average, and its first reconstruction has only 27.2% mask
recall and 21.8% precision. A global similarity fit against the 17 training
silhouettes raises raw point-cloud F1 from 0.49 to 0.74. On the five untouched
cameras it independently raises F1 from 0.40 to 0.66, so the fit does recover
real geometric alignment rather than merely memorizing the training views.
Deterministic averaging to 2,500 points gives the strongest balanced rendered
candidate:

| Geometry | Held OLAT 000 whole / foreground | Held OLAT 086 whole / foreground |
| --- | ---: | ---: |
| Selected learned surface | 28.94/27.18; 20.48/18.49 dB | 22.81/21.01; 13.39/10.87 dB |
| Aligned broad-light dense cloud | 28.45/25.94; 20.15/17.72 dB | 23.31/21.11; 13.88/10.95 dB |

The dense candidate improves every OLAT-086 cell but regresses every OLAT-000
cell, especially the held-camera tail, so it is not selected. Its silhouette
is substantially more complete while its appearance remains visibly speckled.
The useful conclusion is a capture constraint: broad-light geometry images
must be taken in the same rigid session at the exact measured-light camera
poses, not on a separately mounted object followed by a global registration.
`etc/colmap.sh --dense-images PATH` now supports that separation without using
the broad images to fit the captured appearance. The ignored experiment is
under `target/audit-runs/openillumination/{lighting-pattern-audit,joint-pattern-dense,silhouette-align}/`;
all GPU work stayed healthy inside the 12 GiB scope.

### Same-session grouped-light reconstruction

The follow-up removes the registration ambiguity above. It uses
`lighting_patterns/obj_63-fabric-friends-cup`, where geometry and every
lighting condition were recorded in one rigid session with identical exposure.
The first six published patterns are disjoint 23-LED spatial groups; pattern
013 is the broad all-142-LED geometry light. The checked-in fetcher carries the
group memberships recovered by registering the official pattern diagram to the
pinned 142-light calibration. The assignment remains identical under 12
projection and sampling perturbations. The importer accepts those groups
without a dataset-specific rendering path: `LABEL=i,j,...` sums ordinary
calibrated emitters, while `LABEL=all` selects all 142.

```bash
etc/fetch_openillumination.sh --capture lighting_patterns
# Run the import_openillumination command printed by the fetcher.
```

The leakage-free SfM stage registers 24 training cameras and no evaluation
camera. Its centers differ from the official calibration by 0.0104 scene units
on average and rotations by 0.91 degrees. Geometrically consistent PatchMatch
and strict five-view fusion use only the broad pattern-013 training images and
produce 26,223 oriented points. No pattern-001..006 evaluation image enters
matching, stereo, fusion, or fitting. Deterministic spatial averaging retains
2,500 points for the measured reconstruction.

Patterns 001--004 plus the all-LED pattern 013 are the five known lights.
Patterns 005 and 006 and ten camera poses are excluded from fitting. Refinement
settings are selected only from those held cameras under known pattern 001;
the excluded-light captures are loaded after the asset is final. Three matched
repeats select `--render-refine-radii --render-refine-normals`: relative to the
clean medians, scalar foreground PSNR moves 15.32/13.99→15.99/14.45 dB and
recall 84.5%→87.8%; Gaussian foreground moves 16.33/14.53→16.47/14.80 dB.
Whole-frame tails change by -0.11 and -0.08 dB respectively, so this is a
support-focused final-quality schedule rather than a universal default.

The selected persisted assets then give the following untouched-light result.
The scalar asset is trained with eight visibility-plus-bounce rays and scored
with 64 through `--score-diffuse-samples 64`; the Gaussian path remains
analytic and is unchanged by that reporting option.

| Asset | Excluded pattern 005 whole / foreground | Excluded pattern 006 whole / foreground |
| --- | ---: | ---: |
| Scalar point surface | 23.78/22.66; 14.62/13.74 dB | 22.97/20.81; 13.52/11.80 dB |
| Full-covariance Gaussian | 22.96/20.69; 13.82/11.92 dB | 22.30/19.30; 12.92/10.62 dB |
| Strongest black/capture-copy baseline | 22.78/20.66; 12.35/10.71 dB | 23.43/19.11; 12.99/8.48 dB |

The scalar point surface beats both trivial baselines on the object foreground
under both excluded lights. The Gaussian does the same under pattern 005 and
misses the pattern-006 foreground mean by 0.07 dB while beating its tail by
2.14 dB. Both representations still lose the pattern-006 whole-frame mean to
black because roughly 91% of the frame is background; that easy baseline stays
in the gate rather than being discarded. The checked-in pattern-006 PNG shows
the more important failure directly: the relit bowl is recognizable, but its
surface is speckled and its appearance lacks the reference's print and smooth
shading.

Existing controls narrow the next work. Eight simultaneous position rounds
regress known-light Gaussian quality. Final opacity pruning is not responsible:
retaining all 2,500 particles changes recall by only 0.4 point and lowers
foreground PSNR. Per-point material refinement is inapplicable to a 2,500-entry
material table, and exact position-plus-radius coordinate descent exceeds 12
minutes without finishing where the selected bounded pass takes about 12
seconds. Production screens also reject one temporary appearance table per
particle and light: it gains small foreground PSNR but loses 1.3 recall points
and a known-light tail. Applying finite-light visibility plus one bounce as a
fixed per-particle factor raises recall by 0.8 point but loses 0.50/0.56 dB on
known-light Gaussian mean/tail and lowers precision.

Exact missing-ray ownership narrows the boundary further. Updating only the
largest current compositing weight on repeated camera rays does improve recall
and precision, but paints that particle's established appearance over new
support and loses known-light quality. Giving an owner a separate child and a
five-light material still fails because its missing rays are not necessarily
one physical point. Even an owner-depth child whose gain-invariant image patch
is consistent across four cameras fails the complete silhouette screen; when
forced, it lowers both precision and every known/held whole-frame mean. The
next target is therefore not another ownership-weighted update. It is an
explicit correspondence source: mutually match missing-foreground descriptors
or calibrated multi-light signatures across cameras, triangulate coherent
tracks into points, then let the existing point-cloud geometry, support, and
material fits evaluate them. More global radius, opacity, or iteration knobs
are not supported by this gate.

The final converged-evaluation run takes 32.1 seconds and peaks at 801 MB of
scoped host memory, uses no swap, and records no OOM, socket throttling, or GPU
fault. The RTX 5070 finishes at 49 °C. Exact ignored outputs and telemetry are under
`target/audit-runs/openillumination/lighting-pattern-audit/{fit8-score64-final,grouped-five-importer-final,selector-*}`.

### Explicit missing-surface tracks

The first implementation after that boundary is diagnostic-only. It computes
the exact current Gaussian alpha, takes masked foreground below 50% coverage,
and matches 3x3 four-light gain-invariant response patches along calibrated
epipolar lines. Matches must be mutual, triangulate with at least three
cameras, stay below one-pixel reprojection error, and pass a separately
reserved visual-hull screen. It adds no graph operation, shader, model field,
or dependency and does not mutate either production asset.

The fixed training set is split 16/4/4 into matching, selection, and validation
cameras. The ten official test cameras and excluded patterns 005/006 remain
untouched. The rejection funnel is 7,990 eligible descriptors, 2,026 sampled
anchors, 2,114 mutual pair matches, 534 multi-view groups, 390 triangulations,
220 observation-unique tracks, and 219 after the selection-camera visual hull.
Accepted tracks average 3.8 observations, 0.342 pixels reprojection error,
0.050 normalized descriptor error, and 50.0 degrees parallax; the worst mean
reprojection error is 0.751 pixels.

On the four validation cameras, 96.9% of track-cloud coverage stays inside the
foreground and it covers 9.3% of the current Gaussian's missing foreground.
Missing-only precision is 47.4% because about half the finite track footprints
overlap foreground the Gaussian already covers. Requiring missing support in
three of four separate selection cameras retains 81 tracks in a representative
repeat and moves validation foreground precision to 98.8%, missing-only
precision to 62.7%, and missing recall to 4.3%.

The validation renders trace the cup body, rim, and handle from unseen
matching views, with some isolated dust. This passes the correspondence-source
milestone but not the integration gate. Fitting five-light normals/materials
and appending the selected points at 0.25 opacity improves selection recall
`65.2%→66.0%` and precision `92.3%→92.4%`, but the worst foreground PSNR
falls `15.39→15.38` dB, so the append implementation was removed. No official
test camera or excluded pattern was loaded for that decision. The next
experiment must combine neighboring tracks into coherent oriented support and
optimize its complete render. Exact ignored PLYs, renders, and clean 12 GiB
telemetry are under
`target/audit-runs/openillumination/lighting-pattern-audit/{missing-tracks-v4,missing-tracks-v9}/`;
the latest run peaks at 700 MB with zero swap, OOM, throttling, or GPU fault.

A bounded local-covariance follow-up drops isolated tracks and caps inferred
support by the observed pixel footprint. Its adjacent repeat retains 73 of 79
selected tracks and reaches 5.1% missing recall, 66.2% missing-only precision,
and 99.0% foreground precision on the same validation slice. This remains a
track-only diagnostic pending a complete combined-render gate.

The first complete-render probe is still negative: the local patch improves
selection recall/precision by 1.11/0.09 points but loses 0.024/0.016 dB
foreground mean/worst. Subtracting the established render before fitting patch
materials also loses 0.018/0.055 dB because a fixed layer order is not the
renderer's exact per-ray order. Both probes were removed. The next step is a
frozen-prefix, trainable-suffix fit through complete Gaussian compositing.

That exact suffix fit was tested with the established prefix bitwise frozen.
After 625 updates it raises recall 0.83 point, but loses 0.26 point precision
and 0.023/0.016 dB foreground mean/worst on selection cameras. Its code was
removed. Track selection must now operate on coherent shared-view components,
with every component screened by a complete render before joint fitting.

The initial pass must report mean and worst held-camera PSNR, coverage, image
comparisons, point count, training time, peak cgroup memory, and the exact
object/light selection. A result is useful only if it beats a capture-light
colour or average-image baseline on the held light and the images visibly move
shadows and highlights in the right direction.

The current missing-surface diagnostic now groups tracks by adjacency in at
least two shared source views, estimates orientation within each connected
component, and screens each component with selection-only alpha renders. A
complete-render candidate once improved both selection and untouched
validation metrics, but the adjacent end-to-end repeat retrained a different
base Gaussian and regressed selection foreground mean/worst PSNR by
0.013/0.009 dB. Automatic merging remains disabled and no held light was used.
The next OpenIllumination run will persist one exact base Gaussian and demand
two repeat passes from that fixed input before any excluded data is opened.

That replay boundary is now implemented as `--missing-tracks-base`. Two full
runs against the exact v19 PBR Gaussian produce byte-identical 45-point PLYs
and PNGs, with the same 3.3% validation missing recall, 59.1% missing
precision, and 98.9% foreground precision. Peak scoped memory is 689/665 MB,
with no swap, OOM, throttling, or GPU fault. This validates deterministic track
and patch construction, not automatic integration.

The complete-render candidate was then replayed twice from that base. Both
runs selected the same 20-point component and wrote the same PLY (SHA-256
`b375a8c3bf364874e280ec44a55d4e86423e54680742b4fd0ed9c011a5b4a93e`).
Selection foreground mean/worst moved `16.318/15.723→16.326/15.737` dB;
validation moved `16.561/16.218→16.569/16.242` dB, with non-regressing mask
recall and precision. That cleared the predeclared boundary for opening the
official data, but the candidate did not clear the official gate. Relative to
the exact fixed base, known-light held-camera mean rounded down
`23.86→23.85` dB; excluded-light results contained mixed changes at the 0.01 dB
level and no visible recovery of the fragmented surface. Exact track-pixel
normal observations did not change the decision. The candidate and append
implementation were removed. The 12 GiB candidate/base evaluations peaked at
743/666 MB with zero swap, OOM, throttling, or GPU fault; ignored comparisons
are under `missing-tracks-fixed-held{,-base}` and telemetry is in
`/tmp/blade-volume-fixed-{candidate,base}-held.log`.

A diagnostic-only point resampling step now preserves those oriented track
surfels and adds a half-radius sample on each unique two-nearest-neighbor edge
inside a shared-view component. It adds no polygonal intermediate and does not
alter the fixed base. The two replay PLYs and PNG directories are byte
identical; their selected 105-point cloud improves track-only validation
missing recall `3.3%→3.8%`, missing precision `59.1%→59.6%`, and foreground
precision `98.9%→99.0%`. Recomputing all support after interpolation instead
shrinks the evidence points and drops recall to 2.3%, so that control was
removed. The positive resampling remains diagnostic pending a regularized
material and complete-render gate on fresh held data.

### Fresh second-object surface-support gate

`obj_31_painted_toy` is a different, concave painted figurine from the same
lighting-pattern dataset. The importer writes all 48 official poses to
`sparse/0`, but writes only 38 training poses to `sparse/train`; ten official
test cameras are therefore absent from undistortion, PatchMatch, and fusion.
It also canonicalizes the dataset's binary `0/1` masks to `0/255`. Before that
normalization, image decoding interpreted foreground as 0.0039 coverage and
produced zero usable observations.

Pose-only COLMAP fusion contributes one-view samples by design. Filtering the
fused cloud through at least 80% of the selected training silhouettes before
voxel downsampling removes only 68,278 of 1,254,170 inputs, but prevents a
small distant tail from consuming the fixed point budget. At 2,500 points the
number observed by training photographs rises from 1,887 to 2,385.

| 2,500-point scalar surface | Known-light held cameras foreground | Recall / precision | Excluded 005 foreground | Excluded 006 foreground |
| --- | ---: | ---: | ---: | ---: |
| Unfiltered | 16.46 / 15.38 dB | 98.4% / 78.9% | 15.00 / 13.82 dB | 13.48 / 11.98 dB |
| Training-mask soft hull | 17.54 / 16.87 dB | 96.8% / 93.1% | 15.32 / 14.06 dB | 13.61 / 12.06 dB |
| Strongest trivial baseline | — | — | 15.68 / 13.04 dB (capture copy) | 17.89 / 13.48 dB (black) |

The surface cleanup transfers to both excluded lights but does not solve
relighting. Pattern 006 is the useful counterexample: the photograph contains
strong directional self-shadowing, while the render is broadly gray. This is
not well described by a scalar exposure error, and it remains a transport and
closed-support failure.

A selection-only 5,000-point scalar run improves known-light held-camera
foreground to 18.15/17.52 dB, 97.7% recall, and 93.5% precision. Its final
complete run is mixed: excluded 005 changes `15.32/14.06→15.47/14.01` dB and
excluded 006 changes `13.61/12.06→13.68/12.11` dB. The mean gains do not excuse
the 005 tail regression. The matching Gaussian PBR path is not ready for that
density: its fixed opacity persistence rule either removes most particles or,
under a stricter guard control, over-expands them and loses precision. A
3,500-point control is also worse than the selected 2,500-point Gaussian.
Consequently, the soft-hull filter is retained, 2,500 remains the Gaussian
gate, and density-stable Gaussian support is explicit work rather than a
hidden point-count knob.

Nearest-camera PatchMatch source selection produced a larger one-view cloud
but did not improve the known-light reconstruction, so its importer changes
were dropped. Global radius inflation, equal silhouette weighting in the local
radius pass, and bypassing the five-light material fit were likewise rejected.
The selected run peaks at 823 MB in its 8 GiB scope with no swap, OOM, or GPU
fault. Outputs are under
`target/audit-runs/openillumination/obj31-hull-final/`; the filtered scalar
pattern-006 image is also shown in the README.

The fresh object also closes several post-processing routes. An
all-foreground version of the calibrated matcher finds 1,827 tracks from
47,621 descriptors; 1,711 pass selection-camera silhouettes and 1,654 belong
to coherent image-space patches. The 3,804-sample diagnostic has a recognizable
held-view outline and 93.2% foreground precision, but using it as geometry
falls to 53.5% recall and 62.7% precision. Applying the ordinary all-camera
soft hull leaves 882 samples and only 63.0% recall. The temporary selector and
initializer were removed; the missing-alpha diagnostic remains unchanged.

Three later controls improve the known-light render but fail excluded lights:

- aligned-light descriptor agreement rejects 2,632 of 20,000 spatial depth
  candidates and reaches `17.63/17.00` dB known-light foreground, but excluded
  005 reaches only `15.40/13.95` and excluded 006 `13.58/11.99` dB;
- one bounded normal-plane relaxation moves every point by 0.000713 world units
  on average, raises observed support to 2,424, and reaches
  `17.70/16.90` dB known-light foreground, but excluded lights fall to
  `15.21/13.75` and `13.36/11.90` dB;
- a visibility-aware multi-light albedo solve passes an exact cast-shadow unit
  control, yet reaches only `17.48/16.72`, `15.26/14.05`, and
  `13.52/11.96` dB on known/005/006. A one-radius shadow bias similarly gives
  mixed hundredth-decibel changes and is rejected.

Changing the capture schedule is not the escape hatch. Dropping the broad
all-LED pattern and keeping the joint material solve reduces excluded 005/006
foreground to `14.11/12.38` and `12.06/10.17` dB. Making the broad pattern the
primary capture reaches only `14.58/13.00` and `13.08/11.61` dB. Five known
lights with pattern 001 primary therefore remains the selected schedule.

These controls leave a narrower next step: derive local depth from per-view
photometric normals, anchor it with verified tracks, and require another
camera to confirm the patch before fusion. A response descriptor, a normal, or
the current cloud's visibility is evidence about a surface, but none alone is
an independent depth measurement.

## Capture direction

For our own controlled capture, use one locked camera at a set of repeatable
poses and cycle several independently controlled lights plus one broad diffuse
geometry light without moving the camera. A turntable or pose marks are
adequate for the first object gate; a synchronized camera/light rig is better.
Record light position, RGB power, exposure, ISO, aperture, white balance, and a
gray-card frame in every session. Reserve one light and several poses before
training.

A practical phone workflow eventually cannot require pixel-aligned repeated
orbits. After the aligned gate passes, the trainer should accept a set of
`(image, camera, light)` observations with different pose subsets per light.
That is the correct next capture abstraction for a moving phone with lights
cycled or strobed during one trajectory.

## Implementation order

1. Expand the verified sparse tracks into continuous oriented point-cloud
   patches. Infer tangent support from shared-view neighborhoods, share or
   regularize material parameters within each patch, and select additions on
   withheld training cameras using foreground PSNR and mask precision/recall.
2. Once that surface gate passes, revisit visibility and indirect transport
   together. The current scalar control proves they are not useful while the
   support is fragmented; enabling only one half can also trade an over-bright
   error for an under-bright one rather than recover transport.
3. Require every candidate to beat black and capture-light-copy baselines on
   foreground mean/worst PSNR, mask recall/precision, covered-pixel quality,
   and visible shadow/highlight motion.
4. Generalize aligned directories to sparse `(camera, light)` observations so
   every light need not be photographed at every pose.
5. Fit diffuse albedo and normals first. Enable roughness and reflectance only
   after multiple lights improve held-light quality consistently.
6. Confirm the selected method on DiLiGenT-MV or ReNé, then scale to
   OLATverse/OpenSubstance only if the smaller gates justify it.
