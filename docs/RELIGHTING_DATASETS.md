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
matrices to pose-only COLMAP binaries, copies PNG masks, writes the official
five-camera split, and creates distant-light `.f32` environments:

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
    --sparse etc/data/openillumination/prepared/obj_16_friends_cup/sparse/0 \
    --images etc/data/openillumination/OLAT/obj_16_friends_cup/Lights/062/raw_undistorted \
    --masks etc/data/openillumination/prepared/obj_16_friends_cup/masks \
    --test-list etc/data/openillumination/prepared/obj_16_friends_cup/test.txt \
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

The exact ignored runs are under
`target/audit-runs/openillumination/{four-training-lights,four-training-lights-015,five-training-lights-015,six-axis-training-lights,three-training-lights-held-086,four-training-lights-139-held-086,five-training-lights-held-086,five-training-lights-fixed-budget-held-000,five-training-lights-fixed-budget-held-086,five-training-two-held}/`.
All complete runs used the guarded 12 GiB scope; the final two-held run peaked
at 1.1 GiB including its release rebuild, with zero swap, memory pressure, OOM,
or GPU fault.

The initial pass must report mean and worst held-camera PSNR, coverage, image
comparisons, point count, training time, peak cgroup memory, and the exact
object/light selection. A result is useful only if it beats a capture-light
colour or average-image baseline on the held light and the images visibly move
shadows and highlights in the right direction.

## Capture direction

For our own controlled capture, use one locked camera at a set of repeatable
poses and cycle several independently controlled lights without moving the
camera. A turntable or pose marks are adequate for the first object gate; a
synchronized camera/light rig is better. Record light position, RGB power,
exposure, ISO, aperture, white balance, and a gray-card frame in every session.
Reserve one light and several poses before training.

A practical phone workflow eventually cannot require pixel-aligned repeated
orbits. After the aligned gate passes, the trainer should accept a set of
`(image, camera, light)` observations with different pose subsets per light.
That is the correct next capture abstraction for a moving phone with lights
cycled or strobed during one trajectory.

## Implementation order

1. Replace global support widening with per-ray surface responsibility. A
   candidate must identify which depth layer owns a foreground observation,
   using multi-view mask and foam evidence without treating every silhouette
   pixel as a request to widen every overlapping point. Select it on withheld
   training cameras using foreground PSNR and mask precision/recall.
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
