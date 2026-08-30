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

## Hugging Face availability

As of 2026-08-29, the only official Hugging Face copy among the requested
datasets is
[`OpenIllumination/OpenIllumination`](https://huggingface.co/datasets/OpenIllumination/OpenIllumination).
No official Hub repository was found for MIT Multi-Illumination, FAU
Multi-Illuminant, Flash/Ambient, MILL, LUCES-MV, ReNé, DiLiGenT-MV, Objects
With Lighting, or OpenSubstance; use their official project releases instead.
The additional official
[`cyberagent/mvscps`](https://huggingface.co/datasets/cyberagent/mvscps)
release provides six multi-view OLAT scenes, masks, and camera projection
matrices. It is a useful later unknown-light capture gate, but its 65.1 GB
release deliberately has no light calibration and its photogrammetry mesh is
qualitative only.
The Hub-hosted [`whcfang/M2AD`](https://huggingface.co/datasets/whcfang/M2AD)
has angle and illumination labels but no published camera or light calibration,
so it cannot support the camera/light reconstruction gate without additional
metadata.

## Ranked datasets

| Dataset | Camera/light coverage | Best use here | Limitation |
| --- | --- | --- | --- |
| [LUCES-MV](https://arxiv.org/abs/2412.16737) | Public calibrated subset: 10 objects, 12 views by 15 near-field LEDs | **Current controlled-light gate.** It has linear 16-bit RGB, masks, camera/LED calibration, depth, normals, and ground-truth shape. One object is about 1.9 GB. | Non-commercial research licence; the calibrated public subset is smaller than the paper's complete capture. |
| [DiLiGenT-MV](https://sites.google.com/site/photometricstereodata/mv) | 5 objects, 20 views by 96 calibrated lights | **Second active controlled-light gate.** The pure-Rust Bear route below reconstructs and scores a fixed 16/4-camera, 24/8-light split. | Only five object-centric scenes; the lights are distant and the objects are mostly diffuse. |
| [MVSCPS capture](https://huggingface.co/datasets/cyberagent/mvscps) | 6 scenes, generally 24 camera poses under each of 6 moving-rig OLATs | Later unknown-light reconstruction and capture-practice gate with RAW/JPEG, masks, and camera projection matrices. | 65.1 GB, CC BY-NC; no measured lights and no quantitative ground-truth mesh. |
| [OpenIllumination](https://oppo-us-research.github.io/OpenIllumination/) | 64 objects, 70 views, 13 multi-LED patterns and 142 OLAT conditions | Existing broad-light real gate, with camera poses, masks, and official train/test views. | Roughly 900 GB in full; its public positions do not provide the complete finite-light radiometric calibration needed here. |
| [Objects With Lighting](https://github.com/isl-org/objects-with-lighting) | 64 input cameras under one unknown natural environment, plus nine official camera/environment test pairs per object | **First independent natural-environment gate.** Compact, calibrated, masked, and directly compatible with the distant-HDR renderer. | One input environment leaves material and illumination strongly ambiguous; it is an evaluation gate, not repeated-light training data. |
| [ReNé](https://eyecan-ai.github.io/rene/) | 20 objects, 50 views by 40 OLAT conditions | Possible later cross-check for calibrated camera/light poses. | The public object capture has no masks and fills a textured enclosure; the referenced empty-scene capture is not in the public archive, so honest background subtraction is currently blocked. |
| [Stanford-ORB](https://stanfordorb.github.io/) | 14 objects captured in multiple real environments with HDR environment maps and registered poses | Best match for the current distant-environment renderer and an important in-the-wild relighting check. | It does not provide a dense aligned camera/light grid for each background environment. |
| [DTU robot data](https://roboimagedata.compute.dtu.dk/?page_id=24) | 60 scenes, 119 cameras by 19 LEDs | Larger, more scene-like multi-view/multi-light stress test. | Approximately 730 GB in full and built around local LEDs. |
| [OpenSubstance](https://opensubstance.github.io/) | 187 objects, 270 views and 1,637 lighting conditions | Later high-resolution material and specular benchmark. | Multi-terabyte scale and access by request. |

OLATverse is not part of the active plan. Its registration-gated release may
never be available to this project, so no milestone or quality claim depends
on it.

## Current alternative route: LUCES-MV

LUCES-MV replaces OLATverse and the blocked public ReNé capture for the next
controlled-light work. The official calibrated release provides twelve poses
per object and fifteen individually calibrated LEDs per pose. Each LED has a
camera-local position, brightest outgoing direction, RGB scale, and
cosine-power exponent. This is exactly the missing observation model: a
finite emitter whose direction and inverse-square attenuation vary over the
point cloud and whose world pose changes with the camera rig.

The first implementation slice is complete:

- `blade_volume::relight::PointLight` defines the finite light and its rigid
  camera-to-world transform;
- analytical normal/material refinement accepts either a distant irradiance or
  one calibrated point light per view;
- the Gaussian support optimizer evaluates the same light model inside one
  mutated Meganeura graph. It adds no Meganeura operation, shader group, or
  shader-entry variant;
- exact CPU/GPU and end-to-end synthetic tests cover distance, angular falloff,
  position gradients, and view-specific moving lights.

The checked-in fetch is intentionally one-object and licence gated. It pins the
official Owl archive and both camera calibration files by SHA-256, and writes
only below ignored `target/` storage:

```bash
# Read the licence linked by the script before accepting it.
etc/cgroup_run.sh --mem 2G -- \
    etc/fetch_luces_mv_owl.sh --accept-license
```

The downloaded Owl object contains 12 × 15 RGB16 images plus masks, camera
extrinsics, ground-truth depth and normals. A pure-Rust calibration sanity
check fits a diffuse normal independently at every fourth masked pixel from
all fifteen images. Against ground truth it obtains:

| View | Samples | Mean angular error | Median | P90 |
| --- | ---: | ---: | ---: | ---: |
| 000 | 9,466 | 13.69° | 10.01° | 28.58° |
| 018 | 8,964 | 16.77° | 13.04° | 36.39° |
| 060 | 9,583 | 19.69° | 17.41° | 34.89° |

This is not the reconstruction result. It is an intentionally simple linear
diffuse oracle which confirms the published units, LED orientation, RGB
normalization, and 16-bit image convention before they supervise geometry.
Generated normal images remain ignored under
`target/audit-runs/luces-mv/`; dataset imagery cannot be copied into the
repository under its licence.

The Rust adapter groups the official directory as fifteen aligned
`Capture`s, parses the stored NumPy camera extrinsics without adding a ZIP/NPY
dependency, transforms each camera-local LED into the existing world frame,
and downsamples RGB/masks without display decoding. A real all-image load
produces 15 × 12 views at 80×60, with a 0.329 normalized peak and plausible
camera/light baselines. The warm loader peaks at 1,138,884,608 bytes in a 2 GiB
cgroup with zero swap, limit, OOM, or GPU event. The preceding cold build was
not used as evidence because it touched its 3 GiB limit while compiling a
separate audit target.

The fixed gate is now executable. Cameras 000/024/048 and LEDs 03/09/15 are
excluded; the other nine cameras and twelve lights fit the model. The importer
writes only normalized photographs, masks, pose-only COLMAP binaries, and the
four split lists. Ground-truth shape, depth, and normals are neither read nor
converted by this path:

```bash
cargo run --release -p blade-volume-train --bin import_luces_mv -- \
    --input target/audit-runs/luces-mv/data/Owl \
    --camera-one target/audit-runs/luces-mv/source/cam1_params.txt \
    --camera-two target/audit-runs/luces-mv/source/cam2_params.txt \
    --output target/audit-runs/luces-mv/prepared-320 --width 320

cargo run --release -p blade-volume-train --bin train_colmap -- \
    --sparse target/audit-runs/luces-mv/prepared-320/sparse/0 \
    --images target/audit-runs/luces-mv/prepared-320/light-08/images \
    --masks target/audit-runs/luces-mv/prepared-320/masks \
    --test-list target/audit-runs/luces-mv/prepared-320/test-views.txt \
    --output target/audit-runs/luces-mv/far-foreground-16k-rust-v1/foam.ply \
    --initialization camera-lattice --max-points 16384 \
    --width 128 --height 96 --views 0 --far-plane 1000 --max-steps 384 \
    --pixel-batch 1024 --views-per-batch 9 --steps-per-view 200 \
    --sh-degree 2 --foreground-fraction 0.5
```

The explicit far plane is required because LUCES uses millimetre-like world
units and its cameras are roughly 400 units from the object. The old fixed
100-unit plane ended every ray in the unbounded black camera cell; its 21.56 dB
score was only the 95% black background. With the corrected plane, uniform
sampling reaches 32.43 dB on the three held cameras. Drawing half of each batch
from mask foreground reaches 33.21 dB, with 35.74 dB on construction cameras,
on the matched 4,096-site ablation. The final 16,384-site run uses the stock
Rust Delaunay implementation—no Qhull feature or new dependency—and reaches
33.19 dB on the same held cameras.

Extract the point surface, then fit the calibrated lights:

```bash
cargo run --release -p blade-volume-train --bin reconstruct -- \
    --sparse target/audit-runs/luces-mv/prepared-320/sparse/0 \
    --images target/audit-runs/luces-mv/prepared-320/light-08/images \
    --masks target/audit-runs/luces-mv/prepared-320/masks \
    --test-list target/audit-runs/luces-mv/prepared-320/test-views.txt \
    --width 128 --stride 1 --voxel-factor 2 \
    --foam target/audit-runs/luces-mv/far-foreground-16k-rust-v1/foam.ply \
    --no-shadows \
    --output target/audit-runs/luces-mv/far-foreground-16k-rust-v1/reconstruct-vf2/scene.rply

cargo run --release -p blade-volume-train --bin fit_luces_mv -- \
    --input target/audit-runs/luces-mv/data/Owl \
    --camera-one target/audit-runs/luces-mv/source/cam1_params.txt \
    --camera-two target/audit-runs/luces-mv/source/cam2_params.txt \
    --surface target/audit-runs/luces-mv/far-foreground-16k-rust-v1/reconstruct-vf2/scene.rply \
    --output target/audit-runs/luces-mv/far-foreground-16k-rust-v1/calibrated-production/scene.rply \
    --gaussian-output target/audit-runs/luces-mv/far-foreground-16k-rust-v1/calibrated-production/pbr.ply \
    --dump target/audit-runs/luces-mv/far-foreground-16k-rust-v1/calibrated-production/images \
    --width 128 --rounds 3
```

The two-pixel point merge produces 589 surfels, compared with 114 at the old
five-pixel default. Under the source light it reaches 29.46/28.86 dB
whole-frame and 20.54/20.00 dB foreground mean/worst on held cameras, with
98.4% recall and 91.0% precision. Calibrated normal/material fitting uses only
the twelve construction LEDs and nine construction cameras. The held-light
data is opened only after both point clouds are serialized.

The ordinary production tracer now evaluates the same finite point-light model
as the CPU solve. It mutates one uniform at runtime rather than adding a shader
group, shader entry, or point-light-specific pipeline. Direct point-light
shading is diffuse-only, matching the identified material model; environment
specular and sampled indirect transport are bypassed. Complete image-space
scores are:

| Backend | Lights / cameras | sRGB mean/worst | Foreground mean/worst | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| Surface | fitted / fitted | 35.63 / 32.70 dB | 26.44 / 22.54 dB | 98.2% | 93.8% |
| Surface | fitted / held | 35.50 / 32.16 dB | 26.17 / 22.68 dB | 98.0% | 93.3% |
| Surface | held / fitted | 35.19 / 33.07 dB | 25.95 / 23.17 dB | 98.2% | 93.8% |
| Surface | held / held | 34.96 / 32.51 dB | 25.51 / 23.28 dB | 98.0% | 93.3% |
| Gaussian | fitted / fitted | 34.77 / 32.51 dB | 25.38 / 22.45 dB | 94.5% | 88.8% |
| Gaussian | fitted / held | 34.39 / 32.20 dB | 24.62 / 22.21 dB | 94.9% | 88.5% |
| Gaussian | held / fitted | 34.41 / 32.90 dB | 24.81 / 23.20 dB | 94.5% | 88.8% |
| Gaussian | held / held | 34.01 / 32.66 dB | 24.13 / 22.66 dB | 94.9% | 88.5% |

The final surface-material pass solves all 589 diffuse colours together through
the same coverage-weighted particle groups as the runtime. It uses only the
construction lights/cameras and improves every surface row without changing
geometry. The Gaussian keeps its backend-specific material fit rather than
receiving parameters optimized for the surface compositor.

A complete-render point-light normal probe improved this Owl table, but did not
survive the fresh Cow control below and is not retained. Matched localized
radius and normal-axis center probes are also rejected: they lower construction
loss and improve whole-frame quality, but trade foreground quality or recall
for background fit. Complete image loss can refine a used split; it cannot
replace independent cross-view depth/support evidence.

The Gaussian continuation runs 1,200 updates, improves its construction loss
from 0.003958 to 0.002280, and retains all 589 particles. These numbers are
full production renders, not projected-center samples; the held/held row is
nine camera/light combinations excluded from every fitting stage. The result
clears the finite-light transport and split-integrity gate. It is still visibly
soft at the object's ridges, making spatial surface detail and correspondence
precision—not another light representation—the next controlled target.
DiLiGenT-MV remains the fallback/control; Stanford-ORB remains the later
distant-HDR cross-check.

### LUCES spatial-detail screen

The first adjacent surface screen is closed. It changes one source of spatial
capacity at a time while keeping the 9/3-camera and 12/3-light split fixed:

- Training the same 16,384 cells at 320 rather than 128 pixels raises static
  held-camera PSNR from `33.19` to `35.12` dB, but the 3,925-surfel calibrated
  surface falls to `23.81/22.34` dB held-light/held-camera foreground
  mean/worst. Pooling that field at the old physical support scale is also
  worse at `23.97/22.19` dB. Projected ground-truth depth is worse, not merely
  more finely sampled: median absolute discrepancy moves from `5.31` to
  `6.02` world units (`7.10` after matched-scale pooling).
- Extracting 1,028 rather than 589 surfels from the selected 128-pixel field
  exposes the radius tradeoff directly. A 2.5-cell footprint reaches
  `94.9%/92.6%` source-light recall/precision, but its final Gaussian reaches
  only `23.08/21.51` dB held-light/held-camera foreground with `92.1%` recall.
  Fixed, doubled, intermediate, and four-neighbour adaptive footprints all
  fail a complete scalar gate.
- Observation diagnostics explain why density alone stops transferring. Only
  `10.4%` of the selected 589-surface samples share an exact center pixel; the
  3,925-surface raises that to `82.0%`. Bilinear subpixel reads improve several
  means but lose the fitted-light/held-camera whole-frame tail by `0.09` dB and
  the held/held tail by `0.11` dB, so the prototype is removed rather than
  treating interpolation as correspondence.
- Averaging only the twelve construction LEDs raises static held-camera PSNR
  to `33.86` dB. The final scalar held/held row becomes
  `34.21/32.89` dB whole-frame and `24.76/22.59` dB foreground: better tails
  in some cells, but lower whole-frame mean and foreground worst. Its Gaussian
  similarly trades mean for tail and loses recall/precision, so the importer
  change is removed.
- The existing strict epipolar tracker finds only five accepted 128-pixel
  tracks from a four-light response. Captured-light RGB gives 72 tracks at
  320 pixels, but their projected-depth discrepancy is worse than the foam
  surface. No matching threshold is weakened and no tracks are merged.
- Inverting all twelve calibrated construction LEDs at each pixel against a
  coarse depth plane does not create a useful invariant descriptor. Diffuse
  albedo yields 19 strict tracks at 128 pixels, but their ground-truth depth
  discrepancy is `27.31` world units at the median versus `5.31` for the
  selected surface; recovered world normals yield only three tracks. The
  ignored prototype is removed rather than weakening mutual-match or
  reprojection thresholds.
- Sharing appearance across 64 materials slightly improves the scalar
  held/held foreground mean/tail to `24.76/22.76` dB, but loses whole-frame
  tail and recall; its Gaussian falls to `24.03/22.46` dB foreground. A
  32-material control is also mixed. An exact temporary image-space solve then
  optimized those 32 diffuse colours through complete finite-light surface
  and Gaussian renders. The scalar cloud reaches `34.57/32.19` dB whole-frame
  and `24.87/22.67` dB foreground, but precision drops to `93.2%`. Optimizing
  the Gaussian compositor raises foreground to `24.37/23.12` dB while lowering
  whole-frame mean/tail to `33.90/32.54` dB and precision to `88.4%`. This is a
  backend-specific appearance trade rather than recovered geometry, so the
  solver, renderer hook, and test are removed.

These are ignored diagnostics, not vendored benchmark artifacts. Heavy runs
stay in 4--8 GiB scopes; representative peaks are 0.16 GB for 320-pixel
training, 2.04 GB for calibrated scoring, and 1.64 GB for the averaged-light
Gaussian continuation, with zero swap, OOM, throttle, or GPU fault. The
production code is unchanged by every rejected arm.

The selected joint material solve now preserves continuous render
responsibility without moving points. The remaining problem is a
correspondence-aware surface proposal: several calibrated cameras must agree
on a point before adding it. It must be selected on construction
lights/cameras, then improve whole-frame and foreground mean/worst plus recall
and precision on both excluded axes. More resolution, a global radius sweep,
or a looser pair matcher is not a new experiment.

## Second controlled-light gate: DiLiGenT-MV

DiLiGenT-MV is now an independent distant-light control, not a contingency on
OLATverse. The checked-in fetch pins the official 6.85 GB archive and extracts
only Bear. The loader parses its compressed MATLAB camera calibration in Rust,
divides linear RGB16 by the published per-channel light intensity, and maps the
photometric directions into the existing point-light renderer as emitters at
effectively infinite distance. No shader, Meganeura operation, or model variant
is added. Ground-truth normals and the released mesh are never opened by the
training path.

```bash
etc/cgroup_run.sh --mem 2G -- \
    etc/fetch_diligent_mv_bear.sh --accept-license

cargo run --release -p blade-volume-train --bin import_diligent_mv -- \
    --input target/datasets/diligent-mv/data/DiLiGenT-MV/mvpmsData/bearPNG \
    --output target/audit-runs/diligent-mv/prepared-320 --width 320

cargo run --release -p blade-volume-train --bin train_colmap -- \
    --sparse target/audit-runs/diligent-mv/prepared-320/sparse/0 \
    --images target/audit-runs/diligent-mv/prepared-320/light-004/images \
    --masks target/audit-runs/diligent-mv/prepared-320/masks \
    --test-list target/audit-runs/diligent-mv/prepared-320/test-views.txt \
    --output target/audit-runs/diligent-mv/bear-16k/foam.ply \
    --initialization camera-lattice --max-points 16384 \
    --width 128 --height 107 --views 0 --far-plane 4000 --max-steps 384 \
    --pixel-batch 1024 --views-per-batch 16 --steps-per-view 200 \
    --sh-degree 2 --foreground-fraction 0.5

cargo run --release -p blade-volume-train --bin reconstruct -- \
    --sparse target/audit-runs/diligent-mv/prepared-320/sparse/0 \
    --images target/audit-runs/diligent-mv/prepared-320/light-004/images \
    --masks target/audit-runs/diligent-mv/prepared-320/masks \
    --test-list target/audit-runs/diligent-mv/prepared-320/test-views.txt \
    --foam target/audit-runs/diligent-mv/bear-16k/foam.ply \
    --width 128 --far-plane 4000 --stride 1 --voxel-factor 2 --no-shadows \
    --output target/audit-runs/diligent-mv/bear-16k/reconstruct-vf2/scene.rply

cargo run --release -p blade-volume-train --bin fit_diligent_mv -- \
    --input target/datasets/diligent-mv/data/DiLiGenT-MV/mvpmsData/bearPNG \
    --surface target/audit-runs/diligent-mv/bear-16k/reconstruct-vf2/scene.rply \
    --output target/audit-runs/diligent-mv/bear-16k/calibrated-production/scene.rply \
    --gaussian-output target/audit-runs/diligent-mv/bear-16k/calibrated-production/scene.ply \
    --dump target/audit-runs/diligent-mv/bear-16k/calibrated-production/images \
    --width 128 --rounds 3 --normal-candidates 1024
```

The split is fixed before fitting: cameras 1/6/11/16 and lights
1/13/25/37/49/61/73/85 are excluded. Twenty-four evenly spaced remaining
lights keep the real gate practical without weakening its angular coverage.
The source-light field reaches `29.63/29.30` dB whole-frame and
`24.28/23.66` dB foreground mean/worst on the four excluded cameras. The
2-pixel surface merge produces 2,267 surfels. After fitting, complete
production renders are:

| Backend | Lights / cameras | sRGB mean/worst | Foreground mean/worst | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| Surface | fitted / fitted | 29.80 / 25.47 dB | 21.99 / 16.44 dB | 96.2% | 93.9% |
| Surface | fitted / held | 29.14 / 25.73 dB | 21.11 / 16.79 dB | 95.0% | 94.4% |
| Surface | held / fitted | 30.16 / 26.52 dB | 22.08 / 17.67 dB | 96.2% | 93.9% |
| Surface | held / held | 29.35 / 26.89 dB | 21.16 / 18.12 dB | 95.0% | 94.4% |
| Gaussian | fitted / fitted | 30.14 / 25.44 dB | 21.67 / 16.32 dB | 93.1% | 97.0% |
| Gaussian | fitted / held | 29.40 / 25.49 dB | 20.84 / 16.41 dB | 92.0% | 97.1% |
| Gaussian | held / fitted | 30.12 / 26.49 dB | 21.47 / 17.45 dB | 93.1% | 97.0% |
| Gaussian | held / held | 29.32 / 26.67 dB | 20.65 / 17.71 dB | 92.0% | 97.1% |

The coupled sparse material solve improves every surface mean/tail across both
axes while preserving geometry. The Gaussian runs 2,400 geometry updates and
retains 2,233 of 2,267 particles.
It improves held/held whole-frame mean and precision, but loses foreground
quality and recall, so it does not replace the scalar result. The complete run
peaks at 1.09 GB host memory with no swap, cgroup event, or GPU fault.

The dataset also localizes the next error. An ignored per-pixel Lambertian
control, using the same 24 construction and eight excluded lights, reaches
`33.57` dB foreground on excluded lights. The production surface is roughly
12.5 dB behind even before asking it to move to another camera. A one-pixel
merge raises the surface to 5,511 points but drops held/held foreground to
`19.27/17.05` dB and recall to 84.8%; per-camera normal combination is also
slightly worse. Using all 88 available construction lights improves the oracle
by 0.37 dB but gives the surface only `+0.10` dB foreground mean while losing
its worst view and recall. All three controls are removed or remain ignored.
Continuous material ownership is now selected. The active target is therefore
cross-view geometry correspondence and surface coverage, not more lights,
points, or shader capacity.

### Fresh-object guard: DiLiGenT-MV Cow

Cow was extracted from the already pinned archive only after the Owl/Bear
normal proposal was fixed. It uses the identical 16/4-camera, 24/8-light,
16,384-cell route and never reads released normals, depth, or mesh geometry.
The 1,428-surfel production surface reaches `30.86/29.16` dB whole-frame and
`20.34/18.25` dB foreground mean/worst on the 32 held-light/held-camera pairs,
with 97.6% recall and 88.8% precision. Its Gaussian reaches `30.81/28.90` and
`19.74/17.55` dB, with 84.7% recall and 94.5% precision.

The complete-render point-light normal proposal accepts all four construction
rounds and improves most means, but loses four independently reported tails:
fitted-light/fitted-camera whole-frame worst `28.33→28.31` dB and foreground
worst `17.69→17.64` dB, fitted-light/held-camera whole-frame worst
`28.83→28.82` dB, and held-light/held-camera whole-frame worst
`29.16→29.15` dB. It is therefore removed together with its point-light batch
plumbing and test; no dormant shader, renderer branch, or fitting option
remains. The retained outputs are under
`target/audit-runs/diligent-mv/cow/16k/calibrated-no-rendered-normal/`. The
complete fit peaks at 239 MiB; all extraction, training, reconstruction, and
fit scopes report zero OOM, throttle, or GPU fault.

Skipping calibrated multi-light Gaussian continuation is also mixed, not a
valid rollback. It restores held/held recall `84.7%→89.9%` and foreground
worst `17.55→17.78` dB, but loses whole-frame mean/worst
`30.81/28.90→30.46/28.64` dB and lowers most other complete-matrix cells. The
continuation log is therefore explicit that its `0.004036→0.004405` scalar is
an audit loss for one deterministic light/view batch, not the complete fitting
objective. No single-sample rollback is added.

Cow also supplies the first missing-surface proposal to pass the complete
camera/light gate. Twenty-four calibrated construction lights are reduced to a
robust diffuse-albedo correspondence image without reading the released mesh,
depth, or normals. For each of eight matching cameras, a diagnostic pass uses
that camera's missing pixels and the measured foreground in the other cameras,
retains tracks containing the selected hole, and globally deduplicates the
eight result sets. An internal 8/4/4 match/selection/validation camera split
produces 137 unique tracks and seven shared-view patches. The existing
two-pixel support cap leaves five
foreground-safe patches; complete production renders select patches 1/4/5,
32 Gaussian surfels with one shared diffuse material per patch.

The subset is fixed before the four official held cameras or eight held lights
are scored:

| Lights / cameras | Gaussian control whole | Candidate whole | Gaussian control foreground | Candidate foreground | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fitted / fitted | 32.427 / 28.839 | **32.444 / 28.855** | 21.471 / 17.951 | **21.490 / 17.964** | 84.18% → **84.92%** | 96.28% → **96.30%** |
| fitted / held | 31.410 / 29.047 | **31.426 / 29.075** | 20.377 / 17.847 | **20.396 / 17.894** | 84.68% → **85.50%** | 94.50% → **94.52%** |
| held / fitted | 32.136 / 29.089 | **32.150 / 29.111** | 21.140 / 18.115 | **21.156 / 18.131** | 84.18% → **84.92%** | 96.28% → **96.30%** |
| held / held | 30.812 / 28.899 | **30.827 / 28.912** | 19.737 / 17.548 | **19.754 / 17.592** | 84.68% → **85.50%** | 94.50% → **94.52%** |

The binary candidate is at
`target/audit-runs/diligent-mv/cow/16k/missing-surface-candidate/scene-gaussian.ply`.
It has 1,460 rather than 1,428 particles. The complete run peaks at 266.5 MiB
with zero swap, OOM, throttle, or GPU fault. Only the calibrated distant-light
albedo estimator moves into production. The multi-pass correspondence search,
subset search, and merge remain ignored diagnostics.

A direct one-pass source/qualifier API does not reproduce that result. It finds
67 Cow tracks and patches of 10/5/5 tracks. An internally selected 10-surfel
subset passes selection and validation, but loses final foreground tails by
0.004–0.009 dB and one whole-frame tail by 0.0004 dB. The same one-pass form on
Bear finds 26 qualifying tracks, 24 in the visual hull, and no five-track
patch.

Repeating the exact eight-pass Cow policy on Bear proves that alternative loss
is not the whole failure. It finds 71 unique tracks, retains 67 in the visual
hull, and forms one six-track patch. None of the 67 tracks has missing support
in at least three of four selection cameras. The patch covers 0.4% of selection
holes and 0% of validation holes; after merging its six surfels, selection
recall changes by +0.009 points but validation recall changes by -0.010. The
official held cameras and lights remain closed. The clean run peaks at 190 MiB
with zero swap, OOM, throttle, or GPU fault. The proposed matching API and
reconstruction integration remain removed; a production path needs points
that predict missing support beyond the cameras used to triangulate them.

Pot2 supplies a third fresh control from the same pinned archive. Only its
image/calibration directory is extracted; released mesh, normal, and depth
products remain unread. The unchanged 16,384-cell route produces 1,870 surface
points and 1,850 fitted Gaussians. Its held-light/held-camera Gaussian baseline
is `33.242/30.711` dB whole-frame and `23.599/20.523` dB foreground, with 90.50%
recall and 95.18% precision.

The exact eight-pass missing-surface search finds 70 unique tracks, 67 inside
the selection visual hull, and patches of six and nine surfels. The first has
no selection-hole coverage and is rejected. The second raises selection and
validation coverage. A selection-only material screen fixes `1.5×` calibrated
albedo because it gives the best means among scales that also improve both
selection tails. On validation it improves whole mean/worst and foreground
mean, but foreground worst changes `20.0594→20.0543` dB, so the post-hoc cloud
is rejected before its official matrix.

A one-shot seeded control inserts the same nine surfels into the unfitted
surface, then reruns the ordinary calibrated normal/material solve and Gaussian
continuation. The result persists 1,879 Gaussians, 1,859 above the reported
opacity-retention threshold, and improves all four final means, recall, and
precision. Its exact Gaussian matrix is:

| Lights / cameras | Control whole | Seeded whole | Control foreground | Seeded foreground | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fitted / fitted | 34.423 / 30.142 | **34.487 / 30.175** | 24.957 / **20.059** | **25.035** / 20.054 | 90.88% → **91.10%** | 95.30% → **95.33%** |
| fitted / held | 33.429 / 29.947 | **33.491 / 29.956** | 23.909 / 19.715 | **23.989 / 19.724** | 90.50% → **90.75%** | 95.18% → **95.21%** |
| held / fitted | 34.225 / 31.256 | **34.274 / 31.276** | 24.643 / **21.156** | **24.701** / 21.153 | 90.88% → **91.10%** | 95.30% → **95.33%** |
| held / held | 33.242 / 30.711 | **33.295 / 30.714** | 23.599 / 20.523 | **23.664 / 20.526** | 90.50% → **90.75%** | 95.18% → **95.21%** |

The mean improvements do not excuse the 0.0054/0.0028 dB fitted-camera
foreground-tail losses, so this candidate is also rejected. The fresh route
peaks at 1.22 GiB during import; candidate fitting peaks at 309 MiB. Every
post-extraction scope reports zero swap, OOM, throttle, or GPU fault. Archive
extraction hit its initial 2 GiB `memory.max` through page cache without an OOM,
so all subsequent scopes used a raised cap. The next experiment keeps the same
points and appearance policy and changes only continuation acceptance so an
average loss cannot trade away a camera/light tail.

A bounded construction-only checkpoint audit then separates geometry from
opacity. The 87.5% all-parameter checkpoint reaches
`34.446/30.171;25.020/20.071` dB, 90.96% recall, and 95.10% precision: every
quality/recall number beats the unseeded control, but precision does not. Full
trained opacity with checkpointed position reaches 95.31% precision but lowers
foreground worst to 20.053 dB. Normal interpolation is inert after final PBR
attachment. No checkpoint is retained, and all instrumentation is removed.
The next narrow diagnostic fits only the proposed patch's material in the
final Gaussian compositor; it does not reopen geometry, opacity, or matching.

That material-only screen is also negative. Geometry, opacity, normals, and
all 1,870 base-particle materials stay frozen; only one RGB gain shared by the
nine proposed particles is coordinate-searched on selection cameras. The
selected material raises validation whole/foreground means to
`33.8036/24.2991` dB, but foreground worst falls from `20.0594` to `20.0516`
dB. It is rejected before any production parameter-partitioning API.

A separate one-shot geometry route uses the landed 24-light photometric-albedo
estimator before the ordinary trainer. Pure albedo produces 1,940 particles;
the exact fitted/held matrix is:

| Lights / cameras | Control whole | Albedo whole | Control foreground | Albedo foreground | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fitted / fitted | 34.423 / 30.142 | 34.368 / **30.171** | **24.957 / 20.059** | 24.932 / 19.962 | 90.88% → **91.53%** | 95.30% → **95.37%** |
| fitted / held | **33.429 / 29.947** | 33.235 / 29.922 | **23.909** / 19.715 | 23.759 / **19.771** | **90.50%** → 90.33% | **95.18%** → 94.98% |
| held / fitted | **34.225 / 31.256** | 34.137 / 31.128 | **24.643 / 21.156** | 24.566 / 21.079 | 90.88% → **91.53%** | 95.30% → **95.37%** |
| held / held | **33.242 / 30.711** | 32.989 / 30.570 | **23.599** / 20.523 | 23.362 / **20.668** | **90.50%** → 90.33% | **95.18%** → 94.98% |

Fixed 25% and 50% linear blends with the original shadowed images test whether
the two cues are complementary. The 50% blend dominates every metric on both
internal construction-camera splits, but the official held-light/held-camera
whole/foreground means are only `33.1191/23.5073` dB, despite a best worst
foreground of `20.8497` dB. The 25% blend reaches
`33.2411/23.6685;20.5791` dB there, but loses fitted-camera tails and precision
(`94.82%` on held cameras). A selection-fixed 12/16 center-mask support prune
fails internal validation foreground mean. Matching the blend's merge budget
to the base (1,888 versus 1,870 particles) yields only
`33.0210/23.4398;20.6425` dB held/held. All arms are rejected. Artifacts remain
ignored under `target/audit-runs/diligent-mv/pot2-albedo-*`; all scoped runs
finish without swap, OOM, pressure, throttle, or GPU fault, with a 491 MiB peak
for calibrated fitting.

Pot2 has now served its one adaptive proxy-image audit. Further work must not
tune another precombined image against this official split. The next fresh
gate should optimize a shared cloud against the calibrated light stack itself,
keeping light-conditioned appearance separate while geometry remains common.

A silhouette-only lower bound and an alternating multi-light prototype do not
change that conclusion. Using masks as both RGB and opacity supervision
produces a 1,942-particle Pot2 surface but only `33.09/23.53` dB held/held
Gaussian whole/foreground mean and 94.7% precision. An alternating four-light
foam fit stores separate SH nuisance fields while sharing positions/density.
With density trainable it collapses to 1,829 particles and 88.9% held-camera
recall. With density frozen it retains 1,850 fitted particles and its exact
matrix improves every mean; held/held moves to
`33.3631/23.6920;20.5791` dB with 90.70% recall and 95.58% precision. It still
loses four fitted-camera tails by `0.010–0.063` dB. The unchanged replay on Cow
falls to `30.64/19.63` dB held/held whole/foreground mean and 83.9% recall.
This rejects sequential light windows and the temporary density-rate control.
The next implementation needs one optimizer session in which multiple lights
contribute to the same shared-geometry update.

A 64-view control places the four aligned light stacks in one session with one
Adam state but retains one global SH appearance. Trainable density collapses
Pot2 to 1,780 retained particles and about 89.2% held-camera recall. Frozen
density retains 1,861 and improves held-camera means, but still loses
fitted-camera tails. On Cow, the same frozen-density control reaches only about
`30.77/19.72` dB held/held whole/foreground mean, 83.8% recall, and 94.9%
precision. It is rejected and the temporary density control is removed. This
tests joint batching, not per-light nuisance appearance.

The final control supplies that missing degree of freedom in the same session:
four packed SH tables are selected by a per-ray one-hot basis while geometry,
density, and Adam state stay shared. Trainable density retains 1,826 fitted
particles, but held-camera recall falls to 89.77% and several worst-view cells
regress by as much as 0.1765 dB. Frozen density retains 1,861 particles and
improves every mean, recall, and precision cell. It still lowers
fitted-light/fitted-camera whole worst `30.1415→30.0712` dB and
held-light/fitted-camera whole worst `31.2564→31.1970` dB; the latter
foreground worst moves `21.1559→21.1513` dB. Held-light/held-camera
foreground improves `23.5994/20.5232→23.6535/20.6136` dB, so the signal is
real but does not satisfy the complete gate. Both arms are rejected and all
grouping/rate plumbing is removed. Scoped training, fitting, and scoring peak
at about 1.1 GiB with zero swap, memory event, or GPU fault. The next route is
a calibrated-light image-formation objective with shared material response,
not another unconstrained appearance table.

The existing physical Gaussian objective is deterministic on the persisted
Pot2 surface: a clean replay matches the selected baseline to six decimals.
Halving its position rate lowers every complete-matrix metric. A more useful
control alternates one extra measured-light geometry round after refreshing
the shared normal/material fit. The full second round over-prunes support, but
5%, 10%, and 15% interpolated checkpoints improve every construction selection
and validation mean, tail, recall, and precision; 20% first loses selection
recall. The 15% checkpoint is fixed before held data is opened. It then
improves every whole-frame mean/tail, every recall/precision cell, every
foreground tail, and seven foreground means. Held-light/held-camera foreground
mean alone moves `23.5994→23.5953` dB while its tail improves
`20.5232→20.5408` dB. The checkpoint is rejected, and the extra round and
interpolation are removed. This narrows the next physical route to bounded
point-patch updates selected on construction cameras, not another global
continuation.

An eight-region localization control turns that near miss into a complete Pot2
pass. Median planes define fixed spatial octants; each proposal applies 15% of
the second-round position, opacity, and normal delta while preserving the
established material table. Selection admits octants 1/2/4, 718 particles in
their union, and untouched construction validation improves every metric. The
frozen union then improves all final fitted/held aggregates. Held/held
whole-frame mean/worst moves `33.241601/30.711132→33.243049/30.720442` dB,
foreground `23.599410/20.523180→23.599503/20.529617` dB, recall
`90.502525→90.534994%`, and precision `95.184561→95.216615%`.

This remains diagnostic rather than production code. Under the identical
policy Cow admits no octant and is an exact six-decimal no-op. LUCES-MV Owl
admits octants 2/3 independently but rejects their union on construction
validation, also restoring the exact selected result. A generic guarded
implementation adds more than 300 lines and about 7.5 seconds (47% of the
clean Pot2 fitter runtime) for a held/held foreground-mean gain of only
`0.00009` dB. It is removed along with the extra second round, scoring helper,
and test. The next useful unit is a connected point patch generated from
calibrated-light residual/displacement, not an exhaustive octant partition.

That connected-component control is now measured. It keeps points above the
median support-normalized second-round displacement, joins nearby updates only
when their directions agree within 60 degrees, and discards components below
five points. Pot2 produces 31 components; selection admits four totaling 52
points, and the fixed union improves every final aggregate. Held/held
foreground mean/worst reaches `23.600485/20.523332` dB with 90.512968% recall
and 95.193802% precision, all above the base. Cow produces 21 components and
admits none. Owl admits two of 13 components (21 points), but their union
lowers construction-validation whole mean `34.443301→34.441890` dB and
precision `89.65755→89.65186%`. The component route remains ignored: it is a
cleaner Pot2 explanation, not a two-object reconstruction improvement, and 31
complete-render probes take about 13.4 seconds. The next control needs
per-point physical loss/gradient attribution from the existing optimizer, not
another external component sweep.

That one-pass attribution control is also complete. Meganeura first accumulates
each position row's temporal gradient norm inside the existing Adam dispatch.
Ranking coherent components by that signal picks a 17-point Pot2 patch, but it
loses construction selection and validation foreground tails and recall. A
strict two-sigma construction-mask footprint removes 27 of 31 components; the
top surviving 15-point patch still lowers validation worst-case PSNR
`30.141529→30.140759` dB.

A more direct zero-forward probe then assigns each Gaussian its exact detached
L1 residual times its compositing weight. It does not change the physical
forward loss and adds no per-step readback. This correctly ranks component 13,
one of the four Pot2 patches admitted by the earlier expensive selection
screen. Its 14-point proposal improves every selection metric and five of six
validation metrics, but validation worst-case PSNR still moves
`30.141529→30.141397` dB. On Cow, means, recall, and precision improve while
selection/validation worst-case PSNR moves `30.113402→30.112116` and
`28.942884→28.942081` dB; foreground selection tail also declines. Owl has no
component whose entire two-sigma footprint reaches the established 97.5%
construction-mask support threshold. Peak host RSS stays below 1.1 GiB with
zero swap. Both attribution branches and their public diagnostic API are
removed. Average residual magnitude is useful localization evidence, but the
next proposal must require consistent evidence across independent camera
groups before it can move a point.

Buddha is a fourth untouched DiLiGenT-MV control, extracted without its
released mesh, normals, or depth. The ordinary 16k route produces 2,143 point
surfels and retains 2,118 fitted Gaussians. Before any candidate held split is
opened, the exact calibrated-albedo matcher finds 94 unique missing-region
tracks and keeps 89 in the selection visual hull. Its default four-pixel
shared-view graph has one seven-point component; that component reaches only
3.7% missing-region precision and falls just below the 98% foreground-safety
gate. Matching world-space photometric normals instead leaves 28 tracks and no
component. Six- and eight-pixel neighborhoods produce 22 and 62
foreground-safe surfels respectively, but every bounded patch/appearance
candidate trades a selection mean against a tail. No candidate reaches
validation or official cameras/lights. Scoped runs peak near 1.1 GiB with no
swap, memory event, or GPU fault. The Buddha control closes descriptor and
patch-radius tuning rather than weakening the fixed safety gate.

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

### Independent soft-hull validation

`obj_29_fabric_toy` is a third geometry/material class and uses the same
leakage-free protocol: 38 construction poses enter the pose-only PatchMatch
graph, while ten official test poses remain absent from undistortion, stereo,
fusion, fitting, and selection. The native fusion cache contains 213,642
groups from 1,511,977 observations (3.6 distinct views per group). Applying
the selected training-mask hull rejects 5,772 groups before the fixed
2,500-point reduction and raises the number of photograph-observed surfels
from 2,289 to 2,378.

| Fabric-toy surface | Scalar known-light whole / foreground | Scalar recall / precision | Gaussian known-light whole / foreground | Gaussian recall / precision |
| --- | ---: | ---: | ---: | ---: |
| Unfiltered | `27.27/25.26; 18.22/16.00` dB | `97.0/91.1%` | `26.85/26.02; 18.04/16.71` dB | `95.8/90.0%` |
| Training-mask hull | `27.58/25.70; 18.27/16.14` dB | `96.8/93.2%` | `26.73–26.81/25.73–25.82; 18.10–18.14/16.77` dB | `96.0–96.1/88.9–89.2%` |

Two filtered repeats agree within 0.01 dB for the scalar checkpoint. Every
scalar whole-frame and foreground mean/tail improves, precision rises by 2.1
points, and recall falls by 0.2 points. That independently passes the filter's
intended surface-cleanup gate. It does not pass a Gaussian gate: foreground
tail and recall improve, but whole-frame quality and precision regress.

The excluded-light result is also deliberately mixed. Filtered scalar pattern
006 improves `23.77/22.81;14.01/12.72→23.99/22.97;14.12/12.75` dB. Pattern
005 changes `24.55/22.82;14.90/13.17→24.62/22.82;14.81/13.11` dB: the
whole-frame mean rises while foreground quality falls. Both remain below
their strongest trivial foreground baselines. The hull therefore stays as an
upstream scalar-cloud cleanup; representation transfer and light recovery
remain separate open problems.

The ignored inputs, cache, runs, and telemetry are under
`target/audit-runs/openillumination/obj29-{prepared,dense,native-cache}*` and
the matching filtered/unfiltered output directories. PatchMatch ran in a
10 GiB container scope; reconstruction peaked at 1.11 GiB with no swap, OOM,
or GPU fault. The temporary skip-filter control was removed after the exact
comparison.

### Final Gaussian diffuse transfer

A raw final-compositor attribution on the fabric toy assigns 70.9% of held
known-light RGB error to foreground pixels with at least half Gaussian
coverage. Missing foreground accounts for only 1.5%; false support accounts
for 18.7%. Covered foreground is substantially too bright, so adding more
points is not the first correction for this residual.

A large table created in individual-material mode now receives one bounded final-Gaussian
proposal automatically. Current and zero diffuse renders define a secant
surrogate under identical transport samples. A scalar is fitted only on covered
construction foreground, half of its displacement from identity is applied,
and an exact complete construction render accepts or rejects it. Small shared
palettes retain their existing explicit joint solver. Scalar geometry and
large-table material assignments do not move.

| Gaussian transfer | Known-light whole / foreground | Excluded 005 whole / foreground | Excluded 006 whole / foreground | Recall / precision |
| --- | ---: | ---: | ---: | ---: |
| Fabric toy, before | `26.73–26.81/25.73–25.82;18.10–18.14/16.77` | `23.55–23.59/22.74–22.81;13.97–14.01/12.99–13.02` | `23.72–23.75/22.21;14.08–14.10/12.76–12.79` | `96.0–96.1/88.9–89.2%` |
| Fabric toy, gain `0.631581` | `27.54/26.47;18.76/17.60` | `24.35/23.24;14.80/13.39` | `25.04/22.71;15.43/13.25` | `96.1/89.1%` |
| Painted toy, matched before | `26.38/25.60;16.56/15.77` | `24.98/24.47;14.87/14.07` | `23.45/22.87;13.19/12.38` | `93.3/89.5%` |
| Painted toy, gain `0.765983` | `26.91/25.98;17.03/16.18` | `25.63/25.15;15.52/14.58` | `24.32/23.24;14.07/12.97` | `93.3/89.5%` |
| Fresh metal sculpture, exact before | `25.57/23.33;16.39/14.36` | `23.78/22.16;13.33/12.75` | `24.02/22.56;13.55/12.73` | `87.2/74.5%` |
| Fresh metal sculpture, gain `0.616812` | `26.46/24.04;17.05/14.80` | `24.77/23.27;14.33/13.26` | `25.11/23.09;14.64/13.34` | `87.2/74.5%` |

Every matched image mean/tail improves on all three objects and geometry
metrics are identical. The full painted-toy training optimum was rejected
after a 0.05 dB known-light whole-frame tail loss; the halfway correction clears that
tail. These gains still leave both objects below their strongest trivial
excluded-light foreground baselines. This is a representation-transfer fix,
not evidence that the recovered lighting and surface properties are yet good.
The third object, `obj_45_metal_lizard`, was selected before inspecting any
photographs or results. Its 61,648 native depth groups come from 233,072
construction-only observations; the hull rejects 17,480 groups and retains a
fixed 2,499-point surface. Only 1,664 particles are observed by the training
photographs. The exact control removes the accepted gain from the same
persisted Gaussian and scores both variants in one process, so its unchanged
87.2/74.5% recall/precision is not obscured by atomic fitting variance. It
selects the automatic large-individual-table path, but the render still misses
thin disconnected parts and loses badly to both excluded-light foreground
baselines. The next problem is surface ownership and specular material, not
another global colour correction.

The exact runs and telemetry are under
`target/audit-runs/openillumination/obj{29,31}-conservative-gaussian-material/`;
the fresh exact comparison is under `obj45-global-material/`, and its no-flag
automatic-path confirmation is under `obj45-auto-confirm2/`. Reconstruction
peaks at 1.00/0.93/0.82 GiB with zero swap, OOM, or GPU fault. PatchMatch runs
in a separate 10 GiB container scope and completes in 6.9 minutes without a
CUDA or GPU fault.

These controls leave a narrower next step: derive local depth from per-view
photometric normals, anchor it with verified tracks, and require another
camera to confirm the patch before fusion. A response descriptor, a normal, or
the current cloud's visibility is evidence about a surface, but none alone is
an independent depth measurement.

The metal sculpture changes the immediate ordering. A 5,000-group replay gains
3.0 recall points but loses 6.4 precision points and several whole-frame
scores. A three-view fusion threshold halves the number of occluded particles
but loses the known-light gate. At 256 px, 48 independently triangulated tracks
survive the selection visual hull and every one already overlaps the fixed
Gaussian alpha. The remaining visible failure is therefore not repaired by a
point-budget or source-count rule.

An exact-cloud response bound keeps roughness fixed at one and allocates 25%
of base colour to the broad specular term. Known-light whole/foreground moves
`26.46/24.01;17.08/14.92→27.25/25.03;17.56/15.49` dB. Patterns 005/006 move
to `26.05/24.57;15.59/14.37` and `26.39/24.31;15.88/14.46` dB, respectively;
both beat capture-light-copy on foreground mean/tail for the first time, but
remain below black. The clean integrated run is under the ignored diagnostics
directory at `target/audit-runs/openillumination/obj45-specular-transfer-clean/`
and peaks at 0.86 GiB with no swap, OOM, or GPU fault. Because the same
proposal regresses a painted-toy primary-light tail, it is available only as
`--render-transfer-specular` and must not be read as recovered metalness.

The spatial follow-up does not repair that tail. Four deterministic regions
are fitted on patterns 001--003 and screened on patterns 004/013 plus reserved
training cameras; every region is selected on both paint and metal, reducing
to the same global proposal. Painted view `CB8` loses 0.56 dB only under
pattern 001 while improving under the other four known lights, and every
region alone hurts that 001 view. Roughness 0.75/0.5 is substantially worse.
This is directional transport error rather than evidence for a bad spatial
material cluster.

An attempted three-fifths-of-correction diffuse safety factor improves 83 of 84 exact
whole/foreground mean/tail metrics across fabric, paint, and metal. The missing
metric is still disqualifying: painted pattern 001 whole-frame worst falls
from `25.98202` dB, and even a 1% extra darkening changes it to `25.97649` dB.
The production factor remains one half. Ignored tools and image dumps are under
`target/audit-tools/{spatial_lobe_gate,obj45_lobe_sweep}/` and
`target/audit-runs/openillumination/spatial-lobe-view-31/`.

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
4. Run the LUCES-MV adapter through the implemented per-view finite-light
   contract, then generalize aligned directories to sparse `(camera, light)`
   observations so every light need not be photographed at every pose.
5. Fit diffuse albedo and normals first. Enable roughness and reflectance only
   after multiple lights improve held-light quality consistently.
6. Confirm the selected method on LUCES-MV Owl and then DiLiGenT-MV. Use
   Stanford-ORB for the independent distant-HDR check; do not make progress
   contingent on OLATverse access.
