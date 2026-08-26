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

### Current result: plumbing passes, quality does not

The first current-tree run on `obj_16_friends_cup` establishes an honest
failure baseline:

| Output | Held condition | Mean / worst PSNR | Coverage | Baseline comparison |
| --- | --- | ---: | ---: | --- |
| Static Gaussian | OLAT 062, official held cameras | 31.51 / 28.99 dB | — | Black: 31.10 / 28.46 dB; a narrow numerical win, visibly still a dark blur. |
| Relightable scalar cloud | unseen OLAT 000, official held cameras | 23.58 / 20.58 dB | 14.1% | Loses badly to black at 29.69 / 28.11 dB and capture-light copy at 30.30 / 28.16 dB. |
| Relightable Gaussian | unseen OLAT 000, official held cameras | 25.74 / 23.43 dB | 9.4% | All 638 particles persist after rejecting a degenerate support fit. The cloud is visible but blurry and still loses both baselines. |

The full run, including static and held-light reference/render pairs, is under
`target/audit-runs/openillumination/official-full-support-guard/`; the foam
training run is under `target/audit-runs/openillumination/official-static-v1/`.
Both are ignored research artifacts. Peak cgroup memory was 946 MB for training
and 728 MB for the guarded reconstruction, with no memory pressure, OOM,
socket-throttling, or GPU-fault marker.

The initial PBR support optimizer kept only 22 of 638 particles above the
persistence threshold. The reconstruction now treats retaining less than one
quarter of the established cloud as a failed fit and restores its opacity and
scale before radius feedback, calibrated-light continuation, and persistence.
The same guard runs after the multi-light pass. It does not activate on the
established synthetic control, which retained 2,184 of 2,343 particles and
kept 55.1% held-light coverage. This is a safety bound, not a quality claim:
the guarded OpenIllumination render exposes that normals, materials, and the
light model remain wrong.

The generated lights are explicit approximations, not claimed calibration.
The public `light_pos.npy` gives the 142 OLAT directions used by the official
photometric code, but not an absolute radiometric power for this pipeline.
The importer normalizes each direction and assigns unit normal-incidence
irradiance to a finite distant lobe. It therefore omits finite-distance
falloff, emitter size, RGB power variation, camera response calibration, and
interreflection. Passing the real gate requires improving this representation,
not tuning against OLAT 000.

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

1. Continue fixing the surface bottleneck exposed above. The near-empty
   Gaussian now loses to its established cloud support. Next derive a more
   accurate boundary/support cloud from the multi-view mask hull and foam
   density, then require held-training-camera coverage before learned support
   can replace that control.
2. Represent the light-stage LEDs honestly. Preserve calibrated finite light
   positions, fit per-light RGB/radiometric gains from training lights only,
   and add analytic point-emitter falloff if it beats the distant control. No
   polygonal light geometry is needed.
3. Require every candidate to beat black and capture-light-copy baselines on
   mean, worst-view, covered-pixel quality, and visible shadow/highlight motion.
4. Generalize aligned directories to sparse `(camera, light)` observations so
   every light need not be photographed at every pose.
5. Fit diffuse albedo and normals first. Enable roughness and reflectance only
   after multiple lights improve held-light quality consistently.
6. Confirm the selected method on DiLiGenT-MV or ReNé, then scale to
   OLATverse/OpenSubstance only if the smaller gates justify it.
