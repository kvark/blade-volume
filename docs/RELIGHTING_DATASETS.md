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

Start with one selectively downloaded OpenIllumination object, not the full
corpus. Materialize all selected camera names under four lighting patterns:
three known training lights and one untouched evaluation light. Keep the
official masks, combine enough official train and test cameras to make one
posed capture, and reserve the test cameras before any geometry work.

The first importer should emit the existing COLMAP-plus-aligned-directories
contract and calibrated `.f32` lights. Then run the primary light plus two
aligned fitting lights through `reconstruct`:

```bash
cargo run --release -p blade-volume-train --bin reconstruct -- \
    --sparse capture/sparse/0 \
    --images capture/light-001 \
    --masks capture/masks \
    --environment capture/light-001.f32 \
    --normal-images capture/light-004 \
    --normal-environment capture/light-004.f32 \
    --normal-images capture/light-007 \
    --normal-environment capture/light-007.f32 \
    --held-out-images capture/light-010 \
    --held-out-environment capture/light-010.f32 \
    --pbr-gaussian-output target/audit-runs/openillumination/scene.ply \
    --dump target/audit-runs/openillumination/images
```

The ordinary `test` and `g-test` rows measure novel cameras under the capture
light. `relight` and `g-relight` use the same held cameras under light 010,
which is loaded only for scoring. Its comparison images are written under
`images/held-light/{scalar,gaussian}/`. No dataset images or generated results
belong in Git.

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

1. Import one OpenIllumination object and reproduce the held-camera/held-light
   score above.
2. Represent the light-stage LEDs honestly. First quantify the error from a
   distant directional approximation; add calibrated point emitters if that
   error is material. The emitters remain analytic or point-sampled—no polygon
   geometry is needed.
3. Generalize aligned directories to sparse `(camera, light)` observations so
   every light need not be photographed at every pose.
4. Fit diffuse albedo and normals first. Enable roughness and reflectance only
   after multiple lights improve held-light quality consistently.
5. Confirm the selected method on DiLiGenT-MV or ReNé, then scale to
   OLATverse/OpenSubstance only if the smaller gates justify it.
