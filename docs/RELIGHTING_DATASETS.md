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
No official Hub repository was found for OLATverse, MIT Multi-Illumination,
FAU Multi-Illuminant, Flash/Ambient, MILL, LUCES-MV, ReNé, DiLiGenT-MV,
Objects With Lighting, or OpenSubstance; use their official project releases
instead.
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
| [OLATverse](https://github.com/xilongzhou/OLATverse) | 42 validation and 767 training objects, each with 35 cameras and 331 measured finite lights | **Active large calibrated gate.** The official split gives a dense, independently held camera/light cross-product for geometry, material, and transport. | Registration-gated, about 21.2 GiB compressed for validation and 284.4 GiB for all archives; five polarized cameras need separate treatment. |
| [LUCES-MV](https://arxiv.org/abs/2412.16737) | Public calibrated subset: 10 objects, 12 views by 15 near-field LEDs | **Current controlled-light gate.** It has linear 16-bit RGB, masks, camera/LED calibration, depth, normals, and ground-truth shape. One object is about 1.9 GB. | Non-commercial research licence; the calibrated public subset is smaller than the paper's complete capture. |
| [DiLiGenT-MV](https://sites.google.com/site/photometricstereodata/mv) | 5 objects, 20 views by 96 calibrated lights | **Second active controlled-light gate.** The pure-Rust Bear route below reconstructs and scores a fixed 16/4-camera, 24/8-light split. | Only five object-centric scenes; the lights are distant and the objects are mostly diffuse. |
| [MVSCPS capture](https://huggingface.co/datasets/cyberagent/mvscps) | 6 scenes, generally 24 camera poses under each of 6 moving-rig OLATs | Later unknown-light reconstruction and capture-practice gate with RAW/JPEG, masks, and camera projection matrices. | 65.1 GB, CC BY-NC; no measured lights and no quantitative ground-truth mesh. |
| [OpenIllumination](https://oppo-us-research.github.io/OpenIllumination/) | 64 objects, 70 views, 13 multi-LED patterns and 142 OLAT conditions | Existing broad-light real gate, with camera poses, masks, and official train/test views. | Roughly 900 GB in full; its public positions do not provide the complete finite-light radiometric calibration needed here. |
| [Objects With Lighting](https://github.com/isl-org/objects-with-lighting) | 64 input cameras under one unknown natural environment, plus nine official camera/environment test pairs per object | **First independent natural-environment gate.** Compact, calibrated, masked, and directly compatible with the distant-HDR renderer. | One input environment leaves material and illumination strongly ambiguous; it is an evaluation gate, not repeated-light training data. |
| [ReNé](https://eyecan-ai.github.io/rene/) | 20 objects, 50 views by 40 OLAT conditions | Possible later cross-check for calibrated camera/light poses. | The public object capture has no masks and fills a textured enclosure; the referenced empty-scene capture is not in the public archive, so honest background subtraction is currently blocked. |
| [Stanford-ORB](https://stanfordorb.github.io/) | 14 objects captured in multiple real environments with HDR environment maps and registered poses | Best match for the current distant-environment renderer and an important in-the-wild relighting check. | It does not provide a dense aligned camera/light grid for each background environment. |
| [DTU robot data](https://roboimagedata.compute.dtu.dk/?page_id=24) | 60 scenes, 119 cameras by 19 LEDs | Larger, more scene-like multi-view/multi-light stress test. | Approximately 730 GB in full and built around local LEDs. |
| [OpenSubstance](https://opensubstance.github.io/) | 187 objects, 270 views and 1,637 lighting conditions | Later high-resolution material and specular benchmark. | Multi-terabyte scale and access by request. |

## Active large calibrated route: OLATverse

Access is now available. The validation archive contains 42 objects; each raw
object supplies a complete 35-camera by 331-light grid plus two full-bright
frames. The published benchmark excludes the five polarized cameras and uses
24 construction plus six held cameras. Its light split uses 104 construction
indices `(0..310).step_by(3)` and 103 held indices
`(1..310).step_by(3)`. Both axes are disjoint, while the full grid provides
all four fitted/held camera-light quadrants needed for an honest score.

The project adapter is deliberately cloud-only and compact:

- `import_olatverse` materializes frame 14, masks, calibrated pose-only COLMAP
  files, an explicit construction-camera PatchMatch graph, and split lists. It
  can additionally materialize one construction OLAT for the aligned-light
  geometry stage. It does not read the released mesh or pseudo-PBR products.
- `fit_olatverse` reads the selected OLAT AVIF files directly instead of
  copying thousands of PNGs. A small MIT-licensed pure-Rust decoder adds one
  transitive crate. The loader converts sRGB to linear radiance and removes
  the release's documented 2× visualization scale.
- Camera transforms convert millimetres to metres and OpenGL camera axes to
  Blade's `+Y`-down, `+Z`-forward convention. Every view retains its own focal
  lengths and principal point.

For one extracted validation object and the official shared metadata:

```bash
cargo run --release -p blade-volume-train --bin import_olatverse -- \
    --input /mnt/data/OLATverse/validation/OLATverse_Upload_Val/data-042325-C276 \
    --output target/audit-runs/olatverse/C276/prepared-320 \
    --width 320 \
    --lights /mnt/data/OLATverse/reference/shared/all_lights.json \
    --geometry-light 0

# Fit the full-bright field, then move its fixed topology for 200 updates per
# construction camera under that one aligned, construction-only OLAT.
cargo run --release -p blade-volume-train --bin train_colmap -- \
    --sparse target/audit-runs/olatverse/C276/prepared-320/sparse/0 \
    --images target/audit-runs/olatverse/C276/prepared-320/images \
    --geometry-images target/audit-runs/olatverse/C276/prepared-320/geometry-images \
    --geometry-steps-per-view 200 \
    --masks target/audit-runs/olatverse/C276/prepared-320/masks \
    --test-list target/audit-runs/olatverse/C276/prepared-320/test-views.txt \
    --output target/audit-runs/olatverse/C276/foam.ply \
    --initialization camera-lattice --max-points 16384 \
    --width 128 --height 243 --views 0 --far-plane 10 --max-steps 384 \
    --pixel-batch 1024 --views-per-batch 16 --steps-per-view 200 \
    --sh-degree 2 --foreground-fraction 0.5

# After the ordinary pose-only point reconstruction writes surface.f32:
cargo run --release -p blade-volume-train --bin fit_olatverse -- \
    --input /mnt/data/OLATverse/validation/OLATverse_Upload_Val/data-042325-C276 \
    --lights /mnt/data/OLATverse/reference/shared/all_lights.json \
    --surface target/audit-runs/olatverse/C276/surface.f32 \
    --output target/audit-runs/olatverse/C276/relightable.f32 \
    --gaussian-output target/audit-runs/olatverse/C276/relightable.ply \
    --dump target/audit-runs/olatverse/C276/held-renders
```

The first complete point-only gate is C276. A 16,384-cell full-bright foam
produces 7,417 relightable surfels; fitting at width 128 and evaluating every
official cross-product gives:

| Lights / cameras | sRGB mean / worst | Foreground mean / worst | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| fitted / fitted | 34.22 / 24.12 dB | 26.28 / 15.18 dB | 99.4% | 89.3% |
| fitted / held | 33.72 / 26.88 dB | 26.69 / 19.20 dB | 99.5% | 91.0% |
| held / fitted | 34.16 / 26.49 dB | 26.21 / 16.70 dB | 99.4% | 89.3% |
| held / held | 33.67 / 26.92 dB | 26.63 / 19.45 dB | 99.5% | 91.0% |

The held-light quadrants match their fitted-light counterparts closely. That
passes the split and finite-light transfer gate; it does not pass the visual
quality gate. The handbag silhouette is recognizable, but quilting, flap
edges, chain, and gold clasp remain soft, and the worst grazing-light render is
nearly black.

The finite-light renderer now evaluates the existing roughness/F0 material
fields with direct GGX, in the same shader path and without another bind group
or entry point. This closes a runtime gap, not the reconstruction gate. A
bounded center-sample solve selected 12,436 of 15,667 supported materials and
reduced its sampled linear RMSE `0.016449→0.009711`, but complete
fitted-light/held-camera quality fell from `34.51/27.28` to `33.80/26.64` dB
whole-frame/foreground. Even using the old placeholder dielectric F0 without
fitting fell to `33.89/26.76` dB. Both material experiments are rejected and
their fitting code is removed. Calibrated reconstruction now writes explicit
zero F0, preserving the selected diffuse result exactly while leaving the
runtime ready for a later compositor-aware material solve. Direct-light
visibility is available separately through `--point-light-visibility`.

That single deterministic finite-distance ray is a strong but not selected
C276 candidate. On the 31,932-surfel width-64 control it changes:

| Lights / cameras | Baseline whole / foreground | Visibility whole / foreground |
| --- | ---: | ---: |
| fitted / fitted | 35.6179 / 27.4820 dB | 37.5367 / 28.9653 dB |
| fitted / held | 34.5119 / 27.2796 dB | 36.6593 / 28.9774 dB |
| held / fitted | 35.5659 / 27.4186 dB | 37.4426 / 28.8540 dB |
| held / held | 34.4636 / 27.2177 dB | 36.5655 / 28.8732 dB |

Every mean and whole-frame tail improves, and coverage is unchanged. The
strict gate still rejects making it automatic: held-light foreground worst
changes `16.9353→16.9342` and `19.6083→19.6060` dB on fitted and held cameras.
Those 0.0011/0.0023 dB losses were measured at higher precision rather than
rounded away, and the held data is not used to tune the fixed two-radius bias.

One aligned construction OLAT provides a stronger geometry result without
changing the model or renderer. `import_olatverse --lights ...
--geometry-light 0` writes only the 24 construction-camera PNGs and refuses
any held light. The existing `train_colmap --geometry-images ...
--geometry-steps-per-view 200` stage then moves fixed cloud topology under that
second capture. Light 0 and 200 updates were frozen before the second-object
replay; no held camera or held light selects the stage.

On the fresh C452 plaster statue, the 16,384-cell extracted surface changes its
held-camera full-bright score from `21.45/20.20` to `24.35/23.60` dB
whole-frame mean/worst and from `21.87/19.75` to `23.52/22.81` dB foreground.
Recall changes `97.2→99.7%` and precision `75.6→86.3%`. The complete
finite-light comparison is:

| Lights / cameras | C452 base whole / foreground | Aligned-light whole / foreground |
| --- | ---: | ---: |
| fitted / fitted | `34.2762 / 26.8906` dB | `36.2056 / 27.5976` dB |
| fitted / held | `33.4325 / 26.6292` dB | `35.4441 / 27.4855` dB |
| held / fitted | `34.2309 / 26.8425` dB | `36.1557 / 27.5443` dB |
| held / held | `33.4496 / 26.6294` dB | `35.4370 / 27.4816` dB |

Held-camera tails, recall, and precision all improve; held/held foreground
worst moves `20.9275→21.6954` dB and precision `79.101→90.139%`. Three
construction-camera minima move by only `-0.0001` to `-0.0010` dB. C276 then
repeats the held-camera result at width 64 and one material/normal round:
fitted-light whole/foreground means move `34.3543/27.2910→34.6864/27.6721`
dB, while held-light means move `34.3042/27.2253→34.6455/27.6198` dB.
Every held-camera tail and coverage metric improves. Construction-camera means
and coverage improve too, while four minima move by `-0.0001` to `-0.0013`
dB. Keep this as the explicit OLATverse surface recipe; the post-continuation
SH describes the secondary light, so retain or refit the pre-continuation
cloud separately when a static capture-light asset is also required.

Combining that C452 geometry with the still-opt-in visibility ray improves all
four means again:

| Lights / cameras | Aligned-light whole / foreground | With visibility whole / foreground |
| --- | ---: | ---: |
| fitted / fitted | `36.2056 / 27.5976` dB | `37.5926 / 28.6722` dB |
| fitted / held | `35.4441 / 27.4855` dB | `36.9359 / 28.6875` dB |
| held / fitted | `36.1557 / 27.5443` dB | `37.5254 / 28.6057` dB |
| held / held | `35.4370 / 27.4816` dB | `36.9021 / 28.6494` dB |

Held/held whole-frame and foreground worst improve from `29.3903/21.6954` to
`30.9228/22.0447` dB, with identical coverage. This fresh-object result makes
visibility more promising, but it does not erase the small C276 tail losses or
the DiLiGenT-MV Bear regression; the flag therefore remains explicit.

Making only the final diffuse equations visibility-aware is not the missing
coupling. A temporary CPU BVH reproduced the production compositor's
finite-distance translucent shadow ray and passed an exact physical-GPU
pixel test. On the same eight construction lights and 24 construction cameras
used by the region screen, it lowers the sparse linear fit loss
`0.0001541→0.0001237`. Rendered with that same visibility, however, it loses
`0.0826/0.0589` dB whole/foreground mean against the existing material and as
much as `0.4136/0.4358` dB on one image. Against the analytic no-visibility
baseline, its means still rise by `1.4226/1.2026` dB but individual-image
changes reach `-0.0146/-0.3201` dB. The implementation and BVH are removed.
Real shadowed pixels contain indirect light; deleting their direct term while
offering the solve no bounce simply moves the compensation into albedo. The
matching bounded-bounce control is also negative. A first exact-direct-blocker
bounce is identically zero for one-sided diffuse surfaces. Replacing it with a
cosine-hemisphere estimate of one unshadowed secondary hit raises the one-sample
whole/foreground construction means by `1.2671/1.1541` dB over analytic direct
lighting, below the visibility-only gains of `1.5052/1.2615` dB. Four samples
reach `1.3557/1.2407` dB, but individual images still lose as much as
`2.0101/1.8389` dB. The global worst case improves, yet both means and these
large per-image tails fail the gate; more samples reduce variance rather than
the incorrect recovered-radiance attribution. The shader prototype is removed.
Before changing transport again, attribute complete-render residuals to their
camera support and owning cloud points.

The first complete-render normal attribution bounds the simpler version of
that idea. Four `2.5°` antithetic rounds over 4,631 observed surfels and the
same 192 construction images lower pooled sRGB loss by 2.3% and raise
whole/foreground mean PSNR by `0.0582/0.0419` dB. They nevertheless lose
`0.3239/0.2890` dB on light 51 at Cam32. Requiring the plus/minus direction in
each projected surfel rectangle to agree in every construction image accepts
no round and changes no surfel. Both prototypes are removed. A screen-space
rectangle includes other overlapping and occluding particles; the next normal
proposal must retain the existing exact compact-disc blend terms and their
surfel identities instead of approximating ownership by projection. Both runs
peak at 1 GiB with no swap, cgroup event, or GPU fault.

Retaining those identities confirms the limit rather than producing a selected
normal pass. An analytic diffuse-normal gradient through the exact blend lowers
its four-round sparse objective `0.00019483→0.00019116` and raises production
whole/foreground construction means by `0.0398/0.0569` dB, but light 3 at
Cam32 loses `0.3287/0.4315` dB. The strict decomposition is more informative:
of 6,840 supported surfels, only 418 have one gradient direction after grouping
by light, 198 after grouping by camera, and one across all 192 individual
images. Four steps of that one surfel still lose `0.000008/0.000011` dB on a
different pair. The implementation is removed; its 1.5 GiB cgroup records no
swap, memory event, or GPU fault. Both light response and camera ownership are
mixed inside individual surfels, with the camera contradiction stronger. A
single physical normal cannot satisfy the data regardless of how accurately
its residual is attributed. The next cloud-only proposal must split or
densify contradictory support before refitting normals, then pass the same
per-image production gate.

The first no-code-change densification control establishes that ordinary
capacity growth is insufficient. Starting from the selected 16,384-cell C452
aligned-light cloud, the existing four-response geometry objective performs
one 15% split at update 500 and continues to 4,800 updates. It finishes with
18,842 cells and improves the static held-camera surface from
`24.35/23.52` to `24.92/23.94` dB whole/foreground mean while raising
precision from 86.3% to 87.7%. After the unchanged two-round calibrated
material fit, the predeclared construction-only screen over lights
`3/51/99/153/201/255/303/309` and all 24 construction cameras improves mean
whole/foreground PSNR `36.1559/27.6205→36.5813/27.8117` dB, recall by 0.04
points, and precision by 1.11 points. It nevertheless regresses 68 of 192
foreground images. Light 51 at Cam32 loses 1.45 dB foreground and 1.11 dB
whole-frame PSNR. The aggregate-positive candidate is rejected and adds no
training option. Because the generic fitter also printed the official held
quadrants before this per-image screen was run, those held results are not
independent evidence and are not used for selection. Future splitting must be
driven by the contradictory exact-blend support itself, not by a global
position-gradient densification score.

Three untouched training-shard objects now make that next gate independent of
C276/C452. They were predeclared from one masked full-bright thumbnail each,
before any fit: faceted C713, thin plant C769, and dark manufactured C777. Only
`all_cam.json`, masks, and raw masked OLATs were extracted; released meshes,
PBR products, and normal benchmarks were not opened by the pipeline. The
matched width-64, two-round Gaussian held-light/held-camera results are:

| Object | Static whole / foreground | Aligned whole / foreground | Static → aligned recall | Static → aligned precision | Strict failure |
| --- | ---: | ---: | ---: | ---: | --- |
| C713 | `38.8210 / 26.2121` dB | `39.0155 / 26.2735` dB | `96.867→96.889%` | `76.381→80.192%` | held whole/foreground worst `-0.0009/-0.0008` dB; construction foreground means `-0.0143/-0.0060` dB |
| C769 | `37.1710 / 26.1786` dB | `37.3290 / 26.3950` dB | `97.486→96.985%` | `73.366→77.334%` | recall `-0.501` points; held whole/foreground worst `-0.0023/-0.0028` dB |
| C777 | `44.2090 / 31.2836` dB | `44.4693 / 31.2838` dB | `95.241→96.018%` | `67.548→79.233%` | construction foreground means `-0.2395/-0.2385` dB |

Thus aligned-light geometry transfers broadly and can sharply remove false
extent, but it still moves the support/photometry frontier rather than closing
it. The new-object evidence strengthens the next requirement: subdivide only
exact-blend support with contradictory camera/light evidence, conserve its
optical support, and reject the candidate if any construction image or fresh
object loses. Complete outputs are under `/mnt/data/OLATverse/runs/{C713,C769,C777}/`;
each aligned `olat-light000-baseline64-r2-v1/` contains all 2,472 held-camera
reference/render PNGs. The three raw selected objects occupy about 1.1 GiB at
`/mnt/data/OLATverse/training/Tr0281-0350-selected-v1/`.

A direct exact-blend support split is rejected on C769 before either remaining
fresh object is opened. All 104 fitted lights accumulate per-camera tangent
normal gradients and their residual-weighted disc hit points. Of 3,793
surfels, 3,778 have multi-camera support and 3,691 have an opposing,
spatially separated camera pair. Replacing the top 19 parents with two
contained discs of radius `r/√2` lowers construction-validation whole and
foreground means by `0.0161/0.0134` dB and regresses 122/107 of 192 images.
The one-parent bound raises the means by `0.0021/0.0016` dB but still regresses
43/35 images, loses `0.0223/0.0306` dB on the worst whole/foreground image,
and lowers recall. A zero-split replay is bit-exact, isolating the change to
support rather than the material refit or scorer. Equal summed disc area is
not equal optical support: two smaller radial kernels cannot partition a
larger radial kernel without overlap or holes. The host prototype and its
diagnostic binary are removed, leaving no API, shader, operation, or model
field. The three scoped runs peak at 1.47--1.53 GiB with zero swap, OOM, or
GPU fault. C769 is now a development object; C713 and C777 remain unopened by
this rule. The next proposal should triangulate distinct sites from the
104-light multi-view response before surface fusion rather than subdividing an
already merged radial primitive.

That response-track proposal is informative but does not justify additive
support. Gain-normalized 104-light descriptors from 16 construction cameras
produce 2,314 one-way and 899 mutual matches. Of 248 multi-view groups, 80
triangulate and 63 pass the reprojection and construction-only visual-hull
screens. Fifty-seven also survive the disjoint validation-camera masks. Local
connectivity leaves three coherent components containing 25 sites, the largest
with nine. They average 3.22 observations, 0.395-pixel reprojection error,
0.086 descriptor error, and 78.3 degrees of parallax.

Appending those 25 sites after the established fit lowers the exact joint
material objective, but its 192-image construction-validation replay loses
`0.0426/0.0333` dB whole/foreground mean, regresses 141/118 images, and trades
`+0.0331` recall points for `-0.0359` precision points. Inserting the same
sites before the established two-round normal/material fit is much closer but
still fails: whole/foreground means fall `0.0043/0.0045` dB, 98 foreground
images regress, recall loses 0.0106 points, and the worst foreground loss is
0.333 dB. The exact material system improves from `9.87e-5` to `9.71e-5` in
the post-fit arm, so neither rejection is caused by a failed solver.

A final spatial audit identifies the ambiguity. Twenty-three of the 25 sites
are already inside the nearest fitted surfel's radial support; all lie within
one radius of its tangent plane. Their median centre distance is 0.52 radii,
but their median unoriented normal disagreement is 52.4 degrees. The matches
therefore identify competing surface ownership inside support already claimed
by the radial model, not an independent missing layer. Both temporary
diagnostics are removed. The fit and score scopes peak at 1.77 and 1.01 GiB
with zero swap, OOM, or GPU fault. The next bounded experiment should insert
the same verified sites where ownership is partitioned by a PowerFoam cell,
not represented by another composited radial kernel. It must pass C769's same
192-image gate before C713 or C777 is opened.

The exact partition control is negative too. It maps every fitted Gaussian's
stored three-sigma radius to its analytically fixed half-opacity radius
`sqrt(2 ln 2) / 3`, uses an opaque oriented PowerFoam cell, and evaluates the
same diffuse material and calibrated point light at each cell. Thus a new site
partitions overlapping support through radical planes instead of layering more
alpha. All 25 sites raise whole-frame mean by 0.0087 dB but lower foreground
mean by 0.0052 dB, lower recall, and regress 92/192 foreground images; the
worst loses 1.07 dB. The three source-connected components also fail alone:
their foreground mean changes are `-0.0148`, `+0.0115`, and `+0.0141` dB,
with 61/63/30 regressions and worst losses of 0.888/0.826/0.226 dB. The latter
component raises recall but still lowers precision. Rendering its eight sites
through the actual radial production path confirms the rejection: whole and
foreground means lose 0.0012/0.0011 dB, precision falls 0.0075 points, and
63/192 foreground images regress.

The temporary hard-cell renderer and component writer are removed. Each score
scope peaks at approximately 1.0 GiB with zero swap, OOM, or GPU fault. This
closes post-fusion insertion for both overlapping radial and exactly
partitioned support: credible sites still need correct source-observation
ownership before their centers, radii, and materials are fused. The next
proposal must use response compatibility while forming the surface, and pass
C769 before the two untouched objects are opened.

Applying that compatibility during foam-depth fusion is also rejected. The
current extraction is first reproduced at width 128: 24 construction cameras,
the 16,384-cell light-000 foam, and voxel factor two yield the same 3,793
particles. A direct 3x3 photometric-albedo descriptor then keeps the dominant
compatible source-camera subset in each voxel. The unrestricted arm changes
826 cells, removes 2,055 observations, and leaves 3,551 particles. Before any
material fit it gains 0.055/0.035 dB whole/foreground mean and 0.366 precision
points, but loses 0.090 recall points, regresses 72/192 foreground images, and
loses 0.672 dB on the worst. A conservative arm falls back to the ordinary
voxel whenever its compatible subset cannot form a valid multi-view surfel. It
keeps 3,795 particles while changing 476 cells, yet loses every aggregate:
`-0.0015/-0.0056` dB whole/foreground, `-0.0064/-0.0163` recall/precision
points, 103 foreground regressions, and a 0.432 dB worst loss. Geometry fixes
those alpha metrics, so a material fit cannot rescue either arm.

The stronger upstream proxy fails for the complementary reason. A robust
photometric-albedo image is recovered from all 104 calibrated construction
lights at every camera after normalizing finite-light radiance at the cloud
centre. Its foreground values are bounded (median 0.043, p99 0.382, maximum
0.480). The established 200-update-per-view fixed-topology continuation lowers
its training loss `0.1195→0.0542` in 35.6 seconds, but its extracted surface
shrinks to 3,518 particles and traced-hit rate falls `99.1→97.8%`. Before
material fitting it raises whole/foreground mean by 0.787/0.198 dB and
precision by 3.57 points, while losing 0.94 recall points, regressing 103/192
foreground images, and losing up to 2.20 dB. This repeats the earlier
four-light response-field boundary with much stronger data: a light-invariant
proxy sharpens likely foreground but is not the image-formation objective and
suppresses valid low-response support.

All fusion and albedo-training prototypes are removed. The response scopes
peak at 1.9--2.0 GiB, the foam continuation at 1.8 GiB, and every run records
zero swap, OOM, or GPU fault. Do not tune another response threshold or train
another proxy image.

The release layout rules out the previously written C769 geometry check:
training shards contain `masked_olat/`, `mask/`, `pbr/`, and `all_cam.json`,
but only validation shards contain `model/`. Instead, the released C452
validation point stream was used strictly after candidate selection as a
point-cloud evaluator; its faces were never parsed. At 128 pixels, the
light-000 foam's per-view density depth is already displaced from the released
surface by 3.02 source-pixel footprints at the median and 8.28 at p90. The
fused 6,180-point surface has 1.28-pixel median nearest-truth distance and
49.34-degree median normal error. The fitted 6,840-point Gaussian improves
those to 1.11 pixels and 32.35 degrees, respectively. Fusion therefore does
not create the position error, while the later fit improves but does not fix
surface orientation.

Moving the selected absorption segment from its midpoint to the conditional
exponential mean improves the evaluator-only median depth error from 3.02 to
2.59 pixels and the fused median from 1.28 to 1.24 pixels. It still fails the
real C769 construction gate before material fitting: whole/foreground mean
rises by 0.847/0.368 dB, but precision falls 0.234 points, 84 of 192
foreground images regress, and the worst loses 2.01 dB. The implementation is
removed. A better average truth distance is not sufficient when it moves
optical ownership between images.

C769 does provide an independent normal evaluator. Its five polarized cameras
(`Cam07/10/17/22/39`) are excluded from the published 24-construction/6-held
camera split and carry diffuse-normal `_nd` maps. At 320 pixels, the direct
RGB-to-world-XYZ convention is the uniquely strongest camera-facing decoding.
Against these maps, unsigned median normal error is 54.69 degrees directly at
derivative-qualified density depths, 54.04 degrees at the pre-fit surface,
and 47.37 degrees after the calibrated surface fit. The 3,793 surfel centers
are bit-identical before and after fitting. This independently confirms that
the first failing stage is shared density/first-surface geometry, not fusion;
normal fitting recovers part of the loss but inherits a badly associated
surface. The pseudo-GT maps remain evaluator-only. C713 and C777 remain
untouched transfer gates.

The ordinary construction captures contain a substantially stronger usable
signal. Solving a world-space Lambertian normal independently at every density
hit from all 104 calibrated finite lights gives 24.66-degree median unsigned
error against the construction `_ncg` maps, compared with 56.29 degrees for
the density-depth derivative. The fit corrects each observation with the exact
finite point-light radiance at that density hit; it uses only construction
photographs and masks. The `_ncg` maps are loaded afterward for measurement,
never for normal recovery or candidate construction. A brighter-observation
robustification is worse at 26.01 degrees and is rejected.

Perspective log-depth integration proves that this normal signal is real but
also that applying it after density tracing is the wrong boundary. With a weak
anchor, median depth moves only 0.06% and the integrated normal is 52.12
degrees; after the unchanged calibrated fit, the 192-image screen loses the
foreground mean, 0.51 recall points, and 86 image tails. The strongest
predeclared anchor moves depth by 0.22% at the median and improves the excluded
polarized normal check from 47.37 to 44.65 degrees after fitting, but it loses
1.18 recall points and regresses 46 foreground images, by as much as 1.97 dB.
Both candidates improve precision, demonstrating a familiar shrinkage trade,
not a better shared surface. No production option or asset format is added.
The next implementation must put this calibrated normal constraint on
differentiable first-surface depth inside density training, before fusion.

That direct differentiable test is now closed as well. A contiguous-patch loss
formed the expected absorption depth on each ray, unprojected it to a
world-space point patch, and compared its camera-facing normal with the
construction-only 104-light estimate. It used only existing Meganeura tensor
operations. On C769, a density-only `0.05`-weight candidate produces 3,786
surfels instead of 3,793, retains 99.1% traced hits, but changes the exact
192-pair construction screen by `-0.021059/-0.001506` dB whole/foreground
mean. It regresses 106 images and loses as much as 0.472782 dB. Allowing a
`0.01` position learning-rate ratio is worse at the geometry gate: traced-hit
coverage falls to 97.0% and extraction retains only 3,693 surfels, so material
fitting is not run. Both scoped runs have zero swap, OOM, or GPU fault; peak
memory is 2.10 GB.

The failure is structural rather than a weight-selection result. A normal made
from neighboring expected depths constrains slope, but not the absolute layer
that owns the RGB or the optical support that preserves the mask. The public
API, CLI, graph path, and tests are removed. The next density experiment must
retain the full calibrated per-light RGB and mask residual in one shared
first-surface/material objective. Per-light latent appearance may help
optimization but must remain scratch-only and cannot enter the result model.

That full-image experiment is now closed as well. A temporary Meganeura graph
reused the RadFoam absorption weights and existing finite-emitter equations to
fit shared density against all 104 construction lights and 24 construction
cameras (2,496 RGB-and-mask observations). Shared per-cell normals and diffuse
albedo were initialized from the established fitted surface and optimized only
as scratch state; SH appearance was disabled. The prototype added no operation,
shader variant, dependency, persisted model field, or released-data target.

At width 32, one update per observation, a 512-ray batch, eight views per batch,
and a 384-step path limit, all three density rates lower the objective from
`0.009590`: rates `0.01/0.001/0.0001` end at
`0.004469/0.008302/0.008760`. Absolute scratch normal/albedo rate is `0.001`
throughout. The corresponding extractions retain `3,545/3,764/3,790` surfels
and `97.6/99.0/99.1%` traced hits, versus `3,793` and `99.1%` for the control.
The least destructive arm was passed through the unchanged two-round material
fit and exact 192-image construction screen. Whole/foreground mean changes by
`+0.013338/+0.011536` dB and precision by `+0.0820` points, but recall changes
by `-0.0256` points, 94 foreground images regress, and the worst pair loses
`0.328559` dB. Smaller steps converge toward no geometry change; larger steps
delete support.

The failure identifies missing image formation, not a density-rate choice.
The implementation is removed. A construction-only attribution diagnostic then
compared the control and least-destructive density arm at every eligible
foreground pixel after normalizing each measured light's radiance. Of 40,569
samples, 40,250 retain stable opacity, 160 lose opacity, and 159 gain it. The
mean negative residual is `0.0070` for stable pixels but only `0.0045` for lost
pixels; their dark-residual fractions are `2.96%` and `1.65%`. Positive residual
dominates instead: `0.1228` for stable, `0.1214` for lost, and `0.1395` for
gained pixels. Cast-shadow deficit therefore does not preferentially select the
support that density removes. The diagnostic peaks at 2.95 GB with zero swap,
OOM, or GPU fault.

A second scratch run tests optimization order rather than changing the image
model. It first freezes density and warms shared normals/albedo for 2,496
updates, lowering the physical loss from `0.004580` to `0.003380`; a fresh Adam
session then enables density for another 2,496 updates and reaches `0.002973`.
The extraction appears healthier than the control by a coarse support count:
`99.4%` versus `99.1%` traced hits, with 3,761 versus 3,793 surfels. After the
unchanged two-round material fit, however, the exact 192-image construction
screen changes whole/foreground mean by `-0.150134/-0.174586` dB, recall by
`-0.0377` points, and precision by `+0.4411` points. It regresses 116 foreground
images and loses 1.606390 dB in the worst pair. The fit and score peak at
1.75/1.13 GB with zero swap, OOM, or GPU fault. This rejects material warmup,
and the temporary graph/API/CLI are removed.

The next geometry objective should not begin by adding light visibility. The
measured failure is dominated by unexplained positive response—compatible with
highlight, indirect-light, or wrong-layer ownership—and a lower average
physical loss does not identify which. First establish one observation-
consistent surface identity across cameras, anchor its absolute first-surface
depth, and conserve optical support; only then may a robust light residual move
that shared surface. Selection must continue to include every construction
image tail.

A balanced residual decomposition narrows that ambiguity without training a
candidate. It projects the established 3,793-surfel fitted surface into all 104
construction captures and groups 865,488 calibrated observations on 2,250
well-supported surfels. After removing each surfel's mean, camera identity
explains 5.082% of residual variance and light identity explains 62.689%.
Neither is a global exposure correction: effects shared across the whole image
explain only 0.502% by camera and 0.595% by light. Half-vector lobe proxies from
powers 1 through 128 explain at most 0.0598%. The remaining mismatch is thus a
spatially local point/light response—consistent with local transport, a wrong
normal/layer, or both—not global calibration or an ordinary view-dependent
highlight. The complete diagnostic peaks at 2.02 GB with zero swap, OOM,
throttle, or GPU fault.

This makes a scratch light-response field the next discriminating experiment,
not a released representation. One RGB response per density cell and each of
the 104 construction lights is shared across every camera. A first session
freezes density and lowers response loss `0.001153→0.000051`; a fresh session
then trains response at `0.001` absolute rate and density at `0.0001`, with a
ten-weight mask term, lowering its combined loss `0.095665→0.086937`. The
table uses the existing embedding/reduction primitives and is discarded before
serialization; no shader, Meganeura operation, dependency, or persisted field
is added.

The result is closer to the control but still rejected. Extraction retains
3,787 rather than 3,793 surfels and ties traced-hit support at 99.1%. After the
unchanged two-round material fit, the exact 192-image construction screen
changes whole/foreground mean by `-0.002586/+0.001861` dB, recall by `-0.0147`
points, and precision by `+0.0704` points. It regresses 106 foreground images
and loses 0.317244 dB in the worst pair. The fit, extraction, material fit, and
score peak at 1.20/0.15/1.75/1.10 GB with zero swap, OOM, throttle, or GPU
fault. The temporary graph/API/CLI are removed.

This closes radiance-driven density on the current official split. A zero
density rate is exactly the unchanged foam; the first meaningful response-
conditioned step already has broad contradictory tails, and selecting another
rate against those same 192 images would tune the validation screen. The next
geometry input must therefore supply absolute cross-view first-surface
correspondence upstream—triangulated response tracks or another independent
multi-view depth cue—then use the renderer only to validate and polish the
point cloud. C713 and C777 remain untouched transfer gates.

The first upstream test confirms that calibrated response tracks are a useful
absolute cue, while rejecting their first propagation mechanism. Four fixed
construction OLATs (`0/102/204/306`) are matched across 16 cameras; four more
construction cameras select tracks by a 3/4 visual-hull rule and a disjoint
four-camera subset checks them. At width 128, C452 yields 579 selected tracks,
of which 565 (97.58%) pass that camera check. Only after persisting the tracks
is released point-only truth opened: median nearest error is 0.350 source
pixels and 93.96% lie within one pixel. Local covariance normals remain noisy
at 26.49 degrees, so the positions—not their inferred normals—are the useful
measurement. The same construction-only protocol yields 197 selected C769
tracks, of which 193 (97.97%) pass the disjoint-camera check. No released
geometry, normal map, held light, or held camera participates in matching or
selection.

A scratch C452 continuation then freezes positions and adds an L1 loss between
each matched ray's absolute track depth and the density field's conditional
expected depth. The selected `100` weight and `0.01` density rate lower the
track-relative median/p90 error from `0.01045/0.02181` to
`0.00827/0.01870`. Evaluation truth subsequently moves in the intended
direction: same-pixel density-depth median improves from `3.0166` to `2.9257`
source pixels, raw nearest truth distance from `1.2652` to `1.2502`, and fused
surface distance from `1.2783` to `1.2681` pixels.

The independent construction render gate nevertheless rejects the result.
After identical two-round material fits, whole/foreground mean changes
`36.1905/27.6589→36.0632/27.6208` dB, recall changes
`98.3420→98.4506%`, precision changes `90.8236→90.3682%`, and 102/192
foreground images regress; the worst pair loses 1.5679 dB. This is an
easy-track-biased sparse measurement spread through a globally shared optical
field: it gains a little depth and recall while disturbing unrelated support.
The temporary depth-loss path is removed. Tracking, training, truth audit,
extraction, and the complete gate peak at 1.00/1.02/0.10/1.1/3.5 GB with zero
swap, OOM, or GPU fault. Keep the accurate sparse anchors, but next propagate
them through bounded local point-cloud corrections that conserve all
unconstrained support; do not tune another global density rate on this screen.

Post-extraction local propagation is now closed as well. The first candidate
leaves alpha/hit decisions unchanged, applies each observed depth correction
within a compact data-derived 2--8-pixel footprint, and re-fuses the maps. It
touches 21.21% of valid samples and creates 7,446 rather than 6,838 surfels.
Foreground mean and recall improve by 0.0924 dB and 0.4727 points, but whole
mean changes by -0.0817 dB, precision by -0.9865 points, and 62/192 foreground
images regress. The extra cells expose the problem: even a local per-view warp
fragments shared support during voxel fusion.

Exact-count surface variants remove that confound. Assigning each persisted
track to a surfel only inside its existing radial support, moving only along
the established normal, and capping displacement at half a radius improves
whole/foreground/covered means by `0.0144/0.0519/0.0292` dB and recall by
0.1170 points, but loses 0.0986 precision points and 55/192 foreground tails.
An inner-half assignment with a quarter-radius cap reduces the precision loss
to 0.0513 points but regresses 69 tails. Replacing only the normals of those
inner correspondences is worse: whole/foreground mean changes
`-0.0427/-0.0325` dB and 125 tails regress. Accurate sparse normals are not a
drop-in replacement for the jointly fitted shading normals.

Seeding every pixel rather than every other pixel increases the width-128
selection from 579 to 1,332 tracks (1,297 oriented sites); 97.37% pass the
disjoint-camera hull check, and a later truth audit puts the median at 0.361
source pixels, 91.74% within one pixel, and the local-normal median at 26.40
degrees. The unchanged inner-half/quarter-radius rule then moves 627/6,840
surfels and improves all three PSNR aggregates by `0.0342/0.0514/0.0414` dB,
but still loses 0.0616 precision points and 58 tails. Spreading those offsets
smoothly over two support radii moves 4,912 surfels and improves foreground,
recall, and covered quality, yet changes whole mean by -0.0024 dB, precision by
-0.1354 points, and regresses 88 tails. Local smoothness is therefore not the
missing constraint.

At width 256 the same fixed matcher yields 3,831 selected tracks, 3,735
disjoint-camera hull passes, and 3,689 oriented sites. Evaluator truth opened
after persistence reports 0.464 working-pixel median distance and 21.64-degree
median normal error. As a standalone cloud, however, a conservative two-pixel
radius covers only 27.58% of the mask. Standard point-cloud radii (1.4 times
mean eight-neighbor spacing, robustly capped) raise recall to 57.82% while
keeping 96.20% precision, but whole/foreground mean remain 1.558/2.735 dB
below control. Response texture does not cover textureless surface regions.

These variants peak at 3.94 GB with zero swap, OOM, or GPU fault. The tracks
remain unusually accurate independent geometry evidence; what is rejected is
retrofitting them onto a density-derived finished surface. Next use them as
fixed or seeded sites before the ordinary full-image point-cloud training, so
neighboring sites and optical support are optimized around the absolute
anchors. Do not add them after extraction, inflate them across evidence gaps,
or tune another displacement radius on C452's construction screen.

That upstream initialization is now tested too. The width-128 stride-one
tracks replace 1,297 low-density sites in a 16,384-site camera lattice, retain
the exact site budget, and initialize their ordinary density/SH from the
measured response. With every site free during the unchanged single-light plus
geometry continuation, the seeded surface grows from 6,759 to 6,986 surfels
and traced support from 99.6% to 99.8%. Against its paired freshly fitted
control, construction whole/foreground mean improves by `0.2056/0.3226` dB
and recall by 0.8787 points, while precision loses 0.6586 points and 51/192
foreground pairs regress. Against the established fitted control, the same
candidate remains positive by `0.1600/0.1474` dB and 0.6556 recall points, but
loses 0.2729 precision points, regresses 69/192 pairs, and loses 1.9178 dB in
the worst pair. The aggregate signal is real, but it does not clear the
complete-render gate.

Freezing the 1,297 seeded positions tests whether training merely forgets the
measurement. A temporary stop-gradient prefix holds every center exactly (all
position-drift quantiles are zero) without adding an operation or shader. It
instead prevents the power cells from adapting: foam held-view precision falls
to 74.3%, extraction traces only 96.7% of rays, and the fitted construction
screen changes whole/foreground/covered mean by
`-1.3555/-0.5664/-0.9012` dB, recall/precision by
`-0.6458/-7.4095` points, and regresses 153/192 foreground pairs. Restoring
the accurate centers only after free training is also invalid: traversal falls
to 92.6% and the surface to 6,165 surfels because topology and optical support
were learned around the moved sites. Training, extraction, and the full gate
peak at 1.0/0.15/1.6 GB with zero swap, OOM, throttle, or GPU fault. The
temporary gradient mask is removed.

The reordered all-sites-trainable control reproduces the earlier result
(99.8% tracing, 19.173 dB held foam PSNR, and 86.8% precision), ruling out
point order as the cause of the fixed-site collapse. A soft dynamic variant
then projects the 1,297 tracks to 20,435 construction pixels and applies the
previously selected weight-100 relative loss to conditional expected
absorption depth while both density and positions move. The ordinary RGB and
weight-one mask objective remains active. Nevertheless held foam
recall/precision collapses to 76.9/68.8%, extraction traces only 91.9% of rays,
and the surface shrinks to 5,608 points. Conditional mean depth can move
optical mass without establishing a coherent first surface; no lower weight is
tuned and the scratch graph is removed.

Changing which lattice site is replaced isolates a useful but still
unaccepted initialization signal. Replacing the nearest site, rather than the
nearest low-density site, reduces median displacement from 0.0783 to 0.0441
world units. The freely trained 6,900-surfel surface retains 99.8% tracing and,
against the established fitted control, improves whole/foreground/covered mean
by `0.4714/0.3101/0.4220` dB, recall/precision by `0.6560/0.6090` points, and
every per-light mean. It still regresses 52/192 foreground pairs; 26 lose more
than 0.25 dB and the worst loses 1.7089 dB. The tails span cameras and lights,
so this is not one bad calibration entry.

A topology-local selection accepts only the 523 tracks whose displacement is
within the source site's median Delaunay-neighbor distance. It raises precision
by 0.9102 points, but loses 0.0238 dB foreground mean, 0.0698 recall points,
and 94/192 tails. Appending all tracks preserves every lattice site at a 7.9%
point-count cost (16,384→17,681), yet loses 0.1261 precision points and 65/192
tails despite gains of 0.2838/0.2251 dB whole/foreground and 0.6182 recall
points. Neither topology locality nor extra capacity fixes ownership.

The predeclared scale check closes response-track initialization completely.
At 64K/256, the same nearest-site rule replaces 3,689 of 65,536 lattice sites
and reduces median displacement from 0.0441 to 0.0265 world units. That is
still 2.65 times the source site's median Delaunay-neighbor distance. Against
the paired freshly trained control, held foam whole/foreground PSNR falls
`19.5080/10.0652→19.0924/9.6181` dB, recall `94.5→93.6%`, and precision
`83.9→82.6%`. Extraction creates 31,199 rather than 30,286 surfels, but held
surface whole-frame PSNR falls `24.17→23.81` dB and precision `85.6→84.6%`;
recall rises only `99.3→99.5%`.

Identically fitting both raw surfaces on all 104 construction lights leaves a
small aggregate signal: whole/foreground/covered mean changes
`+0.0389/+0.2124/+0.0815` dB and recall rises 0.7071 points. It does not pass
the gate: precision loses 0.4595 points and 60/192 foreground pairs regress;
51 lose more than 0.05 dB, 31 more than 0.25 dB, and three more than 1 dB. The
worst pair loses 1.3079 dB. Every light has a positive mean while six camera
means are negative, localizing the failure to view-dependent cell ownership
rather than light calibration. Training and the full fit peak at 0.28 and
1.61 GiB with zero cgroup swap, memory event, OOM, throttle, or GPU fault.

No C769 or transfer object is opened, and no subset, weight, or threshold is
tuned. The complete initialization family is rejected without an API, graph
option, shader, operation, dependency, or persisted field. Keep the tracks as
independent sparse geometry evidence, but do not insert, fix, softly supervise,
or append them in the camera lattice. The next geometry route must construct
continuous cross-view point-cloud support rather than transfer isolated site
ownership.

That continuous-support follow-up is rejected as well. The first fixed
construction treats the 3,689 width-256 tracks as robust eight-neighbor
oriented discs and keeps the established 64K surface only outside each disc's
one-radius tangent-and-plane support. It replaces 11,959 of 30,286 fallback
surfels and produces 22,016 points. After the same material fit, precision
rises 0.7268 points, but whole/foreground/covered mean loses
`0.2564/0.4029/0.2851` dB, recall loses 0.5048 points, and 152/192 foreground
pairs regress. Twenty lose more than 1 dB and the worst loses 1.5895 dB.

A stronger fixed construction reuses the repository's existing missing-track
patch rule rather than tuning another threshold: observations within four
pixels in at least two shared source cameras form a component, components need
five tracks, orientation and support are estimated within each component, and
one point-only midpoint sample is added on each unique two-nearest-neighbor
edge. Forty-four components retain 3,531/3,831 selected tracks and produce
8,103 samples. Their small observed-pixel support replaces only 3,060 fallback
surfels, yielding a 35,329-point cloud at a 16.7% point cost. Recall and
precision rise by only 0.0361/0.0503 points, while whole/foreground/covered
mean loses `0.2999/0.3170/0.3078` dB and 152/192 tails regress. Every light and
camera mean is negative.

Released point-only C452 truth, opened strictly after both candidates were
fixed and rejected, rules out poor interpolation as the main cause. Midpoint
resampling changes track-only median nearest-truth distance
`0.001842→0.001793` world units and normal error `21.91→26.85` degrees. The
combined patch surface improves over fallback from `0.005312→0.004236` world
units and `51.08→45.75` degrees, yet renders worse. Better isolated geometry
therefore still changes radial optical support and per-view compositing
ownership incorrectly. Track matching, construction, fitting, and truth
diagnosis peak at 1.06/1.60/0.09 GiB with zero cgroup swap, memory event, OOM,
throttle, or GPU fault. No experimental code is retained.

This closes post-formation response-track geometry, including broad discs,
shared-view point patches, and point-only resampling. The next bounded route
must estimate dense depth from per-pixel multi-light signatures along
calibrated epipolar lines before any cloud surface or support is formed. It
must retain masks and source observations and must not introduce polygonal
geometry.

The first dense-depth prerequisite is now established. Replacing the spatial
3×3 four-light response patch by only its normalized centre pixel increases
width-128 selection to 2,894 tracks, but released truth rejects it before any
render: median error is 1.000 source pixel, only 49.97% lie within one pixel,
and median normal error is 46.83 degrees. A local lighting signature alone is
not unique enough.

A fixed higher-dimensional control packs the centered log responses of 27
evenly spaced construction lights into the matcher's existing 3×3×RGB
descriptor. This is an audit encoding, not an image format or production API;
the mutual epipolar, match-ratio, three-view, reprojection, construction-hull,
and disjoint-camera validation rules are unchanged. At logical width 64 it
selects 881 observation-unique tracks, 99.32% within one source pixel of
truth at a 0.302-pixel median. At width 128 it selects 2,381 tracks, persists
2,366 oriented sites, reaches 96.18% within one pixel and a 0.482-pixel median,
and keeps median normal error at 21.95 degrees. The resolution replay takes
87.0 seconds for matching and peaks at 1.21 GiB.

Observation uniqueness, not triangulation, is the immediate density boundary.
At width 64, 3,415 candidates already pass mutual multi-view triangulation;
keeping all of them before spatial fusion leaves 3,223 after the
construction-only hull and persists 3,194 oriented sites. Accuracy is
unchanged at 99.32% within one pixel and a 0.299-pixel median. The existing
shared-view component rule forms one coherent component and point-only
resampling yields 7,351 samples.

That cloud is still not a finished surface. After identical material fitting,
standalone mask recall is only 51.66% despite 96.87% precision; foreground
mean loses 2.9378 dB and 171/192 tails regress. Replacing the 12,773 fallback
surfels inside its support produces 24,864 points and improves precision by
0.7243 points, but loses `0.6018/0.8359/0.6322` dB
whole/foreground/covered mean, 1.3662 recall points, and 185/192 foreground
tails. Every light mean is negative and the worst pair loses 2.2104 dB. The
full integration gate peaks at 1.43 GiB with zero cgroup swap, memory event,
OOM, throttle, or GPU fault.

The selected result is therefore correspondence evidence, not a renderable
asset or retained production implementation. Dense disparity must propagate
the verified multi-light matches across low-response foreground while camera
rays and source observations still define ownership. Only after cross-view
consistency and mask-hull validation should the pipeline assign point radii,
normals, and materials.

### Dense 27-light disparity propagation

An ignored follow-up now performs that propagation without loading released
truth during construction or selection. It normalizes the 27-light
log-luminance response at every foreground pixel, seeds each source view from
the triangulated tracks, and searches scalar depth along the source camera ray.
Each proposal must agree in response with four nearby construction cameras,
remain in the construction mask hull, and agree in depth with independently
solved camera maps before ordinary point-cloud fusion. No polygonal geometry
or persisted light-specific appearance is introduced.

At width 128, using all 24 construction cameras produces 21,303 selected
tracks, 69,614 consistent depth pixels, and 9,683 fused C452 surfels. Released
point truth is opened only after selection: median nearest error is 0.594
source pixel, the 90th/99th percentiles are 1.053/1.732 pixels, and median
normal error is 28.72 degrees. A construction-selected circular radius of
`1.68` times the base footprint improves the fixed 192-image construction
score by `+1.7123/+1.0300` dB whole/foreground mean, `+0.0902` recall point,
`+4.9528` precision points, and `+1.6131` dB covered mean. Twenty-four images
still regress and the worst loses 2.1769 dB, so it does not pass the strict
every-image gate.

Only after freezing that candidate, the official 103 held lights by 6 held
cameras are opened. Relative to the established surface, whole/foreground
mean improves `+2.3561/+1.4863` dB and precision improves 6.8410 points. All
103 per-light means and all six per-camera means improve, including every
camera omitted from construction. Recall loses 0.3122 point, however, and
48/618 individual images regress, with a 2.1802 dB worst loss. The worst
construction frame is a dark cast-shadow case where both diffuse models are
too bright while the candidate silhouette is visibly sharper, localizing the
remaining tail to appearance/transport rather than a geometry collapse.
Photo, control, and candidate dumps remain outside git under
`/mnt/data/OLATverse/runs/C452/packed27-dense-radius-1.68-w128-all24-v1/light099-dump/`.

The predeclared C769 construction transfer rejects a universal circular
support rule before held data is opened. Its 8,609 selected tracks yield only
2,793 fused surfels. A global `3.4` radius multiplier recovers raw recall but
leaves 82/192 fitted images worse; an eight-neighbor local radius recovers raw
recall at factor `2.0` but still leaves 69/192 images worse and loses 0.8278
recall point after fitting. C769 held images and polarized truth remain
untouched. Dense multi-light disparity is therefore selected as the upstream
geometry signal, while isotropic support inflation is rejected.

The minimal fusion correction also fails its bounded gate. Every retained
depth pixel already agrees with at least two independently solved camera maps,
so a scratch replay changes only the later voxel merge from factor `1.5` with
two contributing views to factor `1.0` with one. It preserves the same 26,972
verified depth pixels and grows C769 from 2,793 to 7,619 oriented surfels. The
smallest pre-fit local support that beats both control coverage measures uses
`2.2` times eight-neighbor spacing: recall changes
`98.4748%→98.6612%` and precision `77.7121%→79.3418%`.

After identical fitting, whole/foreground/covered mean changes
`+0.5063/+0.0221/+0.2244` dB and precision gains 5.4192 points, but recall
loses 0.5145 point and 94/192 frames regress. A frozen 1.15 post-fit radius
scale restores recall (`+0.0969` point) and still gains 3.0367 precision
points, yet reduces the whole-frame gain to 0.3068 dB and leaves 90/192
foreground frames worse. Four of eight light means and ten of 24 camera means
are negative; the worst pair loses 2.9577 dB. This closes redundant voxel
filtering and isotropic support resizing together. C769 held data remains
unopened. The next experiment must project each accepted sample's actual
source-pixel footprint onto its tangent plane and retain that anisotropic
point support through material fitting. It remains a point cloud, with no
polygonal fallback or per-light stored appearance.

Two stricter support controls close that geometry branch. A layer-aware voxel
merge retains a second aggregate when normals disagree by more than 60 degrees
or their normal-plane depths differ by half a voxel. The same 26,972 verified
depth pixels yield 18,726 oriented samples, 7,655 occupied voxels, and 9,698
surfels; 1,802 voxels genuinely split into multiple sheets. A construction-
selected local radius of `2.25` ties pre-fit recall within 0.0028 point while
gaining 2.8581 precision points. After identical material fitting, however,
whole/foreground/covered mean changes `+0.4670/-0.1541/+0.1037` dB, recall
loses 0.8891 point, and 98/192 frames regress. The worst loses 3.6184 dB.
Conflicting layers exist, but preserving them does not recover a transferable
surface.

The existing anisotropic PBR Gaussian path then isolates support shape without
adding an operation or shader. Relative to the same 7,619-center isotropic
Gaussian, an area-preserving eight-neighbor covariance ellipse changes
whole/foreground mean by `-0.0625/-0.0263` dB, recall/precision by
`+0.2583/-1.2159` points, and loses 109/192 tails. Aligning only points within
2.5 pixels of a construction-mask boundary is less negative, but still changes
whole/foreground by `-0.0325/+0.0069` dB, recall/precision by
`+0.1343/-0.4348` points, and loses 90/192 tails. Neither anisotropy heuristic
is retained.

The corresponding fixed visibility replay identifies the next dependency.
Without visibility, the recall-restored dense candidate beats the established
C769 surface by `+0.3068/+0.0200/+0.1592` dB whole/foreground/covered and
3.0367 precision points, but has 90/192 foreground regressions. Rendering both
frozen models with the existing finite-distance visibility ray changes those
gains to `+0.6044/+0.4282/+0.4853` dB and cuts regressions to 52/192. Seven of
eight light means and 20/24 camera means become positive. The remaining worst
loss is 1.7836 dB, so visibility remains opt-in and C769 held data stays
closed. A local worst-frame dump shows both no-visibility models much brighter
than the dark, cast-shadowed photograph while the dense silhouette is sharper.
The next bounded experiment is therefore a low-order point-local transfer
basis fitted on one subset of construction light directions and validated on
another. It may model smooth indirect/shadow residual, but must not store a
light-indexed table or alter geometry; only an internal split can justify a
later held-light gate.

That first transfer oracle is rejected without indexing or scoring held data. It
freezes the 7,619-point C769 geometry, refits normals and diffuse material on
27 evenly spaced construction lights and 16 construction cameras, and fits a
per-point RGB degree-2 SH residual from calibrated point-to-light direction.
Eight different construction lights and eight disjoint construction cameras
form the validation cross-product. The basis contains no light ID, and its
evaluation changes material only; mask coverage is therefore exactly fixed.

With relative ridge `0.1`, fitted-light/disjoint-camera whole/foreground PSNR
changes by `-0.0772/-0.0767` dB and 172/216 foreground images regress. On the
disjoint-light/disjoint-camera split it changes by `-0.1040/-0.1039` dB and
54/64 images regress, with a 0.4457 dB worst loss. Raising the predeclared
ridge to `1.0` does not reverse the result: disjoint-light whole/foreground
changes by `-0.0685/-0.0718` dB and 59/64 images regress. The implementation
uses the same world-space finite-light direction and SH ordering as the
renderer; the likely mismatch is its center-pixel, one-owner observation
solve versus the renderer's depth-ordered overlapping-surface blend.

No runtime field, shader, operation, dependency, or model format is added.
The next bounded oracle keeps the same frozen geometry and light/camera split,
but places the directional coefficients inside the existing sparse
compositor-aware material equations. It must improve every disjoint-light
aggregate and tail before C452 or any official held observation is opened.

The exact-blend follow-up validates the formulation but not yet a production
representation. A new construction-only loader ensures that these subsequent
runs do not open or decode held-camera photographs at all; the earlier general
loader had decoded those files even though the fit and score never indexed
them. Replaying C769 through the stricter loader is byte-for-byte metric
identical.

The oracle copies the established depth-ordered compact-disc blend, keeps the
7,619 centers, normals, radii, opacity, and material ownership frozen, and
solves four degree-1 SH coefficients per point and RGB channel inside the
coupled pixel equations. At relative ridge `0.1`, its fitted linear RMSE moves
`0.011458→0.005750`. On fitted lights and disjoint cameras it gains
`+1.5151/+1.7971` dB whole/foreground with one 0.00009 dB regression among
216 images. More importantly, the eight disjoint construction lights by eight
disjoint cameras gain `+1.3666/+1.6019/+1.3901` dB
whole/foreground/covered, and all 64 foreground images improve. Coverage is
identical by construction. The scope peaks at 1.1 GiB with no swap, memory
event, throttle, or GPU fault.

The frozen C452 transfer is mixed. Ridge `0.1` gains
`+0.5769/+0.4791` dB whole/foreground on the same 64-image internal split but
regresses 13 images by up to 0.6234 dB. The one predeclared stronger ridge of
`1.0` reduces this to five regressions and a 0.1802 dB worst loss while
preserving `+0.4080/+0.4037/+0.4205` dB whole/foreground/covered gains. Every
light mean and every camera mean is positive; the failures are isolated
light/camera interactions. A four-coefficient achromatic transport factor
behaves almost identically, so light-dependent colour is not responsible.
Rendering the RGB candidate with existing visibility reduces its gain and
leaves 13/64 regressions, rejecting a visibility toggle as the repair.

This is useful evidence, not a selected field. The fit clamps roughly 4% of
C452 channel evaluations at ridge `1.0`, and its unconstrained per-point
coefficients can redistribute energy between nearby overlapping supports.
The bounded spatial follow-up connects each point to at most four nearby
normal-compatible points. With relative spatial strength `0.25`, a scalar
degree-1 field reduces coefficient magnitude but leaves the same five C452
tails. A degree-2 RGB field improves all 64 C769 disjoint-light images by
`+1.2390/+1.4512/+1.2601` dB whole/foreground/covered. On C452 it gains
`+0.6412/+0.7045/+0.6657` dB, but light 156 still loses Cam09 and Cam16 by up
to 0.2863 dB. More SH order and neighbor smoothing therefore improve the
mean without resolving the contradictory camera evidence.

A broad exact-compositor material oracle jointly refits diffuse albedo and
the existing GGX F0 at fixed roughness `1.0`, retaining explicit zero F0 for
diffuse-only points. It gains `+4.4130/+4.3181` dB whole/foreground on C769's
disjoint split and improves every image, but changes C452 by
`-0.2864/-0.2497` dB and regresses 60/64 images. This object-specific result
rejects a larger direct material fit. A brightened dump of the remaining
C452 light-156/Cam09 failure is under
`/mnt/data/OLATverse/runs/C452/blended-degree2-sp025-light156/`: the photograph
contains a dim but coherent back-facing statue while both direct-light renders
contain only sparse grazing slivers. The current directional-albedo family
still multiplies its result by `max(n·l, 0)` and therefore cannot represent
that observation.

The final response-family oracle removes that cosine only from an additive
low-order transport residual while retaining the production direct GGX base.
It uses the same exact compositor, 27 fit lights by 16 fit cameras, eight
disjoint construction lights by eight disjoint construction cameras, relative
ridge `1.0`, spatial strength `0.25`, and a global half step. The CPU evaluator
includes all pixels rather than only the fit mask and is gated against the
production GPU baseline: maximum score disagreement is 0.0069 dB on C769 and
0.0014 dB on C452.

With pooled linear fitting, degree-2 light-only transport improves every C769
disjoint image by `+1.0150/+1.1794/+1.0420` dB, but C452 gains only
`+0.1019/+0.1303/+0.1149` dB and loses 10/64 images, one by 6.8037 dB.
First-order sRGB weighting reduces that C452 worst loss to 0.3128 dB and gains
`+0.3403/+0.3790/+0.3483` dB, but the same 10/64 tail count remains. A
continuous degree-1 light-direction by degree-1 view-direction field also
passes every C769 image (`+0.8485/+0.9763/+0.8705` dB) and fails C452:
`+0.2824/+0.3341/+0.2871` dB with 13/64 regressions, worst 0.9685 dB. This is
not a light-indexed table, yet its additional view capacity makes transfer
worse.

No runtime field, shader, operation, dependency, or file-format change is
added for any of these rejected oracles. C713 and C777 remain unopened by this
response family. The next experiment must move the physical evidence upstream:
jointly optimize oriented point geometry with visibility/indirect transport
through complete composited images, weight the per-image tails explicitly, and
hold both light and camera directions out. Another frozen-surface response
field is not justified.

The direct photometric-normal follow-up closes three narrower alternatives on
C769 before C452 is opened again. Dense disparity supplies 16,826 normal
samples from the 16 fitting cameras. Assigning samples to the established
7,619-point surface, requiring at least two samples at 0.9 directional
concentration, and accepting only candidates within 30 degrees of the current
normal changes 3,027 points. After the same exact-blend material fit, the eight
construction lights by eight disjoint construction cameras gain
`+1.0864/+1.1816/+1.1236` dB whole/foreground/covered, but four of 64
foreground images regress; light 099 / Cam09 loses 0.4046 dB. Existing finite
visibility changes the mean without removing that tail.

A stronger independence check estimates each pixel normal separately from two
interleaved construction-light folds. It requires two assigned samples per
fold, 0.9 concentration inside each fold, agreement within 15 degrees between
folds, and the same 30-degree geometry bound. Only 285 points remain. A 25%
normal step still gains `+1.0749/+1.1746` dB whole/foreground but loses four
of 64 images, worst 0.1453 dB; a 12.5% step loses three, worst 0.1410 dB.
This is not support-intersection coupling alone. An exact CPU replay uses the
unchanged geometric discs and coverage while applying the candidate as a
separate shading normal. It gains `+1.0386/+1.1759/+1.0776` dB on a second
eight-light construction split but regresses 10/64 pairs, worst 0.1036 dB.
The control agrees with the production GPU within 0.0008 dB.

Using the fold-consistent evidence before point formation is also mixed. A
strict 16-camera build keeps the other eight construction cameras out of
matching and depth. Its ordinary fine fusion contains 4,640 cells. Selecting
one two-camera, 30-degree-consistent photometric cluster per voxel replaces 243
cells and adds 12, falling back exactly everywhere else. After identical local
radius and normal/material fitting, the 4,652-point candidate changes
whole/foreground/covered by `-0.0131/+0.0035/+0.0091` dB and recall/precision
by `+0.4965/-0.1942` percentage points. It regresses 29/64 images, worst
0.1953 dB. No normal field, alternate fusion path, or threshold follows.

The repeated bad frames are not explained by camera exposure. Across the same
eight lights, Cam09's per-image optimal linear gain spans `0.4540--1.9006`.
Under light 099 the photograph contains 1.9905 times the rendered foreground
luminance, yet its least-squares gain is 0.7125 and its spatial correlation is
only 0.5957: the render has too much energy in narrow grazing slivers and too
little over the broad dim backside. A camera scalar cannot make those changes
at once. Local-only outputs are under
`/mnt/data/OLATverse/runs/C769/packed27-photometric-normal-*`,
`packed27-dense-fit16-vf1-min1-*`, and
`photometric-normal-angle30-screen-dumps/`. The scoped runs peak at 1.23 GiB
with zero swap, OOM, throttle, or GPU fault.

Two direct attempts to generalize geometry across fitted lights are rejected.
The existing physical Gaussian multi-light objective, replayed from the
improved C452 surface, changes its construction audit loss
`0.002379→0.004154`; its guard restores geometry before serialization. The
restored Gaussian remains below the surface on held/held whole-frame mean
(`34.6499` versus `35.4370` dB), recall (`96.227%` versus `98.848%`), and
precision (`89.501%` versus `90.139%`).

A four-light log-response field is a stronger but still mixed control. It
uses frozen construction lights `0/102/204/306`, centers log luminance across
the four observations so scalar albedo cancels, and stores three response
coordinates only as temporary geometry supervision. The resulting response
appearance is not copied into the relightable surface. On C452, held-camera
full-bright whole-frame mean/worst improves `24.35/23.60→25.25/24.41` dB,
foreground `23.52/22.81→23.72/23.31` dB, and precision `86.3→88.5%`, with
recall tied at `99.7%`. After two material rounds, held/held whole-frame
mean/worst improves `35.4370/29.3903→35.8559/30.7297` dB and precision
`90.139→92.235%`, but foreground mean changes `27.4816→27.4738` dB and recall
`98.848→98.815%`.

The frozen C276 replay rejects promotion more clearly: held-camera
full-bright whole-frame mean/worst improves `23.61/20.99→24.44/21.64` dB and
precision `89.1→90.9%`, while foreground regresses
`24.22/22.15→24.08/21.96` dB. Coordinate interpolation is not a safe rollback:
25/50/75% blends between the two healthy foam endpoints all collapse support,
because intermediate Voronoi ownership is not meaningful. Keep the one-light
recipe. Any next multi-light geometry proposal must select coherent whole
topology states or construction-validated spatial regions rather than blend
point indices globally.

The first spatial selection proxy is also rejected. A construction-only
`4×4×4` screen freezes the best C452 full-bright region (region 55, 147 of
16,384 cells) from the four-light response endpoint and leaves everything else
at the aligned-light baseline. Static held-camera foreground improves
`22.81→22.83` dB at its worst view, but the exact finite-light compositor
lowers all four foreground means: fitted/fitted `27.5976→27.5971`,
fitted/held `27.4855→27.4783`, held/fitted `27.5443→27.5440`, and held/held
`27.4816→27.4723` dB. Several minima improve, but a static full-bright image is
not a valid selector for local finite-light transport. Future region proposals
must be selected through construction-light production renders.
The follow-up does that without opening either official held axis: every raw
region receives the same two-round fit from eight fixed construction lights
`3/51/99/153/201/255/303/309`, then all 192 light/camera images are scored
individually by the production renderer. No non-empty region preserves every
image. Region 52 has the best combined mean gain (`+0.0138/+0.0191` dB
whole/foreground) but loses `0.423/0.546` dB on its worst affected image;
region 55 gains `+0.0040/+0.0032` dB while losing `0.154/0.202` dB on a tail.
The reusable scorer now exposes per-image finite-light results so later,
finer proposals can be rejected before their pooled average hides that loss.

An `8×8×8` subdivision of region 52 closes the axis-aligned follow-up. Its
eight children partition all 131 foam cells exactly; five produce a
byte-identical extracted surface. The three effective children contain
25/75/2 cells. After the identical two-round material/normal fit, their
whole/foreground mean changes are respectively `+0.0025/-0.0003`,
`+0.0023/+0.0058`, and `-0.0002/-0.0047` dB. Their worst individual-image
changes are `-0.357/-0.274`, `-0.462/-0.492`, and `-0.132/-0.164` dB. Thus
even the useful 75-cell foreground mean is paid for by a much larger
construction-image loss. The extraction command was recovered and reproduced
both the surface and environment byte-for-byte before this screen; the eight
candidate extractions peaked at 169 MiB, and exact scoring peaked at 990 MiB,
with zero swap, cgroup event, or GPU fault. Do not recurse into smaller boxes
or add a spatial-selector option. The next proposal must remove the mismatch
between fitted equations and the finite-light production compositor.

A display-referred material control reaches the same conclusion from the
appearance side. One Gauss-Newton direction minimizes sRGB error through the
existing sparse particle blend; a 5% trust-region step was frozen using only
C452 fitted-light/fitted-camera results. On C452 it improves every mean by
`0.035–0.043` dB and preserves all reported minima except fitted-light/held-
camera foreground, which changes `19.5091→19.5090` dB. The untouched C276
transfer is:

| Lights / cameras | Baseline whole / foreground | 5% display step whole / foreground |
| --- | ---: | ---: |
| fitted / fitted | `35.4371 / 27.4939` dB | `35.4829 / 27.5441` dB |
| fitted / held | `34.6959 / 27.6798` dB | `34.7384 / 27.7235` dB |
| held / fitted | `35.3807 / 27.4288` dB | `35.4264 / 27.4789` dB |
| held / held | `34.6517 / 27.6240` dB | `34.6941 / 27.6681` dB |

Coverage is identical and all C276 held-light minima improve or tie, but the
fitted-light/held-camera foreground minimum regresses `19.7062→19.7014` dB.
Scaling every update by the fraction of construction cameras that observe its
material reduces that loss to `19.7062→19.7048` dB while retaining smaller
mean gains; it does not close the gate. Both experimental implementations are
removed. Optimizing pooled pixels in the correct transfer function is still
not enough when material ownership changes with camera. A next material pass
needs per-image production-render acceptance and a construction-camera
cross-validation split, not another global objective or trust knob.

A cloud-only 65,536-cell geometry arm at width 256 uses 19.7 million optimizer
rays without truncation and improves static held-camera whole-frame PSNR
`25.84→27.43` dB, foreground `20.03→21.58` dB, recall `85.9→90.7%`, and
precision `74.0→81.2%`. Its 31,932-surfel relight fit at width 64 improves the
held/held whole-frame mean/worst `34.30/27.10→34.48/27.72` dB and precision
`90.7→92.0%`, while foreground remains `27.23/19.61` dB and recall changes
`99.4→99.0%`. Keep it as a promising geometry arm, not an all-metric-selected
replacement.

The matched 31,932-particle measured-light Gaussian continuation is rejected.
Its deterministic audit, cycling across every fitted light and camera, changes
`0.003033→0.010926`; accepting it turns the object into a broad low-precision
fog. The pipeline now restores geometry and normals whenever that audit
regresses. On the official held-light/held-camera cross-product, the restored
Gaussian reaches `33.37/27.45` dB whole-frame mean/worst,
`25.86/19.64` dB foreground, `96.2%` recall, and `92.8%` precision. The
corresponding surface cloud reaches `34.46/27.68`, `27.22/19.61`, `99.0%`, and
`92.0%`. The Gaussian is therefore a safe serialized control but is not the
selected quality result.

The loader now decodes cameras in parallel. On this 12-thread machine that
reduces a complete width-64 fit/evaluation from 30m06s to 5m37s with a
byte-identical output model. The coupled surface solver also discards exact
zero-response light terms: on the 31,932-surfel arm it stores 14,760,383 rather
than 28,693,080 terms and again writes a byte-identical model.
Foreground masks are immutable and now shared across every light from the same
camera. A matched C276 width-64/two-round replay remains byte-identical while
peak cgroup memory falls from 2,095,202,304 to 1,888,456,704 bytes (9.9%); it
uses no swap and records no memory or OOM event.

One source-format caveat remains. `avif-rust 0.0.7` strictly decodes 6,218 of
the 6,240 extracted C276 files; 22 valid files exercise decoder paths it does
not yet implement correctly. The selected C452 protocol has one such file;
eight more failures found in a full partial scan belong to excluded polarized
camera 39. Both libaom and dav1d agree on these images. Disabling entropy
validation produces corrupted blocks and is rejected. The audit preserves the
raw release and losslessly re-encodes only rejected files in hard-linked
derived trees under `/mnt/data/OLATverse/derived/`; the Rust loader itself has
no external-process fallback. Fix the decoder upstream rather than silently
accepting damaged pixels.

Current artifacts are under `/mnt/data/OLATverse/runs/C276/`:

- `olat-surface-128-r3c/scene.rply` and `images/` — complete production-size
  scalar fit and 618 held-light/held-camera reference/render pairs;
- `foam-static-64k-256/foam.ply` and `novel_00.png` … `novel_04.png` — denser
  static cloud and nearest-camera novel-view strip;
- `surface-64k-256-vf2/scene.rply` — denser relightable surface initializer;
- `olat-gaussian-64k-fit64-r1-guard/scene.{rply,ply}` — guarded matched surface
  and Gaussian control; the regressed candidate was restored before writing.
- `olat-directspec-disabled64-r2/scene.rply` — exact diffuse-preservation
  control after adding runtime GGX; all four reported camera/light quadrants
  match the selected 65,536-cell surface baseline.
- `olat-visibility64-r2/scene.rply`, `images/`, and `held-contact.png` — the
  opt-in visibility candidate, all 618 held/held reference-render pairs, and a
  four-light local contact sheet. These stay outside git with the dataset.

Fresh-object artifacts are under `/mnt/data/OLATverse/runs/C452/`:

- `surface-16k-vf2/scene.rply` and `olat-baseline64-r2/scene.rply` — unchanged
  16,384-cell geometry and its exact finite-light control;
- `surface-16k-vf2-light000/scene.rply` and
  `olat-light000-baseline64-r2/scene.rply` — the selected aligned-light surface
  and complete no-visibility gate;
- `olat-light000-visibility64-r2/` — latest combined opt-in transport result
  with all 1,236 held-light/held-camera reference-render images and
  `held-contact.png`.
- `olat-light000-physical-gaussian64-r1/` and the `response4` directories —
  rejected physical-Gaussian, response-field, and endpoint controls retained
  for local audit only.

Dataset imagery remains outside version control. Representative OLATverse
reference/result pairs may be checked in only after the release terms
explicitly permit redistribution; the public repository currently states no
dataset licence.

## Compact calibrated route: LUCES-MV

LUCES-MV remains the compact regression gate alongside OLATverse. The official
calibrated release provides twelve poses
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
group, shader entry, or point-light-specific pipeline. Reconstructed point-light
materials are explicitly diffuse-only; the runtime can evaluate GGX and an
opt-in visibility ray, while environment sampling is bypassed. Complete
image-space scores are:

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

The optional finite-light visibility ray is backend-mixed here. Every Gaussian
whole/foreground mean and tail improves; held/held moves from
`34.01/32.66;24.13/22.66` to `34.89/34.23;24.62/23.14` dB. The scalar
fitted-light foreground means regress by 0.09--0.20 dB even though their tails
improve. It therefore remains opt-in rather than changing the LUCES default.

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

Finite-light visibility is rejected on Bear. The held/held scalar row falls to
`29.08/26.04;20.75/17.15` dB, and every other scalar quadrant also regresses.
This confirms that a valid shadow ray is not automatically a better image when
the fitted surface layers and material solve assumed unshadowed transport.

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

The signed camera-group control closes that follow-up without adding production
code. A zero-forward scalar moves each Gaussian only along its fitted
second-round displacement inside the graph. After geometry is frozen, ordinary
SGD accumulates the negative directional derivative separately for four
interleaved construction-camera groups. A component qualifies only when every
group supports the move and every two-sigma footprint sample remains inside
the masks. Pot2 retains components of 14 and 8 points. Their threshold-free
22-point union improves all six selection metrics and five validation metrics:
validation mean/foreground mean-tail reaches
`33.742951/24.222507/20.059754` dB and recall/precision improves, but whole
worst moves `30.141529→30.141433` dB. The strict gate rejects it. The temporary
branch adds roughly 200 lines and an extra frozen audit pass, so it is removed
without running independent objects. Camera consensus is not enough while its
audit sees only one 512-pixel sequence per view; the next bounded control must
increase hard-ray coverage or put a tail-aware term in the physical objective,
not add another component selector.

Three construction-only sampling controls close the coverage half of that
choice. Pot2 foreground occupies 10.07% of its fitted crop. An exact 50/50
foreground/background batch improves foreground quality and recall but lowers
whole-frame selection/validation means by `0.275/0.304` dB and precision by
roughly 3.3 points. Weighting those samples by their uniform-image probability
removes most of that bias, but selection precision still falls
`95.272→94.767%` and validation worst/precision moves
`30.141536→30.134948` dB and `95.139→94.868%`.

Uniform no-replacement coverage gives the stronger diagnostic. A deterministic
cyclic 512-ray lattice improves every Pot2 selection and validation aggregate;
whole-frame means reach `34.786671/33.836758` dB and foreground tails reach
`21.402112/20.273102` dB. Cow rejects the identical rule: selection recall
falls `83.742→82.864%`, while validation whole mean/worst falls
`31.749391/28.942864→31.700264/28.861278` dB, foreground mean/tail falls
`20.853258/18.220257→20.810467/18.127553` dB, and recall falls
`84.046→83.130%`. Independently jittering the same disjoint strata already
loses Pot2 selection tails. No official camera or excluded light is opened.
All sampling and graph-weight prototypes are removed after scoped runs stay
below 1.1 GiB with zero swap or GPU fault. Broader mean-loss coverage is not a
tail objective; the next control must freeze and optimize a per-camera hard-ray
set rather than tune another sampler.

Frozen residual tails do not fix the transfer. A four-batch/four-light audit
freezes each Pot2 camera's top pixel decile before optimization; replacing one
quarter of every batch with those pixels lowers selection worst
`31.238324→31.227836` dB and validation whole/foreground worst
`30.141536/20.059379→30.024416/20.039498` dB. Selecting the worst quartile of
cameras instead and giving them a 25% balanced-light continuation makes every
validation aggregate better—whole mean/worst reaches
`33.914956/30.307576` dB and foreground tail reaches `20.347184` dB—but
selection worst falls to `31.186001` dB and recall falls
`90.920→90.625%`. The independent split therefore rejects the apparent tail
gain before Cow or any official data is opened. Both ray-only audit paths and
all continuation code are removed. The next useful measurement is camera-group
gradient conflict at the fixed checkpoint, not another sampling or weighting
rule.

That gradient audit finds a real two-object conflict. Four interleaved camera
groups each accumulate four independent 512-ray batches under four evenly
spaced fitted lights without an optimizer step. Pot2 selection group 2 versus
validation group 3 has position/normal flat cosine `0.142/0.251`, with
43.9%/34.5% negative pointwise dots. The median minimum cosine across every
group pair is `-0.751/-0.875`. Cow's corresponding minima are
`-0.721/-0.971`; its normal groups 1/3 have global cosine `-0.328` and 81.4%
negative dot mass.

Two threshold-free uses of that evidence fail Pot2. Keeping standard fitted
centers/normals only where every group pair agrees lowers selection whole and
foreground mean to `34.648098/25.198727` dB and validation whole mean to
`33.720437` dB. Fixed-order PCGrad projection of the existing Adam displacement
still lowers selection mean/worst to `34.667672/31.194666` dB and validation
mean/worst to `33.721094/30.105555` dB. No complete render ranks a point and no
second object or official split is opened after failure. The temporary audit,
result type, projection, and tests are removed. Gradient conflict is a symptom
of the current image model, not permission to discard contradictory views; the
next audit should correlate it with frozen light visibility on Pot2 and Cow
before any new shadow-aware training path is considered.

That visibility audit rejects shadows as the immediate explanation. It freezes
the checkpoint and approximates the sampled near lights by directions from the
point-cloud center, then evaluates the existing point-cloud shadow test without
an optimizer step. Camera-group openness varies for 82.07% of active Pot2
points and 79.26% of Cow points, so the test has ample signal. Nevertheless,
its Pearson correlation with minimum-pair gradient conflict is only
`0.100/0.073` for Pot2 position/normal gradients and `0.076/0.007` for Cow.
The same two-object gate that protected the optimizer therefore rules out a
shadow loss. Peak scoped RSS is 1.1 GiB for Pot2 and 265.3 MiB for Cow, with
zero swap or cgroup memory events. All temporary gradient-audit code is removed.
The next diagnostic asks whether conflict follows which cameras actually own
visible compositing support for each point, rather than which lights can reach
it.

Exact compositing ownership is not the missing explanation either. The audit
replays those same rays through the sorted CPU candidate oracle, accumulates
front-to-back `T×alpha` contribution per point and camera group, and normalizes
by the sampled ray count. Fully 95.12% of active Pot2 points and 97.20% of Cow
points retain at least 10% of their peak ownership in every group. Correlation
between all-group ownership imbalance and position/normal conflict is only
`0.043/0.030` on Pot2 and `0.047/0.033` on Cow; measuring imbalance only for
each point's most contradictory gradient pair remains below `0.059`. The
least- and most-imbalanced quartiles differ only slightly as well. Scoped peak
RSS is 1.06 GiB and 262.8 MiB respectively, with zero swap or memory event.
The temporary ownership/gradient API is removed. The surviving hypothesis is
view-dependent reflectance omitted by the diffuse-only geometry graph, which
must be checked from frozen residual/half-vector evidence before adding BRDF
capacity.

A frozen non-Lambertian audit also fails its predeclared gate. It removes each
surfel's mean residual, evaluates calibrated Blinn half-vector proxies with
powers 1--128, and requires one lobe to explain at least 5% of both individual-
observation and four-camera-group variance on both objects. Pot2's best group
result is 4.42%; Cow reaches 8.30%, but individual observations correlate in
the wrong direction on both objects for every power (`-0.306…-0.083` and
`-0.105…-0.030`). The audit uses 128,184 Pot2 and 90,648 Cow construction
observations, never opens a complete render or held split, peaks below 116 MiB,
and uses no swap. No specular parameter is added. The next known mismatch is
earlier in the pipeline: the CPU normal/material observation pass gives the
entire sampled pixel to every accepted surfel center when several centers
share that pixel. Its existing overlap diagnostics can establish whether that
duplication is large enough and consistent enough to warrant a responsibility
experiment.

The existing observation diagnostics reject the simplest responsibility
change. Before reading the result, the gate required at least 25% of accepted
samples to share their center pixel with another accepted surfel on both
objects. Pot2 reaches only 16.10% and Cow 15.14%, so exclusive center ownership
is not tested. The same pass exposes a different mismatch without authorizing
a fix: 100% of sampled center pixels lie under multiple disc footprints, with
21.54/22.42 supports on average and 38 at worst. A mixture-aware solve could be
relevant, but aggregate overlap is not causal evidence. The next audit must
attribute residual or camera-group gradient conflict to per-surfel mixture
complexity on both objects before a compositing responsibility is allowed.

Per-surfel attribution rejects that broader rule too. Before running it, the
gate required support multiplicity to correlate at least `0.20` with normalized
physical residual and the most-mixed quartile to be at least 10% worse than the
least-mixed quartile on both objects. Pot2 measures only `0.030` and 1.53%.
Cow reaches `0.112` and 10.38%, failing the correlation condition and the
two-object requirement. The scoped Pot2 pass peaks at 1.06 GiB and Cow below
116 MiB, both without swap. The temporary per-surfel support counter is
removed. Universal overlap is therefore a representation fact, not evidence
for demixing the CPU observations. The next frozen attribution separates each
surfel's residual into camera- and light-index effects; only a repeated dominant
axis can decide whether to return to correspondence or image response.

The residual decomposition finds no repeated axis. Pot2 attributes 15.19% of
within-surfel variance to camera index, 8.14% to the predeclared camera groups,
and 19.36% to light index. Cow attributes 25.48%, 16.93%, and 11.93%
respectively. The gate required one axis to explain at least 20%, at least
twice the competing axis, on both objects. Cow is camera-dominant; Pot2 is
coupled and slightly light-heavy. Neither a geometry-only nor light-only
parameter family is licensed. The next bounded experiment instead moves one
physical quantity through the correct coupled observation model: optimize
shared diffuse albedo inside the exact Gaussian compositor while holding every
geometry, support, normal, and calibrated-light value fixed. It must use
separate construction train/selection/validation camera groups and pass every
complete Pot2 metric before Cow is opened.

That coupled albedo bound is rejected before validation. An audit-only graph
embeds the existing shared material table through the exact particle-to-
material indices and performs 2,400 deterministic 512-ray updates on camera
groups 0/1, with geometry, covariance, opacity, normals, and all light values
frozen. Its fixed audit loss moves `0.002753→0.003158`. On group-2 selection,
the smallest 12.5% blend moves whole mean/worst
`34.712337/31.238324→34.664672/31.198527` dB and foreground mean
`25.257424→25.214368` dB; only foreground worst improves. Larger blends get
monotonically worse. Group-3 validation, Cow, and every official split remain
unopened. The graph-construction flag, material parameter, and audit API are
removed.

This closes the one-site parameter branch: sampling, tails, gradient surgery,
shadows, camera ownership, specular response, observation demixing, residual-
axis attribution, and exact-compositor albedo all fail their predeclared
controls. The next quality route is already represented by the cloud model,
not a new geometry type: make the released eight-site surface-detail backward
practical at production batches. Preserve its exact checkpoint result while
reducing the current roughly 6× step cost to at most 2× baseline; only then
repeat the two-scene held-view/held-light gate.

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

Reading is the fifth untouched object control. Its published camera matrices
contain small calibration scale/shear: the largest observed orthogonality error
is `0.0069234`, beyond the loader's original rigid-pose tolerance. The importer
now preserves matrices inside the old `0.001` bound exactly, projects only the
accepted `0.001..0.01` band onto `SO(3)`, and still rejects material
non-rigidity. It then imports all 20 cameras and the same predeclared 32 lights.
The ordinary 16k route trains at `28.5317/28.0048` dB on fitted/held cameras,
extracts 1,479 point surfels, and retains 1,441 fitted Gaussians. Its independent
held-light/held-camera baseline is:

| Backend | sRGB mean/worst | Foreground mean/worst | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| Surface | 28.02 / 25.82 dB | 18.59 / 16.25 dB | 96.2% | 90.4% |
| Gaussian | 28.63 / 26.26 dB | 18.46 / 16.08 dB | 89.8% | 95.9% |

The images preserve gross light direction and silhouette but visibly blur the
book, face, and clothing, matching the low foreground tails. The unchanged
eight-pass calibrated-albedo diagnostic finds 44 unique tracks and three
components of 14/7/7 points. Cloud-only midpoint densification turns the latter
two into 16-surfel proposals without introducing polygons. On four selection
cameras, component 1 at fixed response `0.5` improves all six complete-render
measures. After freezing that choice, disjoint internal validation moves
whole-frame `29.332948/25.599171→29.333330/25.599432` dB, foreground
`19.015456/15.359089→19.015939/15.359382` dB, and recall
`90.759918%→90.760760%`, but precision slips
`96.200196%→96.200024%`. The strict gate therefore rejects it and the candidate
never sees the official held cameras or lights.

A distinct frozen follow-up inserts exactly that selected component into the
unfitted 1,479-surfel cloud, before the unchanged three-round calibrated
normal/material solve and 2,400-update Gaussian continuation. Only after the
result is serialized does it open the official split. The candidate retains
1,455 rather than 1,441 Gaussians and raises recall by `0.014..0.027` percentage
points, but lowers precision and every Gaussian whole-frame/foreground mean and
tail. On held lights and held cameras, whole-frame moves
`28.632456/26.257141→28.618959/26.230927` dB and foreground moves
`18.465189/16.084684→18.452256/16.055984` dB. Early sparse seeding is therefore
rejected without response or optimizer tuning against the official result.

A final diagnostic asks whether calibrated photometric normals can turn the
verified tracks into the denser connected layer that sparse seeding lacks. In
each matching camera, it integrates perspective log-depth over an eight-pixel
foreground neighborhood, anchored only by the selected independent tracks.
Samples must agree with at least three independently integrated cameras within
two source-pixel footprints. This yields 642 confirmed camera samples and 500
one-source voxels; requiring two source cameras leaves 117 surfels and three
leaves 12. At fixed `0.05` opacity and the already selected `0.5` response, the
500- and 117-surfel candidates improve every internal PSNR measure and recall
but lose precision. The strict 12-surfel candidate also lowers validation
whole-frame mean `29.332948→29.332918` dB and precision
`96.200196%→96.199364%`. No asset is written and this candidate never sees the
official split. The bounded prototype stays ignored: calibrated normals carry
local shape, but sparse anchors do not make their integrated depths independent
multi-view measurements.

The independent-depth follow-up runs fixed-pose PatchMatch on one diffuse
albedo image recovered from the 24 construction lights at each construction
camera. The production importer now writes those 16 images, a pose-only sparse
bundle, and an explicit nearest-12 graph under `prepared/albedo/`. The four
held cameras are physically absent and none of the eight held lights enters
the albedo solve. No released depth, normal, mesh, or polygonal intermediate is
used. For the Reading run, the construction-only baseline cloud bounds the
PatchMatch interval to `1400..1700`:

```bash
colmap image_undistorter \
    --image_path prepared/albedo/images --input_path prepared/albedo/sparse \
    --output_path dense-albedo --output_type COLMAP
cp prepared/albedo/patch-match.cfg dense-albedo/stereo/patch-match.cfg
colmap patch_match_stereo --workspace_path dense-albedo \
    --workspace_format COLMAP --PatchMatchStereo.depth_min 1400 \
    --PatchMatchStereo.depth_max 1700 --PatchMatchStereo.geom_consistency true

cargo run --release -p blade-volume-train --bin reconstruct -- \
    --sparse prepared/sparse/0 --images prepared/light-004/images \
    --masks prepared/masks --test-list prepared/test-views.txt \
    --dense-workspace dense-albedo --dense-cache target/reading-albedo.bvf \
    --dense-max-points 16384 --min-views 2 --width 128 --far-plane 4000 \
    --stride 1 --no-shadows --output target/reading-albedo.rply
```

Native Rust fusion forms 15,729 explicit depth groups from 82,676 observations;
the construction masks reject 3,593 groups and retain 12,136 oriented cloud
surfels. The unchanged three-round calibrated fit plus 2,400-update Gaussian
continuation retains 12,110 particles. Compared with the 1,441-particle foam
baseline, the default-support Gaussian result is:

| Lights / cameras | Baseline whole; foreground | Albedo-MVS whole; foreground | Baseline recall / precision | Albedo-MVS recall / precision |
| --- | ---: | ---: | ---: | ---: |
| fitted / fitted | `29.6505/25.5845; 19.4264/15.3437` | `30.0833/25.8072; 19.6006/14.5894` | `90.54/95.75%` | `87.85/94.80%` |
| fitted / held | `28.9394/25.3484; 18.8435/15.0977` | `29.2037/25.5739; 18.7358/14.9028` | `89.81/95.87%` | `87.55/95.46%` |
| held / fitted | `29.5578/26.4509; 19.2462/16.3844` | `29.9086/26.8679; 19.3467/15.6956` | `90.54/95.75%` | `87.85/94.80%` |
| held / held | `28.9146/26.4657; 18.7490/16.3072` | `29.1081/26.4715; 18.6129/15.7842` | `89.81/95.87%` | `87.55/95.46%` |

These values replay both arms after fixing Gaussian serialization to copy the
final blended surface materials, superseding the earlier stale-material
matrix. This is the first route to recover visibly sharper book, face, and
clothing structure. It improves all whole-frame means and tails, but loses two
foreground means, every foreground tail, and Gaussian coverage. A single
mechanism-driven `--radius-factor 1.7` control raises held/held foreground to
`18.9626/16.1446` dB and recall to `93.04%`, but precision falls to `94.61%`
and fitted-camera foreground tails remain below baseline. Do not tune that
factor further on Reading. Keep the construction-only data path and repeat the
frozen dense-depth recipe on a second object before fitting support against
construction renders.

The frozen repeat uses Cow, the same 16/4 cameras and 24/8 lights, the same
`320`-pixel nearest-12 PatchMatch graph, the same `1400..1700` depth interval,
and the same default `1.4` support. Its construction cloud projects to
`1491.81..1590.72`, so the interval is not changed. Albedo stereo forms 8,370
groups from 25,936 observations; the construction hull retains 5,582 surfels
and the fit retains 5,559 Gaussians. Unlike Reading, Cow's recovered diffuse
albedo is nearly textureless and the candidate loses every complete metric:

| Lights / cameras | Baseline whole; foreground | Cow albedo-MVS whole; foreground | Baseline recall / precision | Albedo-MVS recall / precision |
| --- | ---: | ---: | ---: | ---: |
| fitted / fitted | `32.8948/28.5829; 21.8885/17.6992` | `31.3220/27.4133; 20.3834/16.5216` | `84.18/96.28%` | `80.65/94.05%` |
| fitted / held | `31.6914/29.1129; 20.5257/18.0409` | `30.1137/27.8276; 18.9055/16.9808` | `84.68/94.50%` | `79.39/93.56%` |
| held / fitted | `32.6694/29.0790; 21.6235/18.2148` | `31.1447/27.4093; 20.1549/16.5293` | `84.18/96.28%` | `80.65/94.05%` |
| held / held | `31.1249/29.0753; 19.9275/17.6860` | `29.6920/28.0406; 18.4529/17.0849` | `84.68/94.50%` | `79.39/93.56%` |

Four fixed diagnostics rule out a simple proxy or support repair. World-space
photometric-normal images recover 4,382 surfels, but lose 22 of 24 values and
the complete gate. A 50/50 display-space albedo/normal proxy recovers 6,502
and also fails.
Appending the albedo and normal layers raises held-camera recall to `92.56%`,
but reduces held-light/held-camera foreground mean and precision to
`18.8584` dB and `91.66%`. Requiring the appended surfels to agree with an
independently recovered normal within 30 degrees in at least three construction
cameras collapses the layer to 2,214 surfels and held-camera recall to
`50.44%`. These are one-shot mechanism controls, not a threshold sweep.

The cross-object gate therefore rejects RGB proxy stereo and post-hoc support
tuning. The follow-up scores the calibrated diffuse-albedo and world-normal
channels directly at observations supporting a depth hypothesis. Each
construction pixel has at most two candidates: the fixed-pose albedo and
world-normal PatchMatch depths. A hypothesis projects into the explicit
nearest-12 construction-camera graph and receives one support vote when it is
within 1% of either source depth. Distinct-camera support wins first; ties use
the sum of diffuse-albedo L1 and world-normal angular residual, normalized by
the object-wide median source residual. Cow's scales are `0.019550` and
`53.842°`; Reading's are `0.017107` and `75.836°`. No held input, released
geometry, or polygonal intermediate participates.

Cow selects 40,891 albedo-only pixels, 20,397 normal-only pixels, and 26,627
pixels with both hypotheses; 15,672 paired pixels choose albedo and 10,955
choose normal. Native fusion forms 10,523 groups from 31,364 observations; the
construction hull rejects 2,892 and retains 7,631 surfels, of which 7,577
Gaussians survive fitting. The complete corrected-material matrix is:

| Lights / cameras | Corrected sparse whole; foreground | Corrected albedo-MVS whole; foreground | Selected depth whole; foreground | Sparse / albedo / selected recall; precision |
| --- | ---: | ---: | ---: | ---: |
| fitted / fitted | `32.8948/28.5829; 21.8885/17.6992` | `31.3220/27.4133; 20.3834/16.5216` | `32.3553/28.2376; 21.4318/17.3966` | `84.18/96.28; 80.65/94.05; 88.79/95.22%` |
| fitted / held | `31.6914/29.1129; 20.5257/18.0409` | `30.1137/27.8276; 18.9055/16.9808` | `31.2870/28.8054; 20.0475/17.9870` | `84.68/94.50; 79.39/93.56; 88.88/95.08%` |
| held / fitted | `32.6694/29.0790; 21.6235/18.2148` | `31.1447/27.4093; 20.1549/16.5293` | `32.1612/28.3741; 21.1766/17.4500` | `84.18/96.28; 80.65/94.05; 88.79/95.22%` |
| held / held | `31.1249/29.0753; 19.9275/17.6860` | `29.6920/28.0406; 18.4529/17.0849` | `30.7608/28.7634; 19.4819/17.9014` | `84.68/94.50; 79.39/93.56; 88.88/95.08%` |

The selector improves every one of the 24 albedo-MVS values, proving that
separate physical features rank the two depth maps better than a proxy image.
It still improves only 7 of 24 values over the corrected sparse control:
recall rises, but most quality tails or precision do not.

Reading selects 74,855 albedo-only pixels, 3,374 normal-only pixels, and 46,211
paired pixels; the paired decisions split 29,962/16,249 between albedo and
normal. Fusion forms 16,095 groups from 75,101 observations, rejects 3,535 in
the construction hull, and retains 12,560 surfels and 12,524 fitted Gaussians:

| Lights / cameras | Corrected sparse whole; foreground | Corrected albedo-MVS whole; foreground | Selected depth whole; foreground | Sparse / albedo / selected recall; precision |
| --- | ---: | ---: | ---: | ---: |
| fitted / fitted | `29.6505/25.5845; 19.4264/15.3437` | `30.0833/25.8072; 19.6006/14.5894` | `30.2367/25.8340; 19.7752/15.2568` | `90.54/95.75; 87.85/94.80; 90.09/95.31%` |
| fitted / held | `28.9394/25.3484; 18.8435/15.0977` | `29.2037/25.5739; 18.7358/14.9028` | `29.3960/25.4442; 18.9653/15.0811` | `89.81/95.87; 87.55/95.46; 89.82/96.00%` |
| held / fitted | `29.5578/26.4509; 19.2462/16.3844` | `29.9086/26.8679; 19.3467/15.6956` | `30.0795/26.9455; 19.5433/16.2275` | `90.54/95.75; 87.85/94.80; 90.09/95.31%` |
| held / held | `28.9146/26.4657; 18.7490/16.3072` | `29.1081/26.4715; 18.6129/15.7842` | `29.3337/26.3998; 18.8798/16.0184` | `89.81/95.87; 87.55/95.46; 89.82/96.00%` |

Reading improves 22 of 24 values over corrected albedo MVS, but only 15 of 24
over the corrected sparse control. A direct sparse+dense union instead reaches
about 97.5% recall at only 90.1% precision on Cow; three-view-only selection
collapses recall to about 74.4%. Skipping Gaussian continuation trades quality
against coverage on both objects. These one-shot controls reject a union,
minimum-view threshold, or static-transfer repair.

This audit also found an independent serialization bug: calibrated fitting
copied PBR materials into the Gaussian before the final blended surface solve,
then wrote that stale table. The production call now occurs after the final
solve. Sparse, albedo-MVS, radius-1.7, and selected-depth arms above were all
replayed with that fix; geometry and coverage are unchanged. The direct selector
remains ignored. The next bounded route must preserve each dense point's source
observations through surface-to-Gaussian conversion and supervise its support
and visibility from construction cameras, rather than tune another image proxy,
radius, or support threshold.

The selector builds peak at 1,079,422,976 bytes; corrected complete fits peak at
561,520,640 bytes. The COLMAP container keeps its independent 7 GiB hard cap.
Every scope reports zero swap, OOM event, or GPU fault. Generated results live
under `target/audit-runs/diligent-mv/{prepared-reading-320,reading-16k,cow}/`;
the four-column held-camera comparison is tracked at
`etc/relight-diligent-reading-feature-depth.png`.

### Provenance-aware Gaussian transfer

The retained fusion cache can be matched back to every selected surfel exactly,
so a second audit tested whether the dense model fails only because those source
observations are discarded during Gaussian transfer. It used 63,145 Reading and
22,040 Cow source observations without opening a new camera or light split.

- Final optimized centres already satisfy every retained one-percent source
  depth interval. A two-pixel reprojection envelope constrains only 82 of 12,560
  Reading points and 30 of 7,631 Cow points, with numerically neutral or negative
  image changes.
- Almost every target Gaussian appears on at least one of its exact source rays
  (12,448/12,560 Reading and 7,352/7,631 Cow), but 58% and 75% are never the
  strongest contributor. Treating composited ownership as exclusive therefore
  removes valid overlapping cloud support and fails both internal splits.
- Shrinking the normal axis to the RMS thickness measured by the source depth
  samples raises precision but removes 6--10 recall points. Conserving optical
  mass exposes a smooth tradeoff. A frozen `0.4` compensation exponent improves
  all twelve Reading internal metrics, then loses both Cow foreground means and
  recall on both splits. Source-mask bounds on the two tangent axes show the same
  precision/recall tradeoff.
- Replacing the compact fitted material table with per-point diffuse albedo from
  exact source pixels improves some Cow validation tails but lowers both
  selection means and tails. The deficit is not primarily discarded diffuse
  appearance.
- Enabling the existing explicit multi-light mask loss at its established 1.5
  weight raises Cow Gaussian recall from 88.8% to 97.7%, but fitted/fitted whole
  and foreground means fall by 0.29 and 0.26 dB. Held/held foreground mean also
  falls. No smaller weight was tuned after this official result.

No production field, loss, shader, or dependency is retained from these
controls. They close post-fit provenance constraints on the selected dense
cloud: centre, normal thickness, tangent footprint, opacity, and diffuse
appearance each move the same support-versus-photometry frontier rather than
fix it. The next geometry proposal must be upstream. Grow one connected dense
point layer from the high-precision sparse reconstruction, requiring consistent
local tangent orientation and shared source-view depth order before surface
extraction. Then fit and score that fixed cloud through the unchanged Gaussian
path. This remains point-cloud reconstruction; no polygonal intermediate is
needed.

That upstream proposal is now closed as well. The fixed Cow graph seeds 1,828
dense points from sparse surfel support and expands to 5,597/7,631 through
28,344 normal/depth-consistent edges observed within two pixels in a shared
source camera. Fitting only that connected layer raises precision but collapses
held-camera recall to 77.34%. Adding 373 sparse surfels not covered by the dense
layer raises recall to 86.94% but collapses precision to 91.75%; PSNR tails are
mixed or worse. The graph and fill rules were not threshold-tuned.

Running the already selected 300-update-per-view masked PowerFoam continuation
on all 7,631 dense points is healthy but does not survive conversion back to a
Gaussian. Its 4,800 updates reduce the static loss `0.036613→0.000773` in 42
seconds, yet all 24 final Cow light/camera values change by at most about 0.0006
dB or 0.0006 coverage points, with mixed signs. This points to a representation
boundary rather than another initializer: the next bounded experiment is an
ignored relightable PowerFoam scorer that keeps exclusive oriented-cell
ownership through PBR shading. Only a Cow and Reading win should justify moving
PBR attributes into backend-neutral `PointCloudModel` storage and the runtime.

The ignored controls and telemetry are under
`target/audit-runs/diligent-mv/{reading-16k,cow}/feature-depth-selection/`.
Their largest scoped peak is 1,170,440,192 bytes, with zero swap, OOM event, or
GPU fault.

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

### Exclusive oriented-cell reference

An ignored CPU reference now traces the same bounded-ball and radical-plane
constraints as the compute-splat PowerFoam path, accelerates support candidates
with a checked sphere BVH, and returns one nearest oriented-cell plane. It then
uses the calibrated finite-light diffuse model and that cell's fitted material;
geometry hits are cached across all 24 Cow lights. This tests the representation
without adding a persisted field, shader entry, operation, or runtime branch.

The unchanged selected Cow surface at its stored three-sigma radii reaches
`97.36/96.48%` selection/validation recall but only `87.02/86.12%` precision.
Its whole/foreground selection score is `29.24/27.21;18.73/16.37` dB, against
the matched Gaussian's `32.65/29.90;21.85/18.79` dB. The production masked
PowerFoam continuation changes no covered pixel and moves appearance only in
the fourth decimal place.

There is one non-tuned radius conversion with a direct semantic justification:
the relight surface stores a Gaussian's three-sigma radius, while a hard cell
is opaque. Scaling by `sqrt(2 ln 2) / 3` makes the hard support boundary equal
the Gaussian's 50%-coverage contour. This restores selection/validation
recall to `88.20/87.34%` and precision to `94.89/94.55%`, close to the
Gaussian geometry, but foreground means remain `18.48/17.81` dB. Re-solving
albedo from eight disjoint construction cameras raises those means only to
`18.96/18.03` dB. Re-solving normals changes 3,474 of 3,531 supported sites,
drops recall to `85.63/83.24%`, and lowers foreground means to
`18.03/17.01` dB. Reading and official Cow splits therefore stay closed.

This also sharpens the semantics of the result. The calibrated finite-light
fit and current runtime point-light path are diffuse-only; roughness and F0 are
not exercised by this gate. The next reference should retain exclusive
point-cloud geometry while interpolating its shading normal and diffuse
material over the local Cech neighborhood. Only a Cow internal pass would
justify backend-neutral storage, direct finite-light specular work, or WGSL.
The ignored run peaks at 360,214,528 bytes in a 4 GiB cgroup with zero swap,
OOM, throttle, or GPU fault.

That final local-attribute check is also negative. Gaussian radial weights over
the owner and its full-radius Cech neighbors raise selection/validation
foreground means to `19.27/18.43` dB; smoothing normals is neutral to slightly
negative. Rebuilding the hard cells from the exact optimized Gaussian centers,
volume-equivalent scales, normals, and material table reaches
`19.31/18.50` dB, still behind the same Gaussian's `21.85/20.63` dB. Its mask
is already close (`88.35/87.34%` recall and `95.20/94.73%` precision), so this
is not a hidden pre-Gaussian-center mismatch. No PowerFoam PBR storage or
shader follows from this experiment.

### Fixed-Gaussian residual attribution

The next Cow diagnostic returns to one persisted final Gaussian rather than
converting it to another point-cloud representation. Production point-light
renders are paired with exact Gaussian opacity from the CPU oracle. Every pixel
is classified as missed foreground, covered foreground, spilled background, or
empty background. Appearance controls keep the same geometry and opacity. To
avoid an in-sample oracle, each per-pixel RGB gain and photometric normal plus
diffuse material is fit on one interleaved set of 12 construction lights and
scored on the other 12, then the folds are reversed.

The production scorer remains the authoritative baseline at
`32.6496/31.5918` dB selection/validation mean. The diagnostic reads its 8-bit
PNG dumps, so its corresponding baselines are lower at `32.4360/31.3694` dB;
all attribution deltas use that same quantized boundary. Correctly covered
foreground contributes `81.36%/83.54%` of squared error. Missed foreground
contributes only `10.18%/9.63%`, and spill contributes `5.08%/4.04%`.

A perfect coverage oracle reaches `33.1548/32.0079` dB, only `+0.72/+0.64`
dB. Cross-light material scaling reaches `33.3466/32.6786` dB. Replacing that
with the held-light prediction from a photometric normal and bounded diffuse
material reaches `35.4959/34.7027` dB, an additional `+2.15/+2.02` dB. This is
an upper bound at image pixels, not a result model, but it selects the next
loss cleanly: fit particle shading normals from light contrast on the exact
final Gaussian while freezing geometry, opacity, and support. The ignored
tool and 384 render/photo dumps are under
`target/audit-tools/diligent_tracks/` and
`target/audit-runs/diligent-mv/cow/fixed-gaussian-residual/`. The 4 GiB scope
peaks at 427,638,784 bytes with zero swap, memory event, OOM, throttle, or GPU
fault.

The particle lift closes the tempting direct follow-up. Each Gaussian is
sampled at its projected center in eight owner cameras, and its exact
front-to-back `T*alpha` responsibility weights a normal/material estimate from
one 12-light fold. Updating all supported particles changes 7,599 of 7,631 and
regresses. Requiring at least three owner cameras and a fixed 15-degree
cross-view consensus changes 183 particles in one fold and 507 in the other.
The first arm improves most values but loses validation whole-frame worst by
`0.0010` dB; the reverse arm regresses broadly. Requiring the two independently
fit normals to agree inside the same cone leaves 106 particles and moves every
selection/validation PSNR aggregate down by `0.0017--0.0098` dB. Geometry and
coverage remain unchanged, so no production code follows.

Released Cow normal maps are then used strictly as evaluation data. The two
12-light photometric estimates have `23.36/24.78`-degree median error over
owner-camera foreground and `16.78/23.41` degrees at responsible center rays.
That estimation error is smaller than the association failure: across 7,556
particles visible in at least three owner cameras, the weighted resultant of
released world normals has median length only `0.519`. Restricting to the 558
particles that are the exclusive largest `T*alpha` owner of their own center
ray raises it only to `0.556`. One current Gaussian support therefore spans
different physical surface orientations across views. The next bounded gate
must subdivide point support using construction-only normal and source-depth
groups before fitting one relightable normal. Ground-truth normals must remain
an evaluator, never a split input. The diagnostic peaks at 438,280,192 bytes
with zero swap, memory event, OOM, throttle, or GPU fault.

The next association controls locate the failure upstream. Resetting final
Gaussian centers to the dense-fusion initializer changes positions by only
`0.0801` median world units and leaves released-normal consensus effectively
unchanged: all responsible centers move `0.51918→0.51960`, and exclusive owners
move `0.55618→0.55926`. Standard Gaussian densification cannot subdivide this
case either. The maximum fitted sigma is `4.37`, well below the established
`13.96` broad-support threshold, so no disagreement-ranked point qualifies.

At the exact source pixels stored by dense fusion, association is better but
still mixed. Across 7,612 groups with at least two usable observations,
released-normal consensus has median `0.9157`; construction-only photometric
normal consensus is `0.9467`. Calling a group mixed only when both independent
12-light folds fall outside a 15-degree consensus cone selects 4,229 groups.
Against evaluation-only released normals that predicate has `97.07%` precision
and `70.08%` recall. It is a viable upstream grouping signal.

Using it after fusion is too late. Expanding 4,253 predicted-mixed groups back
to their 13,901 stored depth observations produces 17,279 points. Child opacity
preserves overlap compositing and scale preserves total Gaussian volume, but
all-particle truth consensus barely moves `0.51918→0.51928`; exclusive-owner
median reaches only `0.57133`. Selection/validation recall falls from
`88.53/88.50%` to `79.62/79.65%`, foreground means lose `0.40/0.34` dB, and
whole means lose `0.28/0.23` dB. The post-fusion expansion is rejected without
fitting appearance. The next experiment must apply the same photometric
compatibility before fusion claims observations and averages their 3D points.
The complete 4 GiB run peaks at 451,633,152 bytes with zero swap, memory event,
OOM, throttle, or GPU fault.

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

1. Estimate dense depth directly from per-pixel multi-light response signatures
   along calibrated epipolar lines. Fuse only mutually consistent camera rays
   into point samples, retain their source observations, and use the mask hull
   for low-response regions before any optical support is assigned.
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
