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

The current bottleneck is therefore **surface evidence first, light transport
second**. Material, visibility, and illumination can compensate for wrong or
missing geometry, so optimizing all of them together now would make the
decomposition less identifiable rather than more accurate.

## Execution plan

Work advances in this order. Each row must pass its gate before the next row
is allowed to add parameters.

| Phase | Next implementation | Decision gate |
| --- | --- | --- |
| Capture integrity | Keep held cameras physically absent from dense reconstruction; canonicalize masks on import | Rebuilding a training asset cannot read an excluded pose or light |
| Dense support | Reject samples outside the training-mask soft visual hull before spatial downsampling | Better held-camera foreground mean/worst and precision on two objects; report any recall trade |
| Missing support | Recover per-view photometric normal/depth patches anchored by verified tracks, then fuse only depth-consistent oriented points | Recall and covered quality rise without a precision or PSNR regression |
| Representation scale | Make Gaussian opacity/support behavior stable as a filtered surface grows beyond 2,500 points; keep the scalar and Gaussian budgets separate until then | A denser cloud improves both scalar and Gaussian PBR, not just one backend |
| Light transport | Recompute finite-light visibility after each accepted geometry stage, then fit diffuse material and one bounded bounce | Excluded-light shadows move correctly and beat black/capture-copy foreground baselines |
| Radiometry | Measure one scalar exposure residual per capture; fit it only if the residual is view-wide rather than directional | A constrained gain transfers to held lights and does not enter albedo |
| Materials and capture layout | Add spatially shared roughness only after diffuse transfer; then accept sparse `(camera, light, exposure)` observations | Two held lights and a second object improve; otherwise keep the feature off |

The current selected change is the dense-support row. On the fresh
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
`13.61/12.06→13.68/12.11` dB. It is therefore not selected, and its Gaussian
opacity fit is density-sensitive. A stricter guard restores recall by
over-expanding support and lowering foreground quality; a 3,500-point control
is also worse for Gaussian PBR. Fix that scale behavior instead of tuning a
point-count default. Nearest-view PatchMatch source generation, equal-weight
radius-mask loss, and skipping the joint known-light material fit were isolated
and removed after negative controls.

## Latest surface attribution

The fresh-object split now rules out four tempting shortcuts:

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
- Using cast-shadow visibility in the five-light material solve is correct on
  an exact occluder, but the reconstructed visibility makes every real score
  worse. Reducing shadow bias from three radii to one changes openness
  `46%→35%` and is nearly neutral, with mixed held-light tails. Transport must
  follow an accepted surface update rather than supervise the current one.

The next implementation is therefore a point-only normal/depth integration
diagnostic: recover dense per-view photometric normals from aligned lights,
integrate local depth while anchoring it to the verified multi-view tracks,
and fuse only patches that agree in another camera. It must first improve a
construction/selection/validation split; official cameras and excluded lights
remain final gates, not tuning data.

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

Once support is contiguous, alternate small stages instead of freezing a
transport factor:

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
- [ ] Recover a dense photometric normal map independently in each aligned
  training view, with confidence from directional-light conditioning.
- [ ] Integrate only local point-depth patches anchored by verified tracks and
  fuse them after an unused-camera depth/reprojection check.
- [ ] Revisit 5,000-point Gaussian support only after the new surface passes;
  point density is not a substitute for depth evidence.

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
