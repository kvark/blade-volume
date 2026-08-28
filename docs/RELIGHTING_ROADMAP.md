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
- A same-session OpenIllumination capture produces a recognizable relit object,
  but foreground support is sparse and the result is visibly speckled.
- Increasing opacity or radius globally, splitting a broad Gaussian by its
  missing-ray owner, and validating an owner-depth patch all improve mask
  recall but lose known-light or held-light image quality. These experiments
  show that the remaining surface cannot be invented from a current
  particle's footprint: it needs independent cross-view correspondence.

The current bottleneck is therefore **surface evidence first, light transport
second**. Material, visibility, and illumination can compensate for wrong or
missing geometry, so optimizing all of them together now would make the
decomposition less identifiable rather than more accurate.

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
- [ ] Integrate only coherent track patches, then run the complete known/held
  quality gate under the existing 12 GiB cgroup protocol.

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
