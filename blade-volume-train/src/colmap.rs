//! Pure-Rust reader for COLMAP's binary sparse reconstruction.
//!
//! Format reference: `src/colmap/scene/reconstruction_io_binary.cc` in upstream
//! COLMAP. All values are little-endian. The three files we read live under
//! `<scene>/sparse/0/`:
//!
//! - `cameras.bin`: intrinsic-only descriptors keyed by `camera_t`
//! - `images.bin`: per-image pose + name, keyed by `image_t`
//! - `points3D.bin`: sparse 3D point cloud with per-point RGB
//!
//! What we use:
//! - Camera intrinsics + per-image extrinsics → [`vol::CameraParams`]
//! - 3D point positions + RGB → initial [`vol::PointCloudModel`] (SH degree 0,
//!   uniform density, no adjacency, no radii)
//!
//! What we drop on read:
//! - The per-image 2D observation array (`points2D`). It's the bulk of
//!   `images.bin` and we don't need it for initialisation. We seek past it.
//! - The per-point track in `points3D.bin`. We keep its length for stats but
//!   skip the body.

use blade_volume as vol;
use std::{collections::HashMap, fs, io, path};

// --- Camera models ---------------------------------------------------------
//
// Numeric IDs and parameter counts mirror `src/colmap/sensor/models.h`. The
// Training rectifies perspective models onto a pinhole grid before rendering;
// the full parameter vector is therefore consumed by forward projection.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraModel {
    SimplePinhole,
    Pinhole,
    SimpleRadial,
    Radial,
    Opencv,
    OpencvFisheye,
    FullOpencv,
    Fov,
    SimpleRadialFisheye,
    RadialFisheye,
    ThinPrismFisheye,
    RadTanThinPrismFisheye,
    SimpleDivision,
    Division,
    SimpleFisheye,
    Fisheye,
    Eucm,
    Equirectangular,
}

impl CameraModel {
    fn try_from_id(id: i32) -> Option<Self> {
        Some(match id {
            0 => CameraModel::SimplePinhole,
            1 => CameraModel::Pinhole,
            2 => CameraModel::SimpleRadial,
            3 => CameraModel::Radial,
            4 => CameraModel::Opencv,
            5 => CameraModel::OpencvFisheye,
            6 => CameraModel::FullOpencv,
            7 => CameraModel::Fov,
            8 => CameraModel::SimpleRadialFisheye,
            9 => CameraModel::RadialFisheye,
            10 => CameraModel::ThinPrismFisheye,
            11 => CameraModel::RadTanThinPrismFisheye,
            12 => CameraModel::SimpleDivision,
            13 => CameraModel::Division,
            14 => CameraModel::SimpleFisheye,
            15 => CameraModel::Fisheye,
            16 => CameraModel::Eucm,
            17 => CameraModel::Equirectangular,
            _ => return None,
        })
    }

    fn param_count(self) -> usize {
        match self {
            CameraModel::SimplePinhole => 3,
            CameraModel::Pinhole => 4,
            CameraModel::SimpleRadial => 4,
            CameraModel::Radial => 5,
            CameraModel::Opencv => 8,
            CameraModel::OpencvFisheye => 8,
            CameraModel::FullOpencv => 12,
            CameraModel::Fov => 5,
            CameraModel::SimpleRadialFisheye => 4,
            CameraModel::RadialFisheye => 5,
            CameraModel::ThinPrismFisheye => 12,
            CameraModel::RadTanThinPrismFisheye => 16,
            CameraModel::SimpleDivision => 4,
            CameraModel::Division => 5,
            CameraModel::SimpleFisheye => 3,
            CameraModel::Fisheye => 4,
            CameraModel::Eucm => 6,
            CameraModel::Equirectangular => 2,
        }
    }

    pub fn supports_pinhole_rectification(self) -> bool {
        self != CameraModel::Equirectangular
    }

    /// `(fx, fy, cx, cy)` from the raw params, using each model's convention.
    /// "Simple" variants share a single focal length across both axes.
    pub(crate) fn fxfycxcy(self, params: &[f64]) -> (f64, f64, f64, f64) {
        match self {
            CameraModel::SimplePinhole
            | CameraModel::SimpleRadial
            | CameraModel::RadialFisheye
            | CameraModel::SimpleRadialFisheye
            | CameraModel::SimpleFisheye
            | CameraModel::SimpleDivision => (params[0], params[0], params[1], params[2]),
            CameraModel::Pinhole
            | CameraModel::Opencv
            | CameraModel::OpencvFisheye
            | CameraModel::FullOpencv
            | CameraModel::Fov
            | CameraModel::ThinPrismFisheye
            | CameraModel::RadTanThinPrismFisheye
            | CameraModel::Division
            | CameraModel::Fisheye
            | CameraModel::Eucm => (params[0], params[1], params[2], params[3]),
            CameraModel::Radial => (params[0], params[0], params[1], params[2]),
            CameraModel::Equirectangular => {
                panic!("equirectangular cameras do not have pinhole intrinsics")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColmapCamera {
    pub id: u32,
    pub model: CameraModel,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

impl ColmapCamera {
    /// Project a camera-space direction `(u, v, 1)` into the source image,
    /// including the camera model's forward distortion.
    pub fn project_camera_plane(&self, u: f64, v: f64) -> Option<[f64; 2]> {
        if self.model == CameraModel::Equirectangular {
            let length = (u * u + v * v + 1.0).sqrt();
            let theta = u.atan2(1.0);
            let phi = (-v / length).asin();
            return Some([
                self.params[0] * (theta / std::f64::consts::TAU + 0.5),
                self.params[1] * (0.5 - phi / std::f64::consts::PI),
            ]);
        }
        let (fx, fy, cx, cy) = self.model.fxfycxcy(&self.params);
        let extra_start = match self.model {
            CameraModel::SimplePinhole
            | CameraModel::SimpleFisheye
            | CameraModel::SimpleDivision => 3,
            CameraModel::Pinhole | CameraModel::Fisheye | CameraModel::Division => 4,
            CameraModel::SimpleRadial
            | CameraModel::Radial
            | CameraModel::SimpleRadialFisheye
            | CameraModel::RadialFisheye => 3,
            CameraModel::Opencv
            | CameraModel::OpencvFisheye
            | CameraModel::FullOpencv
            | CameraModel::Fov
            | CameraModel::ThinPrismFisheye
            | CameraModel::RadTanThinPrismFisheye => 4,
            CameraModel::Eucm => 4,
            CameraModel::Equirectangular => unreachable!(),
        };
        let extra = &self.params[extra_start..];

        let project = |x: f64, y: f64| [fx * x + cx, fy * y + cy];
        let radial = |x: f64, y: f64, coefficients: &[f64]| {
            let r2 = x * x + y * y;
            let mut power = r2;
            let mut factor = 0.0;
            for coefficient in coefficients {
                factor += coefficient * power;
                power *= r2;
            }
            [x * (1.0 + factor), y * (1.0 + factor)]
        };
        let fisheye = |x: f64, y: f64| {
            let radius = x.hypot(y);
            if radius > f64::EPSILON {
                let scale = radius.atan() / radius;
                [x * scale, y * scale]
            } else {
                [x, y]
            }
        };

        let distorted = match self.model {
            CameraModel::SimplePinhole | CameraModel::Pinhole => [u, v],
            CameraModel::SimpleRadial => radial(u, v, &extra[..1]),
            CameraModel::Radial => radial(u, v, &extra[..2]),
            CameraModel::Opencv => {
                let [k1, k2, p1, p2] = extra[..4] else {
                    unreachable!()
                };
                let u2 = u * u;
                let uv = u * v;
                let v2 = v * v;
                let r2 = u2 + v2;
                let factor = k1 * r2 + k2 * r2 * r2;
                [
                    u * (1.0 + factor) + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2),
                    v * (1.0 + factor) + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2),
                ]
            }
            CameraModel::FullOpencv => {
                let [k1, k2, p1, p2, k3, k4, k5, k6] = extra[..8] else {
                    unreachable!()
                };
                let u2 = u * u;
                let uv = u * v;
                let v2 = v * v;
                let r2 = u2 + v2;
                let r4 = r2 * r2;
                let r6 = r4 * r2;
                let factor =
                    (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);
                [
                    u * factor + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2),
                    v * factor + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2),
                ]
            }
            CameraModel::Fov => {
                let omega = extra[0];
                let radius = u.hypot(v);
                let factor = if omega.abs() < 1e-8 || radius < 1e-8 {
                    1.0
                } else {
                    (radius * 2.0 * (omega * 0.5).tan()).atan() / (radius * omega)
                };
                [u * factor, v * factor]
            }
            CameraModel::SimpleDivision | CameraModel::Division => {
                let k = extra[0];
                let rho = u.hypot(v);
                let discriminant = 1.0 - 4.0 * rho * rho * k;
                if discriminant < 0.0 {
                    return None;
                }
                let scale = 2.0 / (1.0 + discriminant.sqrt());
                [u * scale, v * scale]
            }
            CameraModel::SimpleFisheye | CameraModel::Fisheye => fisheye(u, v),
            CameraModel::SimpleRadialFisheye => {
                let theta = fisheye(u, v);
                radial(theta[0], theta[1], &extra[..1])
            }
            CameraModel::RadialFisheye => {
                let theta = fisheye(u, v);
                radial(theta[0], theta[1], &extra[..2])
            }
            CameraModel::OpencvFisheye => {
                let theta = fisheye(u, v);
                radial(theta[0], theta[1], &extra[..4])
            }
            CameraModel::ThinPrismFisheye => {
                let [k1, k2, p1, p2, k3, k4, sx1, sy1] = extra[..8] else {
                    unreachable!()
                };
                let [x, y] = fisheye(u, v);
                let x2 = x * x;
                let xy = x * y;
                let y2 = y * y;
                let r2 = x2 + y2;
                let r4 = r2 * r2;
                let radial = k1 * r2 + k2 * r4 + k3 * r4 * r2 + k4 * r4 * r4;
                [
                    x * (1.0 + radial) + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2) + sx1 * r2,
                    y * (1.0 + radial) + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2) + sy1 * r2,
                ]
            }
            CameraModel::RadTanThinPrismFisheye => {
                let [k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3] = extra[..12] else {
                    unreachable!()
                };
                let [theta_x, theta_y] = fisheye(u, v);
                let theta2 = theta_x * theta_x + theta_y * theta_y;
                let coefficients = [k0, k1, k2, k3, k4, k5];
                let mut theta_power = theta2;
                let mut theta_radial = 1.0;
                for coefficient in coefficients {
                    theta_radial += coefficient * theta_power;
                    theta_power *= theta2;
                }
                let x = theta_radial * theta_x;
                let y = theta_radial * theta_y;
                let x2 = x * x;
                let xy = x * y;
                let y2 = y * y;
                let r2 = x2 + y2;
                let r4 = r2 * r2;
                [
                    x + 2.0 * p1 * xy + p0 * (r2 + 2.0 * x2) + s0 * r2 + s1 * r4,
                    y + 2.0 * p0 * xy + p1 * (r2 + 2.0 * y2) + s2 * r2 + s3 * r4,
                ]
            }
            CameraModel::Eucm => {
                let [alpha, beta] = extra[..2] else {
                    unreachable!()
                };
                let rho_squared = beta * (u * u + v * v) + 1.0;
                if rho_squared < 0.0 {
                    return None;
                }
                let denominator = alpha * rho_squared.sqrt() + 1.0 - alpha;
                if denominator <= f64::EPSILON {
                    return None;
                }
                [u / denominator, v / denominator]
            }
            CameraModel::Equirectangular => unreachable!(),
        };
        Some(project(distorted[0], distorted[1]))
    }
}

/// Per-image record. We drop the `points2D` array on read since we don't use
/// it for initialisation; if needed later, the seek-past in [`load_images`]
/// can be replaced with a real parse.
#[derive(Clone, Debug)]
pub struct ColmapImage {
    pub id: u32,
    pub camera_id: u32,
    pub name: String,
    /// `cam_from_world` rotation as `(w, x, y, z)`. COLMAP convention.
    pub quat_wxyz: [f64; 4],
    /// `cam_from_world` translation `(tx, ty, tz)`.
    pub translation: [f64; 3],
    /// Number of 2D observations this image had — kept for stats; the actual
    /// array is skipped on read.
    pub num_points2d: u64,
}

#[derive(Clone, Debug)]
pub struct ColmapPoint3D {
    pub id: u64,
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    pub error: f64,
    /// Length of the original track. The track entries themselves are skipped
    /// on read.
    pub track_len: u64,
}

/// Complete sparse reconstruction.
pub struct Reconstruction {
    pub cameras: HashMap<u32, ColmapCamera>,
    pub images: Vec<ColmapImage>,
    pub points: Vec<ColmapPoint3D>,
}

// --- Binary reader helpers -------------------------------------------------

const MAX_IMAGE_NAME_BYTES: usize = 1024 * 1024;

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn read_u8<R: io::Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_u32<R: io::Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_i32<R: io::Read>(r: &mut R) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
fn read_u64<R: io::Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f64<R: io::Read>(r: &mut R) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}
fn read_cstr<R: io::Read>(r: &mut R) -> io::Result<String> {
    let mut out = Vec::with_capacity(64);
    let mut byte = [0u8; 1];
    for _ in 0..MAX_IMAGE_NAME_BYTES {
        r.read_exact(&mut byte)?;
        if byte[0] == 0 {
            return String::from_utf8(out)
                .map_err(|_| invalid_data("COLMAP image name is not UTF-8"));
        }
        out.push(byte[0]);
    }
    Err(invalid_data(format!(
        "COLMAP image name exceeds {MAX_IMAGE_NAME_BYTES} bytes"
    )))
}
fn skip<R: io::Read>(r: &mut R, n: u64) -> io::Result<()> {
    let mut remaining = n;
    let mut scratch = [0u8; 64 * 1024];
    while remaining > 0 {
        let chunk = remaining.min(scratch.len() as u64) as usize;
        r.read_exact(&mut scratch[..chunk])?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn checked_record_count(
    count: u64,
    file_size: u64,
    minimum_record_size: u64,
    record_name: &str,
) -> io::Result<usize> {
    let available = file_size
        .checked_sub(8)
        .ok_or_else(|| invalid_data("COLMAP binary file is shorter than its count header"))?;
    let maximum = available / minimum_record_size;
    if count > maximum {
        return Err(invalid_data(format!(
            "COLMAP {record_name} count {count} exceeds the file-size limit {maximum}"
        )));
    }
    usize::try_from(count)
        .map_err(|_| invalid_data(format!("COLMAP {record_name} count does not fit usize")))
}

fn reserve_records<T>(capacity: usize, record_name: &str) -> io::Result<Vec<T>> {
    let mut records = Vec::new();
    records.try_reserve_exact(capacity).map_err(|error| {
        invalid_data(format!(
            "cannot reserve space for {capacity} COLMAP {record_name} records: {error}"
        ))
    })?;
    Ok(records)
}

// --- Loaders ---------------------------------------------------------------

pub fn try_load_cameras(path: &path::Path) -> io::Result<Vec<ColmapCamera>> {
    let raw_file = fs::File::open(path)?;
    let file_size = raw_file.metadata()?.len();
    let mut file = io::BufReader::new(raw_file);
    let n = read_u64(&mut file)?;
    let capacity = checked_record_count(n, file_size, 40, "camera")?;
    let mut out = reserve_records(capacity, "camera")?;
    for _ in 0..n {
        let id = read_u32(&mut file)?;
        let model_id = read_i32(&mut file)?;
        let model = CameraModel::try_from_id(model_id)
            .ok_or_else(|| invalid_data(format!("unknown COLMAP camera model id {model_id}")))?;
        let width = read_u64(&mut file)?;
        let height = read_u64(&mut file)?;
        let pcount = model.param_count();
        let mut params = Vec::with_capacity(pcount);
        for _ in 0..pcount {
            params.push(read_f64(&mut file)?);
        }
        if width == 0 || height == 0 {
            return Err(invalid_data(format!(
                "COLMAP camera {id} has zero image dimensions"
            )));
        }
        if params.iter().any(|value| !value.is_finite()) {
            return Err(invalid_data(format!(
                "COLMAP camera {id} has non-finite parameters"
            )));
        }
        if model == CameraModel::Equirectangular {
            if params[0] <= 0.0 || params[1] <= 0.0 {
                return Err(invalid_data(format!(
                    "COLMAP equirectangular camera {id} has non-positive dimensions"
                )));
            }
        } else {
            let (fx, fy, _, _) = model.fxfycxcy(&params);
            if fx <= 0.0 || fy <= 0.0 {
                return Err(invalid_data(format!(
                    "COLMAP camera {id} has non-positive focal length"
                )));
            }
        }
        out.push(ColmapCamera {
            id,
            model,
            width,
            height,
            params,
        });
    }
    Ok(out)
}

pub fn load_cameras(path: &path::Path) -> Vec<ColmapCamera> {
    try_load_cameras(path).unwrap_or_else(|error| panic!("load {}: {error}", path.display()))
}

pub fn try_load_images(path: &path::Path) -> io::Result<Vec<ColmapImage>> {
    let raw_file = fs::File::open(path)?;
    let file_size = raw_file.metadata()?.len();
    let mut file = io::BufReader::new(raw_file);
    let n = read_u64(&mut file)?;
    let capacity = checked_record_count(n, file_size, 73, "image")?;
    let mut out = reserve_records(capacity, "image")?;
    for _ in 0..n {
        let id = read_u32(&mut file)?;
        let qw = read_f64(&mut file)?;
        let qx = read_f64(&mut file)?;
        let qy = read_f64(&mut file)?;
        let qz = read_f64(&mut file)?;
        let tx = read_f64(&mut file)?;
        let ty = read_f64(&mut file)?;
        let tz = read_f64(&mut file)?;
        let camera_id = read_u32(&mut file)?;
        let name = read_cstr(&mut file)?;
        let num_points2d = read_u64(&mut file)?;
        // Each point2D record is 8 + 8 + 8 = 24 bytes (xy as f64, point3D_id as u64).
        let observation_bytes = num_points2d
            .checked_mul(24)
            .ok_or_else(|| invalid_data("COLMAP points2D byte count overflows u64"))?;
        skip(&mut file, observation_bytes)?;
        let quat_wxyz = [qw, qx, qy, qz];
        let translation = [tx, ty, tz];
        if quat_wxyz.iter().any(|value| !value.is_finite())
            || translation.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_data(format!(
                "COLMAP image {id} has a non-finite pose"
            )));
        }
        let quat_norm_squared = quat_wxyz.iter().map(|value| value * value).sum::<f64>();
        if !quat_norm_squared.is_finite() || quat_norm_squared <= f64::EPSILON {
            return Err(invalid_data(format!(
                "COLMAP image {id} has a zero-length orientation quaternion"
            )));
        }
        out.push(ColmapImage {
            id,
            camera_id,
            name,
            quat_wxyz,
            translation,
            num_points2d,
        });
    }
    Ok(out)
}

pub fn load_images(path: &path::Path) -> Vec<ColmapImage> {
    try_load_images(path).unwrap_or_else(|error| panic!("load {}: {error}", path.display()))
}

pub fn try_load_points3d(path: &path::Path) -> io::Result<Vec<ColmapPoint3D>> {
    let raw_file = fs::File::open(path)?;
    let file_size = raw_file.metadata()?.len();
    let mut file = io::BufReader::new(raw_file);
    let n = read_u64(&mut file)?;
    let capacity = checked_record_count(n, file_size, 51, "point3D")?;
    let mut out = reserve_records(capacity, "point3D")?;
    for _ in 0..n {
        let id = read_u64(&mut file)?;
        let x = read_f64(&mut file)?;
        let y = read_f64(&mut file)?;
        let z = read_f64(&mut file)?;
        let r = read_u8(&mut file)?;
        let g = read_u8(&mut file)?;
        let b = read_u8(&mut file)?;
        let error = read_f64(&mut file)?;
        let track_len = read_u64(&mut file)?;
        // Each track entry is image_id (u32) + point2D_idx (u32) = 8 bytes.
        let track_bytes = track_len
            .checked_mul(8)
            .ok_or_else(|| invalid_data("COLMAP point3D track byte count overflows u64"))?;
        skip(&mut file, track_bytes)?;
        if !x.is_finite() || !y.is_finite() || !z.is_finite() || !error.is_finite() || error < 0.0 {
            return Err(invalid_data(format!(
                "COLMAP point3D {id} has invalid coordinates or error"
            )));
        }
        out.push(ColmapPoint3D {
            id,
            xyz: [x, y, z],
            rgb: [r, g, b],
            error,
            track_len,
        });
    }
    Ok(out)
}

pub fn load_points3d(path: &path::Path) -> Vec<ColmapPoint3D> {
    try_load_points3d(path).unwrap_or_else(|error| panic!("load {}: {error}", path.display()))
}

/// Read `<sparse_dir>/cameras.bin`, `images.bin`, `points3D.bin`.
pub fn try_load_reconstruction(sparse_dir: &path::Path) -> io::Result<Reconstruction> {
    let cameras_vec = try_load_cameras(&sparse_dir.join("cameras.bin"))?;
    let images = try_load_images(&sparse_dir.join("images.bin"))?;
    let points = try_load_points3d(&sparse_dir.join("points3D.bin"))?;
    let mut cameras = HashMap::with_capacity(cameras_vec.len());
    for c in cameras_vec {
        let id = c.id;
        if cameras.insert(id, c).is_some() {
            return Err(invalid_data(format!("duplicate COLMAP camera id {id}")));
        }
    }
    for image in &images {
        if !cameras.contains_key(&image.camera_id) {
            return Err(invalid_data(format!(
                "COLMAP image {} references unknown camera id {}",
                image.id, image.camera_id
            )));
        }
    }
    Ok(Reconstruction {
        cameras,
        images,
        points,
    })
}

pub fn load_reconstruction(sparse_dir: &path::Path) -> Reconstruction {
    try_load_reconstruction(sparse_dir)
        .unwrap_or_else(|error| panic!("load {}: {error}", sparse_dir.display()))
}

// --- Conversion to native blade-volume types -------------------------------

/// Matches `(rgb/255 - 0.5) / SH_C0` used elsewhere in the codebase. Lets us
/// initialise SH degree 0 so the DC component reproduces the COLMAP RGB.
const SH_C0: f32 = 0.282_094_8;

/// Initial-density for new points in trainer space. Same order of magnitude as
/// PowerFoam's `density = 0.1` constant before softplus.
const DEFAULT_INITIAL_DENSITY: f32 = 0.1;

impl Reconstruction {
    /// Build a starting `PointCloudModel` from the sparse 3D points:
    /// - positions taken straight from `points3D.bin`
    /// - density is a uniform [`DEFAULT_INITIAL_DENSITY`]
    /// - SH degree 0; DC computed so the rendered colour equals the COLMAP RGB
    ///   (modulo the renderer's `0.5 +` bias)
    /// - no transforms, no adjacency, no radii — the trainer fills those in
    pub fn to_initial_model(&self) -> vol::PointCloudModel {
        self.to_initial_model_with_density(DEFAULT_INITIAL_DENSITY)
    }

    /// As [`to_initial_model`] but with a caller-chosen uniform density.
    /// Higher values give larger initial per-cell alpha at typical COLMAP
    /// scales — useful when the training loss plateaus too early.
    pub fn to_initial_model_with_density(&self, initial_density: f32) -> vol::PointCloudModel {
        let n = self.points.len();
        let mut points = Vec::with_capacity(n);
        let mut sh = Vec::with_capacity(n * 3);
        for p in &self.points {
            points.push(glam::Vec4::new(
                p.xyz[0] as f32,
                p.xyz[1] as f32,
                p.xyz[2] as f32,
                initial_density,
            ));
            let r = (p.rgb[0] as f32) / 255.0;
            let g = (p.rgb[1] as f32) / 255.0;
            let b = (p.rgb[2] as f32) / 255.0;
            sh.push((r - 0.5) / SH_C0);
            sh.push((g - 0.5) / SH_C0);
            sh.push((b - 0.5) / SH_C0);
        }
        vol::PointCloudModel {
            points,
            sh_coefficients: sh,
            sh_degree: 0,
            transforms: None,
            adjacency: None,
            radii: None,
        }
    }

    /// Convert one image's pose + its camera's intrinsics into [`vol::CameraParams`].
    ///
    /// COLMAP stores `cam_from_world`; the renderer wants `world_from_cam`
    /// (camera position in world space, orientation that rotates camera-space
    /// rays into world space). We invert: `R_wfc = R_cfw^T`, `t_wfc = -R_wfc t_cfw`.
    ///
    /// `far_plane` is forwarded to `CameraParams.depth`. Distortion parameters
    /// are ignored — the renderer assumes a pinhole projection.
    pub fn camera_params_for(&self, image: &ColmapImage, far_plane: f32) -> vol::CameraParams {
        let cam = self.cameras.get(&image.camera_id).unwrap_or_else(|| {
            panic!(
                "image {} references unknown camera_id {}",
                image.id, image.camera_id
            )
        });
        assert!(
            cam.model.supports_pinhole_rectification(),
            "equirectangular camera {} cannot use the pinhole runtime camera",
            cam.id,
        );

        let q_cfw = glam::DQuat::from_xyzw(
            image.quat_wxyz[1],
            image.quat_wxyz[2],
            image.quat_wxyz[3],
            image.quat_wxyz[0],
        )
        .normalize();
        let t_cfw = glam::DVec3::new(
            image.translation[0],
            image.translation[1],
            image.translation[2],
        );
        let q_wfc = q_cfw.inverse();
        let cam_pos_world = q_wfc * (-t_cfw);

        let (fx, fy, cx, cy) = cam.model.fxfycxcy(&cam.params);
        let fov_x = 2.0 * ((cam.width as f64) / (2.0 * fx)).atan();
        let fov_y = 2.0 * ((cam.height as f64) / (2.0 * fy)).atan();

        vol::CameraParams {
            cam_position: [
                cam_pos_world.x as f32,
                cam_pos_world.y as f32,
                cam_pos_world.z as f32,
            ],
            depth: far_plane,
            cam_orientation: [
                q_wfc.x as f32,
                q_wfc.y as f32,
                q_wfc.z as f32,
                q_wfc.w as f32,
            ],
            fov: [fov_x as f32, fov_y as f32],
            principal: [
                (2.0 * cx / cam.width as f64 - 1.0) as f32,
                (2.0 * cy / cam.height as f64 - 1.0) as f32,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    // --- Writer helpers used only by the test fixture --------------------

    fn put_u8(out: &mut Vec<u8>, v: u8) {
        out.push(v);
    }
    fn put_u32(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_i32(out: &mut Vec<u8>, v: i32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_u64(out: &mut Vec<u8>, v: u64) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_f64(out: &mut Vec<u8>, v: f64) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_cstr(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(s.as_bytes());
        out.push(0);
    }

    fn write_fixture(dir: &path::Path) {
        fs::create_dir_all(dir).unwrap();

        // cameras.bin: one PINHOLE camera (id=1, 640x480, fx=fy=500, cx=320, cy=240).
        let mut buf = Vec::new();
        put_u64(&mut buf, 1);
        put_u32(&mut buf, 1); // camera id
        put_i32(&mut buf, 1); // PINHOLE
        put_u64(&mut buf, 640);
        put_u64(&mut buf, 480);
        for &v in &[500.0_f64, 500.0, 320.0, 240.0] {
            put_f64(&mut buf, v);
        }
        fs::File::create(dir.join("cameras.bin"))
            .unwrap()
            .write_all(&buf)
            .unwrap();

        // images.bin: two images, one with no points2D, one with two.
        let mut buf = Vec::new();
        put_u64(&mut buf, 2);

        // image 1: identity pose at origin
        put_u32(&mut buf, 10);
        for &v in &[1.0_f64, 0.0, 0.0, 0.0] {
            put_f64(&mut buf, v);
        }
        for &v in &[0.0_f64, 0.0, 0.0] {
            put_f64(&mut buf, v);
        }
        put_u32(&mut buf, 1);
        put_cstr(&mut buf, "frame0001.jpg");
        put_u64(&mut buf, 0); // num_points2D = 0

        // image 2: 180° about Y, translated +5z
        put_u32(&mut buf, 11);
        // qw=0, qx=0, qy=1, qz=0
        for &v in &[0.0_f64, 0.0, 1.0, 0.0] {
            put_f64(&mut buf, v);
        }
        for &v in &[0.0_f64, 0.0, 5.0] {
            put_f64(&mut buf, v);
        }
        put_u32(&mut buf, 1);
        put_cstr(&mut buf, "frame0002.jpg");
        put_u64(&mut buf, 2); // num_points2D = 2
        for _ in 0..2 {
            put_f64(&mut buf, 0.0); // x
            put_f64(&mut buf, 0.0); // y
            put_u64(&mut buf, 0); // point3D_id
        }
        fs::File::create(dir.join("images.bin"))
            .unwrap()
            .write_all(&buf)
            .unwrap();

        // points3D.bin: three points with simple colours, varying track lengths.
        let mut buf = Vec::new();
        put_u64(&mut buf, 3);
        struct P {
            id: u64,
            xyz: [f64; 3],
            rgb: [u8; 3],
            err: f64,
            tl: u64,
        }
        let pts = [
            P {
                id: 100,
                xyz: [0.0, 0.0, 0.0],
                rgb: [255, 0, 0],
                err: 0.1,
                tl: 0,
            },
            P {
                id: 101,
                xyz: [1.0, 2.0, 3.0],
                rgb: [0, 255, 0],
                err: 0.2,
                tl: 1,
            },
            P {
                id: 102,
                xyz: [-1.0, 0.5, 4.0],
                rgb: [0, 0, 255],
                err: 0.3,
                tl: 2,
            },
        ];
        for P {
            id,
            xyz,
            rgb,
            err,
            tl,
        } in pts
        {
            put_u64(&mut buf, id);
            for v in xyz {
                put_f64(&mut buf, v);
            }
            for v in rgb {
                put_u8(&mut buf, v);
            }
            put_f64(&mut buf, err);
            put_u64(&mut buf, tl);
            for _ in 0..tl {
                put_u32(&mut buf, 10); // image_id
                put_u32(&mut buf, 0); // point2D_idx
            }
        }
        fs::File::create(dir.join("points3D.bin"))
            .unwrap()
            .write_all(&buf)
            .unwrap();
    }

    fn tmpdir(name: &str) -> path::PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!("blade-volume-train-colmap-{name}"));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn parses_synthetic_fixture() {
        let dir = tmpdir("parse");
        write_fixture(&dir);
        let r = load_reconstruction(&dir);

        assert_eq!(r.cameras.len(), 1);
        let cam = &r.cameras[&1];
        assert_eq!(cam.model, CameraModel::Pinhole);
        assert_eq!(cam.width, 640);
        assert_eq!(cam.height, 480);
        assert_eq!(cam.params, vec![500.0, 500.0, 320.0, 240.0]);

        assert_eq!(r.images.len(), 2);
        assert_eq!(r.images[0].name, "frame0001.jpg");
        assert_eq!(r.images[0].num_points2d, 0);
        assert_eq!(r.images[1].name, "frame0002.jpg");
        assert_eq!(r.images[1].num_points2d, 2);
        assert_eq!(r.images[1].quat_wxyz, [0.0, 0.0, 1.0, 0.0]);

        assert_eq!(r.points.len(), 3);
        assert_eq!(r.points[0].id, 100);
        assert_eq!(r.points[1].xyz, [1.0, 2.0, 3.0]);
        assert_eq!(r.points[2].rgb, [0, 0, 255]);
        assert_eq!(r.points[2].track_len, 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fallible_loaders_reject_implausible_counts_and_models() {
        let dir = tmpdir("invalid-header");
        fs::create_dir_all(&dir).unwrap();

        let count_path = dir.join("count.bin");
        fs::write(&count_path, u64::MAX.to_le_bytes()).unwrap();
        let count_error = try_load_cameras(&count_path).unwrap_err();
        assert_eq!(count_error.kind(), io::ErrorKind::InvalidData);
        assert!(count_error.to_string().contains("file-size limit"));

        let model_path = dir.join("model.bin");
        let mut bytes = Vec::new();
        put_u64(&mut bytes, 1);
        put_u32(&mut bytes, 7);
        put_i32(&mut bytes, 99);
        put_u64(&mut bytes, 640);
        put_u64(&mut bytes, 480);
        put_f64(&mut bytes, 1.0);
        put_f64(&mut bytes, 1.0);
        fs::write(&model_path, bytes).unwrap();
        let model_error = try_load_cameras(&model_path).unwrap_err();
        assert_eq!(model_error.kind(), io::ErrorKind::InvalidData);
        assert!(model_error.to_string().contains("camera model id 99"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fallible_loaders_reject_invalid_numeric_records() {
        let dir = tmpdir("invalid-numeric");
        fs::create_dir_all(&dir).unwrap();

        let camera_path = dir.join("camera.bin");
        let mut bytes = Vec::new();
        put_u64(&mut bytes, 1);
        put_u32(&mut bytes, 1);
        put_i32(&mut bytes, 1);
        put_u64(&mut bytes, 640);
        put_u64(&mut bytes, 480);
        for &value in &[f64::NAN, 500.0, 320.0, 240.0] {
            put_f64(&mut bytes, value);
        }
        fs::write(&camera_path, bytes).unwrap();
        let camera_error = try_load_cameras(&camera_path).unwrap_err();
        assert!(camera_error.to_string().contains("non-finite parameters"));

        let image_path = dir.join("image.bin");
        let mut bytes = Vec::new();
        put_u64(&mut bytes, 1);
        put_u32(&mut bytes, 10);
        for _ in 0..7 {
            put_f64(&mut bytes, 0.0);
        }
        put_u32(&mut bytes, 1);
        put_cstr(&mut bytes, "frame.png");
        put_u64(&mut bytes, 0);
        fs::write(&image_path, bytes).unwrap();
        let image_error = try_load_images(&image_path).unwrap_err();
        assert!(image_error
            .to_string()
            .contains("zero-length orientation quaternion"));

        let point_path = dir.join("point.bin");
        let mut bytes = Vec::new();
        put_u64(&mut bytes, 1);
        put_u64(&mut bytes, 100);
        for &value in &[0.0_f64, f64::INFINITY, 0.0] {
            put_f64(&mut bytes, value);
        }
        bytes.extend_from_slice(&[255, 255, 255]);
        put_f64(&mut bytes, 0.0);
        put_u64(&mut bytes, 0);
        fs::write(&point_path, bytes).unwrap();
        let point_error = try_load_points3d(&point_path).unwrap_err();
        assert!(point_error
            .to_string()
            .contains("invalid coordinates or error"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn variable_length_fields_are_bounded() {
        let dir = tmpdir("invalid-image");
        fs::create_dir_all(&dir).unwrap();

        let mut image_prefix = Vec::new();
        put_u64(&mut image_prefix, 1);
        put_u32(&mut image_prefix, 10);
        for &value in &[1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] {
            put_f64(&mut image_prefix, value);
        }
        put_u32(&mut image_prefix, 1);

        let name_path = dir.join("name.bin");
        let mut bytes = image_prefix.clone();
        bytes.resize(bytes.len() + MAX_IMAGE_NAME_BYTES, b'a');
        fs::write(&name_path, bytes).unwrap();
        let name_error = try_load_images(&name_path).unwrap_err();
        assert_eq!(name_error.kind(), io::ErrorKind::InvalidData);
        assert!(name_error.to_string().contains("image name exceeds"));

        let observations_path = dir.join("observations.bin");
        let mut bytes = image_prefix;
        put_cstr(&mut bytes, "frame.png");
        put_u64(&mut bytes, u64::MAX);
        fs::write(&observations_path, bytes).unwrap();
        let observations_error = try_load_images(&observations_path).unwrap_err();
        assert_eq!(observations_error.kind(), io::ErrorKind::InvalidData);
        assert!(observations_error
            .to_string()
            .contains("byte count overflows"));

        let track_path = dir.join("track.bin");
        let mut bytes = Vec::new();
        put_u64(&mut bytes, 1);
        put_u64(&mut bytes, 100);
        for &value in &[0.0_f64, 0.0, 0.0] {
            put_f64(&mut bytes, value);
        }
        bytes.extend_from_slice(&[255, 255, 255]);
        put_f64(&mut bytes, 0.0);
        put_u64(&mut bytes, u64::MAX);
        fs::write(&track_path, bytes).unwrap();
        let track_error = try_load_points3d(&track_path).unwrap_err();
        assert_eq!(track_error.kind(), io::ErrorKind::InvalidData);
        assert!(track_error
            .to_string()
            .contains("track byte count overflows"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn initial_model_matches_points3d() {
        let dir = tmpdir("model");
        write_fixture(&dir);
        let r = load_reconstruction(&dir);
        let model = r.to_initial_model();
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(model.len(), 3);
        assert_eq!(model.sh_degree, 0);
        assert!(model.transforms.is_none());
        assert!(model.adjacency.is_none());
        assert!(model.radii.is_none());
        assert_eq!(model.sh_coefficients.len(), 3 * 3);

        // SH DC for the red point: (1.0 - 0.5)/C0 on R; (0 - 0.5)/C0 on G/B.
        let dc = model.get_sh_coefficients(0)[0];
        assert!((dc.x - 0.5 / SH_C0).abs() < 1e-4);
        assert!((dc.y - (-0.5 / SH_C0)).abs() < 1e-4);
        assert!((dc.z - (-0.5 / SH_C0)).abs() < 1e-4);

        // Density is the uniform initial value.
        assert!((model.points[0].w - DEFAULT_INITIAL_DENSITY).abs() < 1e-6);
    }

    #[test]
    fn camera_params_for_identity_pose_is_origin_at_origin() {
        let dir = tmpdir("cam-id");
        write_fixture(&dir);
        let r = load_reconstruction(&dir);
        let p = r.camera_params_for(&r.images[0], 100.0);
        let _ = fs::remove_dir_all(&dir);

        // Identity quat + zero translation → camera at origin, identity orientation.
        assert!(p.cam_position[0].abs() < 1e-6);
        assert!(p.cam_position[1].abs() < 1e-6);
        assert!(p.cam_position[2].abs() < 1e-6);
        assert!((p.cam_orientation[3] - 1.0).abs() < 1e-6); // w
        assert!(p.cam_orientation[0].abs() < 1e-6);
        assert!(p.cam_orientation[1].abs() < 1e-6);
        assert!(p.cam_orientation[2].abs() < 1e-6);
        assert_eq!(p.depth, 100.0);

        // fov_x = 2 * atan(640 / (2 * 500)) ≈ 1.1730
        let expected_fov_x = 2.0 * (640.0_f32 / 1000.0).atan();
        assert!((p.fov[0] - expected_fov_x).abs() < 1e-5);
    }

    #[test]
    fn camera_params_for_translated_pose_recovers_world_position() {
        let dir = tmpdir("cam-trans");
        write_fixture(&dir);
        let r = load_reconstruction(&dir);
        // Image 2: 180° about Y in cam_from_world, t = (0, 0, 5).
        // world_from_cam rotation is the same 180° about Y; cam_pos_world =
        // R_wfc * (-t) = rot180_y * (0, 0, -5) = (0, 0, 5).
        let p = r.camera_params_for(&r.images[1], 100.0);
        let _ = fs::remove_dir_all(&dir);

        assert!(p.cam_position[0].abs() < 1e-5);
        assert!(p.cam_position[1].abs() < 1e-5);
        assert!((p.cam_position[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn camera_params_preserve_off_center_principal_point() {
        let dir = tmpdir("cam-principal");
        write_fixture(&dir);
        let mut r = load_reconstruction(&dir);
        let camera = r.cameras.get_mut(&1).unwrap();
        camera.params[2] = 400.0;
        camera.params[3] = 180.0;
        let p = r.camera_params_for(&r.images[0], 100.0);
        let _ = fs::remove_dir_all(&dir);

        assert!((p.principal[0] - 0.25).abs() < 1e-6);
        assert!((p.principal[1] + 0.25).abs() < 1e-6);
    }

    #[test]
    fn simple_radial_projection_matches_colmap_equation() {
        let camera = ColmapCamera {
            id: 1,
            model: CameraModel::SimpleRadial,
            width: 100,
            height: 80,
            params: vec![100.0, 50.0, 40.0, 0.1],
        };
        let projected = camera.project_camera_plane(1.0, 0.5).unwrap();
        assert!((projected[0] - 162.5).abs() < 1e-10);
        assert!((projected[1] - 96.25).abs() < 1e-10);
    }

    #[test]
    fn radial_fisheye_uses_single_focal_intrinsic_layout() {
        let camera = ColmapCamera {
            id: 1,
            model: CameraModel::RadialFisheye,
            width: 640,
            height: 480,
            params: vec![500.0, 321.0, 239.0, 0.0, 0.0],
        };
        assert_eq!(
            camera.model.fxfycxcy(&camera.params),
            (500.0, 500.0, 321.0, 239.0)
        );
        let projected = camera.project_camera_plane(1.0, 0.0).unwrap();
        assert!((projected[0] - (321.0 + 500.0 * std::f64::consts::FRAC_PI_4)).abs() < 1e-10);
        assert_eq!(projected[1], 239.0);
    }

    #[test]
    fn eucm_projection_matches_unified_camera_limits() {
        let mut camera = ColmapCamera {
            id: 1,
            model: CameraModel::Eucm,
            width: 640,
            height: 480,
            params: vec![500.0, 600.0, 320.0, 240.0, 0.0, 1.0],
        };
        let pinhole = camera.project_camera_plane(0.2, -0.1).unwrap();
        assert!((pinhole[0] - 420.0).abs() < 1e-10);
        assert!((pinhole[1] - 180.0).abs() < 1e-10);

        camera.params[4] = 1.0;
        let projected = camera.project_camera_plane(0.2, -0.1).unwrap();
        let norm = (1.0_f64 + 0.2 * 0.2 + 0.1 * 0.1).sqrt();
        assert!((projected[0] - (320.0 + 500.0 * 0.2 / norm)).abs() < 1e-10);
        assert!((projected[1] - (240.0 - 600.0 * 0.1 / norm)).abs() < 1e-10);
    }

    #[test]
    fn equirectangular_projection_maps_forward_and_quarter_turn() {
        assert_eq!(CameraModel::try_from_id(16), Some(CameraModel::Eucm));
        assert_eq!(CameraModel::Eucm.param_count(), 6);
        assert_eq!(
            CameraModel::try_from_id(17),
            Some(CameraModel::Equirectangular)
        );
        assert_eq!(CameraModel::Equirectangular.param_count(), 2);
        let camera = ColmapCamera {
            id: 1,
            model: CameraModel::Equirectangular,
            width: 800,
            height: 400,
            params: vec![800.0, 400.0],
        };
        assert!(!camera.model.supports_pinhole_rectification());
        assert_eq!(camera.project_camera_plane(0.0, 0.0), Some([400.0, 200.0]));
        let projected = camera.project_camera_plane(1.0, 0.0).unwrap();
        assert!((projected[0] - 500.0).abs() < 1e-10);
        assert!((projected[1] - 200.0).abs() < 1e-10);
    }

    #[test]
    fn equidistant_fisheye_projection_uses_theta_radius() {
        let camera = ColmapCamera {
            id: 1,
            model: CameraModel::SimpleFisheye,
            width: 100,
            height: 80,
            params: vec![100.0, 50.0, 40.0],
        };
        let projected = camera.project_camera_plane(1.0, 0.0).unwrap();
        assert!((projected[0] - (50.0 + 100.0 * std::f64::consts::FRAC_PI_4)).abs() < 1e-10);
        assert!((projected[1] - 40.0).abs() < 1e-10);
    }
}
