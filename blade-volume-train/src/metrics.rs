//! Image-quality metrics shared by training/eval code.
//!
//! Pixels are flat `[N*3]` RGB in `[0, 1]`. Mismatched lengths panic — these
//! are sanity checks for our own pipelines, not user-facing validation.

/// Mean squared error over every channel of every pixel.
pub fn mse_rgb(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(pred.len(), target.len(), "mse_rgb: length mismatch");
    if pred.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for (a, b) in pred.iter().zip(target.iter()) {
        let d = (a - b) as f64;
        sum += d * d;
    }
    (sum / pred.len() as f64) as f32
}

/// Mean absolute error over every channel of every pixel.
pub fn mae_rgb(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(pred.len(), target.len(), "mae_rgb: length mismatch");
    if pred.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for (a, b) in pred.iter().zip(target.iter()) {
        sum += (a - b).abs() as f64;
    }
    (sum / pred.len() as f64) as f32
}

/// Peak signal-to-noise ratio in dB, assuming `MAX = 1.0`.
/// `inf` when MSE is exactly zero. Returns `f32::NAN` on empty input.
pub fn psnr(pred: &[f32], target: &[f32]) -> f32 {
    if pred.is_empty() {
        return f32::NAN;
    }
    let mse = mse_rgb(pred, target);
    if mse <= 0.0 {
        return f32::INFINITY;
    }
    -10.0 * mse.log10()
}

/// Drop the alpha channel from RGBA pixels produced by `render_cpu`,
/// returning `N*3` floats in `[0, 1]`.
pub fn rgba_to_rgb(rgba: &[f32]) -> Vec<f32> {
    assert!(rgba.len().is_multiple_of(4), "rgba_to_rgb: not RGBA");
    let n_px = rgba.len() / 4;
    let mut rgb = Vec::with_capacity(n_px * 3);
    for px in 0..n_px {
        rgb.push(rgba[px * 4]);
        rgb.push(rgba[px * 4 + 1]);
        rgb.push(rgba[px * 4 + 2]);
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psnr_of_identical_is_infinite() {
        let a = vec![0.5_f32; 12];
        assert!(psnr(&a, &a).is_infinite());
    }

    #[test]
    fn psnr_drops_with_increasing_mse() {
        let a = vec![0.0_f32; 6];
        let b1 = vec![0.01_f32; 6];
        let b2 = vec![0.1_f32; 6];
        // PSNR should be higher for closer match.
        assert!(psnr(&a, &b1) > psnr(&a, &b2));
    }

    #[test]
    fn psnr_known_value() {
        // MSE = 0.01 → PSNR = -10 * log10(0.01) = 20.0 dB.
        let a = vec![0.0_f32; 4];
        let b = vec![0.1_f32; 4];
        let p = psnr(&a, &b);
        assert!((p - 20.0).abs() < 1e-3, "{p}");
    }

    #[test]
    fn rgba_to_rgb_strips_alpha() {
        let rgba = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let rgb = rgba_to_rgb(&rgba);
        assert_eq!(rgb, vec![0.1, 0.2, 0.3, 0.5, 0.6, 0.7]);
    }
}
