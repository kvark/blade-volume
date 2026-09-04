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

/// PSNR weighted by foreground coverage. Returns `None` for an empty mask.
pub fn foreground_psnr(pred: &[f32], target: &[f32], mask: &[f32]) -> Option<f32> {
    assert_eq!(pred.len(), target.len(), "foreground_psnr: length mismatch");
    assert_eq!(pred.len(), mask.len() * 3, "foreground_psnr: mask mismatch");
    let coverage = mask.iter().map(|&value| value as f64).sum::<f64>();
    if coverage <= f64::EPSILON {
        return None;
    }
    let error = pred
        .as_chunks::<3>()
        .0
        .iter()
        .zip(target.as_chunks::<3>().0)
        .zip(mask)
        .map(|((actual, expected), &weight)| {
            actual
                .iter()
                .zip(expected)
                .map(|(&a, &b)| {
                    let difference = (a - b) as f64;
                    difference * difference * weight as f64
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / (3.0 * coverage);
    Some(if error <= 0.0 {
        f32::INFINITY
    } else {
        -10.0 * error.log10() as f32
    })
}

/// Soft silhouette intersection divided by target and rendered coverage.
pub fn mask_recall_precision(
    rendered_alpha: &[f32],
    target_alpha: &[f32],
) -> (Option<f32>, Option<f32>) {
    assert_eq!(
        rendered_alpha.len(),
        target_alpha.len(),
        "mask_recall_precision: length mismatch"
    );
    let intersection = rendered_alpha
        .iter()
        .zip(target_alpha)
        .map(|(&rendered, &target)| rendered.clamp(0.0, 1.0).min(target.clamp(0.0, 1.0)) as f64)
        .sum::<f64>();
    let rendered = rendered_alpha
        .iter()
        .map(|&value| value.clamp(0.0, 1.0) as f64)
        .sum::<f64>();
    let target = target_alpha
        .iter()
        .map(|&value| value.clamp(0.0, 1.0) as f64)
        .sum::<f64>();
    (
        (target > f64::EPSILON).then_some((intersection / target) as f32),
        (rendered > f64::EPSILON).then_some((intersection / rendered) as f32),
    )
}

/// Drop the alpha channel from RGBA pixels produced by `render_cpu`,
/// returning `N*3` floats in `[0, 1]`.
pub fn rgba_to_rgb(rgba: &[f32]) -> Vec<f32> {
    rgba_over_background(rgba, [0.0; 3])
}

/// Composite premultiplied RGBA cloud output over an sRGB-code-value
/// background. This is the same display-referred domain used by training
/// images, SH appearance, and PSNR evaluation.
pub fn rgba_over_background(rgba: &[f32], background: [f32; 3]) -> Vec<f32> {
    assert!(rgba.len().is_multiple_of(4), "rgba_to_rgb: not RGBA");
    let n_px = rgba.len() / 4;
    let mut rgb = Vec::with_capacity(n_px * 3);
    for px in 0..n_px {
        let remaining = 1.0 - rgba[px * 4 + 3].clamp(0.0, 1.0);
        rgb.push(rgba[px * 4] + remaining * background[0]);
        rgb.push(rgba[px * 4 + 1] + remaining * background[1]);
        rgb.push(rgba[px * 4 + 2] + remaining * background[2]);
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

    #[test]
    fn rgba_composites_premultiplied_color_over_background() {
        let rgb = rgba_over_background(&[0.2, 0.1, 0.0, 0.25], [1.0, 0.5, 0.0]);
        assert_eq!(rgb, vec![0.95, 0.475, 0.0]);
    }

    #[test]
    fn foreground_psnr_ignores_background_error() {
        let target = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0];
        let prediction = [0.4, 0.4, 0.4, 1.0, 1.0, 1.0];
        let score = foreground_psnr(&prediction, &target, &[1.0, 0.0]).unwrap();
        assert!((score - 20.0).abs() < 1.0e-3, "{score}");
        assert!(foreground_psnr(&prediction, &target, &[0.0, 0.0]).is_none());
    }

    #[test]
    fn soft_masks_report_recall_and_precision() {
        let (recall, precision) = mask_recall_precision(&[1.0, 0.5, 0.0], &[0.5, 0.5, 1.0]);
        assert!((recall.unwrap() - 0.5).abs() < 1.0e-6);
        assert!((precision.unwrap() - 2.0 / 3.0).abs() < 1.0e-6);
    }
}
