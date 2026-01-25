// Spherical harmonics evaluation utilities.
//
// Shared between RadFoam and Gaussian backends.

// SH basis constants for degrees 0-3 (16 components total).
fn sh_basis_constants() -> array<f32, MAX_SH_COMPONENTS> {
    return array<f32, MAX_SH_COMPONENTS>(
        0.28209479177387814,   // L0
        -0.4886025119029199,   // L1 m=-1
        0.4886025119029199,    // L1 m=0
        -0.4886025119029199,   // L1 m=1
        1.0925484305920792,    // L2 m=-2
        -1.0925484305920792,   // L2 m=-1
        0.31539156525252005,   // L2 m=0
        -1.0925484305920792,   // L2 m=1
        0.5462742152960396,    // L2 m=2
        -0.5900435899266435,   // L3 m=-3
        2.890611442640554,     // L3 m=-2
        -0.4570457994644658,   // L3 m=-1
        0.3731763325901154,    // L3 m=0
        -0.4570457994644658,   // L3 m=1
        1.445305721320277,     // L3 m=2
        -0.5900435899266435    // L3 m=3
    );
}

// Returns the number of SH components for a given degree: (1+deg)^2
fn sh_component_count(deg: u32) -> u32 {
    let d = deg + 1u;
    return d * d;
}

// Evaluate SH color from coefficients array.
// coeffs: array of RGB vec3 coefficients, one per SH component
// dir: normalized view direction
// deg: SH degree (0-3)
// Returns: evaluated RGB color (without 0.5 bias - caller can add if needed)
fn sh_eval_color(coeffs: array<vec3<f32>, MAX_SH_COMPONENTS>, dir: vec3<f32>, deg: u32) -> vec3<f32> {
    let SH = sh_basis_constants();
    let d2 = dir * dir;

    // L0
    var color = SH[0] * coeffs[0];

    if (deg >= 1u) {
        color += SH[1] * coeffs[1] * dir.y;
        color += SH[2] * coeffs[2] * dir.z;
        color += SH[3] * coeffs[3] * dir.x;
    }

    if (deg >= 2u) {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;
        let xx = d2.x;
        let yy = d2.y;
        let zz = d2.z;

        color += SH[4] * coeffs[4] * x * y;
        color += SH[5] * coeffs[5] * y * z;
        color += SH[6] * coeffs[6] * (3.0 * zz - 1.0);
        color += SH[7] * coeffs[7] * x * z;
        color += SH[8] * coeffs[8] * (xx - yy);
    }

    if (deg >= 3u) {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;
        let xx = d2.x;
        let yy = d2.y;
        let zz = d2.z;

        color += SH[9] * coeffs[9] * y * (3.0 * xx - yy);
        color += SH[10] * coeffs[10] * x * y * z;
        color += SH[11] * coeffs[11] * y * (5.0 * zz - 1.0);
        color += SH[12] * coeffs[12] * z * (5.0 * zz - 3.0);
        color += SH[13] * coeffs[13] * x * (5.0 * zz - 1.0);
        color += SH[14] * coeffs[14] * z * (xx - yy);
        color += SH[15] * coeffs[15] * x * (xx - 3.0 * yy);
    }

    return color;
}
