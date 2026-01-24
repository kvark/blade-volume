// Spherical harmonics evaluation utilities.
//
// These constants and functions are shared between RadFoam and Gaussian backends.
// The actual SH evaluation functions remain backend-specific due to different
// data layouts (packed attributes vs. struct arrays).

// SH basis constants for degrees 0-3 (16 components total).
// These match the conventions used in Gaussian splatting implementations.
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
