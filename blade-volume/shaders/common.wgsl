// Common shader utilities shared between RadFoam and Gaussian backends.

const MAX_SH_COMPONENTS: u32 = 16u; // (1+3)^2

struct Camera {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>, // quaternion (x,y,z,w)
    fov: vec2<f32>,         // (fov_x, fov_y) where local_dir = (ndc * tan(0.5*fov), 1)
    principal: vec2<f32>,   // optical center in NDC; zero is image center
}

// ---- Quaternion helpers ----

fn qmake(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    return vec4<f32>(axis * sin(angle), cos(angle));
}

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    // v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn qinv(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

// ---- Debug visualization ----

// Heatmap color ramp for debug visualization
fn heatmap_color(t: f32) -> vec3f {
    // Blue -> Cyan -> Green -> Yellow -> Red
    let t_clamped = clamp(t, 0.0, 1.0);
    if (t_clamped < 0.25) {
        let s = t_clamped / 0.25;
        return vec3f(0.0, s, 1.0);
    } else if (t_clamped < 0.5) {
        let s = (t_clamped - 0.25) / 0.25;
        return vec3f(0.0, 1.0, 1.0 - s);
    } else if (t_clamped < 0.75) {
        let s = (t_clamped - 0.5) / 0.25;
        return vec3f(s, 1.0, 0.0);
    } else {
        let s = (t_clamped - 0.75) / 0.25;
        return vec3f(1.0, 1.0 - s, 0.0);
    }
}
