// RadFoam blit shader: HDR -> LDR with tonemapping

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
    // Fullscreen triangle
    var out: VSOut;
    let p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0)
    );
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    out.uv = uv[vi];
    return out;
}

var g_src: texture_2d<f32>;
var g_sampler: sampler;

struct Background {
    color: vec3<f32>,
    pad: f32,
};
var<uniform> g_background: Background;

fn tonemap_reinhard(x: vec3<f32>) -> vec3<f32> {
    return x / (1.0 + x);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
    // The compute pass writes RGBA into the HDR texture:
    //   rgb = accumulated radiance
    //   a   = opacity = 1 - transmittance
    //
    // Composite an explicit sky/background using alpha to avoid black patches
    // when rays miss / terminate early.
    let sample = textureSample(g_src, g_sampler, in.uv);
    let hdr_rgb = sample.xyz;
    let alpha = clamp(sample.w, 0.0, 1.0);

    // "Over" compositing with premultiplied assumption:
    // output = rgb + (1 - alpha) * bg
    let hdr_composited = hdr_rgb + (1.0 - alpha) * g_background.color;

    let ldr = tonemap_reinhard(max(hdr_composited, vec3<f32>(0.0)));
    return vec4<f32>(ldr, 1.0);
}
