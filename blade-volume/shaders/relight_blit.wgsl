// Present the relightable renderer's HDR output on a display.
//
// The other backends store display-referred code values and blit them
// straight through. This one produces linear radiance with a sun in it, which
// can be hundreds of times white, so something has to decide what "white"
// means before it reaches an eight-bit surface.
//
// The curve is the one `relight_quality` scores through, so what is on screen
// is what the tone-mapped number is measuring rather than a second opinion.

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
    var out: VSOut;
    let p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let position = p[vi];
    out.pos = vec4<f32>(position, 0.0, 1.0);
    // Blade renders with a flipped viewport, so clip `+Y` is the top of the
    // target while the compute pass writes its first row at texture `y = 0`.
    // Deriving the coordinate here rather than tabulating it keeps the two
    // conventions in one place; getting it wrong renders the whole scene
    // upside down, which reads as an odd camera rather than as a bug.
    out.uv = vec2<f32>(0.5 * (position.x + 1.0), 0.5 * (1.0 - position.y));
    return out;
}

var g_src: texture_2d<f32>;
var g_sampler: sampler;

struct Present {
    // Multiplies radiance before the curve, so a dim environment can be
    // looked at without changing what was rendered.
    exposure: f32,
    // Whether this shader applies the display transfer curve. The surface is
    // usually a linear format presented in an sRGB colour space, in which case
    // nothing downstream will encode and this has to; if the surface is an
    // sRGB format the hardware does it and doing it here would be twice.
    encode_srgb: u32,
    pad: vec2<u32>,
};
var<uniform> g_present: Present;

fn encode_channel(value: f32) -> f32 {
    if value <= 0.0031308 {
        return 12.92 * value;
    }
    return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
    let radiance = max(textureSample(g_src, g_sampler, in.uv).xyz, vec3<f32>(0.0))
        * max(g_present.exposure, 0.0);
    // Reinhard: everything lands in [0, 1) without a clip, so a sun stays
    // distinguishable from a bright surface instead of both being white.
    var mapped = radiance / (1.0 + radiance);
    if g_present.encode_srgb != 0u {
        mapped = vec3<f32>(
            encode_channel(mapped.x),
            encode_channel(mapped.y),
            encode_channel(mapped.z),
        );
    }
    return vec4<f32>(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
