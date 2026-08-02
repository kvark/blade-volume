#![cfg(target_os = "linux")]

//! The viewer, driven the way a person drives it.
//!
//! Everything else about this crate is tested against an offscreen target,
//! which is most of the renderer and none of the application: window creation,
//! the swapchain, input, and the resize path that destroys and rebuilds render
//! targets underneath a live surface. Those went untested for as long as this
//! machine appeared to have no display, and three wrong defaults and one
//! camera defect shipped in that window — the presentation blit sampled upside
//! down, and a drag of a few hundred pixels turned the world over.
//!
//! So this starts a virtual X server, runs the real binary against it, presses
//! keys, drags the mouse and resizes the window, and checks the frame each
//! time. Capture is `XGetImage` on the window itself rather than an external
//! screenshot tool, so the test needs nothing on the machine but `Xvfb`, and
//! what it compares is what the window contains.
//!
//! It skips, loudly, when it cannot run: no `Xvfb`, no X libraries, no GPU.
//! CI has none of those today, so this is a test for a machine with a display
//! adapter and it says so rather than passing vacuously.

use blade_volume as vol;
use std::{path, process, time};

/// How long any single "wait for the frame to do something" step may take.
///
/// Generous because the first frame includes building the acceleration
/// structures and prefiltering an environment, and because a virtual X server
/// presents by copying on the CPU.
const DEADLINE: time::Duration = time::Duration::from_secs(60);
/// How long to wait for the frame to answer an input.
///
/// Shorter than the first frame's budget, because by then everything slow has
/// already happened and this is only how long a failure takes to report.
const RESPONSE: time::Duration = time::Duration::from_secs(20);
const POLL: time::Duration = time::Duration::from_millis(250);

/// Mean absolute difference, per channel, that counts as the image changing.
///
/// The renderer is deterministic with the analytic lighting this test uses, so
/// an unchanged frame differs by exactly zero and anything above the noise of
/// window borders is a real change.
const CHANGED: f64 = 0.5;
const UNCHANGED: f64 = 0.1;

const WIDTH: u32 = 900;
const HEIGHT: u32 = 620;

// --------------------------------------------------------------------- X11

/// The parts of Xlib and XTEST this needs, opened at run time.
///
/// Dynamically loaded rather than linked: the test has to skip on a machine
/// without X rather than fail to build there, and a `dlopen` that returns
/// nothing is exactly that signal.
struct Driver {
    xlib: x11_dl::xlib::Xlib,
    xtest: x11_dl::xtest::Xf86vmode,
    display: *mut x11_dl::xlib::Display,
    window: u64,
}

impl Driver {
    fn open(display_name: &str) -> Option<Self> {
        let xlib = x11_dl::xlib::Xlib::open().ok()?;
        let xtest = x11_dl::xtest::Xf86vmode::open().ok()?;
        let name = std::ffi::CString::new(display_name).unwrap();
        let display = unsafe { (xlib.XOpenDisplay)(name.as_ptr()) };
        if display.is_null() {
            return None;
        }
        Some(Self {
            xlib,
            xtest,
            display,
            window: 0,
        })
    }

    /// The application's window, once it exists.
    ///
    /// There is no window manager, so the only window under the root is the
    /// one the viewer created.
    fn wait_for_window(&mut self) -> bool {
        let deadline = time::Instant::now() + DEADLINE;
        while time::Instant::now() < deadline {
            unsafe {
                let screen = (self.xlib.XDefaultScreen)(self.display);
                let root = (self.xlib.XRootWindow)(self.display, screen);
                let mut returned_root = 0u64;
                let mut parent = 0u64;
                let mut children: *mut u64 = std::ptr::null_mut();
                let mut count = 0u32;
                (self.xlib.XQueryTree)(
                    self.display,
                    root,
                    &mut returned_root,
                    &mut parent,
                    &mut children,
                    &mut count,
                );
                if count > 0 {
                    self.window = *children;
                    (self.xlib.XFree)(children as *mut std::ffi::c_void);
                    return true;
                }
                if !children.is_null() {
                    (self.xlib.XFree)(children as *mut std::ffi::c_void);
                }
            }
            std::thread::sleep(POLL);
        }
        false
    }

    fn size(&self) -> (u32, u32) {
        unsafe {
            let mut root = 0u64;
            let (mut x, mut y) = (0i32, 0i32);
            let (mut width, mut height) = (0u32, 0u32);
            let (mut border, mut depth) = (0u32, 0u32);
            (self.xlib.XGetGeometry)(
                self.display,
                self.window,
                &mut root,
                &mut x,
                &mut y,
                &mut width,
                &mut height,
                &mut border,
                &mut depth,
            );
            (width, height)
        }
    }

    /// The window's own pixels, as one byte per channel per pixel.
    fn capture(&self) -> Option<Frame> {
        let (width, height) = self.size();
        if width == 0 || height == 0 {
            return None;
        }
        unsafe {
            let image = (self.xlib.XGetImage)(
                self.display,
                self.window,
                0,
                0,
                width,
                height,
                !0,
                x11_dl::xlib::ZPixmap,
            );
            if image.is_null() {
                return None;
            }
            let stride = (*image).bytes_per_line as usize;
            let bytes = ((*image).bits_per_pixel / 8) as usize;
            let data = (*image).data as *const u8;
            let mut pixels = Vec::with_capacity((width * height) as usize * 3);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let base = y * stride + x * bytes;
                    // Whatever the channel order is, it is the same in every
                    // frame, and only differences between frames are compared.
                    for channel in 0..3 {
                        pixels.push(*data.add(base + channel));
                    }
                }
            }
            if let Some(destroy) = (*image).funcs.destroy_image {
                destroy(image);
            }
            Some(Frame {
                width,
                height,
                pixels,
            })
        }
    }

    fn focus(&self) {
        unsafe {
            // A bare X server assigns no input focus, so a synthetic key would
            // otherwise be delivered nowhere at all.
            let (width, height) = self.size();
            (self.xlib.XWarpPointer)(
                self.display,
                0,
                self.window,
                0,
                0,
                0,
                0,
                (width / 2) as i32,
                (height * 3 / 4) as i32,
            );
            (self.xlib.XSetInputFocus)(self.display, self.window, x11_dl::xlib::RevertToParent, 0);
            (self.xlib.XFlush)(self.display);
        }
    }

    fn key(&self, keysym: &str) {
        self.focus();
        unsafe {
            let name = std::ffi::CString::new(keysym).unwrap();
            let sym = (self.xlib.XStringToKeysym)(name.as_ptr());
            let code = (self.xlib.XKeysymToKeycode)(self.display, sym) as u32;
            assert!(code != 0, "no keycode for {keysym}");
            (self.xtest.XTestFakeKeyEvent)(self.display, code, 1, 0);
            (self.xtest.XTestFakeKeyEvent)(self.display, code, 0, 0);
            (self.xlib.XFlush)(self.display);
        }
    }

    /// Hold the left button and move, in the steps a hand produces.
    ///
    /// The viewer turns by the difference between consecutive positions, so a
    /// single jump would exercise a case no mouse ever delivers.
    fn drag(&self, dx: i32, dy: i32) {
        self.focus();
        let (width, height) = self.size();
        let (x, y) = ((width / 2) as i32, (height * 3 / 4) as i32);
        std::thread::sleep(time::Duration::from_millis(200));
        unsafe {
            (self.xtest.XTestFakeButtonEvent)(self.display, 1, 1, 0);
            (self.xlib.XFlush)(self.display);
            const STEPS: i32 = 12;
            for step in 1..=STEPS {
                (self.xlib.XWarpPointer)(
                    self.display,
                    0,
                    self.window,
                    0,
                    0,
                    0,
                    0,
                    x + dx * step / STEPS,
                    y + dy * step / STEPS,
                );
                (self.xlib.XFlush)(self.display);
                std::thread::sleep(time::Duration::from_millis(30));
            }
            (self.xtest.XTestFakeButtonEvent)(self.display, 1, 0, 0);
            (self.xlib.XFlush)(self.display);
        }
    }

    fn resize(&self, width: u32, height: u32) {
        unsafe {
            // No window manager to ask, so the window is resized directly.
            (self.xlib.XResizeWindow)(self.display, self.window, width, height);
            (self.xlib.XFlush)(self.display);
        }
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        unsafe {
            (self.xlib.XCloseDisplay)(self.display);
        }
    }
}

struct Frame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Frame {
    /// Mean absolute difference against another frame of the same size.
    ///
    /// The overlay occupies the top left, so it is left out: this compares
    /// what the renderer drew rather than what egui did.
    fn difference(&self, other: &Frame) -> f64 {
        assert_eq!((self.width, self.height), (other.width, other.height));
        const PANEL: u32 = 400;
        let mut total = 0.0f64;
        let mut count = 0usize;
        for y in 0..self.height {
            for x in 0..self.width {
                if x < PANEL && y < PANEL {
                    continue;
                }
                let base = ((y * self.width + x) * 3) as usize;
                for channel in 0..3 {
                    total += (self.pixels[base + channel] as f64
                        - other.pixels[base + channel] as f64)
                        .abs();
                    count += 1;
                }
            }
        }
        total / count.max(1) as f64
    }

    /// Whether anything was drawn at all, rather than a cleared window.
    fn has_content(&self) -> bool {
        let mut lowest = u8::MAX;
        let mut highest = u8::MIN;
        for value in &self.pixels {
            lowest = lowest.min(*value);
            highest = highest.max(*value);
        }
        highest.saturating_sub(lowest) > 8
    }
}

// ---------------------------------------------------------------- processes

/// Kills what it started, including when an assertion unwinds past it.
struct Child(process::Child);

impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Start a virtual X server on a display nobody is using.
fn start_xvfb() -> Option<(Child, String)> {
    for number in 90..100 {
        if path::Path::new(&format!("/tmp/.X{number}-lock")).exists() {
            continue;
        }
        let name = format!(":{number}");
        let Ok(child) = process::Command::new("Xvfb")
            .args([&name, "-screen", "0", "1400x900x24", "-nolisten", "tcp"])
            .stdout(process::Stdio::null())
            .stderr(process::Stdio::null())
            .spawn()
        else {
            return None;
        };
        let child = Child(child);
        // Wait for it to accept connections rather than guessing at a delay.
        let deadline = time::Instant::now() + time::Duration::from_secs(15);
        while time::Instant::now() < deadline {
            if Driver::open(&name).is_some() {
                return Some((child, name));
            }
            std::thread::sleep(POLL);
        }
    }
    None
}

/// A model small enough to build quickly and large enough to fill a frame.
fn write_asset(path: &path::Path) {
    let mut surfels = Vec::new();
    for index in 0..2048 {
        let angle = index as f32 * 2.399_963;
        let height = 1.0 - 2.0 * (index as f32 + 0.5) / 2048.0;
        let radius = (1.0 - height * height).max(0.0).sqrt();
        let normal = glam::Vec3::new(radius * angle.cos(), height, radius * angle.sin());
        surfels.push(vol::relight::Surfel {
            center: normal.into(),
            radius: 0.09,
            normal: normal.into(),
            material: (index % 2) as u32,
        });
    }
    let model = vol::relight::RelightModel {
        kernel: vol::relight::ParticleKernel::Compact,
        surfels,
        materials: vec![
            vol::relight::Material {
                albedo: [0.8, 0.3, 0.2],
                roughness: 0.7,
                specular_f0: [0.04; 3],
                _padding: 0.0,
            },
            vol::relight::Material {
                albedo: [0.2, 0.5, 0.8],
                roughness: 0.3,
                specular_f0: [0.04; 3],
                _padding: 0.0,
            },
        ],
    };
    vol::io::try_save_relight(path, &model).expect("cannot write the test asset");
}

/// Poll until the frame differs from `before`, and say how far it got.
fn wait_for_change(driver: &Driver, before: &Frame, what: &str) -> Frame {
    let deadline = time::Instant::now() + RESPONSE;
    let mut best = 0.0f64;
    while time::Instant::now() < deadline {
        std::thread::sleep(POLL);
        let Some(frame) = driver.capture() else {
            continue;
        };
        if frame.width != before.width || frame.height != before.height {
            continue;
        }
        let difference = frame.difference(before);
        best = best.max(difference);
        if difference > CHANGED {
            println!("{what}: the frame moved by {difference:.2}");
            return frame;
        }
    }
    panic!("{what} did not change the frame; the most it moved was {best:.3}");
}

#[test]
fn the_viewer_opens_a_window_and_answers_the_keyboard_and_mouse() {
    if vol::gpu::access_disabled() {
        println!("Skipping: GPU access disabled");
        return;
    }
    // A context of its own, dropped before the viewer starts one: if ray
    // tracing is unavailable the binary would panic, and a panic in a child
    // process is a worse diagnosis than a skip here.
    match unsafe {
        blade_graphics::Context::init(blade_graphics::ContextDesc {
            ray_tracing: true,
            ..Default::default()
        })
    } {
        Ok(context) => {
            if !context
                .capabilities()
                .ray_query
                .contains(blade_graphics::ShaderVisibility::COMPUTE)
            {
                println!("Skipping: ray_query in compute is not supported");
                return;
            }
        }
        Err(error) => {
            println!("Skipping: no ray tracing context: {error:?}");
            return;
        }
    }
    let Some((_xvfb, display_name)) = start_xvfb() else {
        println!("Skipping: no Xvfb to run a window in");
        return;
    };

    let asset = std::env::temp_dir().join(format!("blade-volume-smoke-{}.surfel", process::id()));
    write_asset(&asset);

    let viewer = process::Command::new(env!("CARGO_BIN_EXE_view"))
        .arg(&asset)
        .args(["--resolution", &format!("{WIDTH},{HEIGHT}")])
        // Small enough that switching a light is quick; this test is about the
        // application and not about the resolution of the specular ladder.
        .args(["--specular-size", "32"])
        .env("DISPLAY", &display_name)
        // The Mesa overlay would draw over the frame being compared.
        .env("VK_LOADER_LAYERS_DISABLE", "*overlay*")
        .stdout(process::Stdio::null())
        .stderr(process::Stdio::null())
        .spawn();
    let Ok(viewer) = viewer else {
        let _ = std::fs::remove_file(&asset);
        panic!("cannot start the viewer binary");
    };
    let mut viewer = Child(viewer);

    let mut driver = Driver::open(&display_name).expect("cannot connect to the virtual display");
    assert!(driver.wait_for_window(), "the viewer never opened a window");

    // A frame with something in it. The first one has to wait for the
    // acceleration structures and the first prefiltered environment.
    let deadline = time::Instant::now() + DEADLINE;
    let mut frame = loop {
        assert!(
            viewer.0.try_wait().ok().flatten().is_none(),
            "the viewer exited before it drew anything"
        );
        if let Some(frame) = driver.capture() {
            if frame.has_content() {
                break frame;
            }
        }
        assert!(
            time::Instant::now() < deadline,
            "the viewer never drew anything"
        );
        std::thread::sleep(POLL);
    };
    assert_eq!((frame.width, frame.height), (WIDTH, HEIGHT));

    // Nothing moves on its own, which is what makes every difference below
    // attributable to the input that preceded it.
    driver.focus();
    std::thread::sleep(POLL);
    let settled = driver.capture().expect("cannot capture");
    std::thread::sleep(POLL);
    let again = driver.capture().expect("cannot capture");
    let drift = again.difference(&settled);
    assert!(
        drift < UNCHANGED,
        "the frame changes on its own by {drift:.3}, so nothing below proves anything"
    );
    frame = again;

    // The light is a control rather than a property of the asset: this is the
    // claim the whole representation rests on, in the window.
    driver.key("l");
    frame = wait_for_change(&driver, &frame, "pressing L");

    // Looking around. Turning to face nothing at all would also count as a
    // change, so the frame still has to have something in it afterwards.
    driver.drag(160, -40);
    frame = wait_for_change(&driver, &frame, "dragging");
    assert!(frame.has_content(), "dragging emptied the frame");

    // Resizing rebuilds the render target underneath a live swapchain, which
    // is where a viewer usually crashes.
    for (width, height) in [(1200u32, 800u32), (420, 300), (64, 64)] {
        driver.resize(width, height);
        let deadline = time::Instant::now() + RESPONSE;
        loop {
            assert!(
                viewer.0.try_wait().ok().flatten().is_none(),
                "the viewer died on resize to {width}x{height}"
            );
            if driver.size() == (width, height) {
                if let Some(resized) = driver.capture() {
                    if resized.has_content() {
                        frame = resized;
                        break;
                    }
                }
            }
            assert!(
                time::Instant::now() < deadline,
                "the window never came back after resizing to {width}x{height}"
            );
            std::thread::sleep(POLL);
        }
        assert_eq!((frame.width, frame.height), (width, height));
    }

    assert!(
        viewer.0.try_wait().ok().flatten().is_none(),
        "the viewer exited during the test"
    );
    let _ = std::fs::remove_file(&asset);
}
