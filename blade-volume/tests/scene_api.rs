//! Integration tests for the Scene API.
//!
//! Tests scene object management, transforms, and software TLAS.
//! GPU tests are skipped if no supported device is found.

use blade_graphics as gpu;
use blade_volume as vol;
use std::sync;

mod radfoam_synth_chain;

/// Create a headless GPU context for tests.
fn make_test_context() -> Option<gpu::Context> {
    if vol::gpu::access_disabled() {
        return None;
    }
    unsafe {
        match gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: cfg!(debug_assertions),
            timing: false,
            capture: false,
            overlay: false,
            ray_tracing: true,
            xr: None,
            device_id: None,
        }) {
            Ok(ctx) => Some(ctx),
            Err(gpu::NotSupportedError::NoSupportedDeviceFound) => None,
            Err(other) => panic!("failed to init GPU context: {:?}", other),
        }
    }
}

fn gpu_test_guard() -> sync::MutexGuard<'static, ()> {
    static GPU_TEST_LOCK: sync::OnceLock<sync::Mutex<()>> = sync::OnceLock::new();
    GPU_TEST_LOCK
        .get_or_init(|| sync::Mutex::new(()))
        .lock()
        .expect("lock gpu test mutex")
}

#[test]
fn scene_new_is_empty() {
    let scene = vol::Scene::new();
    assert_eq!(scene.object_count(), 0);
    assert!(scene.render_data().is_none());
}

#[test]
fn transform_constructors() {
    let t1 = vol::Transform::identity();
    assert_eq!(t1.position, glam::Vec3::ZERO);
    assert_eq!(t1.rotation, glam::Quat::IDENTITY);
    assert_eq!(t1.scale, glam::Vec3::ONE);

    let pos = glam::Vec3::new(1.0, 2.0, 3.0);
    let t2 = vol::Transform::from_position(pos);
    assert_eq!(t2.position, pos);
    assert_eq!(t2.rotation, glam::Quat::IDENTITY);
    assert_eq!(t2.scale, glam::Vec3::ONE);

    let rot = glam::Quat::from_rotation_y(std::f32::consts::PI);
    let t3 = vol::Transform::from_position_rotation(pos, rot);
    assert_eq!(t3.position, pos);
    assert!((t3.rotation.x - rot.x).abs() < 1e-6);
    assert!((t3.rotation.y - rot.y).abs() < 1e-6);
    assert!((t3.rotation.z - rot.z).abs() < 1e-6);
    assert!((t3.rotation.w - rot.w).abs() < 1e-6);
}

#[test]
fn transform_to_matrix_translation() {
    let t = vol::Transform {
        position: glam::Vec3::new(5.0, 10.0, 15.0),
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::ONE,
    };
    let m = t.to_matrix();
    let result = m.transform_point3(glam::Vec3::ZERO);
    assert!((result - glam::Vec3::new(5.0, 10.0, 15.0)).length() < 1e-5);
}

#[test]
fn transform_to_matrix_scale() {
    let t = vol::Transform {
        position: glam::Vec3::ZERO,
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::new(2.0, 3.0, 4.0),
    };
    let m = t.to_matrix();
    let result = m.transform_vector3(glam::Vec3::ONE);
    assert!((result - glam::Vec3::new(2.0, 3.0, 4.0)).length() < 1e-5);
}

#[test]
fn transform_to_matrix_rotation() {
    let t = vol::Transform {
        position: glam::Vec3::ZERO,
        rotation: glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        scale: glam::Vec3::ONE,
    };
    let m = t.to_matrix();
    let result = m.transform_vector3(glam::Vec3::X);
    // Rotating X by 90 degrees around Z gives Y
    assert!((result - glam::Vec3::Y).length() < 1e-5);
}

#[test]
fn scene_add_radfoam_object() {
    let _guard = gpu_test_guard();
    let Some(context) = make_test_context() else {
        eprintln!("Skipping scene GPU test: no supported device found");
        return;
    };

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "scene-test",
        buffer_count: 1,
        manual_barriers: false,
    });

    // Create a simple chain model with adjacency
    let model = radfoam_synth_chain::make_chain_model(10, 0.5, 0.1, 0, glam::Vec3::splat(0.5));

    let mut scene = vol::Scene::new();
    let handle = scene.add_radfoam(&model, &context, &mut encoder);

    assert_eq!(scene.object_count(), 1);

    // Check object type
    assert_eq!(
        scene.get_object_type(handle),
        Some(vol::ObjectType::RadFoam)
    );

    // Check transform
    let t = scene.get_transform(handle).expect("should have transform");
    assert_eq!(t.position, glam::Vec3::ZERO);

    // Set transform
    scene.set_transform(
        handle,
        vol::Transform::from_position(glam::Vec3::new(1.0, 2.0, 3.0)),
    );
    let t2 = scene.get_transform(handle).expect("should have transform");
    assert_eq!(t2.position, glam::Vec3::new(1.0, 2.0, 3.0));

    // Check RadFoam cloud access
    let cloud = scene
        .get_radfoam_cloud(handle)
        .expect("should have RadFoam cloud");
    assert!(cloud.num_points > 0);

    // Cleanup
    scene.destroy(&context);
    context.destroy_command_encoder(&mut encoder);
}

#[test]
fn scene_multiple_radfoam_objects() {
    let _guard = gpu_test_guard();
    let Some(context) = make_test_context() else {
        eprintln!("Skipping scene GPU test: no supported device found");
        return;
    };

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "scene-test",
        buffer_count: 1,
        manual_barriers: false,
    });

    let model1 = radfoam_synth_chain::make_chain_model(5, 0.5, 0.1, 0, glam::Vec3::splat(0.5));
    let model2 = radfoam_synth_chain::make_chain_model(8, 0.3, 0.2, 1, glam::Vec3::splat(0.3));

    let mut scene = vol::Scene::new();
    let h1 = scene.add_radfoam(&model1, &context, &mut encoder);
    let h2 = scene.add_radfoam(&model2, &context, &mut encoder);

    assert_eq!(scene.object_count(), 2);
    assert_ne!(h1, h2); // Different handles

    // Set different transforms
    scene.set_transform(h1, vol::Transform::from_position(glam::Vec3::X));
    scene.set_transform(h2, vol::Transform::from_position(glam::Vec3::Y));

    assert_eq!(
        scene.get_transform(h1).expect("h1 transform").position,
        glam::Vec3::X
    );
    assert_eq!(
        scene.get_transform(h2).expect("h2 transform").position,
        glam::Vec3::Y
    );

    // Cleanup
    scene.destroy(&context);
    context.destroy_command_encoder(&mut encoder);
}

#[test]
fn scene_default_trait() {
    let scene: vol::Scene = Default::default();
    assert_eq!(scene.object_count(), 0);
}

#[test]
fn scene_prepare_creates_buffers() {
    let _guard = gpu_test_guard();
    let Some(context) = make_test_context() else {
        eprintln!("Skipping scene GPU test: no supported device found");
        return;
    };

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "scene-test",
        buffer_count: 1,
        manual_barriers: false,
    });

    let model = radfoam_synth_chain::make_chain_model(10, 0.5, 0.1, 0, glam::Vec3::splat(0.5));

    let mut scene = vol::Scene::new();
    let handle = scene.add_radfoam(&model, &context, &mut encoder);

    // Before prepare, render_data should be None (no buffers)
    assert!(scene.render_data().is_none());

    // After prepare, render_data should be available
    scene.prepare(&context, &mut encoder);
    let render_data = scene.render_data().expect("should have render data");
    assert_eq!(render_data.object_count, 1);
    assert_eq!(render_data.radfoam_clouds.len(), 1);

    // Set transform and re-prepare
    scene.set_transform(
        handle,
        vol::Transform::from_position(glam::Vec3::new(1.0, 2.0, 3.0)),
    );
    scene.prepare(&context, &mut encoder);

    // Cleanup
    scene.destroy(&context);
    context.destroy_command_encoder(&mut encoder);
}

#[test]
fn scene_render_data_reflects_all_objects() {
    let _guard = gpu_test_guard();
    let Some(context) = make_test_context() else {
        eprintln!("Skipping scene GPU test: no supported device found");
        return;
    };

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "scene-test",
        buffer_count: 1,
        manual_barriers: false,
    });

    let model1 = radfoam_synth_chain::make_chain_model(5, 0.5, 0.1, 0, glam::Vec3::splat(0.5));
    let model2 = radfoam_synth_chain::make_chain_model(8, 0.3, 0.2, 1, glam::Vec3::splat(0.3));

    let mut scene = vol::Scene::new();
    let _h1 = scene.add_radfoam(&model1, &context, &mut encoder);
    let _h2 = scene.add_radfoam(&model2, &context, &mut encoder);

    scene.prepare(&context, &mut encoder);
    let render_data = scene.render_data().expect("should have render data");

    assert_eq!(render_data.object_count, 2);
    assert_eq!(render_data.radfoam_clouds.len(), 2);
    assert_eq!(render_data.gaussian_clouds.len(), 0);

    // Cleanup
    scene.destroy(&context);
    context.destroy_command_encoder(&mut encoder);
}

#[test]
fn scene_empty_render_data() {
    let scene = vol::Scene::new();
    assert!(scene.render_data().is_none());
}
