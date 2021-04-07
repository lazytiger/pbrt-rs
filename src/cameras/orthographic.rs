use crate::core::camera::{BaseCamera, Camera, CameraSample};
use crate::core::film::Film;
use crate::core::geometry::{Bounds2f, Point2f, Ray, RayDifferentials, Vector3f};
use crate::core::interaction::Interaction;
use crate::core::light::VisibilityTester;
use crate::core::medium::Medium;
use crate::core::spectrum::Spectrum;
use crate::core::transform::{AnimatedTransform, Transform, Transformf, Vector3Ref};
use crate::{impl_base_camera, Float};
use log::Level::Trace;
use std::any::Any;
use std::sync::Arc;

pub struct OrthographicCamera {
    base: BaseCamera,
    dx_camera: Vector3f,
    dy_camera: Vector3f,
    camera_to_screen: Transformf,
    raster_to_camera: Transformf,
    screen_to_raster: Transformf,
    raster_to_screen: Transformf,
    lens_radius: Float,
    focal_distance: Float,
}

impl OrthographicCamera {
    pub fn new(
        camera_to_world: AnimatedTransform,
        screen_window: Bounds2f,
        shutter_open: Float,
        shutter_close: Float,
        lens_radius: Float,
        focal_distance: Float,
        film: Arc<Film>,
        medium: Arc<Box<dyn Medium>>,
    ) -> OrthographicCamera {
        let camera_to_screen = Transformf::orthographic(0.0, 1.0);
        let screen_to_raster = Transformf::scale(
            film.full_resolution.x as Float,
            film.full_resolution.y as Float,
            1.0,
        ) * Transformf::scale(
            1.0 / (screen_window.max.x - screen_window.min.x),
            1.0 / (screen_window.min.y - screen_window.max.y),
            1.0,
        ) * Transformf::translate(&Vector3f::new(
            -screen_window.min.x,
            -screen_window.max.y,
            0.0,
        ));
        let raster_to_screen = screen_to_raster.inverse();
        let raster_to_camera = camera_to_screen.inverse() * raster_to_screen;
        OrthographicCamera {
            base: BaseCamera::new(camera_to_world, shutter_open, shutter_close, film, medium),
            dx_camera: &raster_to_camera * Vector3Ref(&Vector3f::new(1.0, 0.0, 0.0)),
            dy_camera: &raster_to_camera * Vector3Ref(&Vector3f::new(0.0, 1.0, 0.0)),
            camera_to_screen,
            lens_radius,
            focal_distance,
            screen_to_raster,
            raster_to_camera,
            raster_to_screen: screen_to_raster.inverse(),
        }
    }
}

impl Camera for OrthographicCamera {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f32 {
        unimplemented!()
    }

    fn generate_ray_differential(&self, sample: &CameraSample, rd: &mut RayDifferentials) -> f32 {
        unimplemented!()
    }

    impl_base_camera!();
}
