use crate::core::camera::{BaseCamera, Camera, CameraSample};
use crate::core::film::Film;
use crate::core::geometry::{Bounds2f, Point2f, Point3f, Ray, RayDifferentials, Vector3f};
use crate::core::interaction::Interaction;
use crate::core::light::VisibilityTester;
use crate::core::medium::Medium;
use crate::core::spectrum::Spectrum;
use crate::core::transform::{AnimatedTransform, Point3Ref, Transformf, Vector3Ref};
use crate::{impl_base_camera, Float};
use std::any::Any;
use std::sync::Arc;

pub struct PerspectiveCamera {
    base: BaseCamera,
    dx_camera: Vector3f,
    dy_camera: Vector3f,
    camera_to_screen: Transformf,
    raster_to_camera: Transformf,
    screen_to_raster: Transformf,
    raster_to_screen: Transformf,
    lens_radius: Float,
    focal_distance: Float,
    a: Float,
}

impl PerspectiveCamera {
    pub fn new(
        camera_to_world: AnimatedTransform,
        screen_window: Bounds2f,
        shutter_open: Float,
        shutter_close: Float,
        lens_radius: Float,
        focal_distance: Float,
        fov: Float,
        film: Arc<Film>,
        medium: Arc<Box<dyn Medium>>,
    ) -> PerspectiveCamera {
        let camera_to_screen = Transformf::perspective(fov, 1e-2, 1000.0);
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
        let res = film.full_resolution;
        let mut p_min = &raster_to_camera * Point3Ref(&Point3f::new(0.0, 0.0, 0.0));
        let mut p_max =
            &raster_to_camera * Point3Ref(&Point3f::new(res.x as Float, res.y as Float, 0.0));
        p_min /= p_min.z;
        p_max /= p_max.z;
        let a = ((p_max.x - p_min.x) * (p_max.y - p_min.y)).abs();
        PerspectiveCamera {
            base: BaseCamera::new(camera_to_world, shutter_open, shutter_close, film, medium),
            dx_camera: (&raster_to_camera * Point3Ref(&Vector3f::new(1.0, 0.0, 0.0))
                - &raster_to_camera * Point3Ref(&Point3f::new(0.0, 0.0, 0.0))),
            dy_camera: (&raster_to_camera * Point3Ref(&Vector3f::new(0.0, 1.0, 0.0))
                - &raster_to_camera * Point3Ref(&Point3f::new(0.0, 0.0, 0.0))),
            a,
            camera_to_screen,
            lens_radius,
            focal_distance,
            screen_to_raster,
            raster_to_camera,
            raster_to_screen: screen_to_raster.inverse(),
        }
    }
}

impl Camera for PerspectiveCamera {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f32 {
        unimplemented!()
    }

    fn generate_ray_differential(&self, sample: &CameraSample, rd: &mut RayDifferentials) -> f32 {
        unimplemented!()
    }

    fn we(&self, ray: &Ray, p_raster2: Option<&mut Point2f>) -> Spectrum {
        unimplemented!()
    }

    fn pdf_we(&self, ray: &Ray, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        unimplemented!()
    }

    fn sample_wi(
        &self,
        refer: &Interaction,
        sample: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut f32,
        p_raster: Option<&mut Point2f>,
        vis: &mut VisibilityTester,
    ) -> Spectrum {
        unimplemented!()
    }

    impl_base_camera!();
}
