use crate::core::camera::{BaseCamera, Camera, CameraSample};
use crate::core::film::Film;
use crate::core::geometry::{Bounds2f, Point2f, Point3f, Ray, RayDifferentials, Vector3f};
use crate::core::interaction::Interaction;
use crate::core::light::VisibilityTester;
use crate::core::medium::Medium;
use crate::core::pbrt::lerp;
use crate::core::pbrt::Float;
use crate::core::sampling::concentric_sample_disk;
use crate::core::spectrum::Spectrum;
use crate::core::transform::{AnimatedTransform, Point3Ref, Transform, Transformf, Vector3Ref};
use crate::impl_base_camera;
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

    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> Float {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = &self.raster_to_camera * Point3Ref(&p_film);
        *ray = Ray::new(
            p_camera,
            Vector3f::new(0.0, 0.0, 1.0),
            Float::INFINITY,
            0.0,
            None,
        );
        if self.lens_radius > 0.0 {
            let p_lens = concentric_sample_disk(&sample.p_lens) * self.lens_radius;
            let ft = self.focal_distance / ray.d.z;
            let p_focus = ray.point(ft);
            ray.o = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.d = (p_focus - ray.o).normalize();
        }

        ray.time = lerp(sample.time, self.shutter_open(), self.shutter_close());
        ray.medium = Some(self.medium());
        *ray = (self.camera_to_world(), &*ray).into();
        1.0
    }

    fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        ray: &mut RayDifferentials,
    ) -> Float {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = &self.raster_to_camera * Point3Ref(&p_film);
        *ray = RayDifferentials::new(
            p_camera,
            Vector3f::new(0.0, 0.0, 1.0),
            Float::INFINITY,
            0.0,
            None,
        );
        if self.lens_radius > 0.0 {
            let p_lens = concentric_sample_disk(&sample.p_lens) * self.lens_radius;
            let ft = self.focal_distance / ray.d.z;
            let p_focus = ray.point(ft);
            ray.o = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.d = (p_focus - ray.o).normalize();
        }
        if self.lens_radius > 0.0 {
            let p_lens = concentric_sample_disk(&sample.p_lens) * self.lens_radius;
            let ft = self.focal_distance / ray.d.z;
            let mut p_focus = p_camera + self.dx_camera + Vector3f::new(0.0, 0.0, 1.0) * ft;
            ray.rx_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.rx_direction = (p_focus - ray.rx_origin).normalize();

            p_focus = p_camera + self.dy_camera + Vector3f::new(0.0, 0.0, 1.0) * ft;
            ray.ry_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.ry_direction = (p_focus - ray.ry_origin).normalize();
        } else {
            ray.rx_origin = ray.o + self.dx_camera;
            ray.ry_origin = ray.o + self.dy_camera;
            ray.rx_direction = ray.d;
            ray.ry_direction = ray.d;
        }

        ray.time = lerp(sample.time, self.shutter_open(), self.shutter_close());
        ray.medium = Some(self.medium());
        ray.has_differentials = true;
        *ray = (self.camera_to_world(), &*ray).into();
        1.0
    }

    impl_base_camera!();
}
