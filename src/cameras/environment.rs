use crate::core::camera::{BaseCamera, Camera, CameraSample};
use crate::core::film::Film;
use crate::core::geometry::{Point3f, Ray, Vector3f};
use crate::core::lerp;
use crate::core::medium::Medium;
use crate::core::transform::AnimatedTransform;
use crate::{impl_base_camera, Float, PI};
use std::any::Any;
use std::sync::Arc;

pub struct EnvironmentCamera {
    base: BaseCamera,
}

impl EnvironmentCamera {
    pub fn new(
        camera_to_world: AnimatedTransform,
        shutter_open: Float,
        shutter_close: Float,
        film: Arc<Film>,
        medium: Arc<Box<Medium>>,
    ) -> Self {
        EnvironmentCamera {
            base: BaseCamera::new(camera_to_world, shutter_open, shutter_close, film, medium),
        }
    }
}

impl Camera for EnvironmentCamera {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f32 {
        let theta = PI * sample.p_film.y / self.film().full_resolution.y as Float;
        let phi = 2.0 * PI * sample.p_film.x / self.film().full_resolution.x as Float;
        let dir = Vector3f::new(
            theta.sin() * phi.cos(),
            theta.cos(),
            theta.sin() * phi.sin(),
        );
        *ray = Ray::new(
            Point3f::new(0.0, 0.0, 0.0),
            dir,
            Float::INFINITY,
            lerp(sample.time, self.shutter_open(), self.shutter_close()),
            None,
        );
        ray.medium = Some(self.medium());
        *ray = Ray::from((self.camera_to_world(), &*ray));
        1.0
    }

    impl_base_camera!();
}
