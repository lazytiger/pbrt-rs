use crate::{
    core::{
        camera::{BaseCamera, Camera, CameraSample},
        film::Film,
        geometry::{Point3f, Ray, Vector3f},
        medium::{Medium, MediumDt},
        pbrt::{lerp, Float, PI},
        transform::AnimatedTransform,
    },
    impl_base_camera,
};
use std::{any::Any, sync::Arc};

pub struct EnvironmentCamera {
    base: BaseCamera,
}

impl EnvironmentCamera {
    pub fn new(
        camera_to_world: AnimatedTransform,
        shutter_open: Float,
        shutter_close: Float,
        film: Arc<Film>,
        medium: MediumDt,
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
