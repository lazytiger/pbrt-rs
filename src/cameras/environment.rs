use crate::core::camera::{BaseCamera, Camera, CameraSample};
use crate::core::geometry::Ray;
use crate::impl_base_camera;
use std::any::Any;

pub struct EnvironmentCamera {
    base: BaseCamera,
}

impl Camera for EnvironmentCamera {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f32 {
        unimplemented!()
    }

    impl_base_camera!();
}
