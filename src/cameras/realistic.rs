use crate::core::camera::{BaseCamera, Camera, CameraSample};
use crate::core::geometry::{Bounds2f, Ray};
use crate::core::RealNum;
use crate::{impl_base_camera, Float};
use std::any::Any;

struct LensElementInterface {
    curvature_radius: Float,
    thickness: Float,
    eta: Float,
    aperture_radius: Float,
}
pub struct RealisticCamera {
    base: BaseCamera,
    simple_weighting: bool,
    element_interfaces: Vec<LensElementInterface>,
    exit_pupil_bounds: Vec<Bounds2f>,
}

impl Camera for RealisticCamera {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f32 {
        unimplemented!()
    }

    impl_base_camera!();
}
