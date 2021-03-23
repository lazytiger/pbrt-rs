use crate::core::geometry::Vector3f;
use crate::core::interaction::SurfaceInteraction;
use crate::core::spectrum::Spectrum;
use std::any::Any;

pub trait Light {
    fn as_any(&self) -> &dyn Any;
}

pub trait AreaLight: Light {
    fn l(&self, si: &SurfaceInteraction, v: &Vector3f) -> Spectrum;
}
