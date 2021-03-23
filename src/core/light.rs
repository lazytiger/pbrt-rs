use crate::core::geometry::Vector3f;
use crate::core::interaction::SurfaceInteraction;
use crate::core::spectrum::Spectrum;

pub trait Light {}

pub trait AreaLight: Light {
    fn l(&self, si: &SurfaceInteraction, v: &Vector3f) -> Spectrum;
}
