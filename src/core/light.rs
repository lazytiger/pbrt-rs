use crate::core::geometry::Vector3f;
use crate::core::interaction::SurfaceInteraction;
use crate::core::spectrum::Spectrum;

pub struct Light {}

pub struct AreaLight {}

impl AreaLight {
    pub fn l(&self, si: &SurfaceInteraction, v: &Vector3f) -> Spectrum {
        Spectrum {}
    }
}
