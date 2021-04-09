use crate::core::{
    geometry::Vector3f,
    interaction::{Interaction, SurfaceInteraction},
    spectrum::Spectrum,
};
use std::any::Any;

pub trait Light {
    fn as_any(&self) -> &dyn Any;
}

pub trait AreaLight: Light {
    fn l(&self, si: &SurfaceInteraction, v: &Vector3f) -> Spectrum;
}

pub struct VisibilityTester {
    p0: Interaction,
    p1: Interaction,
}

impl VisibilityTester {
    pub fn new(p0: Interaction, p1: Interaction) -> VisibilityTester {
        VisibilityTester { p0, p1 }
    }
}
