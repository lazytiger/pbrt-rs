use crate::core::interaction::SurfaceInteraction;
use std::any::Any;

pub enum TransportMode {
    Radiance,
    Importance,
}

pub trait Material {
    fn as_any(&self) -> &dyn Any;
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    );
}
