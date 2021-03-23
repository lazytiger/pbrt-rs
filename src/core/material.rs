use crate::core::interaction::SurfaceInteraction;

pub enum TransportMode {
    Radiance,
    Importance,
}

pub trait Material {
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    );
}
