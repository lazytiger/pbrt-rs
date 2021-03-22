use crate::core::geometry::{Bounds3f, Ray};
use crate::core::interaction::SurfaceInteraction;
use crate::core::light::AreaLight;
use crate::core::material::{Material, TransportMode};
use crate::core::shape::Shape;

pub trait Primitive {
    fn world_bound(&self) -> Bounds3f;
    fn intersect(&self, r: &Ray, si: &mut SurfaceInteraction) -> bool;
    fn intersect_p(&self, r: &Ray) -> bool;
    fn get_area_light(&self) -> Option<&AreaLight>;
    fn get_material(&self) -> Option<&Material>;
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    );
}
