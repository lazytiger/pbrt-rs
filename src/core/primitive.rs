use crate::core::geometry::{Bounds3f, Ray};
use crate::core::interaction::SurfaceInteraction;
use crate::core::shape::Shape;

pub trait Primitive {
    fn world_bound(&self) -> Bounds3f;
    fn intersect<T: Shape, P: Primitive>(&self, r: &Ray, si: &mut SurfaceInteraction<T, P>)
        -> bool;
    fn intersect_p(&self, r: &Ray) -> bool;
}
