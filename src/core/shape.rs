use crate::core::geometry::{Bounds3f, Ray, SurfaceInteraction};
use crate::Float;

pub trait Shape {
    fn object_bound(&self) -> Bounds3f;
    fn world_bound(&self) -> Bounds3f;
    fn intersect(
        &self,
        ray: &Ray,
        hit: &mut Option<Float>,
        isect: &mut Option<SurfaceInteraction>,
        test_alpha_texture: bool,
    ) -> bool;
    fn intersectp(&self, ray: &Ray, test_alpha_texture: bool) -> bool {
        self.intersect(ray, &mut None, &mut None, test_alpha_texture)
    }
}
