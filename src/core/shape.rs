use crate::core::geometry::{Bounds3, Bounds3f, Point3f, Ray, SurfaceInteraction};
use crate::core::transform::Transformf;
use crate::Float;

pub trait Shape {
    fn object_bound(&self) -> Bounds3f;
    fn world_bound(&self) -> Bounds3f {
        //self.object_to_world() * self.object_bound()
        Bounds3f::new()
    }
    fn intersect(
        &self,
        ray: &Ray,
        hit: &mut Option<Float>,
        isect: &mut Option<SurfaceInteraction>,
        test_alpha_texture: bool,
    ) -> bool;
    fn intersect_p(&self, ray: &Ray, test_alpha_texture: bool) -> bool {
        self.intersect(ray, &mut None, &mut None, test_alpha_texture)
    }
    fn reverse_orientation(&self) -> bool;
    fn transform_swap_handedness(&self) -> bool;
    fn object_to_world(&self) -> &Transformf;
    fn world_to_object(&self) -> &Transformf;
    fn solid_angle(&self, p: &Point3f, samples: i32) -> Float {
        0.0
    }
}
