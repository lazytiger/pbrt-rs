use crate::core::geometry::{Bounds3, Bounds3f, Point2f, Point3f, Ray, Vector3f};
use crate::core::interaction::{Interaction, SurfaceInteraction};
use crate::core::lowdiscrepancy::radical_inverse;
use crate::core::medium::MediumInterface;
use crate::core::transform::Transformf;
use crate::Float;
use std::any::Any;

pub trait Shape {
    fn as_any(&self) -> &dyn Any;
    fn object_bound(&self) -> Bounds3f;
    fn world_bound(&self) -> Bounds3f {
        self.object_to_world() * &self.object_bound()
    }
    fn intersect(
        &self,
        ray: &Ray,
        hit: &mut Float,
        si: &mut SurfaceInteraction,
        test_alpha_texture: bool,
    ) -> bool;
    fn intersect_p(&self, ray: &Ray, test_alpha_texture: bool) -> bool {
        let mut hit: Float = 0.0;
        let mut si: SurfaceInteraction = Default::default();
        self.intersect(ray, &mut hit, &mut si, test_alpha_texture)
    }
    fn area(&self) -> Float;
    fn sample(&self, u: &Point2f, pdf: &mut Float) -> Interaction;
    fn pdf(&self, si: &Interaction) -> Float {
        1.0 / self.area()
    }
    fn sample2(&self, it: &Interaction, u: &Point2f, pdf: &mut Float) -> Interaction {
        let intr = self.sample(u, pdf);
        let mut wi = intr.p - it.p;
        if wi.length_squared() == 0.0 {
            *pdf = 0.0;
        } else {
            wi = wi.normalize();
            *pdf *= it.p.distance_square(&intr.p) / intr.n.abs_dot(&-wi);
            if pdf.is_infinite() {
                *pdf = 0.0
            }
        }
        intr
    }
    fn pdf2(&self, it: &Interaction, wi: &Vector3f) -> Float {
        let ray = it.spawn_ray(wi);
        let mut hit: Float = 0.0;
        let mut isect_light = Default::default();
        if !self.intersect(&ray, &mut hit, &mut isect_light, false) {
            return 0.0;
        }

        let mut pdf =
            it.p.distance_square(&isect_light.p) / (isect_light.n.abs_dot(&-*wi) * self.area());
        if pdf.is_infinite() {
            pdf = 0.0;
        }
        pdf
    }
    fn reverse_orientation(&self) -> bool;
    fn transform_swap_handedness(&self) -> bool;
    fn object_to_world(&self) -> &Transformf;
    fn world_to_object(&self) -> &Transformf;
    fn solid_angle(&self, p: &Point3f, samples: u64) -> Float {
        let it = Interaction::new(
            *p,
            Default::default(),
            0.0,
            Default::default(),
            Vector3f::new(0.0, 0.0, 1.0),
            Default::default(),
        );
        let mut solid_angle = 0.0;
        for i in 0..samples {
            let u = Point2f::new(radical_inverse(0, i), radical_inverse(1, i));
            let mut pdf = 0.0;
            let it = self.sample(&u, &mut pdf);
            if pdf > 0.0 && !self.intersect_p(&Ray::new(*p, it.p - *p, 0.999, 0.0, None), true) {
                solid_angle += 1.0 / pdf;
            }
        }
        solid_angle / samples as f32
    }
}
