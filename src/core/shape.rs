use crate::core::{
    geometry::{Bounds3, Bounds3f, Point2f, Point3f, Ray, Vector3f},
    interaction::{BaseInteraction, Interaction, InteractionDt, SurfaceInteraction},
    lowdiscrepancy::radical_inverse,
    medium::MediumInterface,
    pbrt::Float,
    transform::Transformf,
};
use std::{
    any::Any,
    fmt::Debug,
    sync::{Arc, Mutex, RwLock},
};

pub trait Shape: Debug {
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
    fn sample(&self, u: &Point2f, pdf: &mut Float) -> InteractionDt;
    fn pdf(&self, _si: InteractionDt) -> Float {
        1.0 / self.area()
    }
    fn sample2(&self, refer: InteractionDt, u: &Point2f, pdf: &mut Float) -> InteractionDt {
        let it = refer.as_base();
        let itdt = self.sample(u, pdf);
        let intr = itdt.as_base();
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
        itdt
    }
    fn pdf2(&self, it: InteractionDt, wi: &Vector3f) -> Float {
        let it = it.as_base();
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
    fn solid_angle(&self, p: Point3f, samples: u64) -> Float {
        let it: InteractionDt = Arc::new(Box::new(BaseInteraction::new(
            p,
            Default::default(),
            0.0,
            Default::default(),
            Vector3f::new(0.0, 0.0, 1.0),
            Default::default(),
        )));
        let mut solid_angle = 0.0;
        for i in 0..samples {
            let u = Point2f::new(radical_inverse(0, i), radical_inverse(1, i));
            let mut pdf = 0.0;
            let p_shape = self.sample2(it.clone(), &u, &mut pdf);
            if pdf > 0.0
                && !self.intersect_p(
                    &Ray::new(p, p_shape.as_base().p - p, 0.999, 0.0, None),
                    true,
                )
            {
                solid_angle += 1.0 / pdf;
            }
        }
        solid_angle / samples as f32
    }
}

pub type ShapeDt = Arc<Box<dyn Shape>>;
pub type ShapeDtMut = Arc<Mutex<Box<dyn Shape>>>;
pub type ShapeDtRw = Arc<RwLock<Box<dyn Shape>>>;
