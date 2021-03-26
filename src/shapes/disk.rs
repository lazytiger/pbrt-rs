use crate::core::efloat::EFloat;
use crate::core::geometry::{Bounds3f, Normal3f, Point2f, Point3f, Ray, Vector3f};
use crate::core::interaction::{Interaction, SurfaceInteraction};
use crate::core::sampling::concentric_sample_disk;
use crate::core::shape::Shape;
use crate::core::transform::{Normal3Ref, Point3Ref, Transformf};
use crate::core::{clamp, radians};
use crate::shapes::BaseShape;
use crate::{impl_base_shape, Float, PI};

pub struct Disk {
    base: BaseShape,
    height: Float,
    radius: Float,
    inner_radius: Float,
    phi_max: Float,
}

impl Disk {
    pub fn new(
        o2w: Transformf,
        w2o: Transformf,
        reverse_orientation: bool,
        height: Float,
        radius: Float,
        inner_radius: Float,
        phi_max: Float,
    ) -> Self {
        Self {
            base: BaseShape::new(o2w, w2o, reverse_orientation),
            height,
            radius,
            inner_radius,
            phi_max: radians(clamp(phi_max, 0.0, 360.0)),
        }
    }

    pub fn intersect_test(&self, r: &Ray) -> (bool, Point3f, Float, Float, Float, Ray) {
        let err = (false, Point3f::default(), 0.0, 0.0, 0.0, Ray::default());

        let mut o_err = Vector3f::default();
        let mut d_err = Vector3f::default();
        let ray = Ray::from((self.world_to_object(), r, &mut o_err, &mut d_err));
        if ray.d.z == 0.0 {
            return err;
        }

        let t_shape_hit = (self.height - ray.o.z) / ray.d.z;
        if t_shape_hit <= 0.0 || t_shape_hit >= ray.t_max {
            return err;
        }

        let p_hit = ray.point(t_shape_hit);
        let dist2 = p_hit.x * p_hit.x + p_hit.y * p_hit.y;
        if dist2 > self.radius * self.radius || dist2 < self.inner_radius * self.inner_radius {
            return err;
        }

        let mut phi = p_hit.y.atan2(p_hit.x);
        if phi < 0.0 {
            phi += 2.0 * PI;
        }
        if phi > self.phi_max {
            return err;
        }

        (true, p_hit, t_shape_hit, dist2, phi, ray)
    }
}

impl Shape for Disk {
    impl_base_shape!();

    fn object_bound(&self) -> Bounds3f {
        Bounds3f::from((
            Point3f::new(-self.radius, -self.radius, self.height),
            Point3f::new(self.radius, self.radius, self.height),
        ))
    }

    fn intersect(
        &self,
        r: &Ray,
        hit: &mut f32,
        si: &mut SurfaceInteraction,
        test_alpha_texture: bool,
    ) -> bool {
        let (ok, mut p_hit, t_shape_hit, dist2, phi, ray) = self.intersect_test(r);
        if !ok {
            return false;
        }

        let u = phi / self.phi_max;
        let r_hit = dist2.sqrt();
        let v = (self.radius - r_hit) / (self.radius - self.inner_radius);
        let dpdu = Vector3f::new(-self.phi_max * p_hit.y, self.phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(p_hit.x, p_hit.y, 0.0) * (self.inner_radius - self.radius) / r_hit;
        let dndu = Normal3f::default();
        let dndv = Normal3f::default();
        p_hit.z = self.height;
        let p_error = Vector3f::default();
        *si = self.object_to_world()
            * &SurfaceInteraction::new(
                p_hit,
                p_error,
                Point2f::new(u, v),
                -ray.d,
                dpdu,
                dpdv,
                dndu,
                dndv,
                ray.time,
                None,
                0,
            );
        *hit = t_shape_hit;
        true
    }

    fn intersect_p(&self, ray: &Ray, test_alpha_texture: bool) -> bool {
        let (ok, _, _, _, _, _) = self.intersect_test(ray);
        ok
    }

    fn area(&self) -> f32 {
        self.phi_max * 0.5 * (self.radius * self.radius - self.inner_radius * self.inner_radius)
    }

    fn sample(&self, u: &Point2f, pdf: &mut f32) -> Interaction {
        let pd = concentric_sample_disk(u);
        let p_obj = Vector3f::new(pd.x * self.radius, pd.y * self.radius, self.height);
        let mut it = Interaction::default();
        it.n = self.object_to_world() * Normal3Ref(&Normal3f::new(0.0, 0.0, 1.0));
        if self.reverse_orientation() {
            it.n *= -1.0;
        }
        it.p = Point3f::from((
            self.object_to_world(),
            Point3Ref(&p_obj),
            &Vector3f::default(),
            &mut it.error,
        ));
        *pdf = 1.0 / self.area();
        it
    }
}
