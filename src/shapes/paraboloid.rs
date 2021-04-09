use crate::{
    core::{
        efloat::EFloat,
        geometry::{Bounds3f, Point2f, Point3f, Ray, Vector3f},
        interaction::{Interaction, SurfaceInteraction},
        pbrt::{clamp, radians, Float, PI},
        shape::Shape,
        transform::Transformf,
    },
    impl_base_shape,
    shapes::{compute_normal_differential, BaseShape},
};
use num::traits::Pow;

pub struct Paraboloid {
    base: BaseShape,
    radius: Float,
    z_min: Float,
    z_max: Float,
    phi_max: Float,
}

impl Paraboloid {
    pub fn new(
        o2w: Transformf,
        w2o: Transformf,
        ro: bool,
        radius: Float,
        z0: Float,
        z1: Float,
        phi_max: Float,
    ) -> Self {
        Self {
            base: BaseShape::new(o2w, w2o, ro),
            radius,
            z_min: z0.min(z1),
            z_max: z0.max(z1),
            phi_max: radians(clamp(phi_max, 0.0, 360.0)),
        }
    }

    fn intersect_test(&self, ray: &Ray) -> (bool, Point3f, Float, Float, Ray, Vector3f) {
        let err = (
            false,
            Point3f::default(),
            0.0,
            0.0,
            Ray::default(),
            Vector3f::default(),
        );

        let mut o_err = Vector3f::default();
        let mut d_err = Vector3f::default();
        let ray = Ray::from((self.world_to_object(), ray, &mut o_err, &mut d_err));

        let (ox, oy, oz, dx, dy, dz) = ray.efloats(&o_err, &d_err);
        let k = EFloat::new(self.z_max, 0.0)
            / (EFloat::new(self.radius, 0.0) * EFloat::new(self.radius, 0.0));
        let a = (dx * dx + dy * dy) * k;
        let b = k * (dx * ox + dy * oy) * 2.0 - dz;
        let c = k * (ox * ox + oy * oy) - oz;

        let (ok, t0, t1) = EFloat::quadratic(a, b, c);
        if !ok {
            return err;
        }

        let mut hit = false;
        let mut t_shape_hit = EFloat::default();
        let mut p_hit = Point3f::default();
        let mut phi = 0.0;
        for t in &[t0, t1] {
            if t.lower_bound() <= 0.0 || t.upper_bound() > ray.t_max {
                continue;
            }
            p_hit = ray.point(t.v);
            phi = p_hit.y.atan2(p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }

            if p_hit.z < self.z_min || p_hit.z > self.z_max || phi > self.phi_max {
                continue;
            }
            t_shape_hit = *t;
            break;
        }
        let p_error = if hit {
            let px = ox + t_shape_hit * dx;
            let py = oy + t_shape_hit * dy;
            let pz = oz + t_shape_hit * dz;
            Vector3f::new(
                px.get_absolute_error(),
                py.get_absolute_error(),
                pz.get_absolute_error(),
            )
        } else {
            Vector3f::default()
        };
        (hit, p_hit, t_shape_hit.v, phi, ray, p_error)
    }
}

impl Shape for Paraboloid {
    impl_base_shape!();

    fn object_bound(&self) -> Bounds3f {
        let p1 = Point3f::new(-self.radius, -self.radius, self.z_min);
        let p2 = Point3f::new(self.radius, self.radius, self.z_max);
        Bounds3f::from((p1, p2))
    }

    fn intersect(
        &self,
        ray: &Ray,
        hit: &mut f32,
        si: &mut SurfaceInteraction,
        test_alpha_texture: bool,
    ) -> bool {
        let (ok, p_hit, t_shape_hit, phi, ray, p_error) = self.intersect_test(ray);
        if !ok {
            return false;
        }

        let u = phi / self.phi_max;
        let v = (p_hit.z - self.z_min) / (self.z_max - self.z_min);
        let dpdu = Vector3f::new(-self.phi_max * p_hit.y, self.phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(p_hit.x / (2.0 * p_hit.z), p_hit.y / (2.0 * p_hit.z), 1.0)
            * (self.z_max - self.z_min);
        let d2pduu = Vector3f::new(p_hit.x, p_hit.y, 0.0) * (-self.phi_max * self.phi_max);
        let d2pduv = Vector3f::new(-p_hit.y / (2.0 * p_hit.z), p_hit.x / (2.0 * p_hit.z), 0.0)
            * (self.z_max - self.z_min)
            * self.phi_max;
        let d2pdvv = Vector3f::new(
            p_hit.x / (4.0 * p_hit.z * p_hit.z),
            p_hit.y / (4.0 * p_hit.z * p_hit.z),
            0.0,
        );

        let (dndu, dndv) = compute_normal_differential(&dpdu, &dpdv, &d2pduu, &d2pduv, &d2pdvv);
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
        let radius2 = self.radius * self.radius;
        let k = 4.0 * self.z_max / radius2;
        (radius2 * radius2 * self.phi_max / (12.0 * self.z_max * self.z_max))
            * ((k * self.z_max + 1.0).powf(1.5) - (k * self.z_min + 1.0).pow(1.5))
    }

    fn sample(&self, u: &Point2f, pdf: &mut f32) -> Interaction {
        unimplemented!()
    }
}
