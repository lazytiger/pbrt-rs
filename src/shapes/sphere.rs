use crate::core::efloat::EFloat;
use crate::core::geometry::{Bounds3f, Normal3, Point2f, Point3f, Ray, Vector3f};
use crate::core::interaction::{Interaction, SurfaceInteraction};
use crate::core::shape::Shape;
use crate::core::transform::Transformf;
use crate::core::{clamp, gamma};
use crate::shapes::BaseShape;
use crate::{impl_base_shape, Float, PI};
use std::any::Any;

pub struct Sphere {
    base: BaseShape,
    radius: Float,
    z_min: Float,
    z_max: Float,
    theta_min: Float,
    theta_max: Float,
    phi_max: Float,
}

impl Shape for Sphere {
    impl_base_shape!();

    fn object_bound(&self) -> Bounds3f {
        Bounds3f::from((
            Point3f::new(-self.radius, -self.radius, self.z_min),
            Point3f::new(self.radius, self.radius, self.z_max),
        ))
    }

    fn intersect(
        &self,
        r: &Ray,
        hit: &mut f32,
        si: &mut SurfaceInteraction,
        test_alpha_texture: bool,
    ) -> bool {
        let mut o_err = Vector3f::default();
        let mut d_err = Vector3f::default();
        let ray = Ray::from((self.world_to_object(), r, &mut o_err, &mut d_err));

        let ox = EFloat::new(ray.o.x, o_err.x);
        let oy = EFloat::new(ray.o.y, o_err.y);
        let oz = EFloat::new(ray.o.z, o_err.z);
        let dx = EFloat::new(ray.d.x, d_err.x);
        let dy = EFloat::new(ray.d.y, d_err.y);
        let dz = EFloat::new(ray.d.z, d_err.z);
        let a = dx * dx + dy * dy + dz * dz;
        let b = (dx * ox + dy * oy + dz * oz) * 2.0;
        let c = ox * ox + oy * oy + oz * oz
            - EFloat::new(self.radius, 0.0) * EFloat::new(self.radius, 0.0);

        let (ok, t0, t1) = EFloat::quadratic(a, b, c);
        if !ok {
            return false;
        }

        if t0.upper_bound() > ray.t_max || t1.lower_bound() <= 0.0 {
            return false;
        }

        let mut t_shape_hit = t0;
        if t_shape_hit.lower_bound() < 0.0 {
            t_shape_hit = t1;
            if t_shape_hit.upper_bound() > ray.t_max {
                return false;
            }
        }

        let mut p_hit = ray.point(t_shape_hit.v);
        p_hit *= self.radius / p_hit.distance(&Point3f::default());
        if p_hit.x == 0.0 && p_hit.y == 0.0 {
            p_hit.x = 1e-5 * self.radius;
        }
        let mut phi = p_hit.y.atan2(p_hit.x);
        if phi < 0.0 {
            phi += 2.0 * PI;
        }

        if self.z_min > -self.radius && p_hit.z < self.z_min
            || self.z_max < self.radius && p_hit.z > self.z_max
            || phi > self.phi_max
        {
            if t_shape_hit == t1 {
                return false;
            }

            if t1.upper_bound() > ray.t_max {
                return false;
            }
            t_shape_hit = t1;
            p_hit = ray.point(t_shape_hit.v);

            p_hit *= self.radius / p_hit.distance(&Point3f::default());
            if p_hit.x == 0.0 && p_hit.y == 0.0 {
                p_hit.x = 1e-5 * self.radius;
            }
            phi = p_hit.y.atan2(p_hit.z);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            if self.z_min > -self.radius && p_hit.z < self.z_min
                || self.z_max < self.radius && p_hit.z > self.z_max
                || phi > self.phi_max
            {
                return false;
            }
        }

        let u = phi / self.phi_max;
        let theta = clamp(p_hit.z / self.radius, -1.0, 1.0).acos();
        let v = (theta - self.theta_min) / (self.theta_max - self.theta_min);
        let z_radius = (p_hit.x * p_hit.x + p_hit.y * p_hit.y).sqrt();
        let inv_z_radius = 1.0 / z_radius;
        let cos_phi = p_hit.x * inv_z_radius;
        let sin_phi = p_hit.y * inv_z_radius;
        let dpdu = Vector3f::new(-self.phi_max * p_hit.y, self.phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(
            p_hit.z * cos_phi,
            p_hit.z * sin_phi,
            -self.radius * theta.sin(),
        ) * (self.theta_max - self.theta_min);

        let d2pduu = Vector3f::new(p_hit.x, p_hit.y, 0.0) * (-self.phi_max * self.phi_max);
        let d2pduv = Vector3f::new(-sin_phi, cos_phi, 0.0)
            * (self.theta_max - self.theta_min)
            * p_hit.z
            * self.phi_max;
        let d2pdvv = Vector3f::new(p_hit.x, p_hit.y, p_hit.z)
            * (self.theta_min - self.theta_max)
            * (self.theta_max - self.theta_min);

        let e = dpdu.dot(&dpdu);
        let f = dpdu.dot(&dpdv);
        let g = dpdv.dot(&dpdv);
        let n = dpdu.cross(&dpdv).normalize();
        let ee = n.dot(&d2pduu);
        let ff = n.dot(&d2pduv);
        let gg = n.dot(&d2pdvv);

        let inv_egf2 = 1.0 / (e * g - f * f);
        let dndu = dpdu * inv_egf2 * (ff * f - ee * g) + dpdv * inv_egf2 * (ee * f - ff * e);
        let dndv = dpdu * inv_egf2 * (gg * f - ff * g) + dpdv * inv_egf2 * (ff * f - gg * e);

        let p_error = p_hit.abs() * gamma(5.0);

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
                None, //FIXME shape
                0,
            );
        *hit = t_shape_hit.v;
        true
    }

    fn area(&self) -> f32 {
        unimplemented!()
    }

    fn sample(&self, u: &Point2f) -> (Interaction, f32) {
        unimplemented!()
    }
}
