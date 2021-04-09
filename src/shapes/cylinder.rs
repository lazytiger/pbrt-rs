use crate::{
    core::{
        efloat::EFloat,
        geometry::{Bounds3f, Point2f, Point3f, Ray, Vector3f},
        interaction::{Interaction, SurfaceInteraction},
        pbrt::{clamp, gamma, radians, Float, PI},
        shape::Shape,
        transform::Transformf,
    },
    impl_base_shape,
    shapes::{compute_normal_differential, BaseShape},
};

pub struct Cylinder {
    base: BaseShape,
    radius: Float,
    z_min: Float,
    z_max: Float,
    phi_max: Float,
}

impl Cylinder {
    pub fn new(
        o2w: Transformf,
        w2o: Transformf,
        reverse_orientation: bool,
        radius: Float,
        z_min: Float,
        z_max: Float,
        phi_max: Float,
    ) -> Self {
        Self {
            base: BaseShape::new(o2w, w2o, reverse_orientation),
            radius,
            z_min: z_min.min(z_max),
            z_max: z_max.max(z_min),
            phi_max: radians(clamp(phi_max, 0.0, 360.0)),
        }
    }

    fn compute_intersect(&self, r: &Ray) -> (bool, Point3f, Float, EFloat, Ray) {
        let err = (
            false,
            Point3f::default(),
            0.0,
            EFloat::default(),
            Ray::default(),
        );

        let mut o_err = Vector3f::default();
        let mut d_err = Vector3f::default();
        let ray = Ray::from((self.world_to_object(), r, &mut o_err, &mut d_err));
        let (ox, oy, _oz, dx, dy, _dz) = ray.efloats(&o_err, &d_err);
        let a = dx * dx + dy * dy;
        let b = (dx * ox + dy * oy) * 2.0;
        let c = ox * ox + oy * oy - EFloat::new(self.radius, 0.0) * EFloat::new(self.radius, 0.0);

        let (ok, t0, t1) = EFloat::quadratic(a, b, c);
        if !ok {
            return err;
        }

        let mut t_shape_hit = EFloat::default();
        let mut p_hit = Point3f::default();
        let mut hit = false;
        let mut phi = 0.0;
        for t in &[t0, t1] {
            if t.lower_bound() <= 0.0 || t.upper_bound() > ray.t_max {
                continue;
            }
            p_hit = ray.point(t.v);
            let hit_rad = (p_hit.x * p_hit.x + p_hit.y * p_hit.y).sqrt();
            p_hit.x *= self.radius / hit_rad;
            p_hit.y *= self.radius / hit_rad;
            phi = p_hit.y.atan2(p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            if p_hit.z < self.z_min || p_hit.z > self.z_max || phi > self.phi_max {
                continue;
            }
            hit = true;
            t_shape_hit = *t;
            break;
        }
        (hit, p_hit, phi, t_shape_hit, ray)
    }
}

impl Shape for Cylinder {
    impl_base_shape!();

    fn object_bound(&self) -> Bounds3f {
        unimplemented!()
    }

    fn intersect(
        &self,
        r: &Ray,
        hit: &mut f32,
        si: &mut SurfaceInteraction,
        _test_alpha_texture: bool,
    ) -> bool {
        let (_ok, p_hit, phi, t_shape_hit, ray) = self.compute_intersect(r);
        let u = phi / self.phi_max;
        let v = (p_hit.z - self.z_min) / (self.z_max - self.z_min);
        let dpdu = Vector3f::new(-self.phi_max * p_hit.y, self.phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(0.0, 0.0, self.z_max - self.z_min);

        let d2pduu = Vector3f::new(p_hit.x, p_hit.y, 0.0) * (-self.phi_max * self.phi_max);
        let d2pduv = Vector3f::default();
        let d2pdvv = Vector3f::default();

        let (dndu, dndv) = compute_normal_differential(&dpdu, &dpdv, &d2pduu, &d2pduv, &d2pdvv);
        let p_error = Vector3f::new(p_hit.x, p_hit.y, 0.0).abs() * gamma(3.0);
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
        *hit = t_shape_hit.v;
        true
    }

    fn intersect_p(&self, ray: &Ray, _test_alpha_texture: bool) -> bool {
        let (ok, _, _, _, _) = self.compute_intersect(ray);
        ok
    }

    fn area(&self) -> f32 {
        unimplemented!()
    }

    fn sample(&self, _u: &Point2f, _pdf: &mut f32) -> Interaction {
        unimplemented!()
    }
}
