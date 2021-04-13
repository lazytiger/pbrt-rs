use crate::{
    core::{
        efloat::EFloat,
        geometry::{Bounds3f, Point2f, Point3f, Ray, Vector3f},
        interaction::{Interaction, InteractionDt, SurfaceInteraction},
        pbrt::{clamp, radians, Float, PI},
        shape::Shape,
        transform::Transformf,
    },
    impl_base_shape,
    shapes::{compute_normal_differential, BaseShape},
};

pub struct Cone {
    base: BaseShape,
    radius: Float,
    height: Float,
    phi_max: Float,
}

impl Cone {
    pub fn new(
        o2w: Transformf,
        w2o: Transformf,
        reverse_orientation: bool,
        height: Float,
        radius: Float,
        phi_max: Float,
    ) -> Cone {
        Cone {
            base: BaseShape::new(o2w, w2o, reverse_orientation),
            radius,
            height,
            phi_max: radians(clamp(phi_max, 0.0, 360.0)),
        }
    }

    fn compute_intersect(&self, r: &Ray) -> (bool, Point3f, Float, EFloat, Vector3f, Ray) {
        let err = (
            false,
            Point3f::default(),
            0.0,
            EFloat::default(),
            Vector3f::default(),
            Ray::default(),
        );
        let mut o_err = Vector3f::default();
        let mut d_err = Vector3f::default();
        let ray = Ray::from((self.world_to_object(), r, &mut o_err, &mut d_err));

        let ox = EFloat::new(ray.o.x, o_err.x);
        let oy = EFloat::new(ray.o.y, o_err.y);
        let oz = EFloat::new(ray.o.z, o_err.z);
        let dx = EFloat::new(ray.d.x, d_err.x);
        let dy = EFloat::new(ray.d.y, d_err.y);
        let dz = EFloat::new(ray.d.z, d_err.z);
        let k = EFloat::new(self.radius, 0.0) / EFloat::new(self.height, 0.0);
        let k = k * k;
        let a = dx * dx + dy * dy - k * dz * dz;
        let b = (dx * ox + dy * oy - k * dz * (oz - self.height)) * 2.0;
        let c = ox * ox + oy * oy - k * (oz - self.height) * (oz - self.height);
        let (ok, t0, t1) = EFloat::quadratic(a, b, c);
        if !ok {
            return err;
        }

        let mut hit = false;
        let mut t_shape_hit = EFloat::default();
        let mut p_hit = Point3f::default();
        let phi = 0.0;
        for t in &[t0, t1] {
            if t.lower_bound() < 0.0 || t.upper_bound() > ray.t_max {
                continue;
            }
            p_hit = ray.point(t.v);
            let mut phi = p_hit.y.atan2(p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            if p_hit.z < 0.0 || p_hit.z > self.height || phi > self.phi_max {
                continue;
            }
            hit = true;
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

        (hit, p_hit, phi, t_shape_hit, p_error, ray)
    }
}

impl Shape for Cone {
    impl_base_shape!();

    fn object_bound(&self) -> Bounds3f {
        let p1 = Point3f::new(-self.radius, -self.radius, 0.0);
        let p2 = Point3f::new(self.radius, self.radius, self.height);
        Bounds3f::from((p1, p2))
    }

    fn intersect(
        &self,
        r: &Ray,
        hit: &mut f32,
        si: &mut SurfaceInteraction,
        _test_alpha_texture: bool,
    ) -> bool {
        let (ok, p_hit, phi, t_shape_hit, p_error, ray) = self.compute_intersect(r);
        if !ok {
            return false;
        }

        let u = phi / self.phi_max;
        let v = p_hit.z / self.height;

        let dpdu = Vector3f::new(-self.phi_max * p_hit.y, self.phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(-p_hit.x / (1.0 - v), -p_hit.y / (1.0 - v), self.height);

        let d2pduu = Vector3f::new(p_hit.x, p_hit.y, 0.0) * (-self.phi_max * self.phi_max);
        let d2pduv = Vector3f::new(p_hit.y, -p_hit.x, 0.0) * (self.phi_max / (1.0 - v));
        let d2pdvv = Vector3f::default();

        let (dndu, dndv) = compute_normal_differential(&dpdu, &dpdv, &d2pduu, &d2pduv, &d2pdvv);
        let osi = SurfaceInteraction::new(
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
        *si = self.object_to_world() * &osi;
        *hit = t_shape_hit.v;
        true
    }

    fn intersect_p(&self, r: &Ray, _test_alpha_texture: bool) -> bool {
        let (ok, _, _, _, _, _) = self.compute_intersect(r);
        ok
    }

    fn area(&self) -> f32 {
        self.radius * (self.height * self.height + self.radius + self.radius).sqrt() * self.phi_max
            / 2.0
    }

    fn sample(&self, _u: &Point2f, _pdf: &mut f32) -> InteractionDt {
        unimplemented!()
    }
}
