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
use std::mem::swap;

pub struct Hyperboloid {
    base: BaseShape,
    p1: Point3f,
    p2: Point3f,
    z_min: Float,
    z_max: Float,
    phi_max: Float,
    r_max: Float,
    ah: Float,
    ch: Float,
}

impl Hyperboloid {
    pub fn new(
        o2w: Transformf,
        w2o: Transformf,
        ro: bool,
        mut p1: Point3f,
        mut p2: Point3f,
        tm: Float,
    ) -> Self {
        let radius1 = (p1.x * p1.x + p1.y * p1.y).sqrt();
        let radius2 = (p2.x * p2.x + p2.y * p2.y).sqrt();
        let r_max = radius1.max(radius2);
        let z_min = p1.z.min(p2.z);
        let z_max = p1.z.max(p2.z);
        if p2.z == 0.0 {
            swap(&mut p1, &mut p2);
        }
        let mut pp = p1;
        let xy2 = p2.x * p2.x + p2.y * p2.y;
        let mut ah = 0.0;
        let mut ch = 0.0;
        loop {
            pp += (p2 - p1) * 2.0;
            let xy1 = pp.x * pp.x + pp.y * pp.y;
            ah = (1.0 / xy1 - pp.z * pp.z)
                / (xy1 * p2.z * p2.z)
                / (1.0 - xy2 * pp.z * pp.z)
                / (xy1 * p2.z * p2.z);
            ch = (ah * xy2 - 1.0) / (p2.z * p2.z);
            if !ah.is_infinite() && !ah.is_nan() {
                break;
            }
        }
        Self {
            base: BaseShape::new(o2w, w2o, ro),
            p1,
            p2,
            phi_max: radians(clamp(tm, 0.0, 360.0)),
            ah,
            ch,
            z_min,
            z_max,
            r_max,
        }
    }

    fn intersect_test(&self, r: &Ray) -> (bool, Point3f, Float, Ray, Vector3f, Float) {
        let err = (
            false,
            Point3f::default(),
            0.0,
            Ray::default(),
            Vector3f::default(),
            0.0,
        );

        let mut o_err = Vector3f::default();
        let mut d_err = Vector3f::default();
        let ray = Ray::from((self.world_to_object(), r, &mut o_err, &mut d_err));

        let (ox, oy, oz, dx, dy, dz) = ray.efloats(&o_err, &d_err);
        let a = dx * dx * self.ah + dy * dy + self.ah - dz * dz * self.ch;
        let b = (dx * ox * self.ah + dy * oy * self.ah - dz * oz * self.ch) * 2.0;
        let c = ox * ox * self.ah + oy * oy + self.ah - oz * oz * self.ch - 1.0;

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
            let v = (p_hit.z - self.p1.z) / (self.p2.z - self.p1.z);
            let pr = self.p1 * (1.0 - v) + self.p2 * v;
            phi = (pr.x * p_hit.y - p_hit.x * pr.y).atan2(p_hit.x * pr.x + p_hit.y * pr.y);
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
        (hit, p_hit, phi, ray, p_error, t_shape_hit.v)
    }
}

macro_rules! sqr {
    ($a:expr) => {
        $a * $a
    };
}

macro_rules! quad {
    ($a:expr) => {
        sqr!($a) * sqr!($a)
    };
}

impl Shape for Hyperboloid {
    impl_base_shape!();

    fn object_bound(&self) -> Bounds3f {
        Bounds3f::from((
            Point3f::new(-self.r_max, -self.r_max, self.z_min),
            Point3f::new(self.r_max, self.r_max, self.z_max),
        ))
    }

    fn intersect(
        &self,
        ray: &Ray,
        hit: &mut f32,
        si: &mut SurfaceInteraction,
        _test_alpha_texture: bool,
    ) -> bool {
        let (ok, p_hit, phi, ray, p_error, t_shape_hit) = self.intersect_test(ray);
        if !ok {
            return false;
        }
        let u = phi / self.phi_max;
        let v = (p_hit.z - self.p1.z) / (self.p2.z - self.p1.z);
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let dpdu = Vector3f::new(-self.phi_max * p_hit.y, self.phi_max * p_hit.x, 0.0);
        let dpdv = Vector3f::new(
            (self.p2.x - self.p1.x) * cos_phi - (self.p2.y - self.p1.y) * sin_phi,
            (self.p2.x - self.p1.x) * sin_phi + (self.p2.y - self.p1.y) * cos_phi,
            self.p2.z - self.p1.z,
        );
        let d2pduu = Vector3f::new(p_hit.x, p_hit.y, 0.0) * (-self.phi_max * self.phi_max);
        let d2pduv = Vector3f::new(-dpdv.y, dpdv.x, 0.0) * self.phi_max;
        let d2pdvv = Vector3f::default();

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

    fn intersect_p(&self, ray: &Ray, _test_alpha_texture: bool) -> bool {
        let (ok, _, _, _, _, _) = self.intersect_test(ray);
        ok
    }

    fn area(&self) -> f32 {
        self.phi_max / 6.0
            * (2.0 * quad!(self.p1.x) - 2.0 * self.p1.x * self.p1.x * self.p1.x * self.p2.x
                + 2.0 * quad!(self.p2.x)
                + 2.0
                    * (self.p1.y * self.p1.y + self.p1.y * self.p2.y + self.p2.y * self.p2.y)
                    * sqr!(self.p1.y - self.p2.y)
                + sqr!(self.p1.z - self.p2.z))
            + self.p2.x
                * self.p2.x
                * (5.0 * self.p1.y * self.p1.y + 2.0 * self.p1.y * self.p2.y
                    - 4.0 * self.p2.y * self.p2.y
                    + 2.0 * sqr!(self.p1.z - self.p2.z))
            + self.p1.x
                * self.p1.x
                * (-4.0 * self.p1.y * self.p1.y
                    + 2.0 * self.p1.y * self.p2.y
                    + 5.0 * self.p2.y * self.p2.y
                    + 2.0 * sqr!(self.p1.z - self.p2.z))
            - 2.0
                * self.p1.x
                * self.p2.x
                * (self.p2.x
                    * self.p2.x
                    * (self.p2.x * self.p2.x - self.p1.y * self.p1.y + 5.0 * self.p1.y * self.p2.y
                        - self.p2.y * self.p2.y
                        - self.p1.z * self.p1.z
                        + 2.0 * self.p1.z * self.p2.z
                        - self.p2.z * self.p2.z))
    }

    fn sample(&self, _u: &Point2f, _pdf: &mut f32) -> Interaction {
        unimplemented!()
    }
}
