use crate::{
    core::{
        efloat::EFloat,
        geometry::{
            offset_ray_origin, spherical_direction, Bounds3f, Point2f, Point3f, Ray, Vector3f,
        },
        interaction::{Interaction, SurfaceInteraction},
        pbrt::{clamp, gamma, radians, Float, PI},
        sampling::{uniform_cone_pdf, uniform_sample_sphere},
        shape::Shape,
        transform::{Point3Ref, Transformf},
    },
    impl_base_shape,
    shapes::{compute_normal_differential, BaseShape},
};

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
        _test_alpha_texture: bool,
    ) -> bool {
        let (ok, p_hit, phi, ray, t_shape_hit) = self.intersect_test(r);
        if !ok {
            return false;
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

        let (dndu, dndv) = compute_normal_differential(&dpdu, &dpdv, &d2pduu, &d2pduv, &d2pdvv);

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

    fn intersect_p(&self, r: &Ray, _test_alpha_texture: bool) -> bool {
        let (ok, _, _, _, _) = self.intersect_test(r);
        ok
    }

    fn area(&self) -> Float {
        self.phi_max * self.radius * (self.z_max - self.z_min)
    }

    fn sample(&self, u: &Point2f, pdf: &mut Float) -> Interaction {
        let mut obj = Point3f::default() + uniform_sample_sphere(u) * self.radius;
        let mut it = Interaction::default();
        it.n = self.object_to_world() * Point3Ref(&obj);
        if self.reverse_orientation() {
            it.n *= -1.0;
        }
        obj *= self.radius / obj.distance(&Point3f::default());
        let obj_error = obj.abs() * gamma(5.0);
        it.p = Point3f::from((
            self.object_to_world(),
            Point3Ref(&obj),
            &obj_error,
            &mut it.error,
        ));
        *pdf = 1.0 / self.area();
        it
    }

    fn sample2(&self, int: &Interaction, u: &Point2f, pdf: &mut Float) -> Interaction {
        let p_center = self.object_to_world() * Point3Ref(&Point3f::default());
        let p_origin = offset_ray_origin(&int.p, &int.error, &int.n, &(p_center - int.p));
        if p_origin.distance_square(&p_center) <= self.radius * self.radius {
            let intr = self.sample(u, pdf);
            let mut wi = intr.p - int.p;
            if wi.length_squared() == 0.0 {
                *pdf = 0.0;
            } else {
                wi = wi.normalize();
                *pdf *= int.p.distance_square(&intr.p) / intr.n.abs_dot(&-wi);
            }
            if pdf.is_infinite() {
                *pdf = 0.0;
            }
            return intr;
        }
        let dc = int.p.distance(&p_center);
        let inv_dc = 1.0 / dc;
        let wc = (p_center - int.p) * inv_dc;
        let (wc_x, wc_y) = wc.coordinate_system();

        let sin_theta_max = self.radius * inv_dc;
        let sin_theta_max2 = sin_theta_max * sin_theta_max;
        let inv_sin_theta_max = 1.0 / sin_theta_max;
        let cos_theta_max = (1.0 - sin_theta_max2).max(0.0).sqrt();

        let mut cos_theta = (cos_theta_max - 1.0) * u[0] + 1.0;
        let mut sin_theta2 = 1.0 - cos_theta * cos_theta;

        if sin_theta_max2 < 0.00068523 {
            //sin<sup>2</sup>(1.5 deg)
            sin_theta2 = sin_theta_max2 * u[0];
            cos_theta = (1.0 - sin_theta2).sqrt();
        }

        let cos_alpha = sin_theta2 * inv_sin_theta_max
            + cos_theta
                * (1.0 - sin_theta2 * inv_sin_theta_max * inv_sin_theta_max)
                    .max(0.0)
                    .sqrt();
        let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();
        let phi = u[1] * 2.0 * PI;
        let n_world = spherical_direction(sin_alpha, cos_alpha, phi, -wc_x, -wc_y, -wc);
        let p_world = p_center + n_world * self.radius;

        let mut it = Interaction::default();
        it.p = p_world;
        it.error = p_world.abs() * gamma(5.0);
        it.n = n_world;
        if self.reverse_orientation() {
            it.n *= -1.0;
        }

        *pdf = 1.0 / (2.0 * PI * (1.0 - cos_theta_max));
        it
    }

    fn pdf2(&self, int: &Interaction, wi: &Vector3f) -> Float {
        let p_center = self.object_to_world() * Point3Ref(&Point3f::default());
        let p_origin = offset_ray_origin(&int.p, &int.error, &int.n, &(p_center - int.p));
        if p_origin.distance_square(&p_center) < self.radius * self.radius {
            Shape::pdf2(self, int, wi)
        } else {
            let sin_theta_max2 = self.radius * self.radius / int.p.distance_square(&p_center);
            let cos_theta_max = (1.0 - sin_theta_max2).max(0.0).sqrt();
            uniform_cone_pdf(cos_theta_max)
        }
    }

    fn solid_angle(&self, p: Point3f, _n_samples: u64) -> Float {
        let p_center = self.object_to_world() * Point3Ref(&Point3f::default());
        if p.distance_square(&p_center) <= self.radius * self.radius {
            4.0 * PI
        } else {
            let sin_theta2 = self.radius * self.radius / p.distance_square(&p_center);
            let cos_theta = (1.0 - sin_theta2).max(0.0).sqrt();
            2.0 * PI * (1.0 - cos_theta)
        }
    }
}

impl Sphere {
    pub fn new(
        object_to_world: Transformf,
        world_to_object: Transformf,
        reverse_orientation: bool,
        radius: Float,
        z_min: Float,
        z_max: Float,
        phi_max: Float,
    ) -> Sphere {
        Sphere {
            base: BaseShape::new(object_to_world, world_to_object, reverse_orientation),
            radius,
            z_min: clamp(z_min.min(z_max), -radius, radius),
            z_max: clamp(z_max.max(z_min), -radius, radius),
            theta_min: clamp(z_min.min(z_max) / radius, -1.0, 1.0).acos(),
            theta_max: clamp(z_max.max(z_min) / radius, -1.0, 1.0).acos(),
            phi_max: radians(clamp(phi_max, 0.0, 360.0)),
        }
    }

    fn intersect_test(&self, r: &Ray) -> (bool, Point3f, Float, Ray, EFloat) {
        let err = (
            false,
            Point3f::default(),
            0.0,
            Ray::default(),
            EFloat::default(),
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
        let a = dx * dx + dy * dy + dz * dz;
        let b = (dx * ox + dy * oy + dz * oz) * 2.0;
        let c = ox * ox + oy * oy + oz * oz
            - EFloat::new(self.radius, 0.0) * EFloat::new(self.radius, 0.0);

        let (ok, t0, t1) = EFloat::quadratic(a, b, c);
        if !ok {
            return err;
        }

        let mut hit = false;
        let mut t_shape_hit = EFloat::default();
        let mut p_hit = Point3f::default();
        let mut phi = 0.0;
        for t in &[t0, t1] {
            if t.lower_bound() < 0.0 || t.upper_bound() > ray.t_max {
                continue;
            }
            p_hit = ray.point(t.v);
            p_hit *= self.radius / p_hit.distance(&Point3f::default());
            if p_hit.x == 0.0 && p_hit.y == 0.0 {
                p_hit.x = 1e-5 * self.radius;
            }
            phi = p_hit.y.atan2(p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            if self.z_min > -self.radius && p_hit.z < self.z_min
                || self.z_max < self.radius && p_hit.z > self.z_max
                || phi > self.phi_max
            {
                continue;
            }
            hit = true;
            t_shape_hit = *t;
            break;
        }
        (hit, p_hit, phi, ray, t_shape_hit)
    }
}
