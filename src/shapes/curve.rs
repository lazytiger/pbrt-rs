use crate::{
    core::{
        geometry::{Bounds3f, Normal3f, Point2f, Point3f, Ray, Union, Vector3f},
        interaction::{Interaction, InteractionDt, SurfaceInteraction},
        pbrt::{clamp, float_to_bits, lerp, Float},
        shape::Shape,
        transform::{Point3Ref, Transformf, Vector3Ref},
    },
    impl_base_shape,
    shapes::BaseShape,
};
use std::sync::Arc;
#[derive(Debug)]
pub enum CurveType {
    Flat,
    Cylinder,
    Ribbon,
}
#[derive(Debug)]
pub struct CurveCommon {
    typ: CurveType,
    cp_obj: [Point3f; 4],
    width: [Float; 2],
    n: [Normal3f; 2],
    normal_angle: Float,
    inv_sin_normal_angle: Float,
}

impl CurveCommon {
    pub fn new(
        cp_obj: [Point3f; 4],
        width0: Float,
        width1: Float,
        typ: CurveType,
        norm: Option<&[Normal3f]>,
    ) -> Self {
        let width = [width0, width1];
        let (n, normal_angle, inv_sin_normal_angle) = if let Some(norm) = norm {
            let n = [norm[0].normalize(), norm[1].normalize()];
            let normal_angle = clamp(n[0].dot(&n[1]), 0.0, 1.0);
            let inv_sin_normal_angle = 1.0 / normal_angle.sin();
            (n, normal_angle, inv_sin_normal_angle)
        } else {
            ([Normal3f::default(); 2], 0.0, 0.0)
        };
        Self {
            typ,
            width,
            cp_obj,
            n,
            normal_angle,
            inv_sin_normal_angle,
        }
    }
}

#[derive(Debug)]
pub struct Curve {
    base: BaseShape,
    common: Arc<CurveCommon>,
    u_min: Float,
    u_max: Float,
}

impl Curve {
    pub fn new(
        o2w: Transformf,
        w2o: Transformf,
        ro: bool,
        common: Arc<CurveCommon>,
        u_min: Float,
        u_max: Float,
    ) -> Self {
        Self {
            base: BaseShape::new(o2w, w2o, ro),
            common,
            u_min,
            u_max,
        }
    }

    fn blossom_bezier(p: &[Point3f], u0: Float, u1: Float, u2: Float) -> Point3f {
        let a = [
            p[0].lerp(u0, p[1]),
            p[1].lerp(u0, p[2]),
            p[2].lerp(u0, p[3]),
        ];
        let b = [a[0].lerp(u1, a[1]), a[1].lerp(u1, a[2])];
        b[0].lerp(u2, b[1])
    }

    fn subdivide_bezier(cp: &[Point3f], cp_split: &mut [Point3f]) {
        cp_split[0] = cp[0];
        cp_split[1] = (cp[0] + cp[1]) / 2.0;
        cp_split[2] = (cp[0] + cp[1] * 2.0 + cp[2]) / 4.0;
        cp_split[3] = (cp[0] + cp[1] * 3.0 + cp[2] * 3.0 + cp[3]) / 8.0;
        cp_split[4] = (cp[1] + cp[2] * 2.0 + cp[3]) / 4.0;
        cp_split[5] = (cp[2] + cp[3]) / 2.0;
        cp_split[6] = cp[3];
    }

    fn eval_bezier(cp: &[Point3f], u: Float, derive: Option<&mut Vector3f>) -> Point3f {
        let cp1 = [
            cp[0].lerp(u, cp[1]),
            cp[1].lerp(u, cp[2]),
            cp[2].lerp(u, cp[3]),
        ];

        let cp2 = [cp1[0].lerp(u, cp1[1]), cp[1].lerp(u, cp[2])];
        if let Some(derive) = derive {
            if (cp2[1] - cp2[0]).length_squared() > 0.0 {
                *derive = (cp2[1] - cp2[0]) * 3.0;
            } else {
                *derive = cp[3] - cp[0];
            }
        }

        cp2[0].lerp(u, cp2[1])
    }

    fn get_blossom_bezier(&self) -> [Point3f; 4] {
        [
            Self::blossom_bezier(&self.common.cp_obj, self.u_min, self.u_min, self.u_min),
            Self::blossom_bezier(&self.common.cp_obj, self.u_min, self.u_min, self.u_max),
            Self::blossom_bezier(&self.common.cp_obj, self.u_min, self.u_max, self.u_max),
            Self::blossom_bezier(&self.common.cp_obj, self.u_max, self.u_max, self.u_max),
        ]
    }

    fn recursive_intersect(
        &self,
        ray: &Ray,
        t_hit: &mut Float,
        si: &mut SurfaceInteraction,
        cp: &[Point3f],
        ray_to_object: &Transformf,
        u0: Float,
        u1: Float,
        depth: i32,
    ) -> bool {
        let ray_length = ray.d.length();
        if depth > 0 {
            let mut cp_split = [Point3f::default(); 7];
            Self::subdivide_bezier(cp, &mut cp_split);

            let mut hit = false;
            let u = [u0, (u0 + u1) / 2.0, u1];
            let mut cps = &cp_split[..];
            for seg in 0..2 {
                let max_width = lerp(u[seg], self.common.width[0], self.common.width[1]).max(lerp(
                    u[seg + 1],
                    self.common.width[0],
                    self.common.width[1],
                ));

                if cps[..3].iter().fold(Float::MIN, |max, p| p.y.max(max)) + 0.5 * max_width < 0.0
                    || cps[..3].iter().fold(Float::MAX, |min, p| p.y.min(min)) + 0.5 * max_width
                        > 0.0
                {
                    continue;
                }

                if cps[..3].iter().fold(Float::MIN, |max, p| p.x.max(max)) + 0.5 * max_width < 0.0
                    || cps[..3].iter().fold(Float::MAX, |min, p| p.x.min(min)) + 0.5 * max_width
                        > 0.0
                {
                    continue;
                }

                let z_max = ray_length * ray.t_max;

                if cps[..3].iter().fold(Float::MIN, |max, p| p.z.max(max)) + 0.5 * max_width < 0.0
                    || cps[..3].iter().fold(Float::MAX, |min, p| p.z.min(min)) + 0.5 * max_width
                        > z_max
                {
                    continue;
                }

                hit |= self.recursive_intersect(
                    ray,
                    t_hit,
                    si,
                    cps,
                    ray_to_object,
                    u[seg],
                    u[seg + 1],
                    depth - 1,
                );
                if hit && *t_hit != 0.0 {
                    return true;
                }
                cps = &cps[3..];
            }
            return hit;
        } else {
            let edge = (cp[1].y - cp[0].y) * -cp[0].y + cp[0].x * (cp[0].x - cp[1].x);
            if edge < 0.0 {
                return false;
            }

            let edge = (cp[2].y - cp[3].y) * -cp[3].y + cp[3].x * (cp[3].x - cp[2].x);
            if edge < 0.0 {
                return false;
            }

            let segment_direction = Point2f::from(cp[3]) - Point2f::from(cp[0]);
            let denom = segment_direction.length_squared();
            if denom == 0.0 {
                return false;
            }
            let w = segment_direction.dot(&-Point2f::from(cp[0])) / denom;
            let u = clamp(lerp(w, u0, u1), u0, u1);
            let mut hit_width = lerp(u, self.common.width[0], self.common.width[1]);
            let mut n_hit = Normal3f::default();
            if let CurveType::Ribbon = self.common.typ {
                let sin0 =
                    ((1.0 - u) * self.common.normal_angle).sin() * self.common.inv_sin_normal_angle;
                let sin1 = (u * self.common.normal_angle).sin() * self.common.inv_sin_normal_angle;
                n_hit = self.common.n[0] * sin0 + self.common.n[1] * sin1;
                hit_width *= n_hit.abs_dot(&ray.d) / ray_length;
            }

            let mut dpcdw = Vector3f::default();
            let pc = Self::eval_bezier(cp, clamp(w, 0.0, 1.0), Some(&mut dpcdw));
            let pt_curve_dist2 = pc.x * pc.x + pc.y * pc.y;
            if pt_curve_dist2 > hit_width * hit_width * 0.25 {
                return false;
            }

            let z_max = ray_length * ray.t_max;
            if pc.z < 0.0 || pc.z > z_max {
                return false;
            }

            let pt_curve_dist = pt_curve_dist2.sqrt();
            let edge_func = dpcdw.x * -pc.y + pc.x * dpcdw.y;
            let v = if edge_func > 0.0 {
                0.5 + pt_curve_dist / hit_width
            } else {
                0.5 - pt_curve_dist / hit_width
            };

            *t_hit = pc.z / ray_length;

            let p_error = Vector3f::new(2.0 * hit_width, 2.0 * hit_width, 2.0 * hit_width);
            let mut dpdu = Vector3f::default();
            let mut dpdv = Vector3f::default();
            Self::eval_bezier(&self.common.cp_obj, u, Some(&mut dpdu));
            if let CurveType::Ribbon = self.common.typ {
                dpdv = n_hit.cross(&dpdu).normalize() * hit_width;
            } else {
                let dpdu_plane = &ray_to_object.inverse() * Vector3Ref(&dpdu);
                let mut dpdv_plane =
                    Vector3f::new(-dpdu_plane.y, dpdu_plane.x, 0.0).normalize() * hit_width;

                if let CurveType::Cylinder = self.common.typ {
                    let theta = lerp(v, -90.0, 90.0);
                    let rot = Transformf::rotate(-theta, &dpdu_plane);
                    dpdv_plane = &rot * Vector3Ref(&dpdv_plane);
                }
                dpdv = ray_to_object * Vector3Ref(&dpdv_plane);
            }
            *si = self.object_to_world()
                * &SurfaceInteraction::new(
                    ray.point(*t_hit),
                    p_error,
                    Point2f::new(u, v),
                    -ray.d,
                    dpdu,
                    dpdv,
                    Normal3f::default(),
                    Normal3f::default(),
                    ray.time,
                    None,
                    0,
                );
        }
        true
    }
}

impl Shape for Curve {
    impl_base_shape!();

    fn object_bound(&self) -> Bounds3f {
        let cp_obj = self.get_blossom_bezier();
        let b =
            Bounds3f::from((cp_obj[0], cp_obj[1])).union(&Bounds3f::from((cp_obj[2], cp_obj[3])));
        let width = [
            lerp(self.u_min, self.common.width[0], self.common.width[1]),
            lerp(self.u_max, self.common.width[0], self.common.width[1]),
        ];
        b.expand(width[0].max(width[1]) * 0.5)
    }

    fn intersect(
        &self,
        r: &Ray,
        hit: &mut Float,
        si: &mut SurfaceInteraction,
        _test_alpha_texture: bool,
    ) -> bool {
        let mut o_err = Vector3f::default();
        let mut d_err = Vector3f::default();
        let ray = Ray::from((self.world_to_object(), r, &mut o_err, &mut d_err));
        let cp_obj = self.get_blossom_bezier();
        let mut dx = ray.d.cross(&(cp_obj[3] - cp_obj[0]));
        if dx.length_squared() == 0.0 {
            (dx, _) = ray.d.coordinate_system();
        }
        let object_to_ray = Transformf::look_at(&ray.o, &(ray.o + ray.d), &dx);
        let cp = [
            &object_to_ray * Point3Ref(&cp_obj[0]),
            &object_to_ray * Point3Ref(&cp_obj[1]),
            &object_to_ray * Point3Ref(&cp_obj[2]),
            &object_to_ray * Point3Ref(&cp_obj[3]),
        ];

        let max_width = lerp(self.u_min, self.common.width[0], self.common.width[1]).max(lerp(
            self.u_max,
            self.common.width[0],
            self.common.width[1],
        ));
        if cp.iter().fold(Float::MIN, |max, p| p.x.max(max)) + 0.5 * max_width < 0.0
            || cp.iter().fold(Float::MAX, |min, p| p.x.min(min)) + 0.5 * max_width > 0.0
        {
            return false;
        }

        let ray_length = ray.d.length();
        let z_max = ray_length * ray.t_max;

        if cp.iter().fold(Float::MIN, |max, p| p.z.max(max)) + 0.5 * max_width < 0.0
            || cp.iter().fold(Float::MAX, |min, p| p.z.min(min)) + 0.5 * max_width > z_max
        {
            return false;
        }

        let mut l0: Float = 0.0;
        for i in 0..2 {
            l0 = l0
                .max((cp[i].x - 2.0 * cp[i + 1].x + cp[i + 2].x).abs())
                .max((cp[i].y - 2.0 * cp[i + 1].y + cp[i + 2].y).abs())
                .max((cp[i].z - 2.0 * cp[i + 1].z + cp[i + 2].z).abs());
        }

        let eps = self.common.width[0].max(self.common.width[1]) * 0.5;
        let log2 = |v: Float| {
            if v < 1.0 {
                0
            } else {
                let bits = float_to_bits(v);
                if (bits >> 23) - 127 + (bits & (1 << 22)) != 0 {
                    1
                } else {
                    0
                }
            }
        };
        let r0 = log2(1.41421356237 * 6.0 * l0 / (8. * eps)) / 2;
        let max_depth = clamp(r0, 0, 10);

        self.recursive_intersect(
            &ray,
            hit,
            si,
            &cp[..],
            &object_to_ray.inverse(),
            self.u_min,
            self.u_max,
            max_depth,
        )
    }

    fn area(&self) -> f32 {
        let cp_obj = self.get_blossom_bezier();
        let (width0, width1) = (
            lerp(self.u_min, self.common.width[0], self.common.width[1]),
            lerp(self.u_max, self.common.width[0], self.common.width[1]),
        );
        let avg_width = (width0 + width1) * 0.5;
        let mut approx_length = 0.0;
        for i in 0..3 {
            approx_length += cp_obj[i].distance(&cp_obj[i + 1]);
        }
        approx_length * avg_width
    }

    fn sample(&self, _u: &Point2f, _pdf: &mut f32) -> InteractionDt {
        unimplemented!()
    }
}
