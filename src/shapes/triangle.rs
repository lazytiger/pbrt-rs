use crate::{
    core::{
        geometry::{Bounds3f, Normal3f, Point2f, Point3f, Ray, Union, Vector3f},
        interaction::{Interaction, SurfaceInteraction},
        pbrt::{clamp, gamma, Float, PI},
        sampling::uniform_sample_triangle,
        shape::Shape,
        transform::{Point3Ref, Transformf},
    },
    impl_base_shape,
    shapes::BaseShape,
};

use crate::core::interaction::{BaseInteraction, InteractionDt};
use std::sync::Arc;

pub struct TriangleMesh {
    vertex_indices: Vec<i32>,
    p: Vec<Point3f>,
    n: Option<Vec<Normal3f>>,
    s: Option<Vec<Vector3f>>,
    uv: Option<Vec<Point2f>>,
    face_indices: Vec<i32>,
    n_triangles: usize,
    n_vertices: usize,
}

pub struct Triangle {
    base: BaseShape,
    mesh: Arc<TriangleMesh>,
    face_index: i32,
    v: [usize; 3],
}

impl Triangle {
    pub fn new(
        o2w: Transformf,
        w2o: Transformf,
        ro: bool,
        mesh: Arc<TriangleMesh>,
        tri_number: usize,
    ) -> Self {
        let mut v: [usize; 3] = Default::default();
        for i in 0..3 {
            v[i] = mesh.vertex_indices[3 * tri_number + i] as usize;
        }
        let face_index = if mesh.face_indices.len() > 0 {
            mesh.face_indices[tri_number]
        } else {
            0
        };
        Self {
            base: BaseShape::new(o2w, w2o, ro),
            mesh,
            v,
            face_index,
        }
    }

    fn get_uvs(&self) -> [Point2f; 3] {
        let mut uv: [Point2f; 3] = Default::default();
        if let Some(uvs) = &self.mesh.uv {
            uv[0] = uvs[self.v[0]];
            uv[1] = uvs[self.v[1]];
            uv[2] = uvs[self.v[2]];
        } else {
            uv[0] = Point2f::default();
            uv[1] = Point2f::new(1.0, 0.0);
            uv[2] = Point2f::new(1.0, 1.0);
        }
        uv
    }

    fn intersect_test(&self, ray: &Ray) -> (bool, Float, Float, Float, Float) {
        let err = (false, 0.0, 0.0, 0.0, 0.0);
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];

        let p0t = p0 - ray.o;
        let p1t = p1 - ray.o;
        let p2t = p2 - ray.o;

        let kz = ray.d.abs().max_dimension();
        let mut kx = kz + 1;
        if kx == 3 {
            kx = 0;
        }
        let mut ky = kx + 1;
        if ky == 3 {
            ky = 0;
        }
        let d = ray.d.permute(kx, ky, kz);

        let mut p0t = p0t.permute(kx, ky, kz);
        let mut p1t = p1t.permute(kx, ky, kz);
        let mut p2t = p2t.permute(kx, ky, kz);

        let sx = -d.x / d.z;
        let sy = -d.y / d.z;
        let sz = 1.0 / d.z;
        p0t.x += sx * p0t.z;
        p0t.y += sy * p0t.z;
        p1t.x += sx * p1t.z;
        p1t.y += sy * p1t.z;
        p2t.x += sx * p2t.z;
        p2t.y += sx * p2t.z;

        let e0 = (p1t.x as f64 * p2t.y as f64 - p1t.y as f64 * p2t.x as f64) as Float;
        let e1 = (p2t.x as f64 * p0t.y as f64 - p2t.y as f64 * p0t.x as f64) as Float;
        let e2 = (p0t.x as f64 * p1t.y as f64 - p0t.y as f64 * p1t.x as f64) as Float;

        if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
            return err;
        }

        let det = e0 + e1 + e2;
        if det == 0.0 {
            return err;
        }

        p0t.z *= sz;
        p1t.z *= sz;
        p2t.z *= sz;

        let t_scaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
        if det < 0.0 && t_scaled >= 0.0 || t_scaled < ray.t_max * det {
            return err;
        } else if det > 0.0 && (t_scaled <= 0.0 || t_scaled > ray.t_max * det) {
            return err;
        }

        let inv_det = 1.0 / det;
        let b0 = e0 * inv_det;
        let b1 = e1 * inv_det;
        let b2 = e2 * inv_det;
        let t = t_scaled * inv_det;

        let max_zt = Vector3f::new(p0t.z, p1t.z, p2t.z).abs().max_component();
        let delta_z = gamma(3.0) * max_zt;

        let max_xt = Vector3f::new(p0t.x, p1t.x, p2t.x).abs().max_component();
        let max_yt = Vector3f::new(p0t.y, p1t.y, p2t.y).abs().max_component();
        let _delta_x = gamma(5.0) * (max_xt + max_zt);
        let delta_y = gamma(5.0) * (max_yt + max_zt);

        let delta_e = 2.0 * (gamma(2.0) * max_xt * max_yt + delta_y * max_xt + delta_y * max_yt);
        let max_e = Vector3f::new(e0, e1, e2).abs().max_component();
        let delta_t = 3.0
            * (gamma(3.0) * max_e * max_zt + delta_e * max_zt + delta_z * max_e)
            * inv_det.abs();
        if t <= delta_t {
            return err;
        }

        //TODO alpha test
        (true, b0, b1, b2, t)
    }
}

impl Shape for Triangle {
    impl_base_shape!();

    fn object_bound(&self) -> Bounds3f {
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];
        let bounds = Bounds3f::from((
            self.world_to_object() * Point3Ref(&p0),
            self.world_to_object() * Point3Ref(&p1),
        ));
        bounds.union(&(self.world_to_object() * Point3Ref(&p2)))
    }

    fn world_bound(&self) -> Bounds3f {
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];
        Bounds3f::from((p0, p1)).union(&p2)
    }

    fn intersect(
        &self,
        ray: &Ray,
        hit: &mut f32,
        si: &mut SurfaceInteraction,
        _test_alpha_texture: bool,
    ) -> bool {
        let (ok, b0, b1, b2, t) = self.intersect_test(ray);
        if !ok {
            return false;
        }
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];

        let uv = self.get_uvs();
        let duv02 = uv[0] - uv[2];
        let duv12 = uv[1] - uv[2];
        let dp02 = p0 - p2;
        let dp12 = p1 - p2;
        let determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
        let degenerate_uv = determinant < 1e-8;
        let mut dpdu = Vector3f::default();
        let mut dpdv = Vector3f::default();
        if !degenerate_uv {
            let inv_det = 1.0 / determinant;
            dpdu = (dp02 * duv12[1] - dp12 * duv02[1]) * inv_det;
            dpdv = (dp02 * -duv12[0] + dp12 * duv02[0]) * inv_det;
        }

        if degenerate_uv || dpdu.cross(&dpdv).length_squared() == 0.0 {
            let ng = (p2 - p0).cross(&(p1 - p0));
            if ng.length_squared() == 0.0 {
                return false;
            }
            (dpdu, dpdv) = ng.normalize().coordinate_system();
        }

        let x_abs_sum = (b0 * p0.x).abs() + (b1 * p1.x).abs() + (b2 * p2.x).abs();
        let y_abs_sum = (b0 * p0.y).abs() + (b1 * p1.y).abs() + (b2 * p2.y).abs();
        let z_abs_sum = (b0 * p0.z).abs() + (b1 * p1.z).abs() + (b2 * p2.z).abs();

        let p_error = Vector3f::new(x_abs_sum, y_abs_sum, z_abs_sum) * gamma(7.0);

        let p_hit = p0 * b0 + p1 * b1 + p2 * b2;
        let uv_hit = uv[0] * b0 + uv[1] * b1 + uv[2] * b2;

        //TODO alpha test

        *si = SurfaceInteraction::new(
            p_hit,
            p_error,
            uv_hit,
            -ray.d,
            dpdu,
            dpdv,
            Normal3f::default(),
            Normal3f::default(),
            ray.time,
            None,
            self.face_index,
        );
        si.shading.n = dp02.cross(&dp12).normalize();
        si.n = si.shading.n;

        if self.reverse_orientation() ^ self.transform_swap_handedness() {
            si.n = -si.n;
            si.shading.n = si.n;
        }

        if self.mesh.n.is_some() || self.mesh.s.is_some() {
            let ns = if let Some(n) = &self.mesh.n {
                let mut ns = n[self.v[0]] * b0 + n[self.v[1]] * b1 + n[self.v[2]] * b2;
                if ns.length_squared() > 0.0 {
                    ns = ns.normalize();
                } else {
                    ns = si.n;
                }
                ns
            } else {
                si.n
            };

            let mut ss = if let Some(s) = &self.mesh.s {
                let ss = s[self.v[0]] * b0 + s[self.v[1]] * b1 + s[self.v[2]] * b2;
                if ss.length_squared() > 0.0 {
                    ss.normalize()
                } else {
                    si.dpdu.normalize()
                }
            } else {
                si.dpdu.normalize()
            };

            let mut ts = ss.cross(&ns);
            if ts.length_squared() > 0.0 {
                ts = ts.normalize();
                ss = ts.cross(&ns);
            } else {
                (ss, ts) = ns.coordinate_system();
            }

            let (dndu, dndv) = if let Some(n) = &self.mesh.n {
                let duv02 = uv[0] - uv[2];
                let duv12 = uv[1] - uv[2];
                let dn1 = n[self.v[0]] - n[self.v[2]];
                let dn2 = n[self.v[1]] - n[self.v[2]];
                let determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
                let _degeneate_uv = determinant < 1e-8;
                if degenerate_uv {
                    let dn = (n[self.v[2]] - n[self.v[0]]).cross(&(n[self.v[1]] - n[self.v[0]]));
                    if dn.length_squared() == 0.0 {
                        (Normal3f::default(), Normal3f::default())
                    } else {
                        let (dnu, dnv) = dn.coordinate_system();
                        (dnu.normalize(), dnv.normalize())
                    }
                } else {
                    let inv_det = 1.0 / determinant;
                    let dndu = (dn1 * duv12[1] - dn2 * duv02[1]) * inv_det;
                    let dndv = (dn1 * -duv12[0] + dn2 * duv02[0]) * inv_det;
                    (dndu, dndv)
                }
            } else {
                (Normal3f::default(), Normal3f::default())
            };
            if self.reverse_orientation() {
                ts = -ts;
            }
            si.set_shading_geometry(ss, ts, dndu, dndv, true);
        }

        *hit = t;
        true
    }

    fn intersect_p(&self, ray: &Ray, _test_alpha_texture: bool) -> bool {
        let (ok, _, _, _, _) = self.intersect_test(ray);
        ok
    }

    fn area(&self) -> f32 {
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];
        (p1 - p0).cross(&(p2 - p0)).length() * 0.5
    }

    fn sample(&self, u: &Point2f, pdf: &mut f32) -> InteractionDt {
        let b = uniform_sample_triangle(u);
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];
        let mut it = BaseInteraction::default();
        it.p = p0 * b[0] + p1 * b[1] + p2 * (1.0 - b[0] - b[1]);
        it.n = (p1 - p0).cross(&(p2 - p0)).normalize();
        if let Some(n) = &self.mesh.n {
            let ns = n[self.v[0]] * b[0] + n[self.v[1]] * b[1] + n[self.v[2]] * (1.0 - b[0] - b[1]);
            it.n = it.n.face_forward(ns);
        } else if self.reverse_orientation() ^ self.transform_swap_handedness() {
            it.n *= -1.0;
        }
        let p_abs_sum = (p0 * b[0]).abs() + (p1 * b[1]).abs() + (p2 * (1.0 - b[0] - b[1])).abs();
        it.error = p_abs_sum * gamma(6.0);
        *pdf = 1.0 / self.area();
        Arc::new(Box::new(it))
    }

    fn solid_angle(&self, p: Point3f, _samples: u64) -> f32 {
        let p0 = self.mesh.p[self.v[0]];
        let p1 = self.mesh.p[self.v[1]];
        let p2 = self.mesh.p[self.v[2]];
        let p_sphere = [
            (p0 - p).normalize(),
            (p1 - p).normalize(),
            (p2 - p).normalize(),
        ];
        let mut cross01 = p_sphere[0].cross(&p_sphere[1]);
        let mut cross12 = p_sphere[1].cross(&p_sphere[2]);
        let mut cross20 = p_sphere[2].cross(&p_sphere[0]);

        if cross01.length_squared() > 0.0 {
            cross01 = cross01.normalize();
        }
        if cross12.length_squared() > 0.0 {
            cross12 = cross12.normalize();
        }
        if cross20.length_squared() > 0.0 {
            cross20 = cross20.normalize();
        }

        (clamp(cross01.dot(&-cross12), -1.0, 1.0).acos()
            + clamp(cross12.dot(&-cross20), -1.0, 1.0).acos()
            + clamp(cross20.dot(&-cross01), -1.0, 1.0).acos()
            - PI)
            .abs()
    }
}
