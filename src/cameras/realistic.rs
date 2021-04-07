use crate::core::camera::{BaseCamera, Camera, CameraSample};
use crate::core::efloat::EFloat;
use crate::core::film::Film;
use crate::core::geometry::{Bounds2f, Normal3f, Point2f, Point3f, Ray, Union, Vector3, Vector3f};
use crate::core::lowdiscrepancy::radical_inverse;
use crate::core::medium::Medium;
use crate::core::reflection::refract;
use crate::core::transform::{AnimatedTransform, Transformf};
use crate::core::{lerp, RealNum};
use crate::{impl_base_camera, quadratic, Float};
use log::Level::Trace;
use std::any::Any;
use std::convert::TryInto;
use std::sync::Arc;

struct LensElementInterface {
    curvature_radius: Float,
    thickness: Float,
    eta: Float,
    aperture_radius: Float,
}
pub struct RealisticCamera {
    base: BaseCamera,
    simple_weighting: bool,
    element_interfaces: Vec<LensElementInterface>,
    exit_pupil_bounds: Vec<Bounds2f>,
}

impl RealisticCamera {
    pub fn new(
        camera_to_world: AnimatedTransform,
        shutter_open: Float,
        shutter_close: Float,
        aperture_diameter: Float,
        focus_distance: Float,
        simple_weighting: bool,
        lens_data: &mut Vec<Float>,
        film: Arc<Film>,
        medium: Arc<Box<Medium>>,
    ) -> Self {
        let mut rc = Self {
            base: BaseCamera::new(
                camera_to_world,
                shutter_open,
                shutter_close,
                film.clone(),
                medium,
            ),
            simple_weighting,
            element_interfaces: Default::default(),
            exit_pupil_bounds: Default::default(),
        };
        for i in (0..lens_data.len()).step_by(4) {
            if lens_data[i] == 0.0 {
                if aperture_diameter > lens_data[i + 3] {
                    log::warn!("specified aperture diameter {} is greater than maximum possible {}. clamping it", aperture_diameter, lens_data[i+3]);
                } else {
                    lens_data[i + 3] = aperture_diameter;
                }
            }
            rc.element_interfaces.push(LensElementInterface {
                curvature_radius: lens_data[i] * 0.001,
                thickness: lens_data[i + 1] * 0.001,
                eta: lens_data[i + 2],
                aperture_radius: lens_data[i + 3] * 0.001 / 2.0,
            });
        }

        let fb = rc.focus_binary_search(focus_distance);
        rc.element_interfaces.last_mut().unwrap().thickness = rc.focus_thick_lens(focus_distance);
        const n_samples: usize = 64;
        rc.exit_pupil_bounds.resize(n_samples, Bounds2f::default());
        for i in 0..n_samples {
            let r0 = i as Float / n_samples as Float * film.diagonal / 2.0;
            let r1 = (i + 1) as Float / n_samples as Float * film.diagonal / 2.0;
            rc.exit_pupil_bounds[i] = rc.bound_exit_pupil(r0, r1);
        }

        if simple_weighting {
            log::warn!("deprecated option");
        }
        rc
    }

    fn trace_lenses_from_film(&self, r_camera: &Ray, r_out: Option<&mut Ray>) -> bool {
        let mut element_z = 0.0;
        let camera_to_lens = Transformf::scale(1.0, 1.0, -1.0);
        let mut r_lens = Ray::from((&camera_to_lens, r_camera));
        for i in (0..self.element_interfaces.len()).rev() {
            let element = &self.element_interfaces[i];
            element_z -= element.thickness;

            let mut t = 0.0;
            let mut n = Normal3f::default();
            let is_stop = element.curvature_radius == 0.0;

            if is_stop {
                if r_lens.d.z >= 0.0 {
                    return false;
                }
                t = (element_z - r_lens.o.z) / r_lens.d.z;
            } else {
                let radius = element.curvature_radius;
                let z_center = element_z + element.curvature_radius;
                if !Self::intersect_spherical_element(radius, z_center, &r_lens, &mut t, &mut n) {
                    return false;
                }
            }

            let p_hit = r_lens.point(t);
            let r2 = p_hit.x * p_hit.x + p_hit.y * p_hit.y;
            if r2 > element.aperture_radius * element.aperture_radius {
                return false;
            }
            r_lens.o = p_hit;

            if !is_stop {
                let mut w = Vector3f::default();
                let eta_i = element.eta;
                let eta_t = if i > 0 && self.element_interfaces[i - 1].eta != 0.0 {
                    self.element_interfaces[i - 1].eta
                } else {
                    1.0
                };
                if refract(&(-r_lens.d).normalize(), &n, eta_i / eta_t, &mut w) {
                    return false;
                }
                r_lens.d = w;
            }
        }
        if let Some(r_out) = r_out {
            let lens_to_camera = Transformf::scale(1.0, 1.0, -1.0);
            *r_out = Ray::from((&lens_to_camera, &r_lens));
        }
        true
    }

    fn lens_rear_z(&self) -> Float {
        self.element_interfaces.last().unwrap().thickness
    }

    fn lens_front_z(&self) -> Float {
        self.element_interfaces
            .iter()
            .fold(0.0, |sum, item| sum + item.thickness)
    }

    fn rear_element_radius(&self) -> Float {
        self.element_interfaces.last().unwrap().aperture_radius
    }

    fn intersect_spherical_element(
        radius: Float,
        z_center: Float,
        ray: &Ray,
        t: &mut Float,
        n: &mut Normal3f,
    ) -> bool {
        let o = ray.o - Vector3f::new(0.0, 0.0, z_center);
        let a = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
        let b = 2.0 * (ray.d.x * o.x + ray.d.y * o.y + ray.d.z * o.z);
        let c = o.x * o.x + o.y * o.y + o.z * o.z - radius * radius;
        let mut t0 = 0.0;
        let mut t1 = 0.0;
        if !quadratic(a, b, c, &mut t0, &mut t1) {
            return false;
        }

        let use_closer_t = (ray.d.z > 0.0) ^ (radius < 0.0);
        *t = if use_closer_t { t0.min(t1) } else { t0.max(t1) };
        if *t < 0.0 {
            return false;
        }

        *n = o + ray.d * *t;
        *n = n.normalize().face_forward(-ray.d);
        true
    }

    fn trace_lenses_from_scene(&self, r_camera: &Ray, r_out: Option<&mut Ray>) -> bool {
        let mut element_z = -self.lens_front_z();
        let camera_to_lens = Transformf::scale(1.0, 1.0, -1.0);
        let mut r_lens = Ray::from((&camera_to_lens, r_camera));
        for i in (0..self.element_interfaces.len()).rev() {
            let element = &self.element_interfaces[i];

            let mut t = 0.0;
            let mut n = Normal3f::default();
            let is_stop = element.curvature_radius == 0.0;

            if is_stop {
                t = (element_z - r_lens.o.z) / r_lens.d.z;
            } else {
                let radius = element.curvature_radius;
                let z_center = element_z + element.curvature_radius;
                if !Self::intersect_spherical_element(radius, z_center, &r_lens, &mut t, &mut n) {
                    return false;
                }
            }

            let p_hit = r_lens.point(t);
            let r2 = p_hit.x * p_hit.x + p_hit.y * p_hit.y;
            if r2 > element.aperture_radius * element.aperture_radius {
                return false;
            }
            r_lens.o = p_hit;

            if !is_stop {
                let mut wt = Vector3f::default();
                let eta_i = if i == 0 || self.element_interfaces[i - 1].eta == 0.0 {
                    1.0
                } else {
                    self.element_interfaces[i - 1].eta
                };
                let eta_t = if self.element_interfaces[i].eta != 0.0 {
                    self.element_interfaces[i].eta
                } else {
                    1.0
                };
                if refract(&(-r_lens.d).normalize(), &n, eta_i / eta_t, &mut wt) {
                    return false;
                }
                r_lens.d = wt;
            }
            element_z += element.thickness;
        }
        if let Some(r_out) = r_out {
            let lens_to_camera = Transformf::scale(1.0, 1.0, -1.0);
            *r_out = Ray::from((&lens_to_camera, &r_lens));
        }
        true
    }

    fn compute_cardinal_points(r_in: &Ray, r_out: &Ray, pz: &mut Float, fz: &mut Float) {
        let tf = -r_out.o.x / r_out.d.x;
        *fz = -r_out.point(tf).z;
        let tp = (r_in.o.x - r_out.o.x) / r_out.d.x;
        *pz = -r_out.point(tp).z;
    }

    fn compute_thick_lens_approximation(&self, pz: &mut [Float; 2], fz: &mut [Float; 2]) {
        let x = 0.001 * self.film().diagonal;
        let r_scene = Ray::new(
            Point3f::new(x, 0.0, self.lens_front_z() + 1.0),
            Vector3f::new(0.0, 0.0, -1.0),
            Float::INFINITY,
            0.0,
            None,
        );
        let r_film = Ray::default();
        Self::compute_cardinal_points(&r_scene, &r_film, &mut pz[0], &mut fz[0]);
        let r_film = Ray::new(
            Point3f::new(x, 0.0, self.lens_rear_z() - 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Float::INFINITY,
            0.0,
            None,
        );
        Self::compute_cardinal_points(&r_film, &r_scene, &mut pz[1], &mut fz[1]);
    }

    fn focus_thick_lens(&self, focus_distance: Float) -> Float {
        let mut pz = [0.0; 2];
        let mut fz = [0.0; 2];
        self.compute_thick_lens_approximation(&mut pz, &mut fz);

        let f = fz[0] - pz[0];
        let z = -focus_distance;
        let c = (pz[1] - z - pz[0]) * (pz[1] - z - 4.0 * f - pz[0]);
        let delta = 0.5 * (pz[1] - z + pz[0] - c.sqrt());
        self.element_interfaces.last().unwrap().thickness + delta
    }

    fn focus_binary_search(&self, focus_distance: Float) -> Float {
        let mut film_distance_lower = 0.0;
        let mut film_distance_upper = 0.0;
        film_distance_lower = self.focus_thick_lens(focus_distance);
        film_distance_upper = film_distance_lower;
        while self.focus_distance(film_distance_lower) > focus_distance {
            film_distance_lower *= 1.005;
        }
        while self.focus_distance(film_distance_upper) > focus_distance {
            film_distance_upper *= 1.005;
        }

        for i in 0..20 {
            let fmid = 0.5 * (film_distance_lower + film_distance_upper);
            let mid_focus = self.focus_distance(fmid);
            if mid_focus < focus_distance {
                film_distance_lower = fmid;
            } else {
                film_distance_upper = fmid;
            }
        }
        0.5 * (film_distance_lower + film_distance_upper)
    }

    fn focus_distance(&self, film_distance: Float) -> Float {
        let bounds = self.bound_exit_pupil(0.0, 0.001 * self.film().diagonal);
        let scaled_factors = [0.1, 0.01, 0.001];
        let mut lu = 0.0;
        let mut ray = Ray::default();
        let mut found_focus_ray = false;
        for scale in &scaled_factors {
            lu = *scale * bounds.max[0];
            if self.trace_lenses_from_film(
                &Ray::new(
                    Point3f::new(0.0, 0.0, self.lens_rear_z() - film_distance),
                    Vector3f::new(lu, 0.0, film_distance),
                    Float::INFINITY,
                    0.0,
                    None,
                ),
                Some(&mut ray),
            ) {
                found_focus_ray = true;
                break;
            }
        }

        if !found_focus_ray {
            log::error!(
                "focus ray at lens {} didn't make it through the lenses with film distance {}",
                lu,
                film_distance
            );
            return Float::INFINITY;
        }

        let t_focus = -ray.o.x / ray.d.x;
        let z_focus = ray.point(t_focus).z;
        if z_focus < 0.0 {
            Float::INFINITY
        } else {
            z_focus
        }
    }

    fn bound_exit_pupil(&self, p_film_x0: Float, p_film_x1: Float) -> Bounds2f {
        let mut pupil_bounds = Bounds2f::default();

        const n_samples: usize = 1024 * 1024;
        let mut n_exiting_rays = 0;
        let rear_radius = self.rear_element_radius();
        let proj_rear_bounds = Bounds2f::from((
            Point2f::new(-1.5 * rear_radius, -1.5 * rear_radius),
            Point2f::new(1.5 * rear_radius, 1.5 * rear_radius),
        ));
        for i in 0..n_samples {
            let p_film = Point3f::new(
                lerp(
                    (i as Float + 0.5) / n_samples as Float,
                    p_film_x0,
                    p_film_x1,
                ),
                0.0,
                0.0,
            );
            let u = [radical_inverse(0, i as u64), radical_inverse(1, i as u64)];
            let p_rear = Point3f::new(
                lerp(u[0], proj_rear_bounds.min.x, proj_rear_bounds.max.x),
                lerp(u[1], proj_rear_bounds.min.y, proj_rear_bounds.max.y),
                self.lens_rear_z(),
            );

            if pupil_bounds.inside(&Point2f::new(p_rear.x, p_rear.y))
                || self.trace_lenses_from_film(
                    &Ray::new(p_film, p_rear - p_film, Float::INFINITY, 0.0, None),
                    None,
                )
            {
                pupil_bounds = pupil_bounds.union(&Point2f::new(p_rear.x, p_rear.y));
                n_exiting_rays += 1;
            }
        }
        if n_exiting_rays == 0 {
            return proj_rear_bounds;
        }

        pupil_bounds = pupil_bounds
            .expand(2.0 * proj_rear_bounds.diagonal().length() / (n_samples as Float).sqrt());
        pupil_bounds
    }

    fn render_exit_pupil(&self, sx: Float, sy: Float, filename: String) {
        unimplemented!()
    }

    fn sample_exit_pupil(
        &self,
        p_film: &Point2f,
        lens_sample: &Point2f,
        sample_bounds_area: Option<&mut Float>,
    ) -> Point3f {
        let r_film = (p_film.x * p_film.x + p_film.y * p_film.y).sqrt();
        let mut r_index =
            r_film / (self.film().diagonal / 2.0) * self.exit_pupil_bounds.len() as Float;
        r_index = r_index.min(self.exit_pupil_bounds.len() as Float - 1.0);
        let pupil_bounds = self.exit_pupil_bounds[r_index as usize];
        if let Some(sample_bounds_area) = sample_bounds_area {
            *sample_bounds_area = pupil_bounds.area();
        }
        let p_lens = pupil_bounds.lerp(lens_sample);

        let sin_theta = if r_film != 0.0 {
            p_film.y / r_film
        } else {
            0.0
        };
        let cos_theta = if r_film != 0.0 {
            p_film.x / r_film
        } else {
            1.0
        };

        Point3f::new(
            cos_theta * p_lens.x - sin_theta * p_lens.y,
            sin_theta * p_lens.x + cos_theta * p_lens.y,
            self.lens_rear_z(),
        )
    }
}

impl Camera for RealisticCamera {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f32 {
        let s = Point2f::new(
            sample.p_film.x / self.film().full_resolution.x as Float,
            sample.p_film.y / self.film().full_resolution.y as Float,
        );

        let p_film2 = self.film().get_physical_extent().lerp(&s);
        let p_film = Point3f::new(-p_film2.x, p_film2.y, 0.0);

        let mut exit_pupil_bounds_area = 0.0;
        let p_rear = self.sample_exit_pupil(
            &Point2f::new(p_film.x, p_film.y),
            &sample.p_lens,
            Some(&mut exit_pupil_bounds_area),
        );
        let r_film = Ray::new(
            p_film,
            p_rear - p_film,
            Float::INFINITY,
            lerp(sample.time, self.shutter_open(), self.shutter_close()),
            None,
        );

        if !self.trace_lenses_from_film(&r_film, Some(ray)) {
            return 0.0;
        }

        *ray = Ray::from((self.camera_to_world(), &*ray));
        ray.d = ray.d.normalize();
        ray.medium = Some(self.medium().clone());

        let cos_theta = r_film.d.normalize().z;
        let cos4theta = (cos_theta * cos_theta) * (cos_theta * cos_theta);
        if self.simple_weighting {
            cos4theta * exit_pupil_bounds_area / self.exit_pupil_bounds[0].area()
        } else {
            (self.shutter_close() - self.shutter_open()) * (cos4theta * exit_pupil_bounds_area)
                / (self.lens_rear_z() * self.lens_rear_z())
        }
    }

    impl_base_camera!();
}
