use crate::core::{
    geometry::{Bounds3f, Bounds3i, IntersectP, Point3f, Point3i, Ray, Vector3i},
    interaction::MediumInteraction,
    medium::{HenyeyGreenstein, Medium},
    pbrt::{lerp, Float},
    sampler::SamplerDtRw,
    spectrum::Spectrum,
    transform::Transformf,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

#[derive(Debug, Clone)]
pub struct GridDensityMedium {
    sigma_a: Spectrum,
    sigma_s: Spectrum,
    g: Float,
    nx: usize,
    ny: usize,
    nz: usize,
    world_to_medium: Transformf,
    density: Vec<Float>,
    sigma_t: Float,
    inv_max_density: Float,
}

impl GridDensityMedium {
    pub fn new(
        sigma_a: Spectrum,
        sigma_s: Spectrum,
        g: Float,
        nx: usize,
        ny: usize,
        nz: usize,
        medium_to_world: Transformf,
        d: &[Float],
    ) -> Self {
        let density: Vec<Float> = d.into();
        let sigma_t = sigma_a[0] + sigma_s[0];
        let max_density = density.iter().fold(
            Float::MIN,
            |ret, item| {
                if ret > *item {
                    ret
                } else {
                    *item
                }
            },
        );
        let inv_max_density = 1.0 / max_density;
        Self {
            sigma_a,
            sigma_s,
            g,
            nx,
            ny,
            nz,
            world_to_medium: medium_to_world.inverse(),
            density,
            sigma_t,
            inv_max_density,
        }
    }

    pub fn density(&self, p: &Point3f) -> Float {
        let p_samples = Point3f::new(
            p.x * self.nx as Float - 0.5,
            p.y * self.ny as Float - 0.5,
            p.z * self.nz as Float - 0.5,
        );
        let pi = p_samples.floor();
        let d = p_samples - pi;
        let pi = pi.into();

        let d00 = lerp(d.x, self.d(&pi), self.d(&(pi + Vector3i::new(1, 0, 0))));
        let d10 = lerp(
            d.x,
            self.d(&(pi + Vector3i::new(0, 1, 0))),
            self.d(&(pi + Vector3i::new(1, 1, 0))),
        );
        let d01 = lerp(
            d.x,
            self.d(&(pi + Vector3i::new(0, 0, 1))),
            self.d(&(pi + Vector3i::new(1, 0, 1))),
        );
        let d11 = lerp(
            d.x,
            self.d(&(pi + Vector3i::new(0, 1, 1))),
            self.d(&(pi + Vector3i::new(1, 1, 1))),
        );
        let d0 = lerp(d.y, d00, d10);
        let d1 = lerp(d.y, d01, d11);
        lerp(d.z, d0, d1)
    }

    pub fn d(&self, p: &Point3i) -> Float {
        let sample_bounds = Bounds3i::from((
            Point3i::default(),
            Point3i::new(self.nx as i32, self.ny as i32, self.nz as i32),
        ));
        if !sample_bounds.inside_exclusive(p) {
            0.0
        } else {
            self.density[((p.z * self.ny as i32 + p.y) * self.nx as i32 + p.x) as usize]
        }
    }
}

impl Medium for GridDensityMedium {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn tr(&self, r_world: &Ray, sampler: SamplerDtRw) -> Spectrum {
        let ray = Ray::from((
            &self.world_to_medium,
            &Ray::new(
                r_world.o,
                r_world.d.normalize(),
                r_world.t_max * r_world.d.length(),
                0.0,
                None,
            ),
        ));
        let b = Bounds3f::from((Point3f::default(), Point3f::new(1.0, 1.0, 1.0)));
        let (ok, t_min, t_max) = b.intersect_p(&ray);
        if !ok {
            return Spectrum::new(1.0);
        }
        let mut tr = 1.0;
        let mut t = t_min;

        loop {
            t -= (1.0 - sampler.write().unwrap().get_1d()).ln() * self.inv_max_density
                / self.sigma_t;
            if t >= t_max {
                break;
            }

            let density = self.density(&ray.point(t));
            tr *= 1.0 - (density * self.inv_max_density).max(0.0);
            const rr_threshold: Float = 0.1;
            if tr < rr_threshold {
                let q = (1.0 - tr).max(0.05);
                if sampler.write().unwrap().get_1d() < q {
                    return 0.0.into();
                }
                tr /= 1.0 - q;
            }
        }
        tr.into()
    }

    fn sample(&self, r_world: &Ray, sampler: SamplerDtRw, mi: &mut MediumInteraction) -> Spectrum {
        let ray = Ray::from((
            &self.world_to_medium,
            &Ray::new(
                r_world.o,
                r_world.d.normalize(),
                r_world.t_max * r_world.d.length(),
                0.0,
                None,
            ),
        ));
        let b = Bounds3f::from((Point3f::default(), Point3f::new(1.0, 1.0, 1.0)));
        let (ok, t_min, t_max) = b.intersect_p(&ray);
        if !ok {
            return Spectrum::new(1.0);
        }
        let mut t = t_min;
        loop {
            t -= (1.0 - sampler.write().unwrap().get_1d()).ln() * self.inv_max_density
                / self.sigma_t;
            if t >= t_max {
                break;
            }
            if self.density(&ray.point(t)) * self.inv_max_density
                > sampler.write().unwrap().get_1d()
            {
                let phase = HenyeyGreenstein::new(self.g);
                *mi = MediumInteraction::new(
                    r_world.point(t),
                    -r_world.d,
                    r_world.time,
                    Arc::new(Box::new(self.clone())), //FIXME
                    Some(phase),
                );
                return self.sigma_s / self.sigma_t;
            }
        }
        Spectrum::new(1.0)
    }
}
