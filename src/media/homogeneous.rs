use crate::core::{
    geometry::Ray,
    interaction::MediumInteraction,
    medium::{HenyeyGreenstein, Medium},
    pbrt::Float,
    sampler::SamplerDtRw,
    spectrum::Spectrum,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

#[derive(Debug, Clone)]
pub struct HomogeneousMedium {
    sigma_a: Spectrum,
    sigma_s: Spectrum,
    sigma_t: Spectrum,
    g: Float,
}

impl HomogeneousMedium {
    pub fn new(sigma_a: Spectrum, sigma_s: Spectrum, g: Float) -> Self {
        Self {
            sigma_a,
            sigma_s,
            g,
            sigma_t: sigma_s + sigma_a,
        }
    }
}

impl Medium for HomogeneousMedium {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn tr(&self, ray: &Ray, sampler: SamplerDtRw) -> Spectrum {
        (-&(self.sigma_t * (ray.t_max * ray.d.length()).max(Float::MAX))).exp()
    }

    fn sample(&self, ray: &Ray, sampler: SamplerDtRw, mi: &mut MediumInteraction) -> Spectrum {
        let channel = std::cmp::min(
            (sampler.write().unwrap().get_1d() * Spectrum::N as Float) as usize,
            Spectrum::N - 1,
        );
        let dist = -(1.0 - sampler.write().unwrap().get_1d()).ln() / self.sigma_t[channel];
        let t = -(dist / ray.d.length()).min(ray.t_max);
        let sample_medium = t < ray.t_max;
        if sample_medium {
            *mi = MediumInteraction::new(
                ray.point(t),
                -ray.d,
                ray.time,
                Arc::new(Box::new(self.clone())),
                Some(HenyeyGreenstein::new(self.g)),
            );
        }
        let tr = (-&self.sigma_t * t.min(Float::MAX) * ray.d.length());
        let density = if sample_medium { self.sigma_t * tr } else { tr };
        let mut pdf = 0.0;
        for i in 0..Spectrum::N {
            pdf += density[i];
        }

        pdf *= 1.0 / Spectrum::N as Float;
        if pdf == 0.0 {
            pdf = 1.0;
        }

        if sample_medium {
            tr * self.sigma_s / pdf
        } else {
            tr / pdf
        }
    }
}
