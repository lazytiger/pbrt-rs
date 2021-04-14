use crate::core::{
    camera::CameraDt,
    geometry::{Bounds2i, RayDifferentials},
    integrator::{BaseSamplerIntegrator, Integrator},
    sampler::SamplerDtRw,
    scene::Scene,
    spectrum::Spectrum,
};
use derive_more::{Deref, DerefMut};
use std::any::Any;

pub struct MLTIntegrator {
    cos_sample: bool,
    n_samples: usize,
}

impl MLTIntegrator {
    pub fn new(
        cos_sample: bool,
        n_samples: usize,
        camera: CameraDt,
        sampler: SamplerDtRw,
        pixel_bounds: &Bounds2i,
    ) -> Self {
        todo!()
    }
}

impl Integrator for MLTIntegrator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&self, scene: &Scene) {
        todo!()
    }

    fn li(
        &self,
        ray: &RayDifferentials,
        scene: &Scene,
        sampler: SamplerDtRw,
        depth: i32,
    ) -> Spectrum {
        unimplemented!()
    }
}
