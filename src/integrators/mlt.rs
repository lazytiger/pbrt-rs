use crate::core::{
    camera::CameraDt,
    geometry::{Bounds2i, RayDifferentials},
    integrator::{Integrator, SamplerIntegrator},
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

    fn render(&mut self, scene: &Scene) {
        todo!()
    }

    fn li(
        &self,
        ray: &mut RayDifferentials,
        scene: &Scene,
        sampler: SamplerDtRw,
        depth: usize,
    ) -> Spectrum {
        unimplemented!()
    }
}
