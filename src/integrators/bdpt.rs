use crate::core::{
    camera::CameraDt,
    geometry::{Bounds2i, RayDifferentials},
    integrator::{BaseSamplerIntegrator, Integrator},
    sampler::SamplerDtRw,
    scene::Scene,
    spectrum::Spectrum,
};
use std::any::Any;

pub struct BDPTIntegrator {
    cos_sample: bool,
    n_samples: usize,
}

impl BDPTIntegrator {
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

impl Integrator for BDPTIntegrator {
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
