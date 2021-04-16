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

#[derive(Deref, DerefMut)]
pub struct WhittedIntegrator {
    #[deref]
    #[deref_mut]
    base: SamplerIntegrator,
    cos_sample: bool,
    n_samples: usize,
}

impl WhittedIntegrator {
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

impl Integrator for WhittedIntegrator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&mut self, scene: &Scene) {
        self.base.render(scene)
    }

    fn li(
        &self,
        ray: &mut RayDifferentials,
        scene: &Scene,
        sampler: SamplerDtRw,
        depth: usize,
    ) -> Spectrum {
        todo!()
    }
}
