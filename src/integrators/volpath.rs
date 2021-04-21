use crate::core::{
    camera::CameraDt,
    geometry::{Bounds2i, RayDifferentials},
    integrator::{Integrator, SamplerIntegrator},
    lightdistrib::{create_light_sample_distribution, LightDistributionDt},
    pbrt::Float,
    sampler::SamplerDtRw,
    scene::Scene,
    spectrum::Spectrum,
};
use derive_more::{Deref, DerefMut};
use std::any::Any;

#[derive(Deref, DerefMut)]
pub struct VolPathIntegrator {
    #[deref]
    #[deref_mut]
    base: SamplerIntegrator,
    max_depth: usize,
    rr_threshold: Float,
    light_sample_strategy: String,
    light_distribution: Option<LightDistributionDt>,
}

impl VolPathIntegrator {
    pub fn new(
        max_depth: usize,
        camera: CameraDt,
        sampler: SamplerDtRw,
        pixel_bounds: Bounds2i,
        rr_threshold: Float,
        light_sample_strategy: String,
    ) -> Self {
        Self {
            base: SamplerIntegrator::new(camera, sampler, pixel_bounds),
            max_depth,
            rr_threshold,
            light_sample_strategy,
            light_distribution: None,
        }
    }
}

impl Integrator for VolPathIntegrator {
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

    fn pre_process(&mut self, scene: &Scene, _sampler: SamplerDtRw) {
        self.light_distribution = Some(create_light_sample_distribution(
            self.light_sample_strategy.clone(),
            scene,
        ));
    }
}
