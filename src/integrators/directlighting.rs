use crate::core::{
    camera::CameraDt,
    geometry::{Bounds2i, RayDifferentials},
    integrator::{
        uniform_sample_all_lights, uniform_sample_one_light, Integrator, SamplerIntegrator,
    },
    interaction::SurfaceInteraction,
    light::is_delta_light,
    material::TransportMode,
    sampler::SamplerDtRw,
    scene::Scene,
    spectrum::Spectrum,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

pub enum LightStrategy {
    UniformSampleAll,
    UniformSampleOne,
}

#[derive(Deref, DerefMut)]
pub struct DirectLightingIntegrator {
    #[deref]
    #[deref_mut]
    base: SamplerIntegrator,
    strategy: LightStrategy,
    max_depth: usize,
    n_light_samples: Vec<usize>,
}

impl DirectLightingIntegrator {
    pub fn new(
        strategy: LightStrategy,
        max_depth: usize,
        camera: CameraDt,
        sampler: SamplerDtRw,
        pixel_bounds: Bounds2i,
    ) -> Self {
        Self {
            base: SamplerIntegrator::new(camera, sampler, pixel_bounds),
            strategy,
            max_depth,
            n_light_samples: vec![],
        }
    }
}

impl Integrator for DirectLightingIntegrator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&mut self, scene: &Scene) {
        self.base.render(scene)
    }

    fn pre_process(&mut self, scene: &Scene, sampler: SamplerDtRw) {
        if let LightStrategy::UniformSampleAll = self.strategy {
            for light in &scene.lights {
                self.n_light_samples
                    .push(sampler.read().unwrap().round_count(light.n_samples()));
            }

            for i in 0..self.max_depth {
                for j in 0..scene.lights.len() {
                    sampler
                        .write()
                        .unwrap()
                        .request_2d_array(self.n_light_samples[j]);
                    sampler
                        .write()
                        .unwrap()
                        .request_2d_array(self.n_light_samples[j]);
                }
            }
        }
    }
    fn li(
        &self,
        ray: &mut RayDifferentials,
        scene: &Scene,
        sampler: SamplerDtRw,
        depth: usize,
    ) -> Spectrum {
        let mut l = Spectrum::new(0.0);
        let mut isect = SurfaceInteraction::default();

        if !scene.intersect(&mut ray.base, &mut isect) {
            for light in &scene.lights {
                l += light.le(ray);
            }
            return l;
        }

        isect.compute_scattering_functions(ray, false, TransportMode::Radiance);
        if isect.bsdf.is_none() {
            self.li(
                &mut isect.spawn_ray(&ray.d).into(),
                scene,
                sampler.clone(),
                depth,
            );
        }
        let wo = isect.wo;
        if scene.lights.len() > 0 {
            match self.strategy {
                LightStrategy::UniformSampleAll => {
                    l += uniform_sample_all_lights(
                        Arc::new(Box::new(isect.clone())),
                        scene,
                        sampler.clone(),
                        &self.n_light_samples,
                        false,
                    );
                }
                LightStrategy::UniformSampleOne => {
                    l += uniform_sample_one_light(
                        Arc::new(Box::new(isect.clone())),
                        scene,
                        sampler.clone(),
                        false,
                        None,
                    );
                }
            }
        }
        if depth + 1 < self.max_depth {
            l += self.specular_reflect(ray, &isect, scene, sampler.clone(), depth);
            l += self.specular_transmit(ray, &isect, scene, sampler.clone(), depth);
        }
        l
    }
}
