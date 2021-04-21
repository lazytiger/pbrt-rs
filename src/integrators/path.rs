use crate::core::{
    camera::CameraDt,
    geometry::{Bounds2i, RayDifferentials, Vector3f},
    integrator::{
        compute_light_power_distribution, uniform_sample_one_light, Integrator, SamplerIntegrator,
    },
    interaction::SurfaceInteraction,
    lightdistrib::{create_light_sample_distribution, LightDistributionDt},
    material::TransportMode,
    pbrt::Float,
    reflection::BxDFType,
    sampler::SamplerDtRw,
    scene::Scene,
    spectrum::Spectrum,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

#[derive(Deref, DerefMut)]
pub struct PathIntegrator {
    #[deref]
    #[deref_mut]
    base: SamplerIntegrator,
    max_depth: usize,
    rr_threshold: Float,
    light_sample_strategy: String,
    light_distribution: Option<LightDistributionDt>,
}

impl PathIntegrator {
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

impl Integrator for PathIntegrator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&mut self, scene: &Scene) {
        self.base.render(scene)
    }

    fn pre_process(&mut self, scene: &Scene, _sampler: SamplerDtRw) {
        self.light_distribution = Some(create_light_sample_distribution(
            self.light_sample_strategy.clone(),
            scene,
        ));
    }

    fn li(
        &self,
        mut ray: &mut RayDifferentials,
        scene: &Scene,
        sampler: SamplerDtRw,
        depth: usize,
    ) -> Spectrum {
        let mut l = Spectrum::new(0.0);
        let mut beta = Spectrum::new(1.0);
        let mut specular_bounce = false;
        let mut bounces = 0;
        let mut eta_scale = 1.0;
        loop {
            let mut isect = SurfaceInteraction::default();
            let found_intersection = scene.intersect(&mut ray.base, &mut isect);
            if bounces == 0 || specular_bounce {
                if found_intersection {
                    l += beta * isect.le(&-ray.d);
                } else {
                    for light in &scene.infinite_lights {
                        l += beta * light.le(ray);
                    }
                }
            }

            if !found_intersection || bounces >= self.max_depth {
                break;
            }

            isect.compute_scattering_functions(ray, true, TransportMode::Radiance);
            if isect.bsdf.is_none() {
                *ray = isect.spawn_ray(&ray.d).into();
                continue;
            }

            let distrib = self.light_distribution.as_ref().unwrap().lookup(&isect.p);
            if isect
                .bsdf
                .as_ref()
                .unwrap()
                .num_components(BxDFType::all() - BxDFType::BSDF_SPECULAR)
                > 0
            {
                let ld = beta
                    * uniform_sample_one_light(
                        &isect,
                        scene,
                        sampler.clone(),
                        false,
                        Some(distrib),
                    );
                l += ld;
            }

            let wo = -ray.d;
            let mut wi = Vector3f::default();
            let mut pdf = 0.0;
            let mut flags = BxDFType::empty();
            let f = isect.bsdf.as_ref().unwrap().sample_f(
                &wo,
                &mut wi,
                &sampler.write().unwrap().get_2d(),
                &mut pdf,
                BxDFType::all(),
                Some(&mut flags),
            );

            if f.is_black() || pdf == 0.0 {
                break;
            }

            beta *= f * (wi.abs_dot(&isect.shading.n) / pdf);
            specular_bounce = flags.contains(BxDFType::BSDF_SPECULAR);
            if flags.contains(BxDFType::BSDF_SPECULAR)
                && flags.contains(BxDFType::BSDF_TRANSMISSION)
            {
                let eta = isect.bsdf.as_ref().unwrap().eta;
                eta_scale *= if wo.dot(&isect.n) > 0.0 {
                    eta * eta
                } else {
                    1.0 / (eta * eta)
                };
            }
            *ray = isect.spawn_ray(&wi).into();

            if isect.bssrdf.is_some() && flags.contains(BxDFType::BSDF_TRANSMISSION) {
                let mut pi = SurfaceInteraction::default();
                let s = isect.bssrdf.as_ref().unwrap().sample_s(
                    scene,
                    sampler.write().unwrap().get_1d(),
                    &sampler.write().unwrap().get_2d(),
                    &mut pi,
                    &mut pdf,
                );

                if s.is_black() || pdf == 0.0 {
                    break;
                }
                beta *= s / pdf;

                l += beta
                    * uniform_sample_one_light(
                        &pi,
                        scene,
                        sampler.clone(),
                        false,
                        Some(self.light_distribution.as_ref().unwrap().lookup(&pi.p)),
                    );

                let f = pi.bsdf.as_ref().unwrap().sample_f(
                    &pi.wo,
                    &mut wi,
                    &sampler.write().unwrap().get_2d(),
                    &mut pdf,
                    BxDFType::all(),
                    Some(&mut flags),
                );

                if f.is_black() || pdf == 0.0 {
                    break;
                }
                beta *= f * (wi.abs_dot(&pi.shading.n) / pdf);
                specular_bounce = flags.contains(BxDFType::BSDF_SPECULAR);
                *ray = pi.spawn_ray(&wi).into();
            }

            let rr_beta = beta * eta_scale;
            if rr_beta.max_component_value() < self.rr_threshold && bounces > 3 {
                let q = (1.0 - rr_beta.max_component_value()).min(0.05);
                if sampler.write().unwrap().get_1d() < q {
                    break;
                }
                beta /= 1.0 - q;
            }

            bounces += 1;
        }

        l
    }
}
