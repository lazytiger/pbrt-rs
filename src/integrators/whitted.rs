use crate::core::{
    camera::CameraDt,
    geometry::{Bounds2i, Ray, RayDifferentials, Vector3f},
    integrator::{Integrator, SamplerIntegrator},
    interaction::SurfaceInteraction,
    light::VisibilityTester,
    material::TransportMode,
    reflection::BxDFType,
    sampler::SamplerDtRw,
    scene::Scene,
    spectrum::Spectrum,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

#[derive(Deref, DerefMut)]
pub struct WhittedIntegrator {
    #[deref]
    #[deref_mut]
    base: SamplerIntegrator,
    max_depth: usize,
}

impl WhittedIntegrator {
    pub fn new(
        max_depth: usize,
        camera: CameraDt,
        sampler: SamplerDtRw,
        pixel_bounds: Bounds2i,
    ) -> Self {
        Self {
            base: SamplerIntegrator::new(camera, sampler, pixel_bounds),
            max_depth,
        }
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
        let mut l = Spectrum::default();
        let mut isect = SurfaceInteraction::default();
        if !scene.intersect(&mut ray.base, &mut isect) {
            for light in &scene.lights {
                l += light.le(ray);
            }
            return l;
        }

        let n = isect.shading.n;
        let mut wo = isect.wo;
        isect.compute_scattering_functions(ray, false, TransportMode::Radiance);
        if isect.bsdf.is_none() {
            let mut ray: RayDifferentials = isect.spawn_ray(&ray.d).into();
            return self.li(&mut ray, scene, sampler, depth);
        }

        l += isect.le(&wo);

        for light in &scene.lights {
            let mut wi = Vector3f::default();
            let mut pdf = 0.0;
            let mut visibility = VisibilityTester::default();
            let li = light.sample_li(
                Arc::new(Box::new(isect.clone())),
                &sampler.write().unwrap().get_2d(),
                &mut wi,
                &mut pdf,
                &mut visibility,
            );
            if li.is_black() || pdf == 0.0 {
                continue;
            }
            let f = isect.bsdf.as_ref().unwrap().f(&wo, &wi, BxDFType::all());
            if !f.is_black() && visibility.un_occluded(scene) {
                l += f * li * wi.abs_dot(&n) / pdf;
            }
        }

        if depth + 1 < self.max_depth {
            l += self.specular_reflect(ray, &isect, scene, sampler.clone(), depth);
            l += self.specular_transmit(ray, &isect, scene, sampler, depth);
        }
        l
    }
}
