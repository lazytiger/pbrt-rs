use crate::core::{
    geometry::{Bounds2f, Bounds3, Bounds3f, Ray},
    interaction::SurfaceInteraction,
    light::{Light, LightFlags},
    primitive::Primitive,
    sampler::Sampler,
    spectrum::Spectrum,
};
use std::sync::Arc;

pub struct Scene {
    pub lights: Vec<Arc<Box<dyn Light>>>,
    pub infinite_lights: Vec<Arc<Box<dyn Light>>>,
    aggregate: Arc<Box<dyn Primitive>>,
    world_bound: Bounds3f,
}

impl Scene {
    pub fn new(aggregate: Arc<Box<dyn Primitive>>, lights: Vec<Arc<Box<dyn Light>>>) -> Self {
        let world_bound = aggregate.world_bound();
        let mut scene = Self {
            lights: lights.clone(),
            aggregate,
            world_bound,
            infinite_lights: vec![],
        };
        for light in lights {
            light.pre_process(&scene);
            if (light.flags() & LightFlags::infinite()).into() {
                scene.infinite_lights.push(light);
            }
        }
        scene
    }

    pub fn world_bound(&self) -> &Bounds3f {
        &self.world_bound
    }

    pub fn intersect(&self, ray: &Ray, isect: &mut SurfaceInteraction) -> bool {
        todo!()
    }

    pub fn intersect_p(&self, ray: &Ray) -> bool {
        todo!()
    }

    pub fn intersect_tr(
        &self,
        ray: &Ray,
        sampler: Arc<Box<dyn Sampler>>,
        isect: &mut SurfaceInteraction,
        transmittance: &mut Spectrum,
    ) {
        todo!()
    }
}
