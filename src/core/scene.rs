use crate::core::{
    geometry::{Bounds3f, Ray},
    interaction::SurfaceInteraction,
    light::{Light, LightDt, LightFlags},
    primitive::{Primitive, PrimitiveDt},
    sampler::{Sampler, SamplerDt, SamplerDtRw},
    spectrum::Spectrum,
};

pub struct Scene {
    pub lights: Vec<LightDt>,
    pub infinite_lights: Vec<LightDt>,
    aggregate: PrimitiveDt,
    world_bound: Bounds3f,
}

impl Scene {
    pub fn new(aggregate: PrimitiveDt, lights: Vec<LightDt>) -> Self {
        let world_bound = aggregate.world_bound();
        let mut scene = Self {
            lights: lights.clone(),
            aggregate,
            world_bound,
            infinite_lights: vec![],
        };
        for light in lights {
            light.pre_process(&scene);
            if (light.flags() & LightFlags::INFINITE).is_empty() {
                scene.infinite_lights.push(light);
            }
        }
        scene
    }

    pub fn world_bound(&self) -> &Bounds3f {
        &self.world_bound
    }

    pub fn intersect(&self, _ray: &Ray, _isect: &mut SurfaceInteraction) -> bool {
        todo!()
    }

    pub fn intersect_p(&self, _ray: &Ray) -> bool {
        todo!()
    }

    pub fn intersect_tr(
        &self,
        _ray: &Ray,
        _sampler: SamplerDtRw,
        _isect: &mut SurfaceInteraction,
        _transmittance: &mut Spectrum,
    ) -> bool {
        todo!()
    }
}
