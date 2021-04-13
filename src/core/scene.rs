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

    pub fn intersect(&self, ray: &mut Ray, isect: &mut SurfaceInteraction) -> bool {
        self.aggregate.intersect(ray, isect)
    }

    pub fn intersect_p(&self, ray: &Ray) -> bool {
        self.aggregate.intersect_p(ray)
    }

    pub fn intersect_tr(
        &self,
        mut ray: Ray,
        sampler: SamplerDtRw,
        isect: &mut SurfaceInteraction,
        transmittance: &mut Spectrum,
    ) -> bool {
        *transmittance = Spectrum::new(1.0);
        loop {
            let hit_surface = self.intersect(&mut ray, isect);
            if let Some(medium) = &ray.medium {
                *transmittance *= medium.tr(&ray, sampler.clone());
            }
            if !hit_surface {
                return false;
            }
            if let Some(primitive) = &isect.primitive {
                if primitive.get_material().is_some() {
                    return true;
                }
            }
            ray = isect.spawn_ray(&ray.d);
        }
    }
}
