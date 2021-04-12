use crate::core::{
    arena::Arena,
    camera::{Camera, CameraDt},
    geometry::{Bounds2i, Point2f},
    interaction::Interaction,
    light::Light,
    sampler::{Sampler, SamplerDt},
    sampling::Distribution1D,
    scene::Scene,
    spectrum::Spectrum,
};
use std::{any::Any, sync::Arc};

pub trait Integrator {
    fn as_any(&self) -> &dyn Any;
    fn render(&self, scene: &Scene);
}

pub fn uniform_sample_all_lights(
    it: &Interaction,
    scene: &Scene,
    sampler: SamplerDt,
    n_light_samples: Vec<usize>,
    handle_media: bool,
) {
    todo!()
}

pub fn uniform_sample_one_light(
    it: &Interaction,
    scene: &Scene,
    sampler: SamplerDt,
    handle_media: bool,
    light_distrib: Option<&Distribution1D>,
) -> Spectrum {
    todo!()
}

pub fn estimate_direct(
    it: &Interaction,
    u_shading: &Point2f,
    light: &Light,
    u_light: Point2f,
    scene: &Scene,
    sampler: SamplerDt,
    handle_media: bool,
    specular: bool,
) -> Spectrum {
    todo!()
}

pub fn compute_light_power_distribution(scene: &Scene) -> Box<Distribution1D> {
    todo!()
}

pub struct SamplerIntegrator {
    pub camera: CameraDt,
    sampler: SamplerDt,
    pixel_bounds: Bounds2i,
}

impl SamplerIntegrator {
    pub fn new(camera: CameraDt, sampler: SamplerDt, pixel_bounds: Bounds2i) -> Self {
        Self {
            camera,
            sampler,
            pixel_bounds,
        }
    }
}
