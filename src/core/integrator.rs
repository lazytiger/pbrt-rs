use crate::core::{arena::Arena, interaction::Interaction, sampler::Sampler, scene::Scene};
use std::sync::Arc;

pub trait Integrator {
    fn render(&self, scene: &Scene);
}

pub fn uniform_sample_all_lights(
    it: &Interaction,
    scene: &Scene,
    sampler: Arc<Box<dyn Sampler>>,
    n_light_samples: Vec<usize>,
    handle_media: bool,
) {
    todo!()
}
