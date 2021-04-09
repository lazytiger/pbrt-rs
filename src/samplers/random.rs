use crate::{
    core::{
        geometry::Point2f,
        rng::RNG,
        sampler::{BaseSampler, Sampler},
    },
    impl_base_sampler,
};
use std::{any::Any, sync::Arc};

pub struct RandomSampler {
    base: BaseSampler,
    rng: RNG,
}

impl RandomSampler {}

impl Sampler for RandomSampler {
    impl_base_sampler!();

    fn get_1d(&mut self) -> f32 {
        todo!()
    }

    fn get_2d(&mut self) -> Point2f {
        todo!()
    }

    fn clone(&self, seed: usize) -> Arc<Box<dyn Sampler>> {
        todo!()
    }

    fn get_index_for_sample(&mut self, sample_num: usize) -> i64 {
        unimplemented!("This method is not supported for RandomSampler")
    }

    fn sample_dimension(&self, index: i64, dimension: usize) -> f32 {
        unimplemented!("This method is not supported for RandomSampler")
    }
}
