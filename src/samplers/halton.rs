use crate::core::lowdiscrepancy::compute_radical_inverse_permutations;
use crate::core::rng::RNG;
use crate::core::sampler::GlobalSampler;

lazy_static::lazy_static! {
    static ref RADICAL_INVERSE_PERMUTATIONS:Vec<u16> = {
        let mut rng = RNG::default();
        compute_radical_inverse_permutations(&mut rng)
    };
}

pub struct HaltonSampler {
    base: GlobalSampler,
}
