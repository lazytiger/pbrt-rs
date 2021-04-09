use crate::{
    core::{
        geometry::{Bounds2i, Point2f, Point2i},
        lowdiscrepancy::{
            compute_radical_inverse_permutations, inverse_radical_inverse, radical_inverse,
            scramble_radical_inverse, PRIME_SUMS, PRIME_TABLE_SIZE,
        },
        pbrt::Float,
        rng::RNG,
        sampler::{GlobalSampler, Sampler},
    },
    impl_global_sampler,
};
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

lazy_static::lazy_static! {
    static ref RADICAL_INVERSE_PERMUTATIONS:Vec<u64> = {
        let mut rng = RNG::default();
        compute_radical_inverse_permutations(&mut rng)
    };
}

#[derive(Clone)]
pub struct HaltonSampler {
    base: GlobalSampler,
    base_scales: Point2i,
    base_exponents: Point2i,
    sample_stride: usize,
    mult_inverse: [u64; 2],
    pixel_for_offset: Point2i,
    offset_for_current_pixel: i64,
    sample_at_pixel_center: bool,
}

const MAX_RESOLUTION: i32 = 128;

fn multiplicative_inverse(a: i64, n: i64) -> u64 {
    let (mut x, mut y) = (0, 0);
    extended_gcd(a as u64, n as u64, &mut x, &mut y);
    let result = x - (x / n) * n;
    if result < 0 {
        (result + n) as u64
    } else {
        result as u64
    }
}

fn extended_gcd(a: u64, b: u64, x: &mut i64, y: &mut i64) {
    if b == 0 {
        *x = 1;
        *y = 0;
        return;
    }
    let (d, mut xp, mut yp) = (a / b, 0, 0);
    extended_gcd(b, a % b, &mut xp, &mut yp);
    *x = yp;
    *y = xp - (d as i64 * yp);
}

impl HaltonSampler {
    pub fn new(
        sample_per_pixel: i64,
        sample_bounds: &Bounds2i,
        sample_at_pixel_center: bool,
    ) -> Self {
        let res = sample_bounds.max - sample_bounds.min;
        let mut base_scales = Point2i::default();
        let mut base_exponents = Point2i::default();
        for i in 0..2 {
            let base = if i == 0 { 2 } else { 3 };
            let (mut scale, mut exp) = (1, 0);
            while scale < std::cmp::min(MAX_RESOLUTION, res[i]) {
                scale *= base;
                exp += 1;
            }
            base_scales[i] = scale;
            base_exponents[i] = exp;
        }
        let sample_stride = (base_scales[0] * base_scales[1]) as usize;

        let mult_inverse = [
            multiplicative_inverse(base_scales[1] as i64, base_scales[0] as i64),
            multiplicative_inverse(base_scales[0] as i64, base_scales[1] as i64),
        ];
        Self {
            base: GlobalSampler::new(sample_per_pixel),
            sample_at_pixel_center,
            base_scales,
            base_exponents,
            sample_stride,
            mult_inverse,
            pixel_for_offset: Point2i::new(i32::MAX, i32::MAX),
            offset_for_current_pixel: 0,
        }
    }

    fn permutation_for_dimension(dim: usize) -> &'static [u64] {
        if dim > PRIME_TABLE_SIZE {
            log::error!(
                "HaltonSampler can only sample {} dimensions",
                PRIME_TABLE_SIZE
            );
        }
        &RADICAL_INVERSE_PERMUTATIONS.as_slice()[PRIME_SUMS[dim] as usize..]
    }
}

impl DerefMut for HaltonSampler {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl Deref for HaltonSampler {
    type Target = GlobalSampler;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl Sampler for HaltonSampler {
    impl_global_sampler!();

    fn clone(&self, _seed: usize) -> Arc<Box<dyn Sampler>> {
        Arc::new(Box::new(Clone::clone(self)))
    }

    fn get_index_for_sample(&mut self, sample_num: usize) -> i64 {
        if self.current_pixel != self.pixel_for_offset {
            self.offset_for_current_pixel = 0;
            if self.sample_stride > 1 {
                let pm = Point2i::new(
                    self.current_pixel[0] % MAX_RESOLUTION,
                    self.current_pixel[1] % MAX_RESOLUTION,
                );
                for i in 0..2 {
                    let dim_offset = if i == 0 {
                        inverse_radical_inverse(2, pm[i] as u64, self.base_exponents[i] as usize)
                    } else {
                        inverse_radical_inverse(3, pm[i] as u64, self.base_exponents[i] as usize)
                    };
                    self.offset_for_current_pixel += (dim_offset
                        * (self.sample_stride / self.base_scales[i] as usize) as u64
                        * self.mult_inverse[i])
                        as i64;
                }
                self.offset_for_current_pixel %= self.sample_stride as i64;
            }
            self.pixel_for_offset = self.current_pixel;
        }
        self.offset_for_current_pixel + (sample_num * self.sample_stride) as i64
    }

    fn sample_dimension(&self, index: i64, dim: usize) -> f32 {
        if self.sample_at_pixel_center && (dim == 0 || dim == 1) {
            return 0.5;
        }
        if dim == 0 {
            radical_inverse(dim as u64, index as u64 >> self.base_exponents[0])
        } else if dim == 1 {
            radical_inverse(dim as u64, index as u64 / self.base_scales[1] as u64)
        } else {
            scramble_radical_inverse(dim, index as u64, Self::permutation_for_dimension(dim))
        }
    }
}
