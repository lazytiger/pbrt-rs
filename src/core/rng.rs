use crate::core::pbrt::{Float, ONE_MINUS_EPSILON};
use std::ops::Sub;

#[derive(Copy, Clone)]
pub struct RNG {
    state: usize,
    inc: usize,
}

const PCG32_DEFAULT_STATE: usize = 0x853c49e6748fea9b;
const PCG32_DEFAULT_STREAM: usize = 0xda3e39cb94b95bdb;
const PCG32_MULT: usize = 0x5851f42d4c957f2d;

impl RNG {
    pub fn new(sequence_index: usize) -> Self {
        let mut rng = RNG::default();
        rng.set_sequence(sequence_index);
        rng
    }

    pub fn set_sequence(&mut self, sequence_index: usize) {
        self.state = 0;
        self.inc = sequence_index << 1 | 1;
        self.uniform_u32();
        self.state += PCG32_DEFAULT_STATE;
        self.uniform_u32();
    }

    pub fn uniform_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state * PCG32_MULT + self.inc;
        let xor_shifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xor_shifted >> rot | xor_shifted << ((!rot + 1) & 31)
    }

    pub fn uniform_u32_u32(&mut self, b: u32) -> u32 {
        let threshold = (!b + 1) % b;
        loop {
            let r = self.uniform_u32();
            if r >= threshold {
                return r % b;
            }
        }
    }
    pub fn uniform_float(&mut self) -> Float {
        ONE_MINUS_EPSILON.min(self.uniform_u32() as Float * 2.3283064365386963e-10)
    }

    pub fn shuffle<T>(&mut self, t: &mut [T]) {
        for i in (1..t.len()).rev() {
            t.swap(i, self.uniform_u32_u32(i as u32 + 1) as usize)
        }
    }

    pub fn advance(&mut self, idelta: i64) {
        let (mut cur_mult, mut cur_plus, mut acc_mult, mut acc_plus, mut delta) =
            (PCG32_MULT, self.inc, 1, 0, idelta as u64);
        while delta > 0 {
            if delta & 1 != 0 {
                acc_mult *= cur_mult;
                acc_plus = acc_plus * cur_mult + cur_plus;
            }
            cur_plus = (cur_mult + 1) * cur_plus;
            cur_mult *= cur_mult;
            delta /= 2;
        }
        self.state = acc_mult * self.state + acc_plus;
    }
}

impl Sub for RNG {
    type Output = i64;

    fn sub(self, rhs: Self) -> Self::Output {
        let (mut cur_mult, mut cur_plus, mut cur_state, mut the_bit, mut distance) =
            (PCG32_MULT, self.inc, rhs.state, 1, 0);
        while self.state != cur_state {
            if self.state & the_bit != cur_state & the_bit {
                cur_state = cur_state * cur_mult + cur_plus;
                distance |= the_bit;
            }
            the_bit <<= 1;
            cur_plus = (cur_mult + 1) * cur_plus;
            cur_mult *= cur_mult;
        }
        distance as i64
    }
}

impl Default for RNG {
    fn default() -> Self {
        Self {
            state: PCG32_DEFAULT_STATE,
            inc: PCG32_DEFAULT_STREAM,
        }
    }
}
