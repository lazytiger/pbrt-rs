use crate::Options;

pub mod bssrdf;
pub mod camera;
pub mod efloat;
pub mod film;
pub mod filter;
pub mod geometry;
pub mod imageio;
pub mod integrator;
pub mod interaction;
pub mod interpolation;
pub mod light;
pub mod lightdistrib;
pub mod lowdiscrepancy;
pub mod material;
pub mod medium;
pub mod memory;
pub mod microfacet;
pub mod mipmap;
pub mod parallel;
pub mod pbrt;
pub mod primitive;
pub mod quaternion;
pub mod reflection;
pub mod rng;
pub mod sampler;
pub mod sampling;
pub mod scene;
pub mod shape;
pub mod sobolmatrices;
pub mod spectrum;
pub mod texture;
pub mod transform;

pub fn pbrt_init(_opts: &Options) {}
pub fn pbrt_parse_file(_f: String) {}
pub fn pbrt_cleanup() {}

use num::{integer::Roots, traits::real::Real, Bounded};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait RealNum<T>:
    Add<Output = T>
    + Sub<Output = T>
    + Mul<Output = T>
    + Div<Output = T>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialOrd
    + PartialEq
    + Bounded
    + Copy
    + Clone
{
    fn one() -> Self;
    fn two() -> Self;
    fn three() -> Self;
    fn zero() -> Self;
    fn min(self, t: Self) -> Self;
    fn max(self, t: Self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn delta() -> Self;
    fn not_one(self) -> bool;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn is_nan(self) -> bool;
    fn machine_epsilon() -> Self;
}

macro_rules! implement_real_num {
    ($t:ident, $sqrt:ident, $zero:expr, $one:expr, $two:expr, $three:expr) => {
        impl RealNum<$t> for $t {
            fn zero() -> Self {
                $zero
            }

            fn one() -> Self {
                $one
            }

            fn two() -> Self {
                $two
            }

            fn three() -> Self {
                $three
            }

            fn sqrt(self) -> Self {
                $sqrt::sqrt(&self)
            }

            fn min(self, t: Self) -> Self {
                std::cmp::min(self, t)
            }

            fn max(self, t: Self) -> Self {
                std::cmp::max(self, t)
            }

            fn abs(self) -> Self {
                $t::abs(self)
            }

            fn delta() -> Self {
                $zero
            }

            fn not_one(self) -> bool {
                self != $one
            }

            fn floor(self) -> Self {
                self
            }

            fn ceil(self) -> Self {
                self
            }

            fn is_nan(self) -> bool {
                false
            }

            fn machine_epsilon() -> Self {
                $zero
            }
        }
    };
    ($t:ident, $sqrt:ident; $zero:expr, $one:expr, $two:expr, $three:expr, $delta:expr, $me:expr) => {
        impl RealNum<$t> for $t {
            fn zero() -> Self {
                $zero
            }

            fn one() -> Self {
                $one
            }

            fn two() -> Self {
                $two
            }

            fn three() -> Self {
                $three
            }

            fn sqrt(self) -> Self {
                $sqrt::sqrt(self)
            }

            fn min(self, t: Self) -> Self {
                $t::min(self, t)
            }

            fn max(self, t: Self) -> Self {
                $t::min(self, t)
            }

            fn abs(self) -> Self {
                $t::abs(self)
            }

            fn delta() -> Self {
                $delta
            }

            fn not_one(self) -> bool {
                (self - $one).abs() > $delta
            }

            fn floor(self) -> Self {
                self.floor()
            }

            fn ceil(self) -> Self {
                self.ceil()
            }

            fn is_nan(self) -> bool {
                self.is_nan()
            }

            fn machine_epsilon() -> Self {
                $me
            }
        }
    };
}

implement_real_num!(f32, f32; 0.0, 1.0, 2.0, 3.0, 0.00001, f32::EPSILON * 0.5);
implement_real_num!(f64, f64; 0.0, 1.0, 2.0, 3.0, 0.00001, f64::EPSILON * 0.5);
implement_real_num!(i8, Roots, 0, 1, 2, 3);
implement_real_num!(i16, Roots, 0, 1, 2, 3);
implement_real_num!(i32, Roots, 0, 1, 2, 3);
implement_real_num!(i64, Roots, 0, 1, 2, 3);
implement_real_num!(i128, Roots, 0, 1, 2, 3);
implement_real_num!(isize, Roots, 0, 1, 2, 3);
implement_real_num!(u8, Roots, 0, 1, 2, 3);
implement_real_num!(u16, Roots, 0, 1, 2, 3);
implement_real_num!(u32, Roots, 0, 1, 2, 3);
implement_real_num!(u64, Roots, 0, 1, 2, 3);
implement_real_num!(u128, Roots, 0, 1, 2, 3);
implement_real_num!(usize, Roots, 0, 1, 2, 3);
