use crate::Options;

pub mod geometry;
pub mod interaction;
pub mod medium;
pub mod primitive;
pub mod shape;
pub mod transform;

pub fn pbrt_init(opts: &Options) {}
pub fn pbrt_parse_file(f: String) {}
pub fn pbrt_cleanup() {}

use num::integer::Roots;
use num::traits::real::Real;
use num::Bounded;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait RealNum<T>:
    Add<Output = T>
    + Sub<Output = T>
    + Mul<Output = T>
    + Div<Output = T>
    + Neg<Output = T>
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
    fn zero() -> Self;
    fn min(self, t: Self) -> Self;
    fn max(self, t: Self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn delta() -> Self;
    fn not_one(self) -> bool;
}

macro_rules! implement_real_num {
    ($t:ident, $sqrt:ident, $zero:expr, $one:expr, $two:expr) => {
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
        }
    };
    ($t:ident, $sqrt:ident; $zero:expr, $one:expr, $two:expr, $delta:expr) => {
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
        }
    };
}

implement_real_num!(f32, f32; 0.0, 1.0, 2.0, 0.00001);
implement_real_num!(f64, f64; 0.0, 1.0, 2.0, 0.00001);
implement_real_num!(i8, Roots, 0, 1, 2);
implement_real_num!(i16, Roots, 0, 1, 2);
implement_real_num!(i32, Roots, 0, 1, 2);
implement_real_num!(i64, Roots, 0, 1, 2);
implement_real_num!(i128, Roots, 0, 1, 2);
implement_real_num!(isize, Roots, 0, 1, 2);

pub fn lerp<T: RealNum<T>>(t: T, v1: T, v2: T) -> T {
    (T::one() - t) * v1 + t * v2
}

#[macro_export]
macro_rules! inherit {
    ($child:ident, $base:ident, $field:ident) => {
        impl Deref for $child {
            type Target = $base;

            fn deref(&self) -> &Self::Target {
                &self.$field
            }
        }
    };
    ($child:ident, $base:ident, $field:ident, $bound:ident) => {
        impl<T: $bound> Deref for $child<T> {
            type Target = $base;

            fn deref(&self) -> &Self::Target {
                &self.$field
            }
        }
    };
}
