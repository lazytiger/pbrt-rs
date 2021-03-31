use crate::{Float, Integer, Options, PI};

pub mod arena;
pub mod efloat;
pub mod geometry;
pub mod interaction;
pub mod light;
pub mod lowdiscrepancy;
pub mod material;
pub mod medium;
pub mod primitive;
pub mod quaternion;
pub mod sampling;
pub mod shape;
pub mod spectrum;
pub mod transform;

pub fn pbrt_init(opts: &Options) {}
pub fn pbrt_parse_file(f: String) {}
pub fn pbrt_cleanup() {}

use num::integer::Roots;
use num::traits::real::Real;
use num::Bounded;
use std::intrinsics::transmute;
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

pub fn lerp<T: RealNum<T>>(t: T, v1: T, v2: T) -> T {
    (T::one() - t) * v1 + t * v2
}

pub fn radians(deg: Float) -> Float {
    PI / 180.0 * deg
}

pub fn degrees(rad: Float) -> Float {
    180.0 / PI * rad
}

pub fn clamp<T: RealNum<T>>(val: T, low: T, high: T) -> T {
    if val < low {
        low
    } else if val > high {
        high
    } else {
        val
    }
}

/// (1+&epsilon;<sub>m</sub>)<sup>n</sup> can be tightly bounded to 1 + &theta;<sub>n</sub>,
/// where &theta;<sub>n</sub> is this gamma function.
pub fn gamma<T: RealNum<T>>(n: T) -> T {
    n * T::machine_epsilon() / (T::one() - n * T::machine_epsilon())
}

pub fn next_float_up(mut n: Float) -> Float {
    if n.is_infinite() && n > 0.0 {
        return n;
    }
    if n == -0.0 {
        n = 0.0;
    }
    // union may cause UB problem
    /*
    let mut u = FloatUnion { f: n };
    unsafe {
        if n >= 0.0 {
            u.u += 1;
        } else {
            u.u -= 1;
        }
        u.f
    }
     */
    unsafe {
        let u: Integer = transmute(n);
        if n >= 0.0 {
            transmute(u + 1)
        } else {
            transmute(u - 1)
        }
    }
}

pub fn next_float_down(mut n: Float) -> Float {
    if n.is_infinite() && n < 0.0 {
        return n;
    }
    if n == 0.0 {
        n = -0.0;
    }

    //union may cause UB problem
    /*
    let mut u = FloatUnion { f: n };
    unsafe {
        if n > 0.0 {
            u.u -= 1;
        } else {
            u.u += 1;
        }
        u.f
    }
     */
    unsafe {
        let u: Integer = transmute(n);
        if n > 0.0 {
            transmute(u - 1)
        } else {
            transmute(u + 1)
        }
    }
}

pub fn float_to_bits(f: Float) -> Integer {
    unsafe { transmute(f) }
}

pub fn find_interval<T: Fn(usize) -> bool>(size: usize, pred: T) -> usize {
    let mut first = 0;
    let mut len = size;
    while len > 0 {
        let half = len >> 1;
        let middle = first + half;
        if pred(middle) {
            first = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    clamp((first - 1) as isize, 0, (size - 2) as isize) as usize
}
