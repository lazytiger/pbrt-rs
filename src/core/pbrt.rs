use crate::core::RealNum;
use num::traits::Pow;
use std::any::Any;
use std::ops::{BitAnd, Not};
use std::raw::TraitObject;
cfg_if::cfg_if! {
   if #[cfg(feature = "float64")] {
        pub type Float = f64;
        pub type Integer = u64;
        pub const PI: f64 = std::f64::consts::PI;
        pub const SHADOWEPSILON:f64 = 0.0001;
        pub const ONE_MINUS_EPSILON:Float = 1.0 - EPSILON;
   } else {
        pub type Float = f32;
        pub type Integer = u32;
        pub const PI: f32 = std::f32::consts::PI;
        pub const INV_PI:f32 = 1.0 /PI;
        pub const INV_2_PI:f32 = INV_PI / 2.0;
        pub const INV_4_PI:f32 = INV_PI / 4.0;
        pub const PI_OVER_2:f32 = PI / 2.0;
        pub const PI_OVER_4:f32 = PI / 4.0;
        pub const SQRT_2:f32 = 1.41421356237309504880;
        pub const SHADOW_EPSILON:f32 = 0.0001;
        pub const EPSILON:f32 = f32::EPSILON;
        pub const MACHINE_EPSILON:f32 = 0.5 * EPSILON;
        pub const ONE_MINUS_EPSILON:Float = 1.0 - EPSILON;
   }
}

#[inline]
pub fn float_to_bits(f: Float) -> Integer {
    unsafe { std::mem::transmute(f) }
}

#[inline]
pub fn bits_to_float(ui: Integer) -> Float {
    unsafe { std::mem::transmute(ui) }
}

#[inline]
pub fn next_float_up(mut n: Float) -> Float {
    if n.is_infinite() && n > 0.0 {
        return n;
    }
    if n == -0.0 {
        n = 0.0;
    }
    unsafe {
        let u: Integer = std::mem::transmute(n);
        if n >= 0.0 {
            std::mem::transmute(u + 1)
        } else {
            std::mem::transmute(u - 1)
        }
    }
}

#[inline]
pub fn next_float_down(mut n: Float) -> Float {
    if n.is_infinite() && n < 0.0 {
        return n;
    }
    if n == 0.0 {
        n = -0.0;
    }

    unsafe {
        let u: Integer = std::mem::transmute(n);
        if n > 0.0 {
            std::mem::transmute(u - 1)
        } else {
            std::mem::transmute(u + 1)
        }
    }
}

#[inline]
pub fn any_equal(a1: &dyn Any, a2: &dyn Any) -> bool {
    let to1: TraitObject = unsafe { std::mem::transmute(a1) };
    let to2: TraitObject = unsafe { std::mem::transmute(a2) };
    to1.data == to2.data
}

/// (1+&epsilon;<sub>m</sub>)<sup>n</sup> can be tightly bounded to 1 + &theta;<sub>n</sub>,
/// where &theta;<sub>n</sub> is this gamma function.
#[inline]
pub fn gamma<T: RealNum<T>>(n: T) -> T {
    n * T::machine_epsilon() / (T::one() - n * T::machine_epsilon())
}

#[inline]
pub fn gamma_correct(value: Float) -> Float {
    if value <= 0.0031308 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
pub fn inverse_gamma_correct(value: Float) -> Float {
    if value <= 0.04045 {
        value * 1.0 / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
pub fn clamp<T: RealNum<T>>(val: T, low: T, high: T) -> T {
    if val < low {
        low
    } else if val > high {
        high
    } else {
        val
    }
}

#[inline]
pub fn mod_num<T: RealNum<T>>(a: T, b: T) -> T {
    let result = a - (a / b) * b;
    if result < T::zero() {
        result + b
    } else {
        result
    }
}

#[inline]
pub fn radians(deg: Float) -> Float {
    PI / 180.0 * deg
}

#[inline]
pub fn degrees(rad: Float) -> Float {
    180.0 / PI * rad
}

#[inline]
pub fn log_2(x: Float) -> Float {
    let inv_log2: Float = 1.442_695_040_888_963_387_004_650_940_071;
    x.ln() * inv_log2
}

#[inline]
pub fn log_2_int_u32(v: u32) -> i32 {
    31 - v.leading_zeros() as i32
}

#[inline]
pub fn log_2_int_i32(v: i32) -> i32 {
    log_2_int_u32(v as u32)
}

#[inline]
pub fn log_2_int_u64(v: u64) -> i64 {
    63 - v.leading_zeros() as i64
}

#[inline]
pub fn log_2_int_i64(v: i64) -> i64 {
    log_2_int_u64(v as u64)
}

#[inline]
pub fn is_power_of_2<T: RealNum<T> + BitAnd<Output = T> + Not<Output = T>>(v: T) -> bool {
    v != T::zero() && (!(v & (v - T::one()))) != T::zero()
}

#[inline]
pub fn round_up_pow2_i32(mut v: i32) -> i32 {
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v + 1
}

#[inline]
pub fn round_up_pow2_i64(mut v: i64) -> i64 {
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v + 1
}

#[inline]
pub fn count_trailing_zeros(v: u32) -> u32 {
    v.trailing_zeros()
}

#[inline]
pub fn quadratic(a: Float, b: Float, c: Float, t0: &mut Float, t1: &mut Float) -> bool {
    let discrim = b as f64 * b as f64 - 4.0 * a as f64 * c as f64;
    if discrim < 0.0 {
        return false;
    }
    let root_discrim = discrim.sqrt() as Float;

    let q = if b < 0.0 {
        -0.5 * (b - root_discrim)
    } else {
        -0.5 * (b + root_discrim)
    };
    *t0 = q / a;
    *t1 = c / q;

    if *t0 > *t1 {
        std::mem::swap(t0, t1)
    }
    true
}

#[inline]
pub fn lerp<T: RealNum<T>>(t: T, v1: T, v2: T) -> T {
    (T::one() - t) * v1 + t * v2
}

#[inline]
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

#[inline]
pub fn erf_inv(mut x: Float) -> Float {
    let (mut w, mut p) = (0.0, 0.0);
    x = clamp(x, -0.99999, 0.99999);
    w = -((1.0 - x) * (1.0 + x)).ln();
    if w < 5.0 {
        w = w - 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06 + p * w;
        p = -4.39150654e-06 + p * w;
        p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w;
        p = -0.00417768164 + p * w;
        p = 0.246640727 + p * w;
        p = 1.50140941 + p * w;
    } else {
        w = w.sqrt() - 3.0;
        p = -0.000200214257;
        p = 0.000100950558 + p * w;
        p = 0.00134934322 + p * w;
        p = -0.00367342844 + p * w;
        p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w;
        p = 0.00943887047 + p * w;
        p = 1.00167406 + p * w;
        p = 2.83297682 + p * w;
    }
    p * x
}

#[inline]
pub fn erf(mut x: Float) -> Float {
    let (a1, a2, a3, a4, a5, p) = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
        0.3275911,
    );

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}
