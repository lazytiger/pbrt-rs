use crate::core::pbrt::{next_float_down, next_float_up, Float, MACHINE_EPSILON};
use std::{
    mem::swap,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[derive(Copy, Clone, Default)]
pub struct EFloat {
    pub v: Float,
    low: Float,
    high: Float,
}

impl EFloat {
    pub fn new(v: Float, err: Float) -> EFloat {
        let mut ef = EFloat::default();
        if err == 0.0 {
            ef.low = v;
            ef.high = v;
        } else {
            ef.low = next_float_down(v - err);
            ef.high = next_float_up(v + err);
        }
        ef
    }

    pub fn get_absolute_error(&self) -> Float {
        next_float_up((self.high - self.v).abs().max(self.v - self.low).abs())
    }

    pub fn upper_bound(&self) -> Float {
        self.high
    }

    pub fn lower_bound(&self) -> Float {
        self.low
    }

    pub fn sqrt(&self) -> EFloat {
        let mut r = EFloat::default();
        r.v = self.v.sqrt();
        r.low = next_float_down(self.low.sqrt());
        r.high = next_float_up(self.high.sqrt());
        r
    }

    pub fn abs(&self) -> EFloat {
        if self.low >= 0.0 {
            *self
        } else if self.high <= 0.0 {
            -*self
        } else {
            let mut r = EFloat::default();
            r.v = self.v.abs();
            r.low = 0.0;
            r.high = self.high.max(-self.low);
            r
        }
    }

    /// The quadratic formula helps us solve any quadratic equation as ax<sup>2</sup> + bx +c = 0;
    /// So the roots x0 = (-b + (b<sup>2</sup> - 4ac)<sup>1/2</sup>)/2a, x1 = (-b - (b<sup>2</sup> - 4ac)<sup>1/2</sup>)/2a, x0 * x1 = c
    pub fn quadratic(a: EFloat, b: EFloat, c: EFloat) -> (bool, EFloat, EFloat) {
        let discrim = b.v as f64 * b.v as f64 - 4.0 * a.v as f64 * c.v as f64;
        if discrim < 0.0 {
            return (false, Default::default(), Default::default());
        }
        let root_discrim = discrim.sqrt();
        let float_root_discrm = EFloat::new(root_discrim as Float, MACHINE_EPSILON);
        let q = if b.v < 0.0 {
            (b - float_root_discrm) * -0.5
        } else {
            (b + float_root_discrm) * -0.5
        };
        let mut t0 = q / a;
        let mut t1 = c / q;
        if t0.v > t1.v {
            swap(&mut t0, &mut t1);
        }
        (true, t0, t1)
    }
}

impl Add for EFloat {
    type Output = EFloat;

    fn add(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        r.v = self.v + rhs.v;
        r.low = next_float_down(self.lower_bound() + rhs.lower_bound());
        r.high = next_float_up(self.upper_bound() + rhs.upper_bound());
        r
    }
}

impl Sub for EFloat {
    type Output = EFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        r.v = self.v - rhs.v;
        r.low = next_float_down(self.lower_bound() - rhs.upper_bound());
        r.high = next_float_up(self.upper_bound() - rhs.lower_bound());
        r
    }
}

impl Mul for EFloat {
    type Output = EFloat;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        r.v = self.v * rhs.v;
        let prod = [
            self.lower_bound() * rhs.lower_bound(),
            self.upper_bound() * rhs.lower_bound(),
            self.lower_bound() * rhs.upper_bound(),
            self.upper_bound() * rhs.upper_bound(),
        ];
        r.low = next_float_down(prod[0].min(prod[1]).min(prod[2]).min(prod[3]));
        r.high = next_float_up(prod[0].max(prod[1]).max(prod[2]).max(prod[3]));
        r
    }
}

impl Div for EFloat {
    type Output = EFloat;

    fn div(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        r.v = self.v / rhs.v;
        if rhs.low < 0.0 && rhs.high > 0.0 {
            r.low = -Float::INFINITY;
            r.high = Float::INFINITY;
        } else {
            let div = [
                self.lower_bound() / rhs.lower_bound(),
                self.upper_bound() / rhs.lower_bound(),
                self.lower_bound() / rhs.upper_bound(),
                self.upper_bound() / rhs.upper_bound(),
            ];
            r.low = next_float_down(div[0].min(div[1]).min(div[2]).min(div[3]));
            r.high = next_float_up(div[0].max(div[1]).max(div[2]).max(div[3]));
        }
        r
    }
}

impl Neg for EFloat {
    type Output = EFloat;

    fn neg(self) -> Self::Output {
        let mut r = Self::Output::default();
        r.v = -self.v;
        r.low = -self.high;
        r.high = -self.low;
        r
    }
}

impl PartialEq for EFloat {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl Add<Float> for EFloat {
    type Output = EFloat;

    fn add(self, rhs: f32) -> Self::Output {
        self + EFloat::new(rhs, 0.0)
    }
}

impl Sub<Float> for EFloat {
    type Output = EFloat;

    fn sub(self, rhs: f32) -> Self::Output {
        self - EFloat::new(rhs, 0.0)
    }
}

impl Mul<Float> for EFloat {
    type Output = EFloat;

    fn mul(self, rhs: f32) -> Self::Output {
        self * EFloat::new(rhs, 0.0)
    }
}

impl Div<Float> for EFloat {
    type Output = EFloat;

    fn div(self, rhs: f32) -> Self::Output {
        self / EFloat::new(rhs, 0.0)
    }
}
