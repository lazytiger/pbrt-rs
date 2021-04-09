use crate::core::geometry::Vector3f;
use crate::core::pbrt::clamp;
use crate::core::pbrt::Float;
use crate::core::transform::{Matrix4x4f, Transformf};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Copy, Clone, Default)]
pub struct Quaternion {
    pub v: Vector3f,
    pub w: Float,
}

impl Quaternion {
    pub fn new(v: Vector3f, w: Float) -> Quaternion {
        Self { v, w }
    }

    pub fn default() -> Quaternion {
        Self::new(Vector3f::new(0.0, 0.0, 0.0), 0.0)
    }

    pub fn normalize(&self) -> Quaternion {
        *self / (self.dot(self)).sqrt()
    }

    pub fn dot(&self, q: &Quaternion) -> Float {
        self.v * q.v + self.w * q.w
    }

    pub fn slerp(&self, t: Float, q: &Quaternion) -> Quaternion {
        let cos_theta = self.dot(q);
        if cos_theta > 0.9995 {
            (*self * (1.0 - t) + *q * t).normalize()
        } else {
            let theta = clamp(cos_theta, -1.0, 1.0).acos();
            let thetap = theta * t;
            let qperp = (*q - *self * cos_theta).normalize();
            *self * thetap.cos() + qperp * thetap.sin()
        }
    }
}

impl Add for Quaternion {
    type Output = Quaternion;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            v: self.v + rhs.v,
            w: self.w + rhs.w,
        }
    }
}

impl AddAssign for Quaternion {
    fn add_assign(&mut self, rhs: Self) {
        self.w += rhs.w;
        self.v += rhs.v;
    }
}

impl Sub for Quaternion {
    type Output = Quaternion;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            v: self.v - rhs.v,
            w: self.w - rhs.w,
        }
    }
}

impl SubAssign for Quaternion {
    fn sub_assign(&mut self, rhs: Self) {
        self.v -= rhs.v;
        self.w -= rhs.w;
    }
}

impl Mul<Float> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Float) -> Self::Output {
        Self::Output {
            v: self.v * rhs,
            w: self.w * rhs,
        }
    }
}

impl Mul for Quaternion {
    type Output = Float;

    fn mul(self, rhs: Self) -> Self::Output {
        self.dot(&rhs)
    }
}

impl MulAssign<Float> for Quaternion {
    fn mul_assign(&mut self, rhs: Float) {
        self.v *= rhs;
        self.w *= rhs;
    }
}

impl Div<Float> for Quaternion {
    type Output = Quaternion;

    fn div(self, rhs: Float) -> Self::Output {
        Self::Output {
            v: self.v / rhs,
            w: self.w / rhs,
        }
    }
}

impl DivAssign<Float> for Quaternion {
    fn div_assign(&mut self, rhs: f32) {
        self.v /= rhs;
        self.w /= rhs;
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;

    fn neg(self) -> Self::Output {
        Self::Output {
            w: -self.w,
            v: -self.v,
        }
    }
}

impl From<Matrix4x4f> for Quaternion {
    fn from(m: Matrix4x4f) -> Self {
        let mut q = Quaternion::default();
        let trace = m.m[0][0] + m.m[1][1] + m.m[2][2];
        if trace > 0.0 {
            let mut s = (trace + 1.0).sqrt();
            q.w = s / 2.0;
            s = 0.5 / s;
            q.v.x = (m.m[2][1] - m.m[1][2]) * s;
            q.v.y = (m.m[0][2] - m.m[2][0]) * s;
            q.v.z = (m.m[1][0] - m.m[0][1]) * s;
        } else {
            let nxt = [1, 2, 0];
            let mut qq = [0.0 as Float; 3];
            let mut i = 0;
            if m.m[1][1] > m.m[0][0] {
                i = 1;
            }
            if m.m[2][2] > m.m[i][i] {
                i = 2;
            }
            let j = nxt[i];
            let k = nxt[j];
            let mut s = (m.m[i][i] - (m.m[j][j] + m.m[k][k]) + 1.0).sqrt();
            qq[i] = s * 0.5;
            if s != 0.0 {
                s = 0.5 / s;
            }
            q.w = (m.m[k][j] - m.m[j][k]) * s;
            qq[j] = (m.m[j][i] + m.m[i][j]) * s;
            qq[k] = (m.m[k][i] + m.m[i][k]) * s;
            q.v.x = qq[0];
            q.v.y = qq[1];
            q.v.z = qq[2];
        }
        q
    }
}
