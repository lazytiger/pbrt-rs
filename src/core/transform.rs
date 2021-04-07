use crate::core::geometry::{Bounds3f, Normal3, Point3, Point3f, Ray, Union, Vector3, Vector3f};
use crate::core::interaction::SurfaceInteraction;
use crate::core::quaternion::Quaternion;
use crate::core::{clamp, lerp, radians, RealNum};
use crate::{Float, PI};
use num::zero;
use std::cmp::Ordering;
use std::mem::swap;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Eq, PartialEq, Default)]
pub struct Matrix4x4<T> {
    pub m: [[T; 4]; 4],
}

impl<T: RealNum<T>> Matrix4x4<T> {
    pub fn new() -> Self {
        Self {
            m: [[T::zero(); 4]; 4],
        }
    }

    pub fn inverse(&self) -> Self {
        let mut indexc = [0usize; 4];
        let mut indexr = [0usize; 4];
        let mut ipiv = [0usize; 4];
        let mut minv = [[T::zero(); 4]; 4];
        for i in 0..4 {
            let mut irow = 0;
            let mut icol = 0;
            let mut big = T::zero();
            for j in 0..4 {
                if ipiv[j] != 1 {
                    for k in 0..4 {
                        if ipiv[k] == 0 {
                            if minv[j][k].abs() >= big {
                                big = minv[j][k].abs();
                                irow = j;
                                icol = k;
                            }
                        } else if ipiv[k] > 1 {
                            panic!("singular matrix in MatrixInvert");
                        }
                    }
                }
            }
            ipiv[icol] += 1;
            if irow != icol {
                for k in 0..4 {
                    let tmp = minv[irow][k];
                    minv[irow][k] = minv[icol][k];
                    minv[icol][k] = tmp;
                }
            }

            indexr[i] = irow;
            indexc[i] = icol;
            if minv[icol][icol] == T::zero() {
                panic!("Singular matrix in MatrixInvert");
            }

            let pivinv = T::one() / minv[icol][icol];
            minv[icol][icol] = T::one();
            for j in 0..4 {
                minv[icol][j] *= pivinv;
            }

            for j in 0..4 {
                if j != icol {
                    let save = minv[j][icol];
                    minv[j][icol] = T::zero();
                    for k in 0..4 {
                        minv[j][k] -= minv[icol][k] * save;
                    }
                }
            }
        }

        for j in (0..4).rev() {
            if indexr[j] != indexc[j] {
                for k in 0..4 {
                    let tmp = minv[k][indexr[j]];
                    minv[k][indexr[j]] = minv[k][indexc[j]];
                    minv[k][indexc[j]] = tmp;
                }
            }
        }

        minv.into()
    }

    pub fn transpose(&self) -> Matrix4x4<T> {
        (
            self.m[0][0],
            self.m[1][0],
            self.m[2][0],
            self.m[3][0],
            self.m[0][1],
            self.m[1][1],
            self.m[2][1],
            self.m[3][1],
            self.m[0][2],
            self.m[1][2],
            self.m[2][2],
            self.m[3][2],
            self.m[0][3],
            self.m[1][3],
            self.m[2][3],
            self.m[3][3],
        )
            .into()
    }
}

impl<T: RealNum<T>> From<[[T; 4]; 4]> for Matrix4x4<T> {
    fn from(m: [[T; 4]; 4]) -> Self {
        Self { m }
    }
}

impl<T: RealNum<T>> From<(T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T)> for Matrix4x4<T> {
    fn from(m: (T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T)) -> Self {
        let mut matrix = Self::new();
        matrix.m[0][0] = m.0;
        matrix.m[0][1] = m.1;
        matrix.m[0][2] = m.2;
        matrix.m[0][3] = m.3;
        matrix.m[1][0] = m.4;
        matrix.m[1][1] = m.5;
        matrix.m[1][2] = m.6;
        matrix.m[1][3] = m.7;
        matrix.m[2][0] = m.8;
        matrix.m[2][1] = m.9;
        matrix.m[2][2] = m.10;
        matrix.m[2][3] = m.11;
        matrix.m[3][0] = m.12;
        matrix.m[3][1] = m.13;
        matrix.m[3][2] = m.14;
        matrix.m[3][3] = m.15;
        matrix
    }
}

impl<T: RealNum<T>> Mul for &Matrix4x4<T> {
    type Output = Matrix4x4<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut r = Matrix4x4::new();
        for i in 0..4 {
            for j in 0..4 {
                r.m[i][j] = self.m[i][0] * rhs.m[0][j]
                    + self.m[i][1] * rhs.m[1][j]
                    + self.m[i][2] * rhs.m[2][j]
                    + self.m[i][3] * rhs.m[3][j];
            }
        }
        r
    }
}

pub type Matrix4x4f = Matrix4x4<Float>;

impl<T: RealNum<T>> Mul for Matrix4x4<T> {
    type Output = Matrix4x4<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

#[derive(PartialEq, Clone, Copy, Default)]
pub struct Transform<T> {
    pub m: Matrix4x4<T>,
    pub m_inv: Matrix4x4<T>,
}

impl<T: RealNum<T>> Transform<T> {
    pub fn new() -> Transform<T> {
        Transform {
            m: Matrix4x4::new(),
            m_inv: Matrix4x4::new(),
        }
    }

    pub fn inverse(&self) -> Transform<T> {
        Self {
            m: self.m_inv,
            m_inv: self.m,
        }
    }

    pub fn transpose(&self) -> Transform<T> {
        Self {
            m: self.m.transpose(),
            m_inv: self.m_inv.transpose(),
        }
    }

    pub fn is_identify(&self) -> bool {
        let m = &self.m.m;
        let one = T::one();
        let zero = T::zero();
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    if m[i][j] != one {
                        return false;
                    }
                } else {
                    if m[i][j] != zero {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn get_matrix(&self) -> &Matrix4x4<T> {
        &self.m
    }

    pub fn get_inverse_matrix(&self) -> &Matrix4x4<T> {
        &self.m_inv
    }

    pub fn has_scale(&self) -> bool {
        let one = T::one();
        let zero = T::zero();
        let x = Vector3::new(one, zero, zero);
        let y = Vector3::new(zero, one, zero);
        let z = Vector3::new(zero, zero, one);
        let la2 = (self * Vector3Ref(&x)).length_squared();
        let lb2 = (self * Vector3Ref(&y)).length_squared();
        let lc2 = (self * Vector3Ref(&z)).length_squared();
        la2.not_one() || lb2.not_one() || lc2.not_one()
    }
}

impl Matrix4x4f {
    pub fn decompose(&self) -> (Vector3f, Quaternion, Matrix4x4f) {
        let t = Vector3f::new(self.m[0][3], self.m[1][3], self.m[2][3]);

        let mut m = *self;
        for i in 0..3 {
            m.m[i][3] = 0.0;
            m.m[3][i] = 0.0;
        }
        m.m[3][3] = 1.0;

        let mut norm = 0.0;
        let mut count = 0;
        let mut r = m;
        while count < 100 && norm > 0.0001 {
            let mut rnext: Matrix4x4f = Default::default();
            let rit = r.transpose().inverse();
            for i in 0..4 {
                for j in 0..4 {
                    rnext.m[i][j] = 0.5 * (r.m[i][j] + rit.m[i][j]);
                }
            }

            norm = 0.0;
            for i in 0..3 {
                let n = (r.m[i][0] - rnext.m[i][0]).abs()
                    + (r.m[i][1] - rnext.m[i][1]).abs()
                    + (r.m[i][2] - rnext.m[i][2]).abs();
                norm = norm.max(n);
            }
            r = rnext;
            count += 1;
        }

        let s = r.inverse() * m;
        let r: Quaternion = r.into();
        (t, r, s)
    }
}

impl<T: RealNum<T>> PartialOrd for Transform<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let l = &self.m.m;
        let r = &other.m.m;
        for i in 0..4 {
            for j in 0..4 {
                if l[i][j] < r[i][j] {
                    return Some(Ordering::Less);
                }
                if l[i][j] > r[i][j] {
                    return Some(Ordering::Greater);
                }
            }
        }
        Some(Ordering::Equal)
    }
}

impl<T: RealNum<T>> From<Matrix4x4<T>> for Transform<T> {
    fn from(m: Matrix4x4<T>) -> Self {
        Transform {
            m_inv: m.inverse(),
            m,
        }
    }
}

impl<T: RealNum<T>> From<(Matrix4x4<T>, Matrix4x4<T>)> for Transform<T> {
    fn from(mm: (Matrix4x4<T>, Matrix4x4<T>)) -> Self {
        Self {
            m: mm.0,
            m_inv: mm.1,
        }
    }
}

impl<T: RealNum<T>> From<[[T; 4]; 4]> for Transform<T> {
    fn from(m: [[T; 4]; 4]) -> Self {
        let m: Matrix4x4<T> = m.into();
        let m_inv = m.inverse();
        Self { m, m_inv }
    }
}

pub struct Point3Ref<'a, T>(pub &'a Point3<T>);
pub struct Normal3Ref<'a, T>(pub &'a Normal3<T>);
pub struct Vector3Ref<'a, T>(pub &'a Vector3<T>);

impl<'a, T: RealNum<T>> Mul<Point3Ref<'a, T>> for &Transform<T> {
    type Output = Point3<T>;

    fn mul(self, rhs: Point3Ref<'a, T>) -> Self::Output {
        let x = rhs.0.x;
        let y = rhs.0.y;
        let z = rhs.0.z;
        let m = &self.m.m;
        let xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
        let yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
        let zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
        let wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
        if wp == T::one() {
            Point3::new(xp, yp, zp)
        } else {
            Point3::new(xp, yp, zp) / wp
        }
    }
}

impl<'a, T: RealNum<T>> Mul<Vector3Ref<'a, T>> for &Transform<T> {
    type Output = Vector3<T>;

    fn mul(self, rhs: Vector3Ref<'a, T>) -> Self::Output {
        let x = rhs.0.x;
        let y = rhs.0.y;
        let z = rhs.0.z;
        let m = &self.m.m;

        Vector3::new(
            m[0][0] * x + m[0][1] * y + m[0][2] * z,
            m[1][0] * x + m[1][1] * y + m[1][2] * z,
            m[2][0] * x + m[2][1] * y + m[2][2] * z,
        )
    }
}

impl<'a, T: RealNum<T>> Mul<Normal3Ref<'a, T>> for &Transform<T> {
    type Output = Normal3<T>;

    fn mul(self, rhs: Normal3Ref<'a, T>) -> Self::Output {
        let x = rhs.0.x;
        let y = rhs.0.y;
        let z = rhs.0.z;
        let m = &self.m_inv.m;

        Vector3::new(
            m[0][0] * x + m[0][1] * y + m[0][2] * z,
            m[1][0] * x + m[1][1] * y + m[1][2] * z,
            m[2][0] * x + m[2][1] * y + m[2][2] * z,
        )
    }
}

pub type Transformf = Transform<Float>;

impl Transformf {
    pub fn translate(delta: &Vector3f) -> Self {
        let m: Matrix4x4f = [
            [1.0, 0.0, 0.0, delta.x],
            [0.0, 1.0, 0.0, delta.y],
            [0.0, 0.0, 1.0, delta.z],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into();
        let m_inv: Matrix4x4f = [
            [1.0, 0.0, 0.0, -delta.x],
            [0.0, 1.0, 0.0, -delta.y],
            [0.0, 0.0, 1.0, -delta.z],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into();
        Self { m, m_inv }
    }

    pub fn scale(x: Float, y: Float, z: Float) -> Self {
        let m: Matrix4x4f = [
            [0.0, 0.0, 0.0, x],
            [0.0, 0.0, 0.0, y],
            [0.0, 0.0, 0.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into();
        let m_inv: Matrix4x4f = [
            [1.0 / x, 0.0, 0.0, 0.0],
            [0.0, 1.0 / y, 0.0, 0.0],
            [0.0, 0.0, 1.0 / z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into();
        Self { m, m_inv }
    }

    pub fn rotate_x(theta: Float) -> Self {
        let theta = radians(theta);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let m: Matrix4x4f = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta, 0.0],
            [0.0, sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into();
        m.into()
    }

    pub fn rotate_y(theta: Float) -> Self {
        let theta = radians(theta);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let m: Matrix4x4f = [
            [cos_theta, 0.0, sin_theta, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_theta, 0.0, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into();
        m.into()
    }

    pub fn rotate_z(theta: Float) -> Self {
        let theta = radians(theta);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let m: Matrix4x4f = [
            [cos_theta, -sin_theta, 0.0, 0.0],
            [sin_theta, cos_theta, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into();
        m.into()
    }

    pub fn rotate(theta: Float, axis: &Vector3f) -> Self {
        let a = axis.normalize();
        let theta = radians(theta);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let mut m = Matrix4x4f::new();
        m.m[0][0] = a.x * a.y + (1.0 - a.x * a.x) * cos_theta;
        m.m[0][1] = a.x * a.y * (1.0 - cos_theta) - a.z * sin_theta;
        m.m[0][2] = a.x * a.z * (1.0 - cos_theta) + a.y * sin_theta;
        m.m[0][3] = 0.0;

        m.m[1][0] = a.x * a.y * (1.0 - cos_theta) + a.z * sin_theta;
        m.m[1][1] = a.y * a.y + (1.0 - a.y * a.y) * cos_theta;
        m.m[1][2] = a.y * a.z * (1.0 - cos_theta) - a.x * sin_theta;
        m.m[1][3] = 0.0;

        m.m[2][0] = a.x * a.z * (1.0 - cos_theta) - a.y * sin_theta;
        m.m[2][1] = a.y * a.z * (1.0 - cos_theta) + a.x * sin_theta;
        m.m[2][2] = a.z * a.z + (1.0 - a.z * a.z) * cos_theta;
        m.m[2][3] = 0.0;

        m.into()
    }

    pub fn look_at(pos: &Point3f, look: &Point3f, up: &Vector3f) -> Self {
        let mut camera_to_world = Matrix4x4f::new();

        camera_to_world.m[0][3] = pos.x;
        camera_to_world.m[1][3] = pos.y;
        camera_to_world.m[2][3] = pos.z;
        camera_to_world.m[3][3] = 1.0;

        let dir = (*look - *pos).normalize();
        let right = up.normalize().cross(&dir);
        if right.length() == 0.0 {
            panic!("invalid parameter");
        }

        let right = right.normalize();
        let up = dir.cross(&right);

        camera_to_world.m[0][0] = right.x;
        camera_to_world.m[1][0] = right.y;
        camera_to_world.m[2][0] = right.z;
        camera_to_world.m[3][0] = 0.;
        camera_to_world.m[0][1] = up.x;
        camera_to_world.m[1][1] = up.y;
        camera_to_world.m[2][1] = up.z;
        camera_to_world.m[3][1] = 0.;
        camera_to_world.m[0][2] = dir.x;
        camera_to_world.m[1][2] = dir.y;
        camera_to_world.m[2][2] = dir.z;
        camera_to_world.m[3][2] = 0.;

        (camera_to_world.inverse(), camera_to_world).into()
    }

    pub fn swap_handedness(&self) -> bool {
        let m = &self.m.m;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        det < 0.0
    }

    pub fn orthographic(near: Float, far: Float) -> Self {
        Self::scale(1.0, 1.0, 1.0 / (far - near)) * Self::translate(&Vector3f::new(0.0, 0.0, -near))
    }

    pub fn perspective(fov: Float, n: Float, f: Float) -> Self {
        let persp: Matrix4x4f = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, f / (f - n), -f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ]
        .into();
        let inv_tan_ang = 1.0 / (radians(fov) / 2.0).tan();
        let trans: Transformf = persp.into();
        Self::scale(inv_tan_ang, inv_tan_ang, 1.0) * trans
    }
}

impl Mul<&Bounds3f> for &Transformf {
    type Output = Bounds3f;

    fn mul(self, rhs: &Bounds3f) -> Self::Output {
        let p = Point3f::new(rhs.min.x, rhs.min.y, rhs.min.z);
        let p = self * Point3Ref(&p);
        let ret: Bounds3f = p.into();

        let p = Point3f::new(rhs.max.x, rhs.min.y, rhs.min.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union(&p);

        let p = Point3f::new(rhs.min.x, rhs.max.y, rhs.min.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union(&p);

        let p = Point3f::new(rhs.min.x, rhs.min.y, rhs.max.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union(&p);

        let p = Point3f::new(rhs.min.x, rhs.max.y, rhs.max.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union(&p);

        let p = Point3f::new(rhs.max.x, rhs.max.y, rhs.min.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union(&p);

        let p = Point3f::new(rhs.max.x, rhs.min.y, rhs.max.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union(&p);

        let p = Point3f::new(rhs.max.x, rhs.max.y, rhs.max.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union(&p);

        ret
    }
}

impl Mul for Transformf {
    type Output = Transformf;

    fn mul(self, rhs: Self) -> Self::Output {
        Transformf {
            m: self.m * rhs.m,
            m_inv: rhs.m_inv * self.m,
        }
    }
}

impl<'a> Mul<&SurfaceInteraction<'a>> for &Transformf {
    type Output = SurfaceInteraction<'a>;

    fn mul(self, rhs: &SurfaceInteraction<'a>) -> Self::Output {
        //TODO
        Default::default()
    }
}

impl From<Quaternion> for Transformf {
    fn from(q: Quaternion) -> Self {
        let xx = q.v.x * q.v.x;
        let yy = q.v.y * q.v.y;
        let zz = q.v.z * q.v.z;
        let xy = q.v.x * q.v.y;
        let xz = q.v.x * q.v.z;
        let yz = q.v.y * q.v.z;
        let wx = q.v.x * q.w;
        let wy = q.v.y * q.w;
        let wz = q.v.z * q.w;

        let mut m = Matrix4x4f::new();
        m.m[0][0] = 1.0 - 2.0 * (yy + zz);
        m.m[0][1] = 2.0 * (xy + wz);
        m.m[0][2] = 2.0 * (xz - wy);
        m.m[1][0] = 2.0 * (xy - wz);
        m.m[1][1] = 1.0 - 2.0 * (xx + zz);
        m.m[1][2] = 2.0 * (yz + wx);
        m.m[2][0] = 2.0 * (xz + wy);
        m.m[2][1] = 2.0 * (yz - wx);
        m.m[2][2] = 1.0 - 2.0 * (xx + yy);

        (m.transpose(), m).into()
    }
}

#[derive(Clone, Copy)]
pub struct Interval {
    low: Float,
    high: Float,
}

impl Interval {
    pub fn new(v0: Float, v1: Float) -> Self {
        Self {
            low: v0.min(v1),
            high: v0.max(v1),
        }
    }

    pub fn sin(&self) -> Interval {
        let mut sin_low = self.low.sin();
        let mut sin_high = self.high.sin();
        if sin_low > sin_high {
            swap(&mut sin_low, &mut sin_high);
        }
        if self.low < PI / 2.0 && self.high > PI / 2.0 {
            sin_high = 1.0;
        }
        if self.low < 3.0 / 2.0 * PI && self.high > 3.0 / 2.0 * PI {
            sin_low = -1.0;
        }
        Self {
            low: sin_low,
            high: sin_high,
        }
    }

    pub fn cos(&self) -> Interval {
        let mut cos_low = self.low.cos();
        let mut cos_high = self.high.cos();
        if cos_low > cos_high {
            swap(&mut cos_high, &mut cos_low);
        }
        if self.low > PI && self.high > PI {
            cos_low = -1.0;
        }
        Interval {
            low: cos_low,
            high: cos_high,
        }
    }
}

impl Add for &Interval {
    type Output = Interval;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            low: self.low + rhs.low,
            high: self.high + rhs.high,
        }
    }
}

impl Add for Interval {
    type Output = Interval;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Sub for &Interval {
    type Output = Interval;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            low: self.low - rhs.low,
            high: self.high - rhs.high,
        }
    }
}

impl Sub for Interval {
    type Output = Interval;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Mul for &Interval {
    type Output = Interval;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::Output {
            low: (self.low * rhs.low)
                .min(self.low * rhs.high)
                .min(self.high * rhs.high)
                .min(self.high * rhs.low),
            high: (self.low * rhs.low)
                .max(self.low * rhs.high)
                .max(self.high * rhs.high)
                .max(self.high * rhs.low),
        }
    }
}

impl Mul for Interval {
    type Output = Interval;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

pub fn interval_find_zeros(
    c1: Float,
    c2: Float,
    c3: Float,
    c4: Float,
    c5: Float,
    theta: Float,
    interval: Interval,
    zeros: &mut Vec<Float>,
    depth: i32,
) {
    let cc1 = Interval::new(c1, c1);
    let cc2 = Interval::new(c2, c2);
    let cc3 = Interval::new(c3, c3);
    let cc4 = Interval::new(c4, c4);
    let cc5 = Interval::new(c5, c5);
    let theta2 = Interval::new(2.0 * theta, 2.0 * theta);
    let range = cc1
        + (cc2 + (cc3 * interval) * (theta2 * interval).cos())
        + (cc4 + (cc5 * interval) * (theta2 * interval).sin());
    if range.low > 0.0 || range.high < 0.0 || range.low == range.high {
        return;
    }
    if depth > 0 {
        let mid = (interval.high + interval.low) * 0.5;
        interval_find_zeros(
            c1,
            c2,
            c3,
            c4,
            c5,
            theta,
            Interval::new(interval.low, mid),
            zeros,
            depth - 1,
        );
        interval_find_zeros(
            c1,
            c2,
            c3,
            c4,
            c5,
            theta,
            Interval::new(mid, interval.high),
            zeros,
            depth - 1,
        );
    } else {
        let mut t_newton = (interval.low + interval.high) * 0.5;
        for i in 0..4 {
            let f_newton = c1
                + (c2 + c3 * t_newton) * (2.0 * theta * t_newton).cos()
                + (c4 + c5 * t_newton) * (2.0 * theta * t_newton).sin();
            let f_prime_newton = (c3 + 2.0 * (c4 + c5 * t_newton) * theta)
                * (2.0 * t_newton * theta).cos()
                + (c5 - 2.0 * (c2 + c3 * t_newton) * theta) * (2.0 * t_newton * theta).sin();
            if f_newton == 0.0 || f_prime_newton == 0.0 {
                break;
            }
            t_newton = t_newton - f_newton / f_prime_newton;
        }
        if t_newton >= interval.low - 1e-3 && t_newton < interval.high + 1e-3 {
            zeros.push(t_newton);
        }
    }
}

#[derive(Default, Clone, Copy)]
struct DerivativeTerm {
    kc: Float,
    kx: Float,
    ky: Float,
    kz: Float,
}

impl DerivativeTerm {
    pub fn new(kc: Float, kx: Float, ky: Float, kz: Float) -> Self {
        Self { kc, kx, ky, kz }
    }

    pub fn eval(&self, p: &Point3f) -> Float {
        self.kc + self.kx * p.x + self.ky * p.y + self.kz * p.z
    }
}

pub struct AnimatedTransform {
    pub(crate) start_transform: Transformf,
    pub(crate) end_transform: Transformf,
    pub(crate) start_time: Float,
    pub(crate) end_time: Float,
    pub(crate) actually_animated: bool,
    t: [Vector3f; 2],
    r: [Quaternion; 2],
    s: [Matrix4x4f; 2],
    has_rotation: bool,
    c1: [DerivativeTerm; 3],
    c2: [DerivativeTerm; 3],
    c3: [DerivativeTerm; 3],
    c4: [DerivativeTerm; 3],
    c5: [DerivativeTerm; 3],
}

impl AnimatedTransform {
    pub fn new(
        start_transform: Transformf,
        start_time: Float,
        end_transform: Transformf,
        end_time: Float,
    ) -> Self {
        let actually_animated = start_transform != end_transform;
        let mut at = Self {
            start_transform,
            end_transform,
            start_time,
            end_time,
            actually_animated,
            t: [Default::default(); 2],
            s: [Default::default(); 2],
            r: [Default::default(); 2],
            has_rotation: false,
            c1: [Default::default(); 3],
            c2: [Default::default(); 3],
            c3: [Default::default(); 3],
            c4: [Default::default(); 3],
            c5: [Default::default(); 3],
        };

        if !actually_animated {
            return at;
        }

        let (t, r, s) = at.start_transform.m.decompose();
        at.t[0] = t;
        at.r[0] = r;
        at.s[0] = s;

        let (t, r, s) = at.end_transform.m.decompose();
        at.t[1] = t;
        at.r[1] = r;
        at.s[1] = s;

        if at.r[0] * at.r[1] < 0.0 {
            at.r[1] = -at.r[1];
        }
        at.has_rotation = at.r[0] * at.r[1] < 0.9995;
        if at.has_rotation {
            let cos_theta = at.r[0] * at.r[1];
            let theta = clamp(cos_theta, -1.0, 1.0).acos();
            let qperp = (at.r[1] - at.r[0] * cos_theta).normalize();

            let t0x = at.t[0].x;
            let t0y = at.t[0].y;
            let t0z = at.t[0].z;
            let t1x = at.t[1].x;
            let t1y = at.t[1].y;
            let t1z = at.t[1].z;
            let q0x = at.r[0].v.x;
            let q0y = at.r[0].v.y;
            let q0z = at.r[0].v.z;
            let q0w = at.r[0].w;
            let qperpx = qperp.v.x;
            let qperpy = qperp.v.y;
            let qperpz = qperp.v.z;
            let qperpw = qperp.w;
            let s000 = at.s[0].m[0][0];
            let s001 = at.s[0].m[0][1];
            let s002 = at.s[0].m[0][2];
            let s010 = at.s[0].m[1][0];
            let s011 = at.s[0].m[1][1];
            let s012 = at.s[0].m[1][2];
            let s020 = at.s[0].m[2][0];
            let s021 = at.s[0].m[2][1];
            let s022 = at.s[0].m[2][2];
            let s100 = at.s[1].m[0][0];
            let s101 = at.s[1].m[0][1];
            let s102 = at.s[1].m[0][2];
            let s110 = at.s[1].m[1][0];
            let s111 = at.s[1].m[1][1];
            let s112 = at.s[1].m[1][2];
            let s120 = at.s[1].m[2][0];
            let s121 = at.s[1].m[2][1];
            let s122 = at.s[1].m[2][2];

            at.c1[0] = DerivativeTerm::new(
                -t0x + t1x,
                (-1.0 + q0y * q0y + q0z * q0z + qperpy * qperpy + qperpz * qperpz) * s000
                    + q0w * q0z * s010
                    - qperpx * qperpy * s010
                    + qperpw * qperpz * s010
                    - q0w * q0y * s020
                    - qperpw * qperpy * s020
                    - qperpx * qperpz * s020
                    + s100
                    - q0y * q0y * s100
                    - q0z * q0z * s100
                    - qperpy * qperpy * s100
                    - qperpz * qperpz * s100
                    - q0w * q0z * s110
                    + qperpx * qperpy * s110
                    - qperpw * qperpz * s110
                    + q0w * q0y * s120
                    + qperpw * qperpy * s120
                    + qperpx * qperpz * s120
                    + q0x * (-(q0y * s010) - q0z * s020 + q0y * s110 + q0z * s120),
                (-1.0 + q0y * q0y + q0z * q0z + qperpy * qperpy + qperpz * qperpz) * s001
                    + q0w * q0z * s011
                    - qperpx * qperpy * s011
                    + qperpw * qperpz * s011
                    - q0w * q0y * s021
                    - qperpw * qperpy * s021
                    - qperpx * qperpz * s021
                    + s101
                    - q0y * q0y * s101
                    - q0z * q0z * s101
                    - qperpy * qperpy * s101
                    - qperpz * qperpz * s101
                    - q0w * q0z * s111
                    + qperpx * qperpy * s111
                    - qperpw * qperpz * s111
                    + q0w * q0y * s121
                    + qperpw * qperpy * s121
                    + qperpx * qperpz * s121
                    + q0x * (-(q0y * s011) - q0z * s021 + q0y * s111 + q0z * s121),
                (-1.0 + q0y * q0y + q0z * q0z + qperpy * qperpy + qperpz * qperpz) * s002
                    + q0w * q0z * s012
                    - qperpx * qperpy * s012
                    + qperpw * qperpz * s012
                    - q0w * q0y * s022
                    - qperpw * qperpy * s022
                    - qperpx * qperpz * s022
                    + s102
                    - q0y * q0y * s102
                    - q0z * q0z * s102
                    - qperpy * qperpy * s102
                    - qperpz * qperpz * s102
                    - q0w * q0z * s112
                    + qperpx * qperpy * s112
                    - qperpw * qperpz * s112
                    + q0w * q0y * s122
                    + qperpw * qperpy * s122
                    + qperpx * qperpz * s122
                    + q0x * (-(q0y * s012) - q0z * s022 + q0y * s112 + q0z * s122),
            );

            at.c2[0] = DerivativeTerm::new(
                0.,
                -(qperpy * qperpy * s000) - qperpz * qperpz * s000 + qperpx * qperpy * s010
                    - qperpw * qperpz * s010
                    + qperpw * qperpy * s020
                    + qperpx * qperpz * s020
                    + q0y * q0y * (s000 - s100)
                    + q0z * q0z * (s000 - s100)
                    + qperpy * qperpy * s100
                    + qperpz * qperpz * s100
                    - qperpx * qperpy * s110
                    + qperpw * qperpz * s110
                    - qperpw * qperpy * s120
                    - qperpx * qperpz * s120
                    + 2.0 * q0x * qperpy * s010 * theta
                    - 2.0 * q0w * qperpz * s010 * theta
                    + 2.0 * q0w * qperpy * s020 * theta
                    + 2.0 * q0x * qperpz * s020 * theta
                    + q0y
                        * (q0x * (-s010 + s110)
                            + q0w * (-s020 + s120)
                            + 2.0 * (-2.0 * qperpy * s000 + qperpx * s010 + qperpw * s020) * theta)
                    + q0z
                        * (q0w * (s010 - s110) + q0x * (-s020 + s120)
                            - 2.0 * (2.0 * qperpz * s000 + qperpw * s010 - qperpx * s020) * theta),
                -(qperpy * qperpy * s001) - qperpz * qperpz * s001 + qperpx * qperpy * s011
                    - qperpw * qperpz * s011
                    + qperpw * qperpy * s021
                    + qperpx * qperpz * s021
                    + q0y * q0y * (s001 - s101)
                    + q0z * q0z * (s001 - s101)
                    + qperpy * qperpy * s101
                    + qperpz * qperpz * s101
                    - qperpx * qperpy * s111
                    + qperpw * qperpz * s111
                    - qperpw * qperpy * s121
                    - qperpx * qperpz * s121
                    + 2.0 * q0x * qperpy * s011 * theta
                    - 2.0 * q0w * qperpz * s011 * theta
                    + 2.0 * q0w * qperpy * s021 * theta
                    + 2.0 * q0x * qperpz * s021 * theta
                    + q0y
                        * (q0x * (-s011 + s111)
                            + q0w * (-s021 + s121)
                            + 2.0 * (-2.0 * qperpy * s001 + qperpx * s011 + qperpw * s021) * theta)
                    + q0z
                        * (q0w * (s011 - s111) + q0x * (-s021 + s121)
                            - 2.0 * (2.0 * qperpz * s001 + qperpw * s011 - qperpx * s021) * theta),
                -(qperpy * qperpy * s002) - qperpz * qperpz * s002 + qperpx * qperpy * s012
                    - qperpw * qperpz * s012
                    + qperpw * qperpy * s022
                    + qperpx * qperpz * s022
                    + q0y * q0y * (s002 - s102)
                    + q0z * q0z * (s002 - s102)
                    + qperpy * qperpy * s102
                    + qperpz * qperpz * s102
                    - qperpx * qperpy * s112
                    + qperpw * qperpz * s112
                    - qperpw * qperpy * s122
                    - qperpx * qperpz * s122
                    + 2.0 * q0x * qperpy * s012 * theta
                    - 2.0 * q0w * qperpz * s012 * theta
                    + 2.0 * q0w * qperpy * s022 * theta
                    + 2.0 * q0x * qperpz * s022 * theta
                    + q0y
                        * (q0x * (-s012 + s112)
                            + q0w * (-s022 + s122)
                            + 2.0 * (-2.0 * qperpy * s002 + qperpx * s012 + qperpw * s022) * theta)
                    + q0z
                        * (q0w * (s012 - s112) + q0x * (-s022 + s122)
                            - 2.0 * (2.0 * qperpz * s002 + qperpw * s012 - qperpx * s022) * theta),
            );

            at.c3[0] = DerivativeTerm::new(
                0.,
                -2.0 * (q0x * qperpy * s010 - q0w * qperpz * s010
                    + q0w * qperpy * s020
                    + q0x * qperpz * s020
                    - q0x * qperpy * s110
                    + q0w * qperpz * s110
                    - q0w * qperpy * s120
                    - q0x * qperpz * s120
                    + q0y
                        * (-2.0 * qperpy * s000
                            + qperpx * s010
                            + qperpw * s020
                            + 2.0 * qperpy * s100
                            - qperpx * s110
                            - qperpw * s120)
                    + q0z
                        * (-2.0 * qperpz * s000 - qperpw * s010
                            + qperpx * s020
                            + 2.0 * qperpz * s100
                            + qperpw * s110
                            - qperpx * s120))
                    * theta,
                -2.0 * (q0x * qperpy * s011 - q0w * qperpz * s011
                    + q0w * qperpy * s021
                    + q0x * qperpz * s021
                    - q0x * qperpy * s111
                    + q0w * qperpz * s111
                    - q0w * qperpy * s121
                    - q0x * qperpz * s121
                    + q0y
                        * (-2.0 * qperpy * s001
                            + qperpx * s011
                            + qperpw * s021
                            + 2.0 * qperpy * s101
                            - qperpx * s111
                            - qperpw * s121)
                    + q0z
                        * (-2.0 * qperpz * s001 - qperpw * s011
                            + qperpx * s021
                            + 2.0 * qperpz * s101
                            + qperpw * s111
                            - qperpx * s121))
                    * theta,
                -2.0 * (q0x * qperpy * s012 - q0w * qperpz * s012
                    + q0w * qperpy * s022
                    + q0x * qperpz * s022
                    - q0x * qperpy * s112
                    + q0w * qperpz * s112
                    - q0w * qperpy * s122
                    - q0x * qperpz * s122
                    + q0y
                        * (-2.0 * qperpy * s002
                            + qperpx * s012
                            + qperpw * s022
                            + 2.0 * qperpy * s102
                            - qperpx * s112
                            - qperpw * s122)
                    + q0z
                        * (-2.0 * qperpz * s002 - qperpw * s012
                            + qperpx * s022
                            + 2.0 * qperpz * s102
                            + qperpw * s112
                            - qperpx * s122))
                    * theta,
            );

            at.c4[0] = DerivativeTerm::new(
                0.,
                -(q0x * qperpy * s010) + q0w * qperpz * s010
                    - q0w * qperpy * s020
                    - q0x * qperpz * s020
                    + q0x * qperpy * s110
                    - q0w * qperpz * s110
                    + q0w * qperpy * s120
                    + q0x * qperpz * s120
                    + 2.0 * q0y * q0y * s000 * theta
                    + 2.0 * q0z * q0z * s000 * theta
                    - 2.0 * qperpy * qperpy * s000 * theta
                    - 2.0 * qperpz * qperpz * s000 * theta
                    + 2.0 * qperpx * qperpy * s010 * theta
                    - 2.0 * qperpw * qperpz * s010 * theta
                    + 2.0 * qperpw * qperpy * s020 * theta
                    + 2.0 * qperpx * qperpz * s020 * theta
                    + q0y
                        * (-(qperpx * s010) - qperpw * s020
                            + 2.0 * qperpy * (s000 - s100)
                            + qperpx * s110
                            + qperpw * s120
                            - 2.0 * q0x * s010 * theta
                            - 2.0 * q0w * s020 * theta)
                    + q0z
                        * (2.0 * qperpz * s000 + qperpw * s010
                            - qperpx * s020
                            - 2.0 * qperpz * s100
                            - qperpw * s110
                            + qperpx * s120
                            + 2.0 * q0w * s010 * theta
                            - 2.0 * q0x * s020 * theta),
                -(q0x * qperpy * s011) + q0w * qperpz * s011
                    - q0w * qperpy * s021
                    - q0x * qperpz * s021
                    + q0x * qperpy * s111
                    - q0w * qperpz * s111
                    + q0w * qperpy * s121
                    + q0x * qperpz * s121
                    + 2.0 * q0y * q0y * s001 * theta
                    + 2.0 * q0z * q0z * s001 * theta
                    - 2.0 * qperpy * qperpy * s001 * theta
                    - 2.0 * qperpz * qperpz * s001 * theta
                    + 2.0 * qperpx * qperpy * s011 * theta
                    - 2.0 * qperpw * qperpz * s011 * theta
                    + 2.0 * qperpw * qperpy * s021 * theta
                    + 2.0 * qperpx * qperpz * s021 * theta
                    + q0y
                        * (-(qperpx * s011) - qperpw * s021
                            + 2.0 * qperpy * (s001 - s101)
                            + qperpx * s111
                            + qperpw * s121
                            - 2.0 * q0x * s011 * theta
                            - 2.0 * q0w * s021 * theta)
                    + q0z
                        * (2.0 * qperpz * s001 + qperpw * s011
                            - qperpx * s021
                            - 2.0 * qperpz * s101
                            - qperpw * s111
                            + qperpx * s121
                            + 2.0 * q0w * s011 * theta
                            - 2.0 * q0x * s021 * theta),
                -(q0x * qperpy * s012) + q0w * qperpz * s012
                    - q0w * qperpy * s022
                    - q0x * qperpz * s022
                    + q0x * qperpy * s112
                    - q0w * qperpz * s112
                    + q0w * qperpy * s122
                    + q0x * qperpz * s122
                    + 2.0 * q0y * q0y * s002 * theta
                    + 2.0 * q0z * q0z * s002 * theta
                    - 2.0 * qperpy * qperpy * s002 * theta
                    - 2.0 * qperpz * qperpz * s002 * theta
                    + 2.0 * qperpx * qperpy * s012 * theta
                    - 2.0 * qperpw * qperpz * s012 * theta
                    + 2.0 * qperpw * qperpy * s022 * theta
                    + 2.0 * qperpx * qperpz * s022 * theta
                    + q0y
                        * (-(qperpx * s012) - qperpw * s022
                            + 2.0 * qperpy * (s002 - s102)
                            + qperpx * s112
                            + qperpw * s122
                            - 2.0 * q0x * s012 * theta
                            - 2.0 * q0w * s022 * theta)
                    + q0z
                        * (2.0 * qperpz * s002 + qperpw * s012
                            - qperpx * s022
                            - 2.0 * qperpz * s102
                            - qperpw * s112
                            + qperpx * s122
                            + 2.0 * q0w * s012 * theta
                            - 2.0 * q0x * s022 * theta),
            );

            at.c5[0] = DerivativeTerm::new(
                0.,
                2.0 * (qperpy * qperpy * s000 + qperpz * qperpz * s000 - qperpx * qperpy * s010
                    + qperpw * qperpz * s010
                    - qperpw * qperpy * s020
                    - qperpx * qperpz * s020
                    - qperpy * qperpy * s100
                    - qperpz * qperpz * s100
                    + q0y * q0y * (-s000 + s100)
                    + q0z * q0z * (-s000 + s100)
                    + qperpx * qperpy * s110
                    - qperpw * qperpz * s110
                    + q0y * (q0x * (s010 - s110) + q0w * (s020 - s120))
                    + qperpw * qperpy * s120
                    + qperpx * qperpz * s120
                    + q0z * (-(q0w * s010) + q0x * s020 + q0w * s110 - q0x * s120))
                    * theta,
                2.0 * (qperpy * qperpy * s001 + qperpz * qperpz * s001 - qperpx * qperpy * s011
                    + qperpw * qperpz * s011
                    - qperpw * qperpy * s021
                    - qperpx * qperpz * s021
                    - qperpy * qperpy * s101
                    - qperpz * qperpz * s101
                    + q0y * q0y * (-s001 + s101)
                    + q0z * q0z * (-s001 + s101)
                    + qperpx * qperpy * s111
                    - qperpw * qperpz * s111
                    + q0y * (q0x * (s011 - s111) + q0w * (s021 - s121))
                    + qperpw * qperpy * s121
                    + qperpx * qperpz * s121
                    + q0z * (-(q0w * s011) + q0x * s021 + q0w * s111 - q0x * s121))
                    * theta,
                2.0 * (qperpy * qperpy * s002 + qperpz * qperpz * s002 - qperpx * qperpy * s012
                    + qperpw * qperpz * s012
                    - qperpw * qperpy * s022
                    - qperpx * qperpz * s022
                    - qperpy * qperpy * s102
                    - qperpz * qperpz * s102
                    + q0y * q0y * (-s002 + s102)
                    + q0z * q0z * (-s002 + s102)
                    + qperpx * qperpy * s112
                    - qperpw * qperpz * s112
                    + q0y * (q0x * (s012 - s112) + q0w * (s022 - s122))
                    + qperpw * qperpy * s122
                    + qperpx * qperpz * s122
                    + q0z * (-(q0w * s012) + q0x * s022 + q0w * s112 - q0x * s122))
                    * theta,
            );

            at.c1[1] = DerivativeTerm::new(
                -t0y + t1y,
                -(qperpx * qperpy * s000) - qperpw * qperpz * s000 - s010
                    + q0z * q0z * s010
                    + qperpx * qperpx * s010
                    + qperpz * qperpz * s010
                    - q0y * q0z * s020
                    + qperpw * qperpx * s020
                    - qperpy * qperpz * s020
                    + qperpx * qperpy * s100
                    + qperpw * qperpz * s100
                    + q0w * q0z * (-s000 + s100)
                    + q0x * q0x * (s010 - s110)
                    + s110
                    - q0z * q0z * s110
                    - qperpx * qperpx * s110
                    - qperpz * qperpz * s110
                    + q0x * (q0y * (-s000 + s100) + q0w * (s020 - s120))
                    + q0y * q0z * s120
                    - qperpw * qperpx * s120
                    + qperpy * qperpz * s120,
                -(qperpx * qperpy * s001) - qperpw * qperpz * s001 - s011
                    + q0z * q0z * s011
                    + qperpx * qperpx * s011
                    + qperpz * qperpz * s011
                    - q0y * q0z * s021
                    + qperpw * qperpx * s021
                    - qperpy * qperpz * s021
                    + qperpx * qperpy * s101
                    + qperpw * qperpz * s101
                    + q0w * q0z * (-s001 + s101)
                    + q0x * q0x * (s011 - s111)
                    + s111
                    - q0z * q0z * s111
                    - qperpx * qperpx * s111
                    - qperpz * qperpz * s111
                    + q0x * (q0y * (-s001 + s101) + q0w * (s021 - s121))
                    + q0y * q0z * s121
                    - qperpw * qperpx * s121
                    + qperpy * qperpz * s121,
                -(qperpx * qperpy * s002) - qperpw * qperpz * s002 - s012
                    + q0z * q0z * s012
                    + qperpx * qperpx * s012
                    + qperpz * qperpz * s012
                    - q0y * q0z * s022
                    + qperpw * qperpx * s022
                    - qperpy * qperpz * s022
                    + qperpx * qperpy * s102
                    + qperpw * qperpz * s102
                    + q0w * q0z * (-s002 + s102)
                    + q0x * q0x * (s012 - s112)
                    + s112
                    - q0z * q0z * s112
                    - qperpx * qperpx * s112
                    - qperpz * qperpz * s112
                    + q0x * (q0y * (-s002 + s102) + q0w * (s022 - s122))
                    + q0y * q0z * s122
                    - qperpw * qperpx * s122
                    + qperpy * qperpz * s122,
            );

            at.c2[1] = DerivativeTerm::new(
                0.,
                qperpx * qperpy * s000 + qperpw * qperpz * s000 + q0z * q0z * s010
                    - qperpx * qperpx * s010
                    - qperpz * qperpz * s010
                    - q0y * q0z * s020
                    - qperpw * qperpx * s020
                    + qperpy * qperpz * s020
                    - qperpx * qperpy * s100
                    - qperpw * qperpz * s100
                    + q0x * q0x * (s010 - s110)
                    - q0z * q0z * s110
                    + qperpx * qperpx * s110
                    + qperpz * qperpz * s110
                    + q0y * q0z * s120
                    + qperpw * qperpx * s120
                    - qperpy * qperpz * s120
                    + 2.0 * q0z * qperpw * s000 * theta
                    + 2.0 * q0y * qperpx * s000 * theta
                    - 4.0 * q0z * qperpz * s010 * theta
                    + 2.0 * q0z * qperpy * s020 * theta
                    + 2.0 * q0y * qperpz * s020 * theta
                    + q0x
                        * (q0w * s020 + q0y * (-s000 + s100) - q0w * s120
                            + 2.0 * qperpy * s000 * theta
                            - 4.0 * qperpx * s010 * theta
                            - 2.0 * qperpw * s020 * theta)
                    + q0w
                        * (-(q0z * s000) + q0z * s100 + 2.0 * qperpz * s000 * theta
                            - 2.0 * qperpx * s020 * theta),
                qperpx * qperpy * s001 + qperpw * qperpz * s001 + q0z * q0z * s011
                    - qperpx * qperpx * s011
                    - qperpz * qperpz * s011
                    - q0y * q0z * s021
                    - qperpw * qperpx * s021
                    + qperpy * qperpz * s021
                    - qperpx * qperpy * s101
                    - qperpw * qperpz * s101
                    + q0x * q0x * (s011 - s111)
                    - q0z * q0z * s111
                    + qperpx * qperpx * s111
                    + qperpz * qperpz * s111
                    + q0y * q0z * s121
                    + qperpw * qperpx * s121
                    - qperpy * qperpz * s121
                    + 2.0 * q0z * qperpw * s001 * theta
                    + 2.0 * q0y * qperpx * s001 * theta
                    - 4.0 * q0z * qperpz * s011 * theta
                    + 2.0 * q0z * qperpy * s021 * theta
                    + 2.0 * q0y * qperpz * s021 * theta
                    + q0x
                        * (q0w * s021 + q0y * (-s001 + s101) - q0w * s121
                            + 2.0 * qperpy * s001 * theta
                            - 4.0 * qperpx * s011 * theta
                            - 2.0 * qperpw * s021 * theta)
                    + q0w
                        * (-(q0z * s001) + q0z * s101 + 2.0 * qperpz * s001 * theta
                            - 2.0 * qperpx * s021 * theta),
                qperpx * qperpy * s002 + qperpw * qperpz * s002 + q0z * q0z * s012
                    - qperpx * qperpx * s012
                    - qperpz * qperpz * s012
                    - q0y * q0z * s022
                    - qperpw * qperpx * s022
                    + qperpy * qperpz * s022
                    - qperpx * qperpy * s102
                    - qperpw * qperpz * s102
                    + q0x * q0x * (s012 - s112)
                    - q0z * q0z * s112
                    + qperpx * qperpx * s112
                    + qperpz * qperpz * s112
                    + q0y * q0z * s122
                    + qperpw * qperpx * s122
                    - qperpy * qperpz * s122
                    + 2.0 * q0z * qperpw * s002 * theta
                    + 2.0 * q0y * qperpx * s002 * theta
                    - 4.0 * q0z * qperpz * s012 * theta
                    + 2.0 * q0z * qperpy * s022 * theta
                    + 2.0 * q0y * qperpz * s022 * theta
                    + q0x
                        * (q0w * s022 + q0y * (-s002 + s102) - q0w * s122
                            + 2.0 * qperpy * s002 * theta
                            - 4.0 * qperpx * s012 * theta
                            - 2.0 * qperpw * s022 * theta)
                    + q0w
                        * (-(q0z * s002) + q0z * s102 + 2.0 * qperpz * s002 * theta
                            - 2.0 * qperpx * s022 * theta),
            );

            at.c3[1] = DerivativeTerm::new(
                0.,
                2.0 * (-(q0x * qperpy * s000) - q0w * qperpz * s000
                    + 2.0 * q0x * qperpx * s010
                    + q0x * qperpw * s020
                    + q0w * qperpx * s020
                    + q0x * qperpy * s100
                    + q0w * qperpz * s100
                    - 2.0 * q0x * qperpx * s110
                    - q0x * qperpw * s120
                    - q0w * qperpx * s120
                    + q0z
                        * (2.0 * qperpz * s010 - qperpy * s020 + qperpw * (-s000 + s100)
                            - 2.0 * qperpz * s110
                            + qperpy * s120)
                    + q0y * (-(qperpx * s000) - qperpz * s020 + qperpx * s100 + qperpz * s120))
                    * theta,
                2.0 * (-(q0x * qperpy * s001) - q0w * qperpz * s001
                    + 2.0 * q0x * qperpx * s011
                    + q0x * qperpw * s021
                    + q0w * qperpx * s021
                    + q0x * qperpy * s101
                    + q0w * qperpz * s101
                    - 2.0 * q0x * qperpx * s111
                    - q0x * qperpw * s121
                    - q0w * qperpx * s121
                    + q0z
                        * (2.0 * qperpz * s011 - qperpy * s021 + qperpw * (-s001 + s101)
                            - 2.0 * qperpz * s111
                            + qperpy * s121)
                    + q0y * (-(qperpx * s001) - qperpz * s021 + qperpx * s101 + qperpz * s121))
                    * theta,
                2.0 * (-(q0x * qperpy * s002) - q0w * qperpz * s002
                    + 2.0 * q0x * qperpx * s012
                    + q0x * qperpw * s022
                    + q0w * qperpx * s022
                    + q0x * qperpy * s102
                    + q0w * qperpz * s102
                    - 2.0 * q0x * qperpx * s112
                    - q0x * qperpw * s122
                    - q0w * qperpx * s122
                    + q0z
                        * (2.0 * qperpz * s012 - qperpy * s022 + qperpw * (-s002 + s102)
                            - 2.0 * qperpz * s112
                            + qperpy * s122)
                    + q0y * (-(qperpx * s002) - qperpz * s022 + qperpx * s102 + qperpz * s122))
                    * theta,
            );

            at.c4[1] = DerivativeTerm::new(
                0.,
                -(q0x * qperpy * s000) - q0w * qperpz * s000
                    + 2.0 * q0x * qperpx * s010
                    + q0x * qperpw * s020
                    + q0w * qperpx * s020
                    + q0x * qperpy * s100
                    + q0w * qperpz * s100
                    - 2.0 * q0x * qperpx * s110
                    - q0x * qperpw * s120
                    - q0w * qperpx * s120
                    + 2.0 * qperpx * qperpy * s000 * theta
                    + 2.0 * qperpw * qperpz * s000 * theta
                    + 2.0 * q0x * q0x * s010 * theta
                    + 2.0 * q0z * q0z * s010 * theta
                    - 2.0 * qperpx * qperpx * s010 * theta
                    - 2.0 * qperpz * qperpz * s010 * theta
                    + 2.0 * q0w * q0x * s020 * theta
                    - 2.0 * qperpw * qperpx * s020 * theta
                    + 2.0 * qperpy * qperpz * s020 * theta
                    + q0y
                        * (-(qperpx * s000) - qperpz * s020 + qperpx * s100 + qperpz * s120
                            - 2.0 * q0x * s000 * theta)
                    + q0z
                        * (2.0 * qperpz * s010 - qperpy * s020 + qperpw * (-s000 + s100)
                            - 2.0 * qperpz * s110
                            + qperpy * s120
                            - 2.0 * q0w * s000 * theta
                            - 2.0 * q0y * s020 * theta),
                -(q0x * qperpy * s001) - q0w * qperpz * s001
                    + 2.0 * q0x * qperpx * s011
                    + q0x * qperpw * s021
                    + q0w * qperpx * s021
                    + q0x * qperpy * s101
                    + q0w * qperpz * s101
                    - 2.0 * q0x * qperpx * s111
                    - q0x * qperpw * s121
                    - q0w * qperpx * s121
                    + 2.0 * qperpx * qperpy * s001 * theta
                    + 2.0 * qperpw * qperpz * s001 * theta
                    + 2.0 * q0x * q0x * s011 * theta
                    + 2.0 * q0z * q0z * s011 * theta
                    - 2.0 * qperpx * qperpx * s011 * theta
                    - 2.0 * qperpz * qperpz * s011 * theta
                    + 2.0 * q0w * q0x * s021 * theta
                    - 2.0 * qperpw * qperpx * s021 * theta
                    + 2.0 * qperpy * qperpz * s021 * theta
                    + q0y
                        * (-(qperpx * s001) - qperpz * s021 + qperpx * s101 + qperpz * s121
                            - 2.0 * q0x * s001 * theta)
                    + q0z
                        * (2.0 * qperpz * s011 - qperpy * s021 + qperpw * (-s001 + s101)
                            - 2.0 * qperpz * s111
                            + qperpy * s121
                            - 2.0 * q0w * s001 * theta
                            - 2.0 * q0y * s021 * theta),
                -(q0x * qperpy * s002) - q0w * qperpz * s002
                    + 2.0 * q0x * qperpx * s012
                    + q0x * qperpw * s022
                    + q0w * qperpx * s022
                    + q0x * qperpy * s102
                    + q0w * qperpz * s102
                    - 2.0 * q0x * qperpx * s112
                    - q0x * qperpw * s122
                    - q0w * qperpx * s122
                    + 2.0 * qperpx * qperpy * s002 * theta
                    + 2.0 * qperpw * qperpz * s002 * theta
                    + 2.0 * q0x * q0x * s012 * theta
                    + 2.0 * q0z * q0z * s012 * theta
                    - 2.0 * qperpx * qperpx * s012 * theta
                    - 2.0 * qperpz * qperpz * s012 * theta
                    + 2.0 * q0w * q0x * s022 * theta
                    - 2.0 * qperpw * qperpx * s022 * theta
                    + 2.0 * qperpy * qperpz * s022 * theta
                    + q0y
                        * (-(qperpx * s002) - qperpz * s022 + qperpx * s102 + qperpz * s122
                            - 2.0 * q0x * s002 * theta)
                    + q0z
                        * (2.0 * qperpz * s012 - qperpy * s022 + qperpw * (-s002 + s102)
                            - 2.0 * qperpz * s112
                            + qperpy * s122
                            - 2.0 * q0w * s002 * theta
                            - 2.0 * q0y * s022 * theta),
            );

            at.c5[1] = DerivativeTerm::new(
                0.,
                -2.0 * (qperpx * qperpy * s000 + qperpw * qperpz * s000 + q0z * q0z * s010
                    - qperpx * qperpx * s010
                    - qperpz * qperpz * s010
                    - q0y * q0z * s020
                    - qperpw * qperpx * s020
                    + qperpy * qperpz * s020
                    - qperpx * qperpy * s100
                    - qperpw * qperpz * s100
                    + q0w * q0z * (-s000 + s100)
                    + q0x * q0x * (s010 - s110)
                    - q0z * q0z * s110
                    + qperpx * qperpx * s110
                    + qperpz * qperpz * s110
                    + q0x * (q0y * (-s000 + s100) + q0w * (s020 - s120))
                    + q0y * q0z * s120
                    + qperpw * qperpx * s120
                    - qperpy * qperpz * s120)
                    * theta,
                -2.0 * (qperpx * qperpy * s001 + qperpw * qperpz * s001 + q0z * q0z * s011
                    - qperpx * qperpx * s011
                    - qperpz * qperpz * s011
                    - q0y * q0z * s021
                    - qperpw * qperpx * s021
                    + qperpy * qperpz * s021
                    - qperpx * qperpy * s101
                    - qperpw * qperpz * s101
                    + q0w * q0z * (-s001 + s101)
                    + q0x * q0x * (s011 - s111)
                    - q0z * q0z * s111
                    + qperpx * qperpx * s111
                    + qperpz * qperpz * s111
                    + q0x * (q0y * (-s001 + s101) + q0w * (s021 - s121))
                    + q0y * q0z * s121
                    + qperpw * qperpx * s121
                    - qperpy * qperpz * s121)
                    * theta,
                -2.0 * (qperpx * qperpy * s002 + qperpw * qperpz * s002 + q0z * q0z * s012
                    - qperpx * qperpx * s012
                    - qperpz * qperpz * s012
                    - q0y * q0z * s022
                    - qperpw * qperpx * s022
                    + qperpy * qperpz * s022
                    - qperpx * qperpy * s102
                    - qperpw * qperpz * s102
                    + q0w * q0z * (-s002 + s102)
                    + q0x * q0x * (s012 - s112)
                    - q0z * q0z * s112
                    + qperpx * qperpx * s112
                    + qperpz * qperpz * s112
                    + q0x * (q0y * (-s002 + s102) + q0w * (s022 - s122))
                    + q0y * q0z * s122
                    + qperpw * qperpx * s122
                    - qperpy * qperpz * s122)
                    * theta,
            );

            at.c1[2] = DerivativeTerm::new(
                -t0z + t1z,
                qperpw * qperpy * s000
                    - qperpx * qperpz * s000
                    - q0y * q0z * s010
                    - qperpw * qperpx * s010
                    - qperpy * qperpz * s010
                    - s020
                    + q0y * q0y * s020
                    + qperpx * qperpx * s020
                    + qperpy * qperpy * s020
                    - qperpw * qperpy * s100
                    + qperpx * qperpz * s100
                    + q0x * q0z * (-s000 + s100)
                    + q0y * q0z * s110
                    + qperpw * qperpx * s110
                    + qperpy * qperpz * s110
                    + q0w * (q0y * (s000 - s100) + q0x * (-s010 + s110))
                    + q0x * q0x * (s020 - s120)
                    + s120
                    - q0y * q0y * s120
                    - qperpx * qperpx * s120
                    - qperpy * qperpy * s120,
                qperpw * qperpy * s001
                    - qperpx * qperpz * s001
                    - q0y * q0z * s011
                    - qperpw * qperpx * s011
                    - qperpy * qperpz * s011
                    - s021
                    + q0y * q0y * s021
                    + qperpx * qperpx * s021
                    + qperpy * qperpy * s021
                    - qperpw * qperpy * s101
                    + qperpx * qperpz * s101
                    + q0x * q0z * (-s001 + s101)
                    + q0y * q0z * s111
                    + qperpw * qperpx * s111
                    + qperpy * qperpz * s111
                    + q0w * (q0y * (s001 - s101) + q0x * (-s011 + s111))
                    + q0x * q0x * (s021 - s121)
                    + s121
                    - q0y * q0y * s121
                    - qperpx * qperpx * s121
                    - qperpy * qperpy * s121,
                qperpw * qperpy * s002
                    - qperpx * qperpz * s002
                    - q0y * q0z * s012
                    - qperpw * qperpx * s012
                    - qperpy * qperpz * s012
                    - s022
                    + q0y * q0y * s022
                    + qperpx * qperpx * s022
                    + qperpy * qperpy * s022
                    - qperpw * qperpy * s102
                    + qperpx * qperpz * s102
                    + q0x * q0z * (-s002 + s102)
                    + q0y * q0z * s112
                    + qperpw * qperpx * s112
                    + qperpy * qperpz * s112
                    + q0w * (q0y * (s002 - s102) + q0x * (-s012 + s112))
                    + q0x * q0x * (s022 - s122)
                    + s122
                    - q0y * q0y * s122
                    - qperpx * qperpx * s122
                    - qperpy * qperpy * s122,
            );

            at.c2[2] = DerivativeTerm::new(
                0.,
                q0w * q0y * s000 - q0x * q0z * s000 - qperpw * qperpy * s000
                    + qperpx * qperpz * s000
                    - q0w * q0x * s010
                    - q0y * q0z * s010
                    + qperpw * qperpx * s010
                    + qperpy * qperpz * s010
                    + q0x * q0x * s020
                    + q0y * q0y * s020
                    - qperpx * qperpx * s020
                    - qperpy * qperpy * s020
                    - q0w * q0y * s100
                    + q0x * q0z * s100
                    + qperpw * qperpy * s100
                    - qperpx * qperpz * s100
                    + q0w * q0x * s110
                    + q0y * q0z * s110
                    - qperpw * qperpx * s110
                    - qperpy * qperpz * s110
                    - q0x * q0x * s120
                    - q0y * q0y * s120
                    + qperpx * qperpx * s120
                    + qperpy * qperpy * s120
                    - 2.0 * q0y * qperpw * s000 * theta
                    + 2.0 * q0z * qperpx * s000 * theta
                    - 2.0 * q0w * qperpy * s000 * theta
                    + 2.0 * q0x * qperpz * s000 * theta
                    + 2.0 * q0x * qperpw * s010 * theta
                    + 2.0 * q0w * qperpx * s010 * theta
                    + 2.0 * q0z * qperpy * s010 * theta
                    + 2.0 * q0y * qperpz * s010 * theta
                    - 4.0 * q0x * qperpx * s020 * theta
                    - 4.0 * q0y * qperpy * s020 * theta,
                q0w * q0y * s001 - q0x * q0z * s001 - qperpw * qperpy * s001
                    + qperpx * qperpz * s001
                    - q0w * q0x * s011
                    - q0y * q0z * s011
                    + qperpw * qperpx * s011
                    + qperpy * qperpz * s011
                    + q0x * q0x * s021
                    + q0y * q0y * s021
                    - qperpx * qperpx * s021
                    - qperpy * qperpy * s021
                    - q0w * q0y * s101
                    + q0x * q0z * s101
                    + qperpw * qperpy * s101
                    - qperpx * qperpz * s101
                    + q0w * q0x * s111
                    + q0y * q0z * s111
                    - qperpw * qperpx * s111
                    - qperpy * qperpz * s111
                    - q0x * q0x * s121
                    - q0y * q0y * s121
                    + qperpx * qperpx * s121
                    + qperpy * qperpy * s121
                    - 2.0 * q0y * qperpw * s001 * theta
                    + 2.0 * q0z * qperpx * s001 * theta
                    - 2.0 * q0w * qperpy * s001 * theta
                    + 2.0 * q0x * qperpz * s001 * theta
                    + 2.0 * q0x * qperpw * s011 * theta
                    + 2.0 * q0w * qperpx * s011 * theta
                    + 2.0 * q0z * qperpy * s011 * theta
                    + 2.0 * q0y * qperpz * s011 * theta
                    - 4.0 * q0x * qperpx * s021 * theta
                    - 4.0 * q0y * qperpy * s021 * theta,
                q0w * q0y * s002 - q0x * q0z * s002 - qperpw * qperpy * s002
                    + qperpx * qperpz * s002
                    - q0w * q0x * s012
                    - q0y * q0z * s012
                    + qperpw * qperpx * s012
                    + qperpy * qperpz * s012
                    + q0x * q0x * s022
                    + q0y * q0y * s022
                    - qperpx * qperpx * s022
                    - qperpy * qperpy * s022
                    - q0w * q0y * s102
                    + q0x * q0z * s102
                    + qperpw * qperpy * s102
                    - qperpx * qperpz * s102
                    + q0w * q0x * s112
                    + q0y * q0z * s112
                    - qperpw * qperpx * s112
                    - qperpy * qperpz * s112
                    - q0x * q0x * s122
                    - q0y * q0y * s122
                    + qperpx * qperpx * s122
                    + qperpy * qperpy * s122
                    - 2.0 * q0y * qperpw * s002 * theta
                    + 2.0 * q0z * qperpx * s002 * theta
                    - 2.0 * q0w * qperpy * s002 * theta
                    + 2.0 * q0x * qperpz * s002 * theta
                    + 2.0 * q0x * qperpw * s012 * theta
                    + 2.0 * q0w * qperpx * s012 * theta
                    + 2.0 * q0z * qperpy * s012 * theta
                    + 2.0 * q0y * qperpz * s012 * theta
                    - 4.0 * q0x * qperpx * s022 * theta
                    - 4.0 * q0y * qperpy * s022 * theta,
            );

            at.c3[2] = DerivativeTerm::new(
                0.,
                -2.0 * (-(q0w * qperpy * s000)
                    + q0x * qperpz * s000
                    + q0x * qperpw * s010
                    + q0w * qperpx * s010
                    - 2.0 * q0x * qperpx * s020
                    + q0w * qperpy * s100
                    - q0x * qperpz * s100
                    - q0x * qperpw * s110
                    - q0w * qperpx * s110
                    + q0z * (qperpx * s000 + qperpy * s010 - qperpx * s100 - qperpy * s110)
                    + 2.0 * q0x * qperpx * s120
                    + q0y
                        * (qperpz * s010 - 2.0 * qperpy * s020 + qperpw * (-s000 + s100)
                            - qperpz * s110
                            + 2.0 * qperpy * s120))
                    * theta,
                -2.0 * (-(q0w * qperpy * s001)
                    + q0x * qperpz * s001
                    + q0x * qperpw * s011
                    + q0w * qperpx * s011
                    - 2.0 * q0x * qperpx * s021
                    + q0w * qperpy * s101
                    - q0x * qperpz * s101
                    - q0x * qperpw * s111
                    - q0w * qperpx * s111
                    + q0z * (qperpx * s001 + qperpy * s011 - qperpx * s101 - qperpy * s111)
                    + 2.0 * q0x * qperpx * s121
                    + q0y
                        * (qperpz * s011 - 2.0 * qperpy * s021 + qperpw * (-s001 + s101)
                            - qperpz * s111
                            + 2.0 * qperpy * s121))
                    * theta,
                -2.0 * (-(q0w * qperpy * s002)
                    + q0x * qperpz * s002
                    + q0x * qperpw * s012
                    + q0w * qperpx * s012
                    - 2.0 * q0x * qperpx * s022
                    + q0w * qperpy * s102
                    - q0x * qperpz * s102
                    - q0x * qperpw * s112
                    - q0w * qperpx * s112
                    + q0z * (qperpx * s002 + qperpy * s012 - qperpx * s102 - qperpy * s112)
                    + 2.0 * q0x * qperpx * s122
                    + q0y
                        * (qperpz * s012 - 2.0 * qperpy * s022 + qperpw * (-s002 + s102)
                            - qperpz * s112
                            + 2.0 * qperpy * s122))
                    * theta,
            );

            at.c4[2] = DerivativeTerm::new(
                0.,
                q0w * qperpy * s000
                    - q0x * qperpz * s000
                    - q0x * qperpw * s010
                    - q0w * qperpx * s010
                    + 2.0 * q0x * qperpx * s020
                    - q0w * qperpy * s100
                    + q0x * qperpz * s100
                    + q0x * qperpw * s110
                    + q0w * qperpx * s110
                    - 2.0 * q0x * qperpx * s120
                    - 2.0 * qperpw * qperpy * s000 * theta
                    + 2.0 * qperpx * qperpz * s000 * theta
                    - 2.0 * q0w * q0x * s010 * theta
                    + 2.0 * qperpw * qperpx * s010 * theta
                    + 2.0 * qperpy * qperpz * s010 * theta
                    + 2.0 * q0x * q0x * s020 * theta
                    + 2.0 * q0y * q0y * s020 * theta
                    - 2.0 * qperpx * qperpx * s020 * theta
                    - 2.0 * qperpy * qperpy * s020 * theta
                    + q0z
                        * (-(qperpx * s000) - qperpy * s010 + qperpx * s100 + qperpy * s110
                            - 2.0 * q0x * s000 * theta)
                    + q0y
                        * (-(qperpz * s010)
                            + 2.0 * qperpy * s020
                            + qperpw * (s000 - s100)
                            + qperpz * s110
                            - 2.0 * qperpy * s120
                            + 2.0 * q0w * s000 * theta
                            - 2.0 * q0z * s010 * theta),
                q0w * qperpy * s001
                    - q0x * qperpz * s001
                    - q0x * qperpw * s011
                    - q0w * qperpx * s011
                    + 2.0 * q0x * qperpx * s021
                    - q0w * qperpy * s101
                    + q0x * qperpz * s101
                    + q0x * qperpw * s111
                    + q0w * qperpx * s111
                    - 2.0 * q0x * qperpx * s121
                    - 2.0 * qperpw * qperpy * s001 * theta
                    + 2.0 * qperpx * qperpz * s001 * theta
                    - 2.0 * q0w * q0x * s011 * theta
                    + 2.0 * qperpw * qperpx * s011 * theta
                    + 2.0 * qperpy * qperpz * s011 * theta
                    + 2.0 * q0x * q0x * s021 * theta
                    + 2.0 * q0y * q0y * s021 * theta
                    - 2.0 * qperpx * qperpx * s021 * theta
                    - 2.0 * qperpy * qperpy * s021 * theta
                    + q0z
                        * (-(qperpx * s001) - qperpy * s011 + qperpx * s101 + qperpy * s111
                            - 2.0 * q0x * s001 * theta)
                    + q0y
                        * (-(qperpz * s011)
                            + 2.0 * qperpy * s021
                            + qperpw * (s001 - s101)
                            + qperpz * s111
                            - 2.0 * qperpy * s121
                            + 2.0 * q0w * s001 * theta
                            - 2.0 * q0z * s011 * theta),
                q0w * qperpy * s002
                    - q0x * qperpz * s002
                    - q0x * qperpw * s012
                    - q0w * qperpx * s012
                    + 2.0 * q0x * qperpx * s022
                    - q0w * qperpy * s102
                    + q0x * qperpz * s102
                    + q0x * qperpw * s112
                    + q0w * qperpx * s112
                    - 2.0 * q0x * qperpx * s122
                    - 2.0 * qperpw * qperpy * s002 * theta
                    + 2.0 * qperpx * qperpz * s002 * theta
                    - 2.0 * q0w * q0x * s012 * theta
                    + 2.0 * qperpw * qperpx * s012 * theta
                    + 2.0 * qperpy * qperpz * s012 * theta
                    + 2.0 * q0x * q0x * s022 * theta
                    + 2.0 * q0y * q0y * s022 * theta
                    - 2.0 * qperpx * qperpx * s022 * theta
                    - 2.0 * qperpy * qperpy * s022 * theta
                    + q0z
                        * (-(qperpx * s002) - qperpy * s012 + qperpx * s102 + qperpy * s112
                            - 2.0 * q0x * s002 * theta)
                    + q0y
                        * (-(qperpz * s012)
                            + 2.0 * qperpy * s022
                            + qperpw * (s002 - s102)
                            + qperpz * s112
                            - 2.0 * qperpy * s122
                            + 2.0 * q0w * s002 * theta
                            - 2.0 * q0z * s012 * theta),
            );

            at.c5[2] = DerivativeTerm::new(
                0.,
                2.0 * (qperpw * qperpy * s000 - qperpx * qperpz * s000 + q0y * q0z * s010
                    - qperpw * qperpx * s010
                    - qperpy * qperpz * s010
                    - q0y * q0y * s020
                    + qperpx * qperpx * s020
                    + qperpy * qperpy * s020
                    + q0x * q0z * (s000 - s100)
                    - qperpw * qperpy * s100
                    + qperpx * qperpz * s100
                    + q0w * (q0y * (-s000 + s100) + q0x * (s010 - s110))
                    - q0y * q0z * s110
                    + qperpw * qperpx * s110
                    + qperpy * qperpz * s110
                    + q0y * q0y * s120
                    - qperpx * qperpx * s120
                    - qperpy * qperpy * s120
                    + q0x * q0x * (-s020 + s120))
                    * theta,
                2.0 * (qperpw * qperpy * s001 - qperpx * qperpz * s001 + q0y * q0z * s011
                    - qperpw * qperpx * s011
                    - qperpy * qperpz * s011
                    - q0y * q0y * s021
                    + qperpx * qperpx * s021
                    + qperpy * qperpy * s021
                    + q0x * q0z * (s001 - s101)
                    - qperpw * qperpy * s101
                    + qperpx * qperpz * s101
                    + q0w * (q0y * (-s001 + s101) + q0x * (s011 - s111))
                    - q0y * q0z * s111
                    + qperpw * qperpx * s111
                    + qperpy * qperpz * s111
                    + q0y * q0y * s121
                    - qperpx * qperpx * s121
                    - qperpy * qperpy * s121
                    + q0x * q0x * (-s021 + s121))
                    * theta,
                2.0 * (qperpw * qperpy * s002 - qperpx * qperpz * s002 + q0y * q0z * s012
                    - qperpw * qperpx * s012
                    - qperpy * qperpz * s012
                    - q0y * q0y * s022
                    + qperpx * qperpx * s022
                    + qperpy * qperpy * s022
                    + q0x * q0z * (s002 - s102)
                    - qperpw * qperpy * s102
                    + qperpx * qperpz * s102
                    + q0w * (q0y * (-s002 + s102) + q0x * (s012 - s112))
                    - q0y * q0z * s112
                    + qperpw * qperpx * s112
                    + qperpy * qperpz * s112
                    + q0y * q0y * s122
                    - qperpx * qperpx * s122
                    - qperpy * qperpy * s122
                    + q0x * q0x * (-s022 + s122))
                    * theta,
            );
        }
        at
    }

    pub fn interpolate(&self, time: Float) -> Transformf {
        let mut t: Transformf = Default::default();
        if !self.actually_animated || time <= self.start_time {
            t = self.start_transform;
            return t;
        }
        if time >= self.end_time {
            t = self.end_transform;
            return t;
        }
        let dt = (time - self.start_time) / (self.end_time - self.start_time);
        let trans = self.t[0] * (1.0 - dt) + self.t[1] * dt;
        let rotate = self.r[0].slerp(dt, &self.r[1]);
        let mut scale: Matrix4x4f = Default::default();
        for i in 0..3 {
            for j in 0..3 {
                scale.m[i][j] = lerp(dt, self.s[0].m[i][j], self.s[1].m[i][j]);
            }
        }
        Transformf::translate(&trans) * Transformf::from(rotate) * Transformf::from(scale)
    }

    pub fn motion_bounds(&self, b: &Bounds3f) -> Bounds3f {
        if !self.actually_animated {
            &self.start_transform * b
        } else if !self.has_rotation {
            (&self.start_transform * b).union(&(&self.end_transform * b))
        } else {
            let mut bounds: Bounds3f = Default::default();
            for corner in 0..8 {
                bounds = bounds.union(&self.bound_point_motion(&b.corner(corner)));
            }
            bounds
        }
    }

    pub fn bound_point_motion(&self, p: &Point3f) -> Bounds3f {
        if !self.actually_animated {
            (&self.start_transform * Point3Ref(p)).into()
        } else {
            let mut bounds: Bounds3f = Default::default();
            bounds.min = &self.start_transform * Point3Ref(p);
            bounds.max = &self.end_transform * Point3Ref(p);
            let cos_theta = self.r[0] * self.r[1];
            let theta = clamp(cos_theta, -1.0, 1.0).acos();
            for c in 0..3 {
                let mut zeros = Vec::new();
                interval_find_zeros(
                    self.c1[c].eval(p),
                    self.c2[c].eval(p),
                    self.c3[c].eval(p),
                    self.c4[c].eval(p),
                    self.c5[c].eval(p),
                    theta,
                    Interval::new(0.0, 1.0),
                    &mut zeros,
                    8,
                );
                for i in 0..zeros.len() {
                    let pz = Point3f::from((
                        self,
                        lerp(zeros[i], self.start_time, self.end_time),
                        Point3Ref(&p),
                    ));
                    bounds = bounds.union(&pz);
                }
            }
            bounds
        }
    }

    pub fn has_scale(&self) -> bool {
        self.start_transform.has_scale() || self.end_transform.has_scale()
    }
}
