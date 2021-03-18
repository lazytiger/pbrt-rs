use crate::core::geometry::{
    Bounds3f, Normal3, Point3, Point3f, SurfaceInteraction, Vector3, Vector3f,
};
use crate::core::quaternion::Quaternion;
use crate::core::{radians, RealNum};
use crate::{Float, PI};
use num::zero;
use std::cmp::Ordering;
use std::mem::swap;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Eq, PartialEq)]
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

#[derive(PartialEq)]
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

        let dir = (look - pos).normalize();
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
        &Self::scale(1.0, 1.0, 1.0 / (far - near))
            * &Self::translate(&Vector3f::new(0.0, 0.0, -near))
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
        &Self::scale(inv_tan_ang, inv_tan_ang, 1.0) * &trans
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
        let ret = ret.union_point(&p);

        let p = Point3f::new(rhs.min.x, rhs.max.y, rhs.min.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union_point(&p);

        let p = Point3f::new(rhs.min.x, rhs.min.y, rhs.max.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union_point(&p);

        let p = Point3f::new(rhs.min.x, rhs.max.y, rhs.max.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union_point(&p);

        let p = Point3f::new(rhs.max.x, rhs.max.y, rhs.min.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union_point(&p);

        let p = Point3f::new(rhs.max.x, rhs.min.y, rhs.max.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union_point(&p);

        let p = Point3f::new(rhs.max.x, rhs.max.y, rhs.max.z);
        let p = self * Point3Ref(&p);
        let ret = ret.union_point(&p);

        ret
    }
}

impl Mul for &Transformf {
    type Output = Transformf;

    fn mul(self, rhs: Self) -> Self::Output {
        Transformf {
            m: self.m * rhs.m,
            m_inv: rhs.m_inv * self.m,
        }
    }
}

impl Mul<&SurfaceInteraction> for &Transformf {
    type Output = SurfaceInteraction;

    fn mul(self, rhs: &SurfaceInteraction) -> Self::Output {
        //TODO
        SurfaceInteraction {}
    }
}

impl From<&Quaternion> for Transformf {
    fn from(q: &Quaternion) -> Self {
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
