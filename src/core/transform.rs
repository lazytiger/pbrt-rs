use crate::core::geometry::{Normal3, Point3, Vector3};
use crate::core::RealNum;
use crate::Float;
use std::cmp::Ordering;
use std::mem::swap;
use std::ops::Mul;

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Matrix4x4<T> {
    m: [[T; 4]; 4],
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

impl<T: RealNum<T>> Mul for Matrix4x4<T> {
    type Output = Matrix4x4<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

#[derive(PartialEq)]
pub struct Transform<T> {
    m: Matrix4x4<T>,
    m_inv: Matrix4x4<T>,
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
