use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use super::RealNum;
use crate::Float;

macro_rules! strip_plus {
    (+ $($rest:expr)+) => {
        $($rest)+
    };
}

macro_rules! match_index {
    ($index:ident, $n:expr,[$($indexes:expr),+], [$($arms:expr),+],) => {
        match $index {
            $(_i if _i == $indexes => $arms,)+
            _ => panic!("Out of index")
        }
    };
    ($index:ident, $n:expr, [$($indexes:expr),*], [$($arms:expr),*], $arm:expr $(;$rest:expr)*) => {
        match_index!($index, $n+1, [$($indexes,)* $n], [$($arms,)* $arm], $($rest);*)
    };
    ($index:ident, $($arms:expr);+) => {
        match_index!($index, 0, [], [], $($arms);+)
    }
}

macro_rules! make_vector {
    (struct $name:ident, $($field:ident),+) => {

        #[derive(Debug)]
        pub struct $name<T> {
            $($field:T,)+
        }

        impl<T:RealNum<T>> $name<T> {
            pub fn new($($field:T),+) ->Self  {
               $name { $($field:$field,)+}
            }

            pub fn length_squared(&self) -> T {
                strip_plus!($(+ self.$field * self.$field)+)
            }

            pub fn length(&self) -> T {
                self.length_squared().sqrt()
            }

            pub fn distance(&self, p:&$name<T>) -> T {
                (self - p).length()
            }
        }

        impl<T: RealNum<T>> Add for &$name<T> {
            type Output = $name<T>;

            fn add(self, rhs: Self) -> Self::Output {
                $name {
                    $($field:self.$field + rhs.$field,)+
                }
            }
        }

        impl<T: RealNum<T>> AddAssign for $name<T> {
            fn add_assign(&mut self, rhs: Self) {
                $(self.$field += rhs.$field;)+
            }
        }

        impl<T: RealNum<T>> Sub for &$name<T> {
            type Output = $name<T>;

            fn sub(self, rhs: Self) -> Self::Output {
                $name {
                    $($field:self.$field - rhs.$field,)+
                }
            }
        }

        impl<T: RealNum<T>> SubAssign for $name<T> {
            fn sub_assign(&mut self, rhs: Self) {
                $(self.$field -= rhs.$field;)+
            }
        }

        impl<T: RealNum<T>> PartialEq for $name<T> {
            fn eq(&self, other: &Self) -> bool {
                $(self.$field == other.$field)&& +
            }
        }

        impl<T: RealNum<T>> Mul for $name<T> {
            type Output = $name<T>;

            fn mul(self, rhs: Self) -> Self::Output {
                $name {
                    $($field:self.$field * rhs.$field,)+
                }
            }
        }

        impl<T: RealNum<T>> Mul<T> for $name<T> {
            type Output = $name<T>;

            fn mul(self, rhs: T) -> Self::Output {
                $name {
                    $($field:self.$field * rhs,)+
                }
            }
        }

        impl<T: RealNum<T>> MulAssign for $name<T> {
            fn mul_assign(&mut self, rhs: Self) {
                $(self.$field *= rhs.$field;)+
            }
        }

        impl<T: RealNum<T>> Div for $name<T> {
            type Output = $name<T>;

            fn div(self, rhs: Self) -> Self::Output {
                $name {
                    $($field:self.$field / rhs.$field,)+
                }
            }
        }

        impl<T: RealNum<T>> Div<T> for $name<T> {
            type Output = $name<T>;

            fn div(self, rhs: T) -> Self::Output {
                $name {
                    $($field:self.$field / rhs,)+
                }
            }
        }

        impl<T: RealNum<T>> DivAssign for $name<T> {
            fn div_assign(&mut self, rhs: Self) {
                $(self.$field /= rhs.$field;)+
            }
        }

        impl<T: RealNum<T>> Neg for $name<T> {
            type Output = $name<T>;

            fn neg(self) -> Self::Output {
                $name {
                    $($field: -self.$field,)+
                }
            }
        }

        impl<T: RealNum<T>> Clone for $name<T> {
               fn clone(&self) -> Self {
                    Self {
                        $($field:self.$field.clone(),)+
                    }
               }
        }

        impl<T> Index<usize> for $name<T> {
            type Output = T;

            fn index(&self, index: usize) -> &Self::Output {
                match_index!(index, $(&self.$field);+)
            }
        }

        impl<T> IndexMut<usize> for $name<T> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                match_index!(index, $(&mut self.$field);+)
            }
        }
    };
}

make_vector!(struct Vector2, x, y);
pub type Vector2f = Vector2<Float>;
pub type Vector2i = Vector2<i32>;
pub type Point2<T> = Vector2<T>;
pub type Point2f = Point2<Float>;
pub type Point2i = Point2<i32>;

make_vector!(struct Vector3, x, y, z);
pub type Vector3f = Vector3<Float>;
pub type Vector3i = Vector3<i32>;
pub type Point3<T> = Vector3<T>;
pub type Point3f = Point3<Float>;
pub type Point3i = Point3<i32>;
pub type Normal3<T> = Vector3<T>;
pub type Normal3f = Normal3<Float>;

pub struct Bounds2<T> {
    min: Point2<T>,
    max: Point2<T>,
}

impl<T: RealNum<T>> Bounds2<T> {
    pub fn new() -> Self {
        Self {
            min: Point2::new(T::min_value(), T::min_value()),
            max: Point2::new(T::max_value(), T::max_value()),
        }
    }

    pub fn diagonal(&self) -> Vector2<T> {
        &self.max - &self.min
    }

    pub fn lerp(&self, t: &Point2<T>) -> Point2<T> {
        Point2 {
            x: super::lerp(t.x, self.min.x, self.max.x),
            y: super::lerp(t.y, self.min.y, self.max.y),
        }
    }

    pub fn offset(&self, p: &Point2<T>) -> Vector2<T> {
        let mut o = p - &self.min;
        if self.max.x > self.min.x {
            o.x /= self.max.x - self.min.x;
        }
        if self.max.y > self.min.y {
            o.y /= self.max.y - self.min.y;
        }
        o
    }

    pub fn inside(&self, pt: &Point2<T>) -> bool {
        pt.x >= self.min.x && pt.x <= self.max.x && pt.y >= self.min.y && pt.y <= self.max.y
    }

    pub fn bounding_sphere(&self, c: &mut Point2<T>, rad: &mut T) {
        *c = (&self.min + &self.max) / T::two();
        if self.inside(c) {
            *rad = c.distance(&self.max);
        } else {
            *rad = T::zero();
        }
    }

    pub fn maximum_extent(&self) -> usize {
        let diag = self.diagonal();
        if diag.x > diag.y {
            0
        } else {
            1
        }
    }
}

impl<T: RealNum<T>> From<Point2<T>> for Bounds2<T> {
    fn from(p: Point2<T>) -> Self {
        Self {
            min: p.clone(),
            max: p,
        }
    }
}

impl<T: RealNum<T>> From<(Point2<T>, Point2<T>)> for Bounds2<T> {
    fn from(ps: (Point2<T>, Point2<T>)) -> Self {
        Self {
            min: Point2::new(ps.0.x.min(ps.1.x), ps.0.y.min(ps.1.y)),
            max: Point2::new(ps.0.x.max(ps.1.x), ps.0.y.max(ps.1.y)),
        }
    }
}
