use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use super::RealNum;
use crate::Float;
use num::Bounded;

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

macro_rules! make_extent {
    ($o:ident, $($fields:ident),+) => {
        make_extent!($o, 0, $($fields),+)
    };
    ($o:ident, $extent:expr, $field:ident, $($rest:ident),+) => {
        if $($o.$field > $o.$rest ) && * {
            $extent
        } else {
            make_extent!($o, $extent + 1, $($rest),+)
        }
    };
    ($o:ident, $extent:expr, $field:ident) => {
        $extent
    };
}

macro_rules! make_component {
    ($o:ident, $m:ident, $left:ident, $right:ident) => {
        $o.$left.$m($o.$right)
    };
    ($o:ident, $m:ident, $field:ident, $($rest:ident),+) => {
        $o.$field.$m(make_component!($o, $m, $($rest),+))
    };
}

macro_rules! make_vector {
    (struct $name:ident, $($field:ident),+) => {

        #[derive(Debug)]
        pub struct $name<T> {
            $(pub $field:T,)+
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

            pub fn max_dimension(&self) -> usize {
                make_extent!(self, $($field),+)
            }

            pub fn max_component(&self) -> T {
                make_component!(self, max, $($field),+)
            }

            pub fn min_component(&self) -> T {
                make_component!(self, min, $($field),+)
            }

            pub fn abs(&self) -> $name<T> {
                $name {
                    $($field: self.$field.abs(),)+
                }
            }

            pub fn dot(&self, v:&$name<T>) -> $name<T> {
                self * v
            }

            pub fn abs_dot(&self, v:&$name<T>) -> $name<T> {
                (self * v).abs()
            }

            pub fn normalize(&self) -> $name<T> {
                self / self.length()
            }

            pub fn min(&self, v: &$name<T>) -> $name<T> {
                $name {
                    $($field: self.$field.min(v.$field),)+
                }
            }

            pub fn max(&self, v: &$name<T>) -> $name<T> {
                $name {
                    $($field: self.$field.max(v.$field),)+
                }
            }

            pub fn permute(&self, $($field:usize),+) -> $name<T> {
                $name::new($(self[$field]),+)
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

        impl<T: RealNum<T>> Add<T> for &$name<T> {
            type Output = $name<T>;

            fn add(self, rhs: T) -> Self::Output {
                $name {
                    $($field:self.$field + rhs,)+
                }
            }
        }

        impl<T: RealNum<T>> Add for $name<T> {
            type Output = $name<T>;

            fn add(self, rhs: Self) -> Self::Output {
                $name {
                    $($field:self.$field + rhs.$field,)+
                }
            }
        }

        impl<T: RealNum<T>> Add<T> for $name<T> {
            type Output = $name<T>;

            fn add(self, rhs: T) -> Self::Output {
                $name {
                    $($field:self.$field + rhs,)+
                }
            }
        }

        impl<T: RealNum<T>> AddAssign for $name<T> {
            fn add_assign(&mut self, rhs: Self) {
                $(self.$field += rhs.$field;)+
            }
        }

        impl<T: RealNum<T>> AddAssign<&$name<T>> for $name<T> {
            fn add_assign(&mut self, rhs: &$name<T>) {
                $(self.$field += rhs.$field;)+
            }
        }

        impl<T: RealNum<T>> AddAssign<T> for $name<T> {
            fn add_assign(&mut self, rhs: T) {
                $(self.$field += rhs;)+
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

        impl<T: RealNum<T>> Sub for $name<T> {
            type Output = $name<T>;

            fn sub(self, rhs: Self) -> Self::Output {
                $name {
                    $($field:self.$field - rhs.$field,)+
                }
            }
        }

        impl<T: RealNum<T>> Sub<T> for $name<T> {
            type Output = $name<T>;

            fn sub(self, rhs: T) -> Self::Output {
                $name {
                    $($field:self.$field - rhs,)+
                }
            }
        }

        impl<T: RealNum<T>> Sub<T> for &$name<T> {
            type Output = $name<T>;

            fn sub(self, rhs: T) -> Self::Output {
                $name {
                    $($field:self.$field - rhs,)+
                }
            }
        }

        impl<T: RealNum<T>> SubAssign for $name<T> {
            fn sub_assign(&mut self, rhs: Self) {
                $(self.$field -= rhs.$field;)+
            }
        }

        impl<T: RealNum<T>> SubAssign<T> for $name<T> {
            fn sub_assign(&mut self, rhs: T) {
                $(self.$field -= rhs;)+
            }
        }

        impl<T: RealNum<T>> SubAssign<&$name<T>> for $name<T> {
            fn sub_assign(&mut self, rhs: &$name<T>) {
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

        impl<T: RealNum<T>> Mul for &$name<T> {
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

        impl<T: RealNum<T>> Mul<T> for &$name<T> {
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

        impl<T: RealNum<T>> MulAssign<T> for $name<T> {
            fn mul_assign(&mut self, rhs: T) {
                $(self.$field *= rhs;)+
            }
        }

        impl<T: RealNum<T>> MulAssign<&$name<T>> for $name<T> {
            fn mul_assign(&mut self, rhs: &$name<T>) {
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

        impl<T: RealNum<T>> Div for &$name<T> {
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

        impl<T: RealNum<T>> Div<T> for &$name<T> {
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

        impl<T: RealNum<T>> DivAssign<T> for $name<T> {
            fn div_assign(&mut self, rhs: T) {
                $(self.$field /= rhs;)+
            }
        }

        impl<T: RealNum<T>> DivAssign<&$name<T>> for $name<T> {
            fn div_assign(&mut self, rhs: &$name<T>) {
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

macro_rules! make_bounds {
    ($name:ident, $p:ident, $v:ident, $($field:ident),+) => {
        pub struct $name<T> {
            min: $p<T>,
            max: $p<T>,
        }

        impl<T: RealNum<T>> $name<T> {
            pub fn new() -> Self {
                Self {
                    min: $p {
                        $($field: T::min_value(),)+
                    },
                    max: $p {
                        $($field: T::max_value(),)+
                    }
                }
            }

            pub fn diagonal(&self) -> $v<T> {
                &self.max - &self.min
            }

            pub fn lerp(&self, t: &$p<T>) -> $p<T> {
                $p {
                    $($field: super::lerp(t.$field, self.min.$field, self.max.$field),)+
                }
            }

            pub fn offset(&self, p: &$p<T>) -> $v<T> {
                let mut o = p - &self.min;

                $(if self.max.$field > self.min.$field {
                    o.$field /= self.max.$field - self.min.$field;
                })+
                o
            }

            pub fn inside(&self, pt: &$p<T>) -> bool {
                $(pt.$field >= self.min.$field && pt.$field <= self.max.$field) && +
            }

            pub fn bounding_sphere(&self, c: &mut $p<T>, rad: &mut T) {
                *c = (&self.min + &self.max) / T::two();
                if self.inside(c) {
                    *rad = c.distance(&self.max);
                } else {
                    *rad = T::zero();
                }
            }

            pub fn maximum_extent(&self) -> usize {
                let diag = self.diagonal();
                diag.max_dimension()
            }

        }

        impl<T: RealNum<T>> From<$p<T>> for $name<T> {
            fn from(p: $p<T>) -> Self {
                Self {
                    min: p.clone(),
                    max: p,
                }
            }
        }
        impl<T: RealNum<T>> From<($p<T>, $p<T>)> for $name<T> {
            fn from(ps: ($p<T>, $p<T>)) -> Self {
                Self {
                    min: $p{
                        $($field:ps.0.$field.min(ps.1.$field),)+
                    },
                    max: $p{
                        $($field:ps.0.$field.max(ps.1.$field),)+
                    },
                }
            }
        }
        impl<T:RealNum<T>> Index<usize> for $name<T> {
            type Output = $p<T>;
            fn index(&self, index:usize) -> &Self::Output {
                match index {
                   0 => &self.min,
                   1 => &self.max,
                   _ => panic!("Out of index"),
                }
            }
        }

        impl<T: RealNum<T>> IndexMut<usize> for $name<T> {
            fn index_mut(&mut self, index:usize) ->&mut Self::Output {
                match index {
                    0 => &mut self.min,
                    1 => &mut self.max,
                    _ => panic!("Out of index"),
                }
            }
        }
    };
}

make_bounds!(Bounds2, Point2, Vector2, x, y);
pub type Bounds2f = Bounds2<Float>;

make_bounds!(Bounds3, Point3, Vector3, x, y, z);
impl<T: RealNum<T>> Bounds3<T> {
    pub fn corner(&self, corner: usize) -> Point3<T> {
        Point3 {
            x: self[(corner & 1)].x,
            y: self[if corner & 2 != 0 { 1 } else { 0 }].y,
            z: self[if corner & 4 != 0 { 1 } else { 0 }].z,
        }
    }

    pub fn surface_area(&self) -> T {
        let d = self.diagonal();
        T::two() * (d.x * d.y + d.x * d.z + d.y * d.z)
    }
    pub fn volume(&self) -> T {
        let d = self.diagonal();
        d.x * d.y * d.z
    }
}
pub type Bounds3f = Bounds3<Float>;

pub struct Differentials {
    rx_origin: Point3f,
    ry_origin: Point3f,
    rx_direction: Vector3f,
    ry_direction: Vector3f,
}

pub struct Ray {
    o: Point3f,
    d: Vector3f,
    pub t_max: Float,
    time: Float,
    differentials: Option<Differentials>,
}

impl Ray {
    pub fn default() -> Ray {
        Ray {
            o: Point3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            d: Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            t_max: Float::max_value(),
            time: 0.0,
            differentials: None,
        }
    }

    pub fn new(o: Point3f, d: Vector3f, t_max: Float, time: Float) -> Ray {
        Ray {
            o,
            d,
            t_max,
            time,
            differentials: None,
        }
    }

    pub fn point(&self, t: Float) -> Point3f {
        &self.o + &(&self.d * t)
    }

    pub fn scale_differentials(&mut self, s: Float) {
        if let Some(diff) = self.differentials.as_mut() {
            diff.rx_origin = &self.o + &((&diff.rx_origin - &self.o) * s);
            diff.ry_origin = &self.o + &((&diff.ry_origin - &self.o) * s);
            diff.rx_direction = &self.d + &((&diff.rx_direction - &self.d) * s);
            diff.ry_direction = &self.d + &((&diff.ry_direction - &self.d) * s);
        }
    }

    pub fn has_differentials(&self) -> bool {
        self.differentials.is_some()
    }
}

pub struct SurfaceInteraction {}
