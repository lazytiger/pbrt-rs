use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use super::RealNum;
use crate::core::medium::Medium;
use crate::core::transform::{AnimatedTransform, Point3Ref, Transformf};
use crate::core::{gamma, next_float_down, next_float_up};
use crate::Float;
use num::Bounded;
use std::mem::swap;
use std::sync::Arc;

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

        #[derive(Debug, Copy, Clone, Default)]
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
                (*self - *p).length()
            }

            pub fn distance_square(&self, p:&$name<T>) -> T {
                (*self - *p).length_squared()
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

            pub fn dot(&self, v:&$name<T>) -> T {
                *self * *v
            }

            pub fn abs_dot(&self, v:&$name<T>) -> T {
                self.dot(v).abs()
            }

            pub fn normalize(&self) -> $name<T> {
                *self / self.length()
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

            pub fn lerp(&self, t:T, v:$name<T>) -> $name<T> {
                *self * (T::one()-t) + v * t
            }

            pub fn floor(&self) -> $name<T> {
                Self {
                    $($field: self.$field.floor()),+
                }
            }

            pub fn ceil(&self) -> $name<T> {
                Self {
                    $($field: self.$field.ceil()),+
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

        impl<T: RealNum<T>> AddAssign<T> for $name<T> {
            fn add_assign(&mut self, rhs: T) {
                $(self.$field += rhs;)+
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

        impl<T: RealNum<T>> PartialEq for $name<T> {
            fn eq(&self, other: &Self) -> bool {
                $(self.$field == other.$field)&& +
            }
        }

        impl<T: RealNum<T>> Mul for $name<T> {
            type Output = T;

            fn mul(self, rhs: Self) -> Self::Output {
                strip_plus!($(+ self.$field * rhs.$field)+)
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

        impl<T: RealNum<T>> MulAssign<T> for $name<T> {
            fn mul_assign(&mut self, rhs: T) {
                $(self.$field *= rhs;)+
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

        impl<T: RealNum<T>> DivAssign<T> for $name<T> {
            fn div_assign(&mut self, rhs: T) {
                $(self.$field /= rhs;)+
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

impl<T: RealNum<T>> Vector3<T> {
    pub fn cross(&self, v: &Vector3<T>) -> Self {
        let v1x = self.x;
        let v1y = self.y;
        let v1z = self.z;
        let v2x = v.x;
        let v2y = v.y;
        let v2z = v.z;
        Vector3::new(
            v1y * v2z - v1z * v2y,
            v1z * v2x - v1x * v2z,
            v1x * v2y - v1y * v2x,
        )
    }

    pub fn coordinate_system(&self) -> (Vector3<T>, Vector3<T>) {
        let v2 = if self.x.abs() > self.y.abs() {
            Vector3::new(-self.z, T::zero(), self.x).normalize()
        } else {
            Vector3::new(T::zero(), self.z, -self.y).normalize()
        };
        let v3 = self.cross(&v2);
        (v2, v3)
    }

    pub fn face_forward(&self, v: Vector3<T>) -> Vector3<T> {
        if self.dot(&v) < T::zero() {
            -v
        } else {
            v
        }
    }
}

impl<'a> From<(&AnimatedTransform, Float, Point3Ref<'a, f32>)> for Point3f {
    fn from(data: (&AnimatedTransform, f32, Point3Ref<'_, f32>)) -> Self {
        let at = data.0;
        let time = data.1;
        let p = data.2;
        if !at.actually_animated || time <= at.start_time {
            &at.start_transform * p
        } else if time > at.end_time {
            &at.end_transform * p
        } else {
            let t = at.interpolate(time);
            &t * p
        }
    }
}

pub trait Union<T> {
    fn union(&self, u: T) -> Self;
}

macro_rules! make_bounds {
    ($name:ident, $p:ident, $v:ident, $($field:ident),+) => {
        #[derive(Copy, Clone)]
        pub struct $name<T> {
            pub min: $p<T>,
            pub max: $p<T>,
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
                self.max - self.min
            }

            pub fn lerp(&self, t: &$p<T>) -> $p<T> {
                $p {
                    $($field: super::lerp(t.$field, self.min.$field, self.max.$field),)+
                }
            }

            pub fn offset(&self, p: &$p<T>) -> $v<T> {
                let mut o = *p - self.min;

                $(if self.max.$field > self.min.$field {
                    o.$field /= self.max.$field - self.min.$field;
                })+
                o
            }

            pub fn inside(&self, pt: &$p<T>) -> bool {
                $(pt.$field >= self.min.$field && pt.$field <= self.max.$field) && +
            }

            pub fn bounding_sphere(&self, c: &mut $p<T>, rad: &mut T) {
                *c = (self.min + self.max) / T::two();
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

            pub fn intersect(&self, b:&Self) -> Self {
                Self {
                    min:self.min.max(&b.min),
                    max:self.max.min(&b.max),
                }
            }

            pub fn overlaps(&self, b:&Self) -> bool {
                $(if self.max.$field < b.min.x || self.min.x > b.max.x { return false })+
                true
            }

            pub fn inside_exclusive(&self, p:&$p<T>) -> bool {
                $(p.$field >= self.min.$field && p.$field < self.max.$field) && +
            }

            pub fn expand(&self, delta:T) -> Self {
                let delta = $v {
                    $($field:delta,)+
                };
                Self {
                    min: self.min - delta,
                    max: self.max + delta,
                }
            }

            pub fn distance_squared(&self, p:&$p<T>) -> T {
                $(let $field = T::zero().max(self.min.$field - p.$field).max(p.$field - self.max.$field);)+
                strip_plus!($(+ $field * $field) +)
            }

            pub fn distance(&self, p:&$p<T>) -> T {
                self.distance_squared(p).sqrt()
            }
        }

        impl<T: RealNum<T>> Union<&$name<T>> for $name<T> {
            fn union(&self, b:&Self) -> Self {
                Self {
                    min:self.min.min(&b.min),
                    max:self.max.max(&b.max),
                }
            }
        }

        impl<T: RealNum<T>> Union<&$p<T>> for $name<T> {
            fn union(&self, p:&$p<T>) -> Self {
                Self {
                    min:self.min.min(p),
                    max:self.max.max(p),
                }
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

        impl<T: RealNum<T>> Default for $name<T> {
            fn default() -> Self {
                Self::new()
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

pub trait IntersectP<T> {
    type Output;
    fn intersect(&self, data: T) -> Self::Output;
}

impl IntersectP<&Ray> for Bounds3f {
    type Output = (bool, Float, Float);

    fn intersect(&self, ray: &Ray) -> Self::Output {
        let mut t0 = 0.0;
        let mut t1 = ray.t_max;
        for i in 0..3 {
            let inv_ray_dir = 1.0 / ray.d[i];
            let mut t_near = (self.min[i] - ray.o[i]) * inv_ray_dir;
            let mut t_far = (self.max[i] - ray.o[i]) * inv_ray_dir;

            if t_near > t_far {
                swap(&mut t_near, &mut t_far);
            }

            t_far *= 1.0 + 2.0 * gamma(3.0);
            if t_near > t0 {
                t0 = t_near;
            }
            if t_far < t1 {
                t1 = t_far;
            }
        }
        (t0 > t1, t0, t1)
    }
}

impl IntersectP<(&Ray, &Vector3f, [usize; 3])> for Bounds3f {
    type Output = bool;

    fn intersect(&self, data: (&Ray, &Vector3f, [usize; 3])) -> Self::Output {
        let ray = data.0;
        let inv_dir = data.1;
        let dir_is_neg = data.2;

        let mut t_min = (self[dir_is_neg[0]].x - ray.o.x) * inv_dir.x;
        let mut t_max = (self[1 - dir_is_neg[0]].x - ray.o.x) * inv_dir.x;
        let mut ty_min = (self[dir_is_neg[1]].y - ray.o.y) * inv_dir.y;
        let mut ty_max = (self[1 - dir_is_neg[1]].y - ray.o.y) * inv_dir.y;

        t_max *= 1.0 + 2.0 * gamma(3.0);
        ty_max *= 1.0 + 2.0 * gamma(3.0);
        if t_min > ty_max || ty_min > t_max {
            return false;
        }

        if ty_min > t_min {
            t_min = ty_min;
        }
        if ty_max < t_max {
            t_max = ty_max;
        }

        let mut tz_min = (self[dir_is_neg[2]].z - ray.o.z) * inv_dir.z;
        let mut tz_max = (self[1 - dir_is_neg[2]].z - ray.o.z) * inv_dir.z;

        tz_max *= 1.0 + 2.0 + gamma(3.0);
        if t_min > tz_max || tz_min > t_max {
            return false;
        }
        if tz_min > t_min {
            t_min = tz_min;
        }
        if tz_max < t_max {
            t_max = tz_max;
        }

        t_min < ray.t_max && t_max > 0.0
    }
}

pub type Bounds3f = Bounds3<Float>;

#[derive(Copy, Clone, Default)]
pub struct Differentials {
    rx_origin: Point3f,
    ry_origin: Point3f,
    rx_direction: Vector3f,
    ry_direction: Vector3f,
}

#[derive(Clone, Default)]
pub struct Ray {
    pub o: Point3f,
    pub d: Vector3f,
    pub t_max: Float,
    pub time: Float,
    pub differentials: Option<Differentials>,
    pub medium: Option<Arc<Box<Medium>>>,
}

impl Ray {
    pub fn new(o: Point3f, d: Vector3f, t_max: Float, time: Float) -> Ray {
        Ray {
            o,
            d,
            t_max,
            time,
            differentials: None,
            medium: None,
        }
    }

    pub fn point(&self, t: Float) -> Point3f {
        self.o + (self.d * t)
    }

    pub fn scale_differentials(&mut self, s: Float) {
        if let Some(diff) = self.differentials.as_mut() {
            diff.rx_origin = self.o + ((diff.rx_origin - self.o) * s);
            diff.ry_origin = self.o + ((diff.ry_origin - self.o) * s);
            diff.rx_direction = self.d + ((diff.rx_direction - self.d) * s);
            diff.ry_direction = self.d + ((diff.ry_direction - self.d) * s);
        }
    }

    pub fn has_differentials(&self) -> bool {
        self.differentials.is_some()
    }
}

impl From<(&Transformf, &Ray)> for Ray {
    fn from(data: (&Transformf, &Ray)) -> Self {
        let t = data.0;
        let r = data.1;
        unimplemented!()
    }
}

pub fn offset_ray_origin(p: &Point3f, p_error: &Vector3f, n: &Normal3f, w: &Vector3f) -> Point3f {
    let d = n.abs().dot(p_error);
    let mut offset = *n * d;
    if w.dot(n) < 0.0 {
        offset = -offset;
    }
    let mut po = *p + offset;
    for i in 0..3 {
        if offset[i] > 0.0 {
            po[i] = next_float_up(po[i]);
        } else if offset[i] < 0.0 {
            po[i] = next_float_down(po[i]);
        }
    }
    po
}
