use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub,
    SubAssign,
};

use super::RealNum;
use crate::core::efloat::EFloat;
use crate::core::medium::Medium;
use crate::core::transform::{AnimatedTransform, Point3Ref, Transform, Transformf, Vector3Ref};
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

            pub fn has_nans(&self) -> bool {
                $(self.$field.is_nan() )||+
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

impl From<Point2i> for Point2f {
    fn from(p: Point2i) -> Self {
        Self {
            x: p.x as Float,
            y: p.y as Float,
        }
    }
}

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

impl<'a> From<(&AnimatedTransform, Float, Point3Ref<'a, Float>)> for Point3f {
    fn from(data: (&AnimatedTransform, f32, Point3Ref<'_, Float>)) -> Self {
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

impl<'a> From<(&AnimatedTransform, Float, Vector3Ref<'a, Float>)> for Vector3f {
    fn from(data: (&AnimatedTransform, f32, Vector3Ref<'a, f32>)) -> Self {
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
pub type Bounds2i = Bounds2<i32>;

impl From<Bounds2i> for Bounds2f {
    fn from(bounds: Bounds2i) -> Self {
        Self {
            min: Point2f::new(bounds.min.x as Float, bounds.min.y as Float),
            max: Point2f::new(bounds.max.x as Float, bounds.max.y as Float),
        }
    }
}

impl From<Bounds2f> for Bounds2i {
    fn from(bounds: Bounds2f) -> Self {
        Self {
            min: Point2i::new(bounds.min.x as i32, bounds.min.y as i32),
            max: Point2i::new(bounds.max.x as i32, bounds.max.y as i32),
        }
    }
}

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
    fn intersect_p(&self, data: T) -> Self::Output;
}

impl IntersectP<&Ray> for Bounds3f {
    type Output = (bool, Float, Float);

    fn intersect_p(&self, ray: &Ray) -> Self::Output {
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

    fn intersect_p(&self, data: (&Ray, &Vector3f, [usize; 3])) -> Self::Output {
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

#[derive(Clone, Default)]
pub struct Ray {
    pub o: Point3f,
    pub d: Vector3f,
    pub t_max: Float,
    pub time: Float,
    pub medium: Option<Arc<Box<Medium>>>,
}

#[derive(Clone, Default)]
pub struct RayDifferentials {
    base: Ray,
    pub has_differentials: bool,
    pub rx_origin: Point3f,
    pub ry_origin: Point3f,
    pub rx_direction: Vector3f,
    pub ry_direction: Vector3f,
}

impl Ray {
    pub fn new(
        o: Point3f,
        d: Vector3f,
        t_max: Float,
        time: Float,
        medium: Option<Arc<Box<Medium>>>,
    ) -> Ray {
        Ray {
            o,
            d,
            t_max,
            time,
            medium,
        }
    }

    pub fn point(&self, t: Float) -> Point3f {
        self.o + (self.d * t)
    }

    pub fn has_nans(&self) -> bool {
        self.o.has_nans() || self.d.has_nans() && self.t_max.is_nan()
    }

    pub fn efloats(
        &self,
        o_err: &Vector3f,
        d_err: &Vector3f,
    ) -> (EFloat, EFloat, EFloat, EFloat, EFloat, EFloat) {
        (
            EFloat::new(self.o.x, o_err.x),
            EFloat::new(self.o.y, o_err.y),
            EFloat::new(self.o.z, o_err.z),
            EFloat::new(self.d.x, d_err.x),
            EFloat::new(self.d.y, d_err.y),
            EFloat::new(self.d.z, d_err.z),
        )
    }
}

impl Deref for RayDifferentials {
    type Target = Ray;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DerefMut for RayDifferentials {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl RayDifferentials {
    pub fn new(
        o: Point3f,
        d: Vector3f,
        t_max: Float,
        time: Float,
        medium: Option<Arc<Box<Medium>>>,
    ) -> RayDifferentials {
        Self {
            base: Ray::new(o, d, t_max, time, medium),
            has_differentials: false,
            rx_origin: Default::default(),
            ry_origin: Default::default(),
            rx_direction: Default::default(),
            ry_direction: Default::default(),
        }
    }
    pub fn scale_differentials(&mut self, s: Float) {
        self.rx_origin = self.o + ((self.rx_origin - self.o) * s);
        self.ry_origin = self.o + ((self.ry_origin - self.o) * s);
        self.rx_direction = self.d + ((self.rx_direction - self.d) * s);
        self.ry_direction = self.d + ((self.ry_direction - self.d) * s);
    }

    pub fn has_nans(&self) -> bool {
        self.base.has_nans()
            || self.has_differentials
                && (self.rx_origin.has_nans()
                    || self.rx_origin.has_nans()
                    || self.rx_direction.has_nans()
                    || self.ry_direction.has_nans())
    }
}

impl From<Ray> for RayDifferentials {
    fn from(ray: Ray) -> Self {
        Self {
            base: ray,
            has_differentials: false,
            rx_origin: Default::default(),
            ry_origin: Default::default(),
            rx_direction: Default::default(),
            ry_direction: Default::default(),
        }
    }
}

impl From<(&Transformf, &Ray)> for Ray {
    fn from(data: (&Transformf, &Ray)) -> Self {
        let t = data.0;
        let r = data.1;
        let mut o_error = Vector3f::default();
        let mut o = Point3f::from((t, Point3Ref(&r.o), &mut o_error));
        let d = t * Vector3Ref(&r.d);
        let length_squared = d.length_squared();
        let mut t_max = r.t_max;
        if length_squared > 0.0 {
            let dt = d.abs().dot(&o_error) / length_squared;
            o += d * dt;
            t_max -= dt;
        }
        Ray::new(o, d, t_max, r.time, r.medium.clone())
    }
}

impl From<(&Transformf, &RayDifferentials)> for RayDifferentials {
    fn from(data: (&Transformf, &RayDifferentials)) -> Self {
        let t = data.0;
        let r = data.1;
        let tr = Ray::from((t, &r.base));
        let mut ret = RayDifferentials::new(tr.o, tr.d, tr.t_max, tr.time, tr.medium);
        ret.has_differentials = r.has_differentials;
        ret.rx_origin = t * Point3Ref(&r.rx_origin);
        ret.ry_origin = t * Point3Ref(&r.ry_origin);
        ret.rx_direction = t * Vector3Ref(&r.rx_direction);
        ret.ry_direction = t * Vector3Ref(&r.ry_direction);
        ret
    }
}

impl<'a, T: RealNum<T>> From<(&Transform<T>, Point3Ref<'a, T>, &mut Vector3<T>)> for Point3<T> {
    fn from(data: (&Transform<T>, Point3Ref<'a, T>, &mut Vector3<T>)) -> Self {
        let t = data.0;
        let p = data.1 .0;
        let p_error = data.2;

        let x = p.x;
        let y = p.y;
        let z = p.z;

        let xp = t.m.m[0][0] * x + t.m.m[0][1] * y + t.m.m[0][2] * z + t.m.m[0][3];
        let yp = t.m.m[1][0] * x + t.m.m[1][1] * y + t.m.m[1][2] * z + t.m.m[1][3];
        let zp = t.m.m[2][0] * x + t.m.m[2][1] * y + t.m.m[2][2] * z + t.m.m[2][3];
        let wp = t.m.m[3][0] * x + t.m.m[3][1] * y + t.m.m[3][2] * z + t.m.m[3][3];

        let x_abs_sum = (t.m.m[0][0] * x).abs()
            + (t.m.m[0][1] * y).abs()
            + (t.m.m[0][2] * z).abs()
            + t.m.m[0][3].abs();
        let y_abs_sum = (t.m.m[1][0] * x).abs()
            + (t.m.m[1][1] * y).abs()
            + (t.m.m[1][2] * z).abs()
            + t.m.m[1][3].abs();
        let z_abs_sum = (t.m.m[2][0] * x).abs()
            + (t.m.m[2][1] * y).abs()
            + (t.m.m[2][2] * z).abs()
            + t.m.m[2][3].abs();
        *p_error = Vector3::new(x_abs_sum, y_abs_sum, z_abs_sum) * gamma(T::three());
        if wp == T::one() {
            Point3::new(xp, yp, zp)
        } else {
            Point3::new(xp, yp, zp) / wp
        }
    }
}

impl<'a, T: RealNum<T>>
    From<(
        &Transform<T>,
        Point3Ref<'a, T>,
        &Vector3<T>,
        &mut Vector3<T>,
    )> for Point3<T>
{
    fn from(
        data: (
            &Transform<T>,
            Point3Ref<'a, T>,
            &Vector3<T>,
            &mut Vector3<T>,
        ),
    ) -> Self {
        let t = data.0;
        let pt = data.1 .0;
        let pt_error = data.2;
        let abs_error = data.3;

        let x = pt.x;
        let y = pt.y;
        let z = pt.z;

        let xp = t.m.m[0][0] * x + t.m.m[0][1] * y + t.m.m[0][2] * z + t.m.m[0][3];
        let yp = t.m.m[1][0] * x + t.m.m[1][1] * y + t.m.m[1][2] * z + t.m.m[1][3];
        let zp = t.m.m[2][0] * x + t.m.m[2][1] * y + t.m.m[2][2] * z + t.m.m[2][3];
        let wp = t.m.m[3][0] * x + t.m.m[3][1] * y + t.m.m[3][2] * z + t.m.m[3][3];
        abs_error.x = (gamma(T::three()) + T::one())
            * ((t.m.m[0][0] * pt_error.x).abs()
                + (t.m.m[0][1] * pt_error.y).abs()
                + (t.m.m[0][2] * pt_error.z).abs())
            + gamma(T::three())
                * ((t.m.m[0][0] * x).abs()
                    + (t.m.m[0][1] * y).abs()
                    + (t.m.m[0][2] * z).abs()
                    + t.m.m[0][3].abs());
        abs_error.y = (gamma(T::three()) + T::one())
            * ((t.m.m[1][0] * pt_error.x).abs()
                + (t.m.m[1][1] * pt_error.y).abs()
                + (t.m.m[1][2] * pt_error.z).abs())
            + gamma(T::three())
                * ((t.m.m[1][0] * x).abs()
                    + (t.m.m[1][1] * y).abs()
                    + (t.m.m[1][2] * z).abs()
                    + t.m.m[1][3].abs());
        abs_error.z = (gamma(T::three()) + T::one())
            * ((t.m.m[2][0] * pt_error.x).abs()
                + (t.m.m[2][1] * pt_error.y).abs()
                + (t.m.m[2][2] * pt_error.z).abs())
            + gamma(T::three())
                * ((t.m.m[2][0] * x).abs()
                    + (t.m.m[2][1] * y).abs()
                    + (t.m.m[2][2] * z).abs()
                    + t.m.m[2][3].abs());

        if wp == T::one() {
            Point3::new(xp, yp, zp)
        } else {
            Point3::new(xp, yp, zp) / wp
        }
    }
}

impl<'a, T: RealNum<T>> From<(&Transform<T>, Vector3Ref<'a, T>, &mut Vector3<T>)> for Vector3<T> {
    fn from(data: (&Transform<T>, Vector3Ref<'a, T>, &mut Vector3<T>)) -> Self {
        let t = data.0;
        let v = data.1 .0;
        let abs_error = data.2;
        let x = v.x;
        let y = v.y;
        let z = v.z;
        abs_error.x = gamma(T::three())
            * ((t.m.m[0][0] * x).abs() + (t.m.m[0][1] * y).abs() + (t.m.m[0][2] * z).abs());
        abs_error.y = gamma(T::three())
            * ((t.m.m[1][0] * x).abs() + (t.m.m[1][1] * y).abs() + (t.m.m[1][2] * z).abs());
        abs_error.z = gamma(T::three())
            * ((t.m.m[2][0] * x).abs() + (t.m.m[2][1] * y).abs() + (t.m.m[2][2] * z).abs());

        let xp = t.m.m[0][0] * x + t.m.m[0][1] * y + t.m.m[0][2] * z;
        let yp = t.m.m[1][0] * x + t.m.m[1][1] * y + t.m.m[1][2] * z;
        let zp = t.m.m[2][0] * x + t.m.m[2][1] * y + t.m.m[2][2] * z;

        Vector3::new(xp, yp, zp)
    }
}

impl<'a, T: RealNum<T>>
    From<(
        &Transform<T>,
        Vector3Ref<'a, T>,
        &Vector3<T>,
        &mut Vector3<T>,
    )> for Vector3<T>
{
    fn from(
        data: (
            &Transform<T>,
            Vector3Ref<'a, T>,
            &Vector3<T>,
            &mut Vector3<T>,
        ),
    ) -> Self {
        let t = data.0;
        let v = data.1 .0;
        let v_error = data.2;
        let abs_error = data.3;

        let x = v.x;
        let y = v.y;
        let z = v.z;

        let xp = t.m.m[0][0] * x + t.m.m[0][1] * y + t.m.m[0][2] * z;
        let yp = t.m.m[1][0] * x + t.m.m[1][1] * y + t.m.m[1][2] * z;
        let zp = t.m.m[2][0] * x + t.m.m[2][1] * y + t.m.m[2][2] * z;
        abs_error.x = (gamma(T::three()) + T::one())
            * ((t.m.m[0][0] * v_error.x).abs()
                + (t.m.m[0][1] * v_error.y).abs()
                + (t.m.m[0][2] * v_error.z).abs())
            + gamma(T::three())
                * ((t.m.m[0][0] * x).abs() + (t.m.m[0][1] * y).abs() + (t.m.m[0][2] * z).abs());
        abs_error.y = (gamma(T::three()) + T::one())
            * ((t.m.m[1][0] * v_error.x).abs()
                + (t.m.m[1][1] * v_error.y).abs()
                + (t.m.m[1][2] * v_error.z).abs())
            + gamma(T::three())
                * ((t.m.m[1][0] * x).abs() + (t.m.m[1][1] * y).abs() + (t.m.m[1][2] * z).abs());
        abs_error.z = (gamma(T::three()) + T::one())
            * ((t.m.m[2][0] * v_error.x).abs()
                + (t.m.m[2][1] * v_error.y).abs()
                + (t.m.m[2][2] * v_error.z).abs())
            + gamma(T::three())
                * ((t.m.m[2][0] * x).abs() + (t.m.m[2][1] * y).abs() + (t.m.m[2][2] * z).abs());

        Vector3::new(xp, yp, zp)
    }
}

impl From<(&Transformf, &Ray, &mut Vector3f, &mut Vector3f)> for Ray {
    fn from(data: (&Transformf, &Ray, &mut Vector3f, &mut Vector3f)) -> Self {
        let t = data.0;
        let r = data.1;
        let o_error = data.2;
        let d_error = data.3;

        let mut error = Vector3f::default();
        let mut o = Point3f::from((t, Point3Ref(&r.o), &mut error));
        *o_error = error;
        let d = Vector3f::from((t, Vector3Ref(&r.d), d_error));
        let t_max = r.t_max;
        let length_squared = d.length_squared();
        if length_squared > 0.0 {
            let dt = d.abs().dot(o_error) / length_squared;
            o += d * dt;
        }
        Ray::new(o, d, t_max, r.time, r.medium.clone())
    }
}

impl
    From<(
        &Transformf,
        &Ray,
        &Vector3f,
        &Vector3f,
        &mut Vector3f,
        &mut Vector3f,
    )> for Ray
{
    fn from(
        data: (
            &Transformf,
            &Ray,
            &Vector3f,
            &Vector3f,
            &mut Vector3f,
            &mut Vector3f,
        ),
    ) -> Self {
        let t = data.0;
        let r = data.1;
        let o_error_in = data.2;
        let d_error_in = data.3;
        let o_error_out = data.4;
        let d_error_out = data.5;

        let mut error = Vector3f::default();
        let mut o = Point3f::from((t, Point3Ref(&r.o), o_error_in, &mut error));
        *o_error_out = error;
        let d = Vector3f::from((t, Vector3Ref(&r.d), d_error_in, d_error_out));
        let t_max = r.t_max;
        let length_squared = d.length_squared();
        if length_squared > 0.0 {
            let dt = d.abs().dot(o_error_out) / length_squared;
            o += d * dt;
        }
        Ray::new(o, d, t_max, r.time, r.medium.clone())
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

pub fn spherical_direction(
    sin_theta: Float,
    cos_theta: Float,
    phi: Float,
    x: Vector3f,
    y: Vector3f,
    z: Vector3f,
) -> Vector3f {
    x * sin_theta * phi.cos() + y * sin_theta * phi.sin() + z * cos_theta
}

pub fn spherical_direction2(sin_theta: Float, cos_theta: Float, phi: Float) -> Vector3f {
    Vector3f::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

impl<T> From<Vector3<T>> for Vector2<T> {
    fn from(v: Vector3<T>) -> Self {
        Self { x: v.x, y: v.y }
    }
}

impl From<(&AnimatedTransform, &Ray)> for Ray {
    fn from(data: (&AnimatedTransform, &Ray)) -> Ray {
        let at = data.0;
        let r = data.1;
        if !at.actually_animated || r.time <= at.start_time {
            (&at.start_transform, r).into()
        } else if r.time > at.end_time {
            (&at.end_transform, r).into()
        } else {
            let t = at.interpolate(r.time);
            (&t, r).into()
        }
    }
}

impl From<(&AnimatedTransform, &RayDifferentials)> for RayDifferentials {
    fn from(data: (&AnimatedTransform, &RayDifferentials)) -> Self {
        let at = data.0;
        let r = data.1;
        let at = data.0;
        let r = data.1;
        if !at.actually_animated || r.time <= at.start_time {
            (&at.start_transform, r).into()
        } else if r.time > at.end_time {
            (&at.end_transform, r).into()
        } else {
            let t = at.interpolate(r.time);
            (&t, r).into()
        }
    }
}
