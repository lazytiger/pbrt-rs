use crate::core::shape::Shape;
use crate::core::transform::Transformf;
use crate::impl_base_shape;
use crate::shapes::BaseShape;

pub struct Sphere {
    base: BaseShape,
}

impl Shape for Sphere {
    impl_base_shape!();
}
