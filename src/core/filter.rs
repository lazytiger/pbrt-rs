use crate::core::geometry::{Point2f, Vector2f};
use crate::Float;
use std::any::Any;

pub trait Filter {
    fn as_any(&self) -> &dyn Any;
    fn evaluate(&self, p: &Point2f) -> Float;
    fn radius(&self) -> &Vector2f;
    fn inv_radius(&self) -> &Vector2f;
}

pub(crate) struct BaseFilter {
    radius: Vector2f,
    inv_radius: Vector2f,
}

#[macro_export]
macro_rules! impl_base_filter {
    () => {
        fn radius(&self) -> &Vector2f {
            &self.radius
        }

        fn inv_radius(&self) -> &Vector2f {
            &self.inv_radius
        }
    };
}
