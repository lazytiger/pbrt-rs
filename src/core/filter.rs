use crate::core::geometry::{Point2f, Vector2f};
use crate::Float;
use std::any::Any;

pub trait Filter {
    fn as_any(&self) -> &dyn Any;
    fn evaluate(&self, p: &Point2f) -> Float;
}

pub(crate) struct BaseFilter {
    radius: Vector2f,
    inv_radius: Vector2f,
}
