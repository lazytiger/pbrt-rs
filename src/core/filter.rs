use crate::core::{
    geometry::{Point2f, Vector2f},
    pbrt::Float,
};
use std::any::Any;

pub trait Filter {
    fn as_any(&self) -> &dyn Any;
    fn evaluate(&self, p: &Point2f) -> Float;
    fn radius(&self) -> &Vector2f;
    fn inv_radius(&self) -> &Vector2f;
}

pub(crate) struct BaseFilter {
    pub radius: Vector2f,
    pub inv_radius: Vector2f,
}

impl BaseFilter {
    pub(crate) fn new(radius: Vector2f) -> BaseFilter {
        let inv_radius = Vector2f::new(1.0 / radius.x, 1.0 / radius.y);
        Self { radius, inv_radius }
    }
}

#[macro_export]
macro_rules! impl_base_filter {
    () => {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn radius(&self) -> &$crate::core::geometry::Vector2f {
            &self.base.radius
        }

        fn inv_radius(&self) -> &$crate::core::geometry::Vector2f {
            &self.base.inv_radius
        }
    };
}
