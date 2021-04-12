use crate::{
    core::{
        filter::{BaseFilter, Filter},
        geometry::{Point2f, Vector2f},
        pbrt::Float,
    },
    impl_base_filter,
};

pub struct BoxFilter {
    base: BaseFilter,
}

impl BoxFilter {
    pub fn new(radius: Vector2f) -> BoxFilter {
        Self {
            base: BaseFilter::new(radius),
        }
    }
}

impl Filter for BoxFilter {
    impl_base_filter!();

    fn evaluate(&self, p: &Point2f) -> Float {
        1.0
    }
}
