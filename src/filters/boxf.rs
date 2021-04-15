use crate::{
    core::{
        filter::{BaseFilter, Filter, FilterDt},
        geometry::{Point2f, Vector2f},
        pbrt::Float,
    },
    impl_base_filter,
};
use std::sync::Arc;

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

    fn evaluate(&self, _p: &Point2f) -> Float {
        1.0
    }
}

pub fn create_box_filter() -> FilterDt {
    //Fixme
    Arc::new(Box::new(BoxFilter::new(Point2f::new(0.5, 0.05))))
}
