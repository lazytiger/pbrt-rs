use crate::{
    core::{
        filter::{BaseFilter, Filter},
        geometry::{Point2f, Vector2f},
    },
    impl_base_filter,
};

pub struct TriangleFilter {
    base: BaseFilter,
}

impl TriangleFilter {
    pub fn new(radius: Vector2f) -> Self {
        Self {
            base: BaseFilter::new(radius),
        }
    }
}

impl Filter for TriangleFilter {
    impl_base_filter!();

    fn evaluate(&self, p: &Point2f) -> f32 {
        (self.base.radius.x - p.x.abs()).max(0.0) * (self.base.radius.y - p.y.abs()).max(0.0)
    }
}
