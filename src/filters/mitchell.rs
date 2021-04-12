use crate::{
    core::{
        filter::{BaseFilter, Filter},
        geometry::{Point2f, Vector2f},
        pbrt::Float,
    },
    impl_base_filter,
};
use std::fs::File;

pub struct MitchellFilter {
    base: BaseFilter,
    b: Float,
    c: Float,
}

impl MitchellFilter {
    pub fn new(radius: Vector2f, b: Float, c: Float) -> Self {
        Self {
            base: BaseFilter::new(radius),
            b,
            c,
        }
    }

    pub fn mitchell_1d(&self, x: Float) -> Float {
        let x = (2.0 * x).abs();
        if x > 1.0 {
            ((-self.b - 6.0 * self.c) * x * x * x
                + (6.0 * self.b + 30.0 * self.c) * x * x
                + (-12.0 * self.b - 48.0 * self.c) * x
                + (8.0 * self.b + 24.0 * self.c))
                * (1.0 / 6.0)
        } else {
            ((12.0 - 9.0 * self.b - 6.0 * self.c) * x * x * x
                + (-18.0 + 12.0 * self.b + 6.0 * self.c) * x * x
                + (6.0 - 2.0 * self.b))
                * (1.0 / 6.0)
        }
    }
}

impl Filter for MitchellFilter {
    impl_base_filter!();

    fn evaluate(&self, p: &Point2f) -> f32 {
        self.mitchell_1d(p.x * self.base.inv_radius.x)
            * self.mitchell_1d(p.y * self.base.inv_radius.y)
    }
}
