use crate::core::geometry::Point2i;

pub fn parallel_for_2d<F: Fn(Point2i)>(func: F, count: &Point2i) {
    for y in 0..count.y {
        for x in 0..count.x {
            func(Point2i::new(x, y));
        }
    }
}
