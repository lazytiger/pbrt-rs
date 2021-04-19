use crate::core::geometry::Point2i;

#[macro_export]
macro_rules! parallel_for_2d {
    ($func:expr, $count:expr) => {
        rayon::scope(|s| {
            for y in 0..$count.y {
                for x in 0..$count.x {
                    let p = Arc::new(Box::new(Point2i::new(x, y)));
                    s.spawn(|_| $func(p));
                }
            }
        });
    };
    ($func:expr, $count:expr, ) => {
        parallel_for_2d!($func, $count);
    };
}

/*
pub fn parallel_for_2d<F: Fn(Point2i)>(func: F, count: &Point2i) {
    for y in 0..count.y {
        for x in 0..count.x {
            func(Point2i::new(x, y));
        }
    }
}
 */

#[macro_export]
macro_rules! parallel_for {
    ($func:expr, $count:expr, $chunk_size:expr) => {
        let max_j = $count / $chunk_size;
        let remaining_j = $count % $chunk_size;
        rayon::scope(|s| {
            for j in 0..max_j {
                let n = Arc::new(j);
                s.spawn(|_| {
                    let j = unsafe { *Arc::into_raw(n) };
                    for i in 0..$chunk_size {
                        let index = j * $chunk_size + i;
                        $func(index);
                    }
                });
            }

            s.spawn(|_| {
                for i in 0..remaining_j {
                    let index = max_j * $chunk_size + i;
                    $func(index);
                }
            });
        });
    };
    ($func:expr, $count:expr) => {
        parallel_for!($func, $count, 1);
    };
    ($func:expr, $count:expr, ) => {
        parallel_for!($func, $count, 1);
    };
    ($func:expr, $count:expr,$chunk_size:expr, ) => {
        parallel_for!($func, $count, $chunk_size);
    };
}

/*
pub fn parallel_for<F: Fn(usize)>(func: F, count: usize, chunk_size: usize) {
    for i in 0..count {
        func(i)
    }
}
 */
