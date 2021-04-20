use crate::core::geometry::Point2i;

#[macro_export]
macro_rules! parallel_for_2d {
    ($func:expr, $count:expr) => {
        rayon::scope(|s| {
            for y in 0..$count.y {
                for x in 0..$count.x {
                    let p = Arc::new(Point2i::new(x, y));
                    s.spawn(|_| {
                        let p = Arc::try_unwrap(p).unwrap();
                        $func(p);
                    });
                }
            }
        });
    };
    ($func:expr, $count:expr, ) => {
        parallel_for_2d!($func, $count);
    };
}

#[macro_export]
macro_rules! parallel_for {
    ($func:expr, $count:expr, $chunk_size:expr) => {
        let max_j = $count / $chunk_size;
        let remaining_j = $count % $chunk_size;
        rayon::scope(|s| {
            for j in 0..max_j {
                let n = Arc::new(j);
                s.spawn(|_| {
                    let j = Arc::try_unwrap(n).unwrap();
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
