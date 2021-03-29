use crate::core::geometry::{offset_ray_origin, Point3f};
use crate::core::transform::Transformf;
use crate::Float;
use std::cmp::Ordering;

struct SDVertex {
    p: Point3f,
    index: i32,
    start_face: i32,
    child: i32,
    regular: bool,
    boundary: bool,
}

impl SDVertex {
    fn new(p: Point3f) -> Self {
        Self {
            p,
            index: -1,
            start_face: -1,
            child: -1,
            regular: false,
            boundary: false,
        }
    }

    fn valence(&self, faces: &[SDFace]) -> i32 {
        if !self.boundary {
            let mut f = self.start_face;
            let mut nf = 1;
            loop {
                f = faces[f as usize].next_face(self.index);
                if f == self.start_face {
                    break;
                }
                nf += 1;
            }
            nf
        } else {
            let mut f = self.start_face;
            let mut nf = 1;
            loop {
                f = faces[f as usize].next_face(self.index);
                if f < 0 {
                    break;
                }
                nf += 1;
            }
            let mut f = self.start_face;
            loop {
                f = faces[f as usize].prev_face(self.index);
                if f < 0 {
                    break;
                }
                nf += 1;
            }
            nf + 1
        }
    }
}

#[derive(Copy, Clone)]
struct SDFace {
    v: [i32; 3],
    f: [i32; 3],
    children: [i32; 4],
}

impl Default for SDFace {
    fn default() -> Self {
        Self {
            v: [-1; 3],
            f: [-1; 3],
            children: [-1; 4],
        }
    }
}

macro_rules! next {
    ($i:expr) => {
        ((($i) + 1) % 3)
    };
}

macro_rules! prev {
    ($i:expr) => {
        ((($i) + 2) % 3)
    };
}

impl SDFace {
    fn vnum(&self, vert: i32) -> i32 {
        for i in 0..3 {
            if self.v[i] == vert {
                return i as i32;
            }
        }
        return -1;
    }

    fn next_face(&self, vert: i32) -> i32 {
        self.f[self.vnum(vert) as usize]
    }

    fn prev_face(&self, vert: i32) -> i32 {
        self.f[prev!(self.vnum(vert)) as usize]
    }

    fn next_vert(&self, vert: i32) -> i32 {
        self.v[next!(self.vnum(vert)) as usize]
    }

    fn prev_vert(&self, vert: i32) -> i32 {
        self.v[prev!(self.vnum(vert)) as usize]
    }

    fn other_vert(&self, v1: i32, v2: i32) -> i32 {
        for i in 0..3 {
            if self.v[i] != v1 && self.v[i] != v2 {
                return self.v[i];
            }
        }
        -1
    }
}

#[derive(Default, Clone, Copy)]
struct SDEdge {
    v: [i32; 2],
    f: [i32; 2],
    f0_edge_num: i32,
}

impl SDEdge {
    fn new(v0: i32, v1: i32) -> Self {
        let mut edge = SDEdge::default();
        edge.v[0] = v0.min(v1);
        edge.v[1] = v1.max(v0);
        edge.f[0] = -1;
        edge.f[1] = -1;
        edge.f0_edge_num = -1;
        edge
    }
}

impl PartialEq for SDEdge {
    fn eq(&self, other: &Self) -> bool {
        self.v[0] == other.v[0] && self.v[1] == other.v[1]
    }
}

impl PartialOrd for SDEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.v[0] == other.v[0] {
            if self.v[1] == other.v[1] {
                Some(Ordering::Equal)
            } else if self.v[1] > other.v[1] {
                Some(Ordering::Greater)
            } else {
                Some(Ordering::Less)
            }
        } else if self.v[0] < other.v[0] {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

#[inline]
fn beta(valence: i32) -> Float {
    if valence == 3 {
        3.0 / 16.0
    } else {
        3.0 / (8.0 * valence as Float)
    }
}

#[inline]
fn loop_gamma(valence: i32) -> Float {
    1.0 / (valence as Float + 3.0 / (8.0 * beta(valence)))
}

fn loop_subdivide(
    o2w: Transformf,
    w2o: Transformf,
    ro: bool,
    n_levels: i32,
    vertex_indices: Vec<i32>,
    p: &[Point3f],
) {
}
