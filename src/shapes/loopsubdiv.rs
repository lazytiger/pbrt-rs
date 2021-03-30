use crate::core::geometry::{offset_ray_origin, Point3f, Vector3f};
use crate::core::transform::Transformf;
use crate::{Float, PI};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Default, Clone, Copy)]
struct SDVertex {
    p: Point3f,
    index: i32,
    start_face: i32,
    child: i32,
    regular: bool,
    boundary: bool,
}

impl SDVertex {
    fn new(p: Point3f, index: i32) -> Self {
        Self {
            p,
            index,
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

    fn one_ring(&self, p: &mut [Point3f], faces: &[SDFace], vertices: &[SDVertex]) {
        let mut index = 0;
        if self.boundary {
            let mut f = self.start_face as usize;
            loop {
                let v = faces[f].next_vert(self.index) as usize;
                p[index] = vertices[v].p;
                index += 1;
                let next_face = faces[f].next_face(self.index);
                if next_face == self.start_face {
                    break;
                }
                f = next_face as usize;
            }
        } else {
            let mut f = self.start_face as usize;
            loop {
                let next_face = faces[f].next_face(self.index);
                if next_face == -1 {
                    break;
                }
                f = next_face as usize;
            }
            let next_vert = faces[f].next_vert(self.index) as usize;
            p[index] = vertices[next_vert].p;
            index += 1;
            loop {
                let prev_vert = faces[f].prev_vert(self.index) as usize;
                p[index] = vertices[prev_vert].p;
                index += 1;
                let prev_face = faces[f].prev_face(self.index);
                if prev_face == -1 {
                    break;
                }
                f = prev_face as usize;
            }
        }
    }
}

#[derive(Copy, Clone)]
struct SDFace {
    index: i32,
    v: [i32; 3],
    f: [i32; 3],
    children: [i32; 4],
}

impl SDFace {
    fn new(index: i32) -> SDFace {
        Self {
            index,
            v: [-1; 3],
            f: [-1; 3],
            children: [-1; 4],
        }
    }
}

impl Default for SDFace {
    fn default() -> Self {
        Self {
            index: -1,
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

#[derive(Default, Clone, Copy, Hash)]
struct SDEdge {
    index: i32,
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

impl Eq for SDEdge {}

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

fn weight_one_ring(
    vertex: &SDVertex,
    beta: Float,
    faces: &[SDFace],
    vertices: &[SDVertex],
) -> Point3f {
    let valence = vertex.valence(faces) as usize;
    let mut rings = vec![Point3f::default(); valence];
    vertex.one_ring(rings.as_mut_slice(), faces, vertices);
    let mut p = vertex.p * (1.0 - valence as Float * beta);
    for i in 0..valence {
        p += rings[i] * beta;
    }
    p
}

fn loop_subdivide(
    o2w: Transformf,
    w2o: Transformf,
    ro: bool,
    n_levels: usize,
    vertex_indices: Vec<i32>,
    p: &[Point3f],
) {
    let mut vertices = Vec::with_capacity(p.len());
    for i in 0..p.len() {
        vertices.push(SDVertex::new(p[i], i as i32));
    }

    let n_faces = p.len() / 3;
    let mut faces = Vec::with_capacity(n_faces);
    for i in 0..n_faces {
        faces.push(SDFace::new(i as i32));
    }

    let mut vp = vertex_indices.as_slice();
    for i in 0..n_faces {
        let f = &mut faces[i];
        for j in 0..3 {
            let v = &mut vertices[vp[j] as usize];
            f.v[j] = v.index;
            v.start_face = f.index;
        }
        vp = &vp[3..];
    }

    let mut edges = HashSet::with_capacity(p.len());
    for i in 0..n_faces {
        for edge_num in 0..3 {
            let v0 = edge_num;
            let v1 = next!(edge_num);
            let mut e = SDEdge::new(v0, v1);
            if edges.contains(&e) {
                e = *edges.get(&e).unwrap();
                faces[e.f[0] as usize].f[e.f0_edge_num as usize] = faces[i].index;
                faces[i].f[edge_num as usize] = e.f[0];
                edges.remove(&e);
            } else {
                e.f[0] = faces[i].index;
                e.f0_edge_num = edge_num;
                edges.insert(e);
            }
        }
    }

    for i in 0..vertices.len() {
        let v = &mut vertices[i];

        let mut f = v.start_face;
        loop {
            f = faces[f as usize].next_face(v.index);
            if f == -1 || f != v.start_face {
                break;
            }
        }
        v.boundary = f == -1;
        if !v.boundary && v.valence(faces.as_slice()) == 6 {
            v.regular = true;
        } else if v.boundary && v.valence(faces.as_slice()) == 4 {
            v.regular = true;
        } else {
            v.regular = false;
        }
    }

    for i in 0..n_levels {
        let mut new_vertices = Vec::with_capacity(vertices.len() * 4);
        let mut new_faces = Vec::with_capacity(faces.len() * 4);

        for vertex in &mut vertices {
            let mut child = SDVertex::new(Point3f::default(), new_vertices.len() as i32);
            child.regular = vertex.regular;
            child.boundary = vertex.boundary;
            new_vertices.push(child);
            vertex.child = child.index;
        }

        for face in &mut faces {
            for k in 0..4 {
                let mut child = SDFace::default();
                child.index = new_faces.len() as i32;
                face.children[k] = child.index;
                new_faces.push(child);
            }
        }

        for vertex in &vertices {
            if !vertex.boundary {
                if vertex.regular {
                    new_vertices[vertex.child as usize].p =
                        weight_one_ring(vertex, 1.0 / 16.0, faces.as_slice(), vertices.as_slice())
                } else {
                    new_vertices[vertex.child as usize].p = weight_one_ring(
                        vertex,
                        beta(vertex.valence(faces.as_slice())),
                        faces.as_slice(),
                        vertices.as_slice(),
                    );
                }
            } else {
                new_vertices[vertex.child as usize].p =
                    weight_boundary(vertex, 1.0 / 8.0, faces.as_slice(), vertices.as_slice());
            }
        }

        let mut edge_verts = HashMap::new();
        for face in &faces {
            for k in 0..3 {
                let edge = SDEdge::new(face.v[k], face.v[next!(k)]);
                let vert = edge_verts.get(&edge);
                if let None = vert {
                    let mut vert = SDVertex::new(Point3f::default(), new_vertices.len() as i32);
                    vert.regular = true;
                    vert.boundary = face.f[k] == -1;
                    vert.start_face = face.children[3];
                    let v0 = &vertices[edge.v[0] as usize];
                    let v1 = &vertices[edge.v[1] as usize];
                    if vert.boundary {
                        vert.p = v0.p * 0.5;
                        vert.p += v1.p * 0.5;
                    } else {
                        vert.p = v0.p * (3.0 / 8.0);
                        vert.p += (v1.p * (3.0 / 8.0));
                        let other_v = face.other_vert(v0.index, v1.index) as usize;
                        vert.p += vertices[other_v].p * (1.0 / 8.0);
                        let other_v =
                            faces[face.f[k] as usize].other_vert(v0.index, v1.index) as usize;
                        vert.p += vertices[other_v].p * (1.0 / 8.0);
                    }
                    edge_verts.insert(edge, vert.index);
                    new_vertices.push(vert);
                }
            }
        }

        for vertex in &vertices {
            let vert_num = faces[vertex.start_face as usize].vnum(vertex.index);
            new_vertices[vertex.child as usize].start_face =
                faces[vertex.start_face as usize].children[vert_num as usize];
        }

        for face in &faces {
            for j in 0..3 {
                new_faces[face.children[3] as usize].f[j] = face.children[next!(j)];
                new_faces[face.children[j] as usize].f[next!(j)] = face.children[3];

                let f2 = face.f[j];
                new_faces[face.children[j] as usize].f[j] = if f2 == -1 {
                    -1
                } else {
                    let f2 = &faces[f2 as usize];
                    f2.children[f2.vnum(face.v[j]) as usize]
                };
                let f2 = face.f[prev!(j)];
                new_faces[face.children[j] as usize].f[prev!(j)] = if f2 == -1 {
                    -1
                } else {
                    let f2 = faces[f2 as usize];
                    f2.children[f2.vnum(face.v[j]) as usize]
                };
            }
        }

        for face in &faces {
            for j in 0..3 {
                new_faces[face.children[j] as usize].v[j] = vertices[face.v[j] as usize].child;
                let vert =
                    if let Some(vert) = edge_verts.get(&SDEdge::new(face.v[j], face.v[next!(j)])) {
                        *vert
                    } else {
                        -1
                    };
                new_faces[face.children[j] as usize].v[next!(j)] = vert;
                new_faces[face.children[next!(j)] as usize].v[j] = vert;
                new_faces[face.children[3] as usize].v[j] = vert;
            }
        }

        faces = new_faces;
        vertices = new_vertices;
    }

    let mut p_limit = vec![Point3f::default(); vertices.len()];
    for i in 0..vertices.len() {
        if vertices[i].boundary {
            p_limit[i] = weight_boundary(
                &vertices[i],
                1.0 / 5.0,
                faces.as_slice(),
                vertices.as_slice(),
            );
        } else {
            p_limit[i] = weight_one_ring(
                &vertices[i],
                loop_gamma(vertices[i].valence(faces.as_slice())),
                faces.as_slice(),
                vertices.as_slice(),
            );
        }
    }

    for i in 0..vertices.len() {
        vertices[i].p = p_limit[i];
    }

    let mut ns = Vec::with_capacity(vertices.len());
    let mut p_ring = vec![Point3f::default(); 16];
    for vertex in &vertices {
        let mut s = Vector3f::default();
        let mut t = Vector3f::default();
        let valence = vertex.valence(faces.as_slice());
        if valence as usize > p_ring.len() {
            p_ring.resize(valence as usize, Point3f::default());
        }
        vertex.one_ring(p_ring.as_mut_slice(), faces.as_slice(), vertices.as_slice());
        if !vertex.boundary {
            for j in 0..valence as usize {
                s += p_ring[j] * (2.0 * PI * j as Float / valence as Float).cos();
                t += p_ring[j] * (2.0 * PI * j as Float / valence as Float).sin();
            }
        } else {
            s = p_ring[valence as usize - 1] - p_ring[0];
            if valence == 2 {
                t = p_ring[0] + p_ring[1] - vertex.p * 2.0;
            } else if valence == 3 {
                t = p_ring[1] - vertex.p;
            } else if valence == 4 {
                t = p_ring[0] * -1.0
                    + p_ring[1] * 2.0
                    + p_ring[2] * 2.0
                    + p_ring[3] * -1.0
                    + vertex.p * -2.0;
            } else {
                let theta = PI / (valence as Float - 1.0);
                t = (p_ring[0] + p_ring[valence as usize - 1]) * theta.sin();
                for k in 1..valence as usize - 1 {
                    let wt = (2.0 * theta.cos() - 2.0) * (k as Float * theta).sin();
                    t += p_ring[k] * wt;
                }
                t = -t;
            }
        }
        ns.push(s.cross(&t));
    }

    //TODO
}

fn weight_boundary(
    vertex: &SDVertex,
    beta: Float,
    faces: &[SDFace],
    vertices: &[SDVertex],
) -> Point3f {
    let valence = vertex.valence(faces) as usize;
    let mut rings = vec![Point3f::default(); valence];
    vertex.one_ring(&mut rings, faces, vertices);
    let mut p = vertex.p * (1.0 - 2.0 * beta);
    p += rings[0] * beta;
    p += rings[valence - 1] * beta;
    p
}
