use crate::core::{
    geometry::{offset_ray_origin, Bounds3f, IntersectP, Point2f, Ray, Union, Vector3f},
    interaction::{Interaction, SurfaceInteraction},
    light::AreaLight,
    material::{Material, TransportMode},
    pbrt::{log_2_int_u64, Float},
    primitive::Primitive,
    shape::Shape,
    transform::Transformf,
};
use std::{any::Any, cmp::Ordering, sync::Arc};

#[derive(Default, Copy, Clone)]
struct KdTodo<'a> {
    node: Option<&'a KdAccelNode>,
    idx: usize,
    t_min: Float,
    t_max: Float,
}

struct KdTreeAccel {
    isect_cost: i32,
    traversal_cost: i32,
    max_prims: usize,
    empty_bonus: Float,
    primitives: Vec<Arc<Box<dyn Primitive>>>,
    primitive_indices: Vec<usize>,
    nodes: Vec<KdAccelNode>,
    n_alloced_nodes: usize,
    next_free_node: usize,
    bounds: Bounds3f,
}

#[repr(C)]
#[derive(Copy, Clone)]
union NodeData {
    split: Float,
    one_primitive: usize,
    primitive_indices_offset: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
union NodeFlag {
    flags: usize,
    n_prims: usize,
    above_child: usize,
}

#[derive(Copy, Clone)]
struct KdAccelNode {
    data: NodeData,
    flag: NodeFlag,
}

impl Default for KdAccelNode {
    fn default() -> Self {
        Self {
            data: NodeData { split: 0.0 },
            flag: NodeFlag { flags: 0 },
        }
    }
}

impl KdAccelNode {
    fn init_leaf(&mut self, prim_nums: &[usize], np: usize, primitive_indices: &mut Vec<usize>) {
        unsafe {
            self.flag.flags = 3;
            self.flag.n_prims |= (np << 2);
            if np == 0 {
                self.data.one_primitive = 0;
            } else if np == 1 {
                self.data.one_primitive = prim_nums[0];
            } else {
                self.data.primitive_indices_offset = primitive_indices.len();
                for i in 0..np {
                    primitive_indices.push(prim_nums[i]);
                }
            }
        }
    }

    fn init_interior(&mut self, axis: usize, ac: usize, s: Float) {
        unsafe {
            self.data.split = s;
            self.flag.flags = axis;
            self.flag.above_child |= (ac << 2);
        }
    }

    fn split_pos(&self) -> Float {
        unsafe { self.data.split }
    }

    fn n_primitives(&self) -> usize {
        unsafe { self.flag.n_prims >> 2 }
    }

    fn split_axis(&self) -> usize {
        unsafe { self.flag.flags & 3 }
    }

    fn is_leaf(&self) -> bool {
        unsafe { self.flag.flags & 3 == 3 }
    }

    fn above_child(&self) -> usize {
        unsafe { self.flag.above_child >> 2 }
    }
}

#[derive(Copy, Clone)]
enum EdgeType {
    Start,
    End,
}

impl PartialEq for EdgeType {
    fn eq(&self, other: &Self) -> bool {
        match self {
            EdgeType::Start => match other {
                EdgeType::Start => true,
                _ => false,
            },
            EdgeType::End => match other {
                EdgeType::End => true,
                _ => false,
            },
        }
    }
}

impl PartialOrd for EdgeType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self {
            EdgeType::Start => match other {
                EdgeType::Start => Some(Ordering::Equal),
                EdgeType::End => Some(Ordering::Less),
            },
            EdgeType::End => match other {
                EdgeType::End => Some(Ordering::Equal),
                EdgeType::Start => Some(Ordering::Greater),
            },
        }
    }
}

impl Default for EdgeType {
    fn default() -> Self {
        EdgeType::Start
    }
}

#[derive(Default, Copy, Clone)]
struct BoundEdge {
    t: Float,
    prim_num: usize,
    typ: EdgeType,
}

impl BoundEdge {
    fn new(t: Float, prim_num: usize, starting: bool) -> Self {
        Self {
            t,
            prim_num,
            typ: if starting {
                EdgeType::Start
            } else {
                EdgeType::End
            },
        }
    }
}

impl KdTreeAccel {
    fn new(
        p: Vec<Arc<Box<dyn Primitive>>>,
        isect_cost: i32,
        traversal_cost: i32,
        empty_bonus: Float,
        max_prims: usize,
        mut max_depth: usize,
    ) -> Self {
        let mut kta = Self {
            isect_cost,
            traversal_cost,
            max_prims,
            empty_bonus,
            primitives: p,
            primitive_indices: vec![],
            n_alloced_nodes: 0,
            next_free_node: 0,
            bounds: Default::default(),
            nodes: Vec::new(),
        };

        if max_depth <= 0 {
            max_depth =
                (log_2_int_u64(kta.primitives.len() as u64) as Float * 1.3 + 8.0).round() as usize;
        }

        let mut prim_bounds = Vec::with_capacity(kta.primitives.len());
        for prim in &kta.primitives {
            let b = prim.world_bound();
            kta.bounds.union(&b);
            prim_bounds.push(b);
        }

        let mut edges = [Vec::new(), Vec::new(), Vec::new()];
        for i in 0..3 {
            edges[i] = vec![BoundEdge::default(); kta.primitives.len() * 2];
        }
        let mut prims0 = vec![0; kta.primitives.len()];
        let mut prims1 = vec![0; kta.primitives.len() * (max_depth as usize + 1)];

        let mut prim_nums = vec![0; kta.primitives.len()];
        for i in 0..kta.primitives.len() {
            prim_nums[i] = i;
        }

        kta.build_tree(
            0,
            kta.bounds,
            &prim_bounds,
            prim_nums.as_slice(),
            kta.primitives.len(),
            max_depth,
            &mut edges[..],
            prims0.as_mut_slice(),
            prims1.as_mut_slice(),
            0,
        );

        kta
    }

    fn build_tree(
        &mut self,
        node_num: usize,
        node_bounds: Bounds3f,
        all_prim_bounds: &Vec<Bounds3f>,
        prim_nums: &[usize],
        n_primitives: usize,
        depth: usize,
        edges: &mut [Vec<BoundEdge>],
        prims0: &mut [usize],
        prims1: &mut [usize],
        mut bad_refines: usize,
    ) {
        if self.next_free_node == self.n_alloced_nodes {
            let n_new_alloc_nodes = std::cmp::max(2 * self.n_alloced_nodes, 512);
            self.nodes.resize(n_new_alloc_nodes, KdAccelNode::default());
            self.n_alloced_nodes = n_new_alloc_nodes;
        }
        self.next_free_node += 1;
        if n_primitives <= self.max_prims || depth == 0 {
            self.nodes[node_num].init_leaf(prim_nums, n_primitives, &mut self.primitive_indices);
            return;
        }

        let mut best_axis: i32 = -1;
        let mut best_offset: i32 = -1;
        let mut best_cost = Float::INFINITY;
        let old_cost = (self.isect_cost * n_primitives as i32) as Float;
        let total_sa = node_bounds.surface_area();
        let inv_total_sa = 1.0 / total_sa;
        let d = node_bounds.max - node_bounds.min;
        let mut axis = node_bounds.maximum_extent();
        let mut retries = 0;

        loop {
            for i in 0..n_primitives {
                let pn = prim_nums[i];
                let bounds = &all_prim_bounds[pn];
                edges[axis][2 * i] = BoundEdge::new(bounds.min[axis], pn, true);
                edges[axis][2 * i + 1] = BoundEdge::new(bounds.max[axis], pn, false);
            }
            edges[axis].as_mut_slice()[0..2 * n_primitives].sort_by(|e1, e2| {
                if e1.t == e2.t {
                    if e1.typ == e2.typ {
                        Ordering::Equal
                    } else if e1.typ < e2.typ {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                } else if e1.t < e2.t {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            });

            let mut n_below = 0;
            let mut n_above = n_primitives;
            for i in 0..2 * n_primitives {
                if let EdgeType::End = edges[axis][i].typ {
                    n_above -= 1;
                }
                let edge_t = edges[axis][i].t;
                if edge_t > node_bounds.min[axis] && edge_t < node_bounds.max[axis] {
                    let other_axis0 = (axis + 1) % 3;
                    let other_axis1 = (axis + 2) % 3;
                    let below_sa = 2.0
                        * (d[other_axis0] * d[other_axis1]
                            + (edge_t - node_bounds.min[axis]) * (d[other_axis0] + d[other_axis1]));
                    let above_sa = 2.0
                        * (d[other_axis0] * d[other_axis1]
                            + (node_bounds.max[axis] - edge_t) * (d[other_axis0] + d[other_axis1]));
                    let p_below = below_sa * inv_total_sa;
                    let p_above = above_sa * inv_total_sa;
                    let eb = if n_above == 0 || n_below == 0 {
                        self.empty_bonus
                    } else {
                        0.0
                    };

                    let cost = self.traversal_cost as Float
                        + self.isect_cost as Float
                            * (1.0 - eb)
                            * (p_below * n_below as Float + p_above * n_above as Float);

                    if cost < best_cost {
                        best_cost = cost;
                        best_axis = axis as i32;
                        best_offset = i as i32;
                    }
                }
                if let EdgeType::Start = edges[axis][i].typ {
                    n_below += 1;
                }
            }
            if best_axis == -1 && retries < 2 {
                retries += 1;
                axis = (axis + 1) % 3;
                continue;
            }
            break;
        }

        if best_cost > old_cost {
            bad_refines += 1;
        }
        if best_cost > 4.0 * old_cost && n_primitives < 16 || best_axis == -1 || bad_refines == 3 {
            self.nodes[node_num].init_leaf(prim_nums, n_primitives, &mut self.primitive_indices);
            return;
        }

        let mut n0 = 0;
        let mut n1 = 0;
        for i in 0..best_offset as usize {
            if let EdgeType::Start = edges[best_axis as usize][i].typ {
                prims0[n0] = edges[best_axis as usize][i].prim_num;
                n0 += 1;
            }
        }
        for i in best_offset as usize + 1..2 * n_primitives {
            if let EdgeType::End = edges[best_axis as usize][i].typ {
                prims1[n1] = edges[best_axis as usize][i].prim_num;
                n1 += 1;
            }
        }

        let t_split = edges[best_axis as usize][best_offset as usize].t;
        let mut bounds0 = node_bounds;
        let mut bounds1 = node_bounds;
        bounds0.max[best_axis as usize] = t_split;
        bounds1.min[best_axis as usize] = t_split;
        let mut prim_nums = Vec::with_capacity(prims0.len());
        prim_nums.extend_from_slice(prims0);
        self.build_tree(
            node_num + 1,
            bounds0,
            all_prim_bounds,
            prim_nums.as_slice(),
            n0,
            depth - 1,
            edges,
            prims0,
            &mut prims1[n_primitives..],
            bad_refines,
        );
        let above_child = self.next_free_node;
        self.nodes[node_num].init_interior(best_axis as usize, above_child, t_split);
        let mut prim_nums = Vec::with_capacity(prims1.len());
        prim_nums.extend_from_slice(prims1);
        self.build_tree(
            above_child,
            bounds1,
            all_prim_bounds,
            prim_nums.as_slice(),
            n1,
            depth - 1,
            edges,
            prims0,
            &mut prims1[n_primitives..],
            bad_refines,
        );
    }
}

impl Primitive for KdTreeAccel {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn world_bound(&self) -> Bounds3f {
        self.bounds
    }

    fn intersect(&self, ray: &mut Ray, si: &mut SurfaceInteraction) -> bool {
        let (ok, mut t_min, mut t_max) = self.bounds.intersect_p(&*ray);
        if !ok {
            return false;
        }

        let inv_dir = Vector3f::new(1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z);
        const MAX_TODO: usize = 64;
        let mut todo = [KdTodo::default(); MAX_TODO];
        let mut todo_pos = 0;
        let mut hit = false;

        let mut node_idx = 0;
        let mut node_opt = self.nodes.get(node_idx);
        while let Some(node) = node_opt {
            if ray.t_max < t_min {
                break;
            }

            if !node.is_leaf() {
                let axis = node.split_axis();
                let t_plane = (node.split_pos() - ray.o[axis]) * inv_dir[axis];

                let mut first_idx = 0;
                let mut second_idx = 0;
                if ray.o[axis] < node.split_pos()
                    || ray.o[axis] == node.split_pos() && ray.d[axis] < 0.0
                {
                    first_idx = node_idx + 1;
                    second_idx = node.above_child();
                } else {
                    first_idx = node.above_child();
                    second_idx = node_idx + 1;
                };
                let first_child = self.nodes.get(first_idx);
                let second_child = self.nodes.get(second_idx);

                if t_plane > t_max || t_plane <= 0.0 {
                    node_opt = first_child;
                    node_idx = first_idx;
                } else if t_plane < t_min {
                    node_opt = second_child;
                    node_idx = second_idx;
                } else {
                    todo[todo_pos].node = second_child;
                    todo[todo_pos].t_min = t_plane;
                    todo[todo_pos].t_max = t_max;
                    todo[todo_pos].idx = second_idx;
                    todo_pos += 1;
                    node_opt = first_child;
                    node_idx = first_idx;
                    t_max = t_plane;
                }
            } else {
                let n_primitives = node.n_primitives();
                if n_primitives == 1 {
                    unsafe {
                        let p = &self.primitives[node.data.one_primitive];
                        if p.intersect(ray, si) {
                            hit = true;
                        }
                    }
                } else {
                    for i in 0..n_primitives {
                        unsafe {
                            let index =
                                self.primitive_indices[node.data.primitive_indices_offset + i];
                            let p = &self.primitives[index];
                            if p.intersect(ray, si) {
                                hit = true;
                            }
                        }
                    }
                }

                if todo_pos > 0 {
                    todo_pos -= 1;
                    node_opt = todo[todo_pos].node;
                    node_idx = todo[todo_pos].idx;
                    t_min = todo[todo_pos].t_min;
                    t_max = todo[todo_pos].t_max;
                } else {
                    break;
                }
            }
        }
        hit
    }

    fn intersect_p(&self, ray: &Ray) -> bool {
        let (ok, mut t_min, mut t_max) = self.bounds.intersect_p(&*ray);
        if !ok {
            return false;
        }

        let inv_dir = Vector3f::new(1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z);
        const MAX_TODO: usize = 64;
        let mut todo = [KdTodo::default(); MAX_TODO];
        let mut todo_pos = 0;

        let mut node_idx = 0;
        let mut node_opt = self.nodes.get(node_idx);
        while let Some(node) = node_opt {
            if ray.t_max < t_min {
                break;
            }

            if !node.is_leaf() {
                let axis = node.split_axis();
                let t_plane = (node.split_pos() - ray.o[axis]) * inv_dir[axis];

                let mut first_idx = 0;
                let mut second_idx = 0;
                if ray.o[axis] < node.split_pos()
                    || ray.o[axis] == node.split_pos() && ray.d[axis] < 0.0
                {
                    first_idx = node_idx + 1;
                    second_idx = node.above_child();
                } else {
                    first_idx = node.above_child();
                    second_idx = node_idx + 1;
                };
                let first_child = self.nodes.get(first_idx);
                let second_child = self.nodes.get(second_idx);

                if t_plane > t_max || t_plane <= 0.0 {
                    node_opt = first_child;
                    node_idx = first_idx;
                } else if t_plane < t_min {
                    node_opt = second_child;
                    node_idx = second_idx;
                } else {
                    todo[todo_pos].node = second_child;
                    todo[todo_pos].t_min = t_plane;
                    todo[todo_pos].t_max = t_max;
                    todo[todo_pos].idx = second_idx;
                    todo_pos += 1;
                    node_opt = first_child;
                    node_idx = first_idx;
                    t_max = t_plane;
                }
            } else {
                let n_primitives = node.n_primitives();
                if n_primitives == 1 {
                    unsafe {
                        let p = &self.primitives[node.data.one_primitive];
                        if p.intersect_p(ray) {
                            return true;
                        }
                    }
                } else {
                    for i in 0..n_primitives {
                        unsafe {
                            let index =
                                self.primitive_indices[node.data.primitive_indices_offset + i];
                            let p = &self.primitives[index];
                            if p.intersect_p(ray) {
                                return true;
                            }
                        }
                    }
                }

                if todo_pos > 0 {
                    todo_pos -= 1;
                    node_opt = todo[todo_pos].node;
                    node_idx = todo[todo_pos].idx;
                    t_min = todo[todo_pos].t_min;
                    t_max = todo[todo_pos].t_max;
                } else {
                    break;
                }
            }
        }
        false
    }

    fn get_area_light(&self) -> Option<Arc<Box<dyn AreaLight>>> {
        unimplemented!(
            "Aggregate does not support get_area_light method, use GeometricPrimitive instead"
        )
    }

    fn get_material(&self) -> Option<Arc<Box<dyn Material>>> {
        unimplemented!(
            "Aggregate does not support get_material method, use GeometricPrimitive instead"
        )
    }

    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    ) {
        unimplemented!("Aggregate does not support compute_scattering_function method, use GeometricPrimitive instead")
    }
}
