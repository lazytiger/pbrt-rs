use crate::core::geometry::{Bounds3, Bounds3f, Point3f, Ray, Union, Vector3f};
use crate::core::interaction::SurfaceInteraction;
use crate::core::light::AreaLight;
use crate::core::material::{Material, TransportMode};
use crate::core::primitive::Primitive;
use crate::core::RealNum;
use crate::Float;
use num::traits::real::Real;
use std::any::Any;
use std::cmp::{max, Ordering};
use std::io::Read;
use std::mem::swap;
use std::sync::atomic::{AtomicI32, AtomicUsize};
use std::sync::Arc;
use typed_arena::Arena;

#[derive(Default, Copy, Clone)]
struct BVHPrimitiveInfo {
    primitive_number: i32,
    bounds: Bounds3f,
    centroid: Point3f,
}

impl BVHPrimitiveInfo {
    fn new(primitive_number: i32, bounds: Bounds3f) -> Self {
        Self {
            primitive_number,
            bounds,
            centroid: bounds.min * 0.5 + bounds.max * 0.5,
        }
    }
}

#[derive(Default, Clone)]
struct BVHBuildNode<'a> {
    bounds: Bounds3f,
    children: Option<[&'a BVHBuildNode<'a>; 2]>,
    split_axis: i32,
    first_prim_offset: i32,
    n_primitives: i32,
}

impl<'a> BVHBuildNode<'a> {
    fn init_leaf(&mut self, first: i32, n: i32, b: Bounds3f) {
        self.first_prim_offset = first;
        self.n_primitives = n;
        self.bounds = b;
    }

    fn init_interior(&mut self, axis: i32, c0: &'a BVHBuildNode, c1: &'a BVHBuildNode) {
        self.bounds = c0.bounds.union(&c1.bounds);
        self.children = Some([c0, c1]);
        self.split_axis = axis;
        self.n_primitives = 0;
    }
}

#[derive(Default, Copy, Clone)]
struct MortonPrimitive {
    primitive_index: i32,
    morton_code: u32,
}

#[derive(Default)]
struct LBVHTreelet<'a> {
    start_index: i32,
    n_primitive: i32,
    build_nodes: &'a mut [BVHBuildNode<'a>],
}

#[derive(Default, Copy, Clone)]
struct LinearBVHNode {
    bounds: Bounds3f,
    primitive_or_second_child_offset: i32,
    n_primitives: u16,
    axis: u8,
}

#[inline]
fn left_shift3(mut x: u32) -> u32 {
    if x == (1 << 10) {
        x -= 1;
    }
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x
}

#[inline]
fn encode_morton3(v: &Vector3f) -> u32 {
    (left_shift3(v.z as u32) << 2) | (left_shift3(v.y as u32) << 1) | left_shift3(v.x as u32)
}

fn radix_sort(v: &mut Vec<MortonPrimitive>) {
    let bits_per_pass = 6;
    let n_bits = 30;
    let n_passes = n_bits / bits_per_pass;
    let mut temp_vector = vec![MortonPrimitive::default(); v.len()];

    for pass in 0..n_passes {
        let low_bit = pass * bits_per_pass;
        let (v_in, v_out) = if (pass & 1) != 0 {
            (&mut temp_vector, &mut *v)
        } else {
            (&mut *v, &mut temp_vector)
        };

        let n_buckets = 1 << bits_per_pass;
        let mut bucket_count = vec![0; n_buckets];
        let bit_mask = (1 << bits_per_pass) - 1;
        for i in 0..v_in.len() {
            let mp = &v_in[i];
            let bucket = (mp.morton_code >> low_bit) & bit_mask;
            bucket_count[bucket as usize] += 1;
        }

        let mut out_index = vec![0; n_buckets];
        out_index[0] = 0;
        for i in 1..n_buckets {
            out_index[i] = out_index[i - 1] + bucket_count[i - 1];
        }

        for i in 0..v_in.len() {
            let mp = &v_in[i];
            let bucket = (mp.morton_code >> low_bit) & bit_mask;
            v_out[out_index[bucket as usize]] = *mp;
            out_index[bucket as usize] += 1;
        }
    }
    if (n_passes & 1) != 0 {
        swap(v, &mut temp_vector)
    }
}

pub enum SplitMethod {
    SAH,
    HLBVH,
    Middle,
    EqualCounts,
}

pub struct BVHAccel {
    max_prims_in_node: i32,
    split_method: SplitMethod,
    primitives: Vec<Arc<Box<dyn Primitive>>>,
    nodes: Option<Vec<LinearBVHNode>>,
}

impl BVHAccel {
    pub fn new(
        p: Vec<Arc<Box<dyn Primitive>>>,
        max_prims_in_node: i32,
        split_method: SplitMethod,
    ) -> BVHAccel {
        let mut accel = BVHAccel {
            max_prims_in_node: std::cmp::min(max_prims_in_node, 255),
            split_method,
            primitives: p,
            nodes: None,
        };

        if accel.primitives.is_empty() {
            return accel;
        }

        let mut primitive_infos = Vec::with_capacity(accel.primitives.len());
        for i in 0..accel.primitives.len() {
            let info = BVHPrimitiveInfo::new(i as i32, accel.primitives[i].world_bound());
            primitive_infos.push(info);
        }

        let arena: Arena<BVHBuildNode> = Arena::with_capacity(1024 * 1024);
        let mut total_nodes = 0;
        let mut ordered_prims = Vec::with_capacity(accel.primitives.len());
        let root = if let SplitMethod::HLBVH = accel.split_method {
            accel.hlbvh_build(
                &arena,
                &primitive_infos,
                &mut total_nodes,
                &mut ordered_prims,
            )
        } else {
            accel.recursive_build(
                &arena,
                primitive_infos.as_mut_slice(),
                0,
                accel.primitives.len(),
                &mut total_nodes,
                &mut ordered_prims,
            )
        };

        accel.primitives = ordered_prims;
        accel.nodes = Some(vec![LinearBVHNode::default(); total_nodes as usize]);
        let mut offset = 0;
        accel.flatten_bvh_tree(&root, &mut offset);
        if total_nodes != offset {
            panic!("flattern_bvh_tree failed");
        }
        accel
    }

    fn recursive_build<'a>(
        &self,
        arena: &'a Arena<BVHBuildNode<'a>>,
        primitive_info: &mut [BVHPrimitiveInfo],
        start: usize,
        end: usize,
        total_nodes: &mut usize,
        ordered_prims: &mut Vec<Arc<Box<Primitive>>>,
    ) -> &'a BVHBuildNode<'a> {
        let node = arena.alloc(BVHBuildNode::default());
        *total_nodes += 1;
        let mut bounds = Bounds3f::default();
        for i in start..end {
            bounds = bounds.union(&primitive_info[i as usize].bounds);
        }
        let n_primitives = end - start;
        if n_primitives == 1 {
            let first_prim_offset = ordered_prims.len();
            for i in start..end {
                let prim_num = primitive_info[i as usize].primitive_number;
                ordered_prims.push(self.primitives[prim_num as usize].clone());
            }
            node.init_leaf(first_prim_offset as i32, n_primitives as i32, bounds);
            return node;
        } else {
            let mut centroid_bounds = Bounds3f::default();
            for i in start..end {
                centroid_bounds.union(&primitive_info[i as usize].centroid);
            }
            let dim = centroid_bounds.maximum_extent();

            let mut mid = (start + end) / 2;
            if centroid_bounds.max[dim] == centroid_bounds.min[dim] {
                let first_prim_offset = ordered_prims.len();
                for i in start..end {
                    let prim_num = primitive_info[i as usize].primitive_number;
                    ordered_prims.push(self.primitives[prim_num as usize].clone());
                }
                node.init_leaf(first_prim_offset as i32, n_primitives as i32, bounds);
                return node;
            } else {
                match self.split_method {
                    SplitMethod::Middle => {
                        let p_mid = (centroid_bounds.min[dim] + centroid_bounds.max[dim]) / 2.0;
                        mid = *&mut primitive_info[start as usize..end as usize + 1]
                            .iter_mut()
                            .partition_in_place(|&pi| pi.centroid[dim] < p_mid)
                            + start;
                        if mid == start || mid == end {
                            let mid = (start + end) / 2;
                            floydrivest::nth_element(
                                primitive_info,
                                mid as usize,
                                &mut |p1, p2| {
                                    if p1.centroid[dim] < p2.centroid[dim] {
                                        Ordering::Less
                                    } else if p1.centroid[dim] == p2.centroid[dim] {
                                        Ordering::Equal
                                    } else {
                                        Ordering::Greater
                                    }
                                },
                            );
                        }
                    }
                    SplitMethod::EqualCounts => {
                        let mid = (start + end) / 2;
                        floydrivest::nth_element(primitive_info, mid as usize, &mut |p1, p2| {
                            if p1.centroid[dim] < p2.centroid[dim] {
                                Ordering::Less
                            } else if p1.centroid[dim] == p2.centroid[dim] {
                                Ordering::Equal
                            } else {
                                Ordering::Greater
                            }
                        });
                    }
                    _ => {
                        if n_primitives <= 2 {
                            let mid = (start + end) / 2;
                            floydrivest::nth_element(
                                primitive_info,
                                mid as usize,
                                &mut |p1, p2| {
                                    if p1.centroid[dim] < p2.centroid[dim] {
                                        Ordering::Less
                                    } else if p1.centroid[dim] == p2.centroid[dim] {
                                        Ordering::Equal
                                    } else {
                                        Ordering::Greater
                                    }
                                },
                            );
                        } else {
                            let n_buckets = 12;
                            let mut buckets = [BucketInfo::default(); 12];

                            for i in start..end {
                                let mut b = n_buckets as Float
                                    * centroid_bounds.offset(&primitive_info[i as usize].centroid)
                                        [dim];
                                if b == n_buckets as Float {
                                    b = (n_buckets - 1) as Float;
                                }

                                buckets[b as usize].count += 1;
                                buckets[b as usize]
                                    .bounds
                                    .union(&primitive_info[i as usize].bounds);
                            }

                            let mut cost = [0.0 as Float; 11];
                            for i in 0..n_buckets - 1 {
                                let mut b0 = Bounds3f::default();
                                let mut b1 = Bounds3f::default();
                                let mut count0 = 0;
                                let mut count1 = 0;
                                for j in 0..i {
                                    b0.union(&buckets[j].bounds);
                                    count0 += buckets[j].count;
                                }
                                for j in i + 1..n_buckets {
                                    b1.union(&buckets[j].bounds);
                                    count1 += buckets[j].count;
                                }

                                cost[i] = 1.0
                                    + (count0 as Float * b0.surface_area()
                                        + count1 as Float * b1.surface_area())
                                        / bounds.surface_area();
                            }

                            let (_, min_cost_split_bucket, min_cost) = cost.iter().fold(
                                (0, 0, Float::MAX),
                                |(current, index, min_cost), cost| {
                                    if *cost < min_cost {
                                        (current + 1, current, *cost)
                                    } else {
                                        (current + 1, index, min_cost)
                                    }
                                },
                            );

                            let leaf_cost = n_primitives as Float;
                            if n_primitives > self.max_prims_in_node as usize
                                || min_cost < leaf_cost
                            {
                                mid = *&mut primitive_info[start as usize..end as usize + 1]
                                    .iter_mut()
                                    .partition_in_place(|pi| {
                                        let mut b = n_buckets as Float
                                            * centroid_bounds.offset(&pi.centroid)[dim];
                                        if b == n_buckets as Float {
                                            b = (n_buckets - 1) as Float;
                                        }
                                        b < min_cost_split_bucket as Float
                                    })
                                    + start;
                            } else {
                                let first_prim_offset = ordered_prims.len();
                                for i in start..end {
                                    let prim_num = primitive_info[i as usize].primitive_number;
                                    ordered_prims.push(self.primitives[prim_num as usize].clone());
                                }
                                node.init_leaf(
                                    first_prim_offset as i32,
                                    n_primitives as i32,
                                    bounds,
                                );
                                return node;
                            }
                        }
                    }
                }
                node.init_interior(
                    dim as i32,
                    self.recursive_build(
                        arena,
                        primitive_info,
                        start,
                        mid,
                        total_nodes,
                        ordered_prims,
                    ),
                    self.recursive_build(
                        arena,
                        primitive_info,
                        mid,
                        end,
                        total_nodes,
                        ordered_prims,
                    ),
                );
            }
        }

        node
    }

    fn hlbvh_build<'a>(
        &self,
        arena: &'a Arena<BVHBuildNode<'a>>,
        primitive_info: &Vec<BVHPrimitiveInfo>,
        total_node: &mut usize,
        ordered_prims: &mut Vec<Arc<Box<Primitive>>>,
    ) -> &'a BVHBuildNode<'a> {
        let mut bounds = Bounds3f::default();
        for pi in primitive_info {
            bounds.union(&pi.centroid);
        }

        let mut morton_prims = vec![MortonPrimitive::default(); primitive_info.len()];

        //TODO parallel
        for i in 0..primitive_info.len() {
            let morton_bits = 10;
            let morton_scale = 1 << morton_bits;
            morton_prims[i].primitive_index = primitive_info[i].primitive_number as i32;
            let centroid_offset = bounds.offset(&primitive_info[i].centroid);
            morton_prims[i].morton_code =
                encode_morton3(&(centroid_offset * morton_scale as Float));
        }

        radix_sort(&mut morton_prims);

        let mut treelets_to_build = Vec::new();
        let mut start = 0;
        for end in 1..morton_prims.len() + 1 {
            let mask = 0b00111111111111000000000000000000;
            if end == morton_prims.len()
                || (morton_prims[start].morton_code & mask)
                    != (morton_prims[end].morton_code & mask)
            {
                let n_primitives = end - start;
                let max_bvh_nodes = 2 * n_primitives;
                let nodes = arena
                    .alloc_extend(std::iter::repeat(BVHBuildNode::default()).take(max_bvh_nodes));
                treelets_to_build.push(LBVHTreelet {
                    start_index: start as i32,
                    n_primitive: n_primitives as i32,
                    build_nodes: nodes,
                });
            }
        }
        let mut atomic_total = AtomicI32::new(0);
        let mut ordered_prim_offset = AtomicUsize::new(0);
        //TODO parallel
        for i in 0..treelets_to_build.len() {
            let mut nodes_created = 0;
            let first_bit_index = 29 - 12;
            let tr = &mut treelets_to_build[i];
            tr.build_nodes = self.emit_lbvh(
                tr.build_nodes,
                primitive_info,
                &morton_prims[tr.start_index as usize..],
                tr.n_primitive as usize,
                &mut nodes_created,
                ordered_prims,
                &mut ordered_prim_offset,
                first_bit_index,
            );
            atomic_total.fetch_add(nodes_created as i32, std::sync::atomic::Ordering::SeqCst);
        }
        *total_node = atomic_total.load(std::sync::atomic::Ordering::SeqCst) as usize;
        let mut finished_treelets = Vec::with_capacity(treelets_to_build.len());
        for treelet in &treelets_to_build {
            finished_treelets.push(&treelet.build_nodes[0]);
        }

        self.build_upper_sah(
            arena,
            finished_treelets.as_mut_slice(),
            0,
            finished_treelets.len(),
            total_node,
        )
    }

    fn emit_lbvh<'a>(
        &self,
        mut build_nodes: &mut [BVHBuildNode<'a>],
        primitive_info: &Vec<BVHPrimitiveInfo>,
        morton_prims: &[MortonPrimitive],
        n_primitives: usize,
        total_nodes: &mut usize,
        ordered_prims: &mut Vec<Arc<Box<Primitive>>>,
        ordered_prims_offset: &mut AtomicUsize,
        bit_index: i32,
    ) -> &mut [BVHBuildNode<'a>] {
        if bit_index == -1 || n_primitives < self.max_prims_in_node as usize {
            *total_nodes += 1;
            let node = &mut build_nodes[0];
            build_nodes = &mut build_nodes[1..];
            let mut bounds = Bounds3f::default();
            let first_prim_offset =
                ordered_prims_offset.fetch_add(n_primitives, std::sync::atomic::Ordering::SeqCst);
            for i in 0..n_primitives {
                let primitive_index = morton_prims[i as usize].primitive_index;
                ordered_prims[first_prim_offset + i] =
                    self.primitives[primitive_index as usize].clone();
                bounds.union(&primitive_info[primitive_index as usize].bounds);
            }
            node.init_leaf(first_prim_offset as i32, n_primitives as i32, bounds);
            build_nodes
        } else {
            let mask = 1 << bit_index;
            if (morton_prims[0].morton_code & mask)
                == (morton_prims[n_primitives as usize - 1].morton_code & mask)
            {
                return self.emit_lbvh(
                    build_nodes,
                    primitive_info,
                    morton_prims,
                    n_primitives,
                    total_nodes,
                    ordered_prims,
                    ordered_prims_offset,
                    bit_index - 1,
                );
            }

            let mut search_start = 0;
            let mut search_end = n_primitives - 1;
            while search_start + 1 != search_end {
                let mid = (search_start + search_end) / 2;
                if (morton_prims[search_start as usize].morton_code & mask)
                    == (morton_prims[mid as usize].morton_code & mask)
                {
                    search_start = mid;
                } else {
                    search_end = mid;
                }
            }

            let split_offset = search_end;
            *total_nodes += 1;
            let node = &mut build_nodes[0];
            build_nodes = &mut build_nodes[1..];
            let lbvh = [
                self.emit_lbvh(
                    build_nodes,
                    primitive_info,
                    morton_prims,
                    split_offset,
                    total_nodes,
                    ordered_prims,
                    ordered_prims_offset,
                    bit_index - 1,
                ),
                self.emit_lbvh(
                    build_nodes,
                    primitive_info,
                    &morton_prims[split_offset..],
                    n_primitives - split_offset,
                    total_nodes,
                    ordered_prims,
                    ordered_prims_offset,
                    bit_index - 1,
                ),
            ];

            let axis = bit_index % 3;
            node.init_interior(axis, &lbvh[0][0], &lbvh[1][0]);
            build_nodes
        }
    }

    fn build_upper_sah<'a>(
        &self,
        arena: &'a Arena<BVHBuildNode<'a>>,
        treelet_roots: &mut [&'a BVHBuildNode<'a>],
        start: usize,
        end: usize,
        total_nodes: &mut usize,
    ) -> &'a BVHBuildNode<'a> {
        let n_nodes = end - start;
        if n_nodes == 1 {
            return treelet_roots[start as usize];
        }
        *total_nodes += 1;
        let mut node = arena.alloc(BVHBuildNode::default());
        let mut bounds = Bounds3f::default();
        for i in start..end {
            bounds.union(&treelet_roots[i as usize].bounds);
        }

        let mut centroid_bounds = Bounds3f::default();
        for i in start..end {
            let centroid =
                (treelet_roots[i as usize].bounds.min + treelet_roots[i as usize].bounds.max) * 0.5;
            centroid_bounds.union(&centroid);
        }
        let dim = centroid_bounds.maximum_extent();
        let n_buckets = 12;
        let buckets = [BucketInfo::default(); 12];

        for i in start..end {
            let centroid = (treelet_roots[i as usize].bounds.min[dim]
                + treelet_roots[i as usize].bounds.max[dim])
                * 0.5;
            let mut b = n_buckets
                * ((centroid - centroid_bounds.min[dim])
                    / (centroid_bounds.max[dim] - centroid_bounds.min[dim]))
                    as usize;
            if b == n_buckets {
                b = n_buckets - 1;
            }
            buckets[b].count += 1;
            buckets[b].bounds.union(&treelet_roots[i as usize].bounds);
        }

        let cost = [0.0 as Float; 11];
        for i in 0..n_buckets - 1 {
            let mut b0 = Bounds3f::default();
            let mut b1 = Bounds3f::default();
            let mut count0 = 0;
            let mut count1 = 0;
            for j in 0..i + 1 {
                b0.union(&buckets[j].bounds);
                count0 += buckets[j].count;
            }
            for j in i + 1..n_buckets {
                b1.union(&buckets[j].bounds);
                count1 += buckets[j].count;
            }
            cost[i] = 0.125
                + (count0 as Float * b0.surface_area() + count1 as Float * b1.surface_area())
                    / bounds.surface_area();
        }

        let (_, min_cost_split_bucket, min_cost) =
            cost.iter()
                .fold((0, 0, Float::MAX), |(current, index, min_cost), cost| {
                    if *cost < min_cost {
                        (current + 1, current, *cost)
                    } else {
                        (current + 1, index, min_cost)
                    }
                });

        let mid = *&mut treelet_roots[start as usize..end as usize + 1]
            .iter_mut()
            .partition_in_place(|node| {
                let centroid = (node.bounds.min[dim] + node.bounds.max[dim]) * 0.5;
                let mut b = n_buckets
                    * ((centroid - centroid_bounds.min[dim])
                        / (centroid_bounds.max[dim] - centroid_bounds.min[dim]))
                        as usize;
                if b == n_buckets {
                    b = n_buckets - 1;
                }
                b < min_cost_split_bucket
            })
            + start;
        node.init_interior(
            dim as i32,
            self.build_upper_sah(arena, treelet_roots, start, mid, total_nodes),
            self.build_upper_sah(arena, treelet_roots, mid, end, total_nodes),
        );
        node
    }

    fn flatten_bvh_tree(&mut self, node: &BVHBuildNode, offset: &mut usize) -> i32 {
        if let Some(nodes) = &mut self.nodes {
            let linear_node = &mut nodes[*offset as usize];
            linear_node.bounds = node.bounds;
            let my_offset = *offset;
            *offset += 1;
            if node.n_primitives > 0 {
                linear_node.primitive_or_second_child_offset = node.first_prim_offset;
                linear_node.n_primitives = node.n_primitives as u16;
            } else {
                linear_node.axis = node.split_axis as u8;
                linear_node.n_primitives = 0;
                if let Some(children) = &node.children {
                    self.flatten_bvh_tree(children[0], offset);
                    linear_node.primitive_or_second_child_offset =
                        self.flatten_bvh_tree(children[1], offset);
                }
            }
            return my_offset as i32;
        }
        0
    }
}

impl Primitive for BVHAccel {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn world_bound(&self) -> Bounds3f {
        if let Some(nodes) = &self.nodes {
            if !nodes.is_empty() {
                return nodes[0].bounds;
            }
        }
        Bounds3f::default()
    }

    fn intersect(&self, r: &mut Ray, si: &mut SurfaceInteraction) -> bool {
        unimplemented!()
    }

    fn intersect_p(&self, r: &Ray) -> bool {
        unimplemented!()
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

#[derive(Default, Copy, Clone)]
struct BucketInfo {
    count: i32,
    bounds: Bounds3f,
}
