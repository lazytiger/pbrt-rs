use crate::core::{
    arena::{Arena, Indexed},
    geometry::{Bounds3f, IntersectP, Point3f, Ray, Union, Vector3f},
    interaction::SurfaceInteraction,
    light::{AreaLight, AreaLightDt},
    material::{Material, MaterialDt, TransportMode},
    pbrt::Float,
    primitive::{Primitive, PrimitiveDt},
    RealNum,
};
use num::traits::real::Real;
use std::{
    any::Any,
    cmp::{max, Ordering},
    io::Read,
    mem::swap,
    sync::{
        atomic::{AtomicI32, AtomicUsize},
        Arc,
    },
};

#[derive(Default, Copy, Clone)]
struct BVHPrimitiveInfo {
    primitive_number: usize,
    bounds: Bounds3f,
    centroid: Point3f,
}

impl BVHPrimitiveInfo {
    fn new(primitive_number: usize, bounds: Bounds3f) -> Self {
        Self {
            primitive_number,
            bounds,
            centroid: bounds.min * 0.5 + bounds.max * 0.5,
        }
    }
}

#[derive(Default, Clone)]
struct BVHBuildNode {
    bounds: Bounds3f,
    index: usize,
    children: [usize; 2],
    split_axis: usize,
    first_prim_offset: usize,
    n_primitives: usize,
}

impl BVHBuildNode {
    fn init_leaf(&mut self, first: usize, n: usize, b: Bounds3f) {
        self.first_prim_offset = first;
        self.n_primitives = n;
        self.bounds = b;
    }

    fn static_init_interior(
        arena: &mut Arena<BVHBuildNode>,
        axis: usize,
        s: usize,
        c0: usize,
        c1: usize,
    ) {
        let (bounds, children) = {
            let c0 = arena.get(c0);
            let c1 = arena.get(c1);
            (c0.bounds.union(&c1.bounds), [c0.index, c1.index])
        };
        let node = arena.get_mut(s);
        node.bounds = bounds;
        node.children = children;
        node.split_axis = axis;
        node.n_primitives = 0;
    }

    fn init_interior(
        &mut self,
        arena: &mut Arena<BVHBuildNode>,
        axis: usize,
        c0: usize,
        c1: usize,
    ) {
        let c0 = arena.get(c0);
        let c1 = arena.get(c1);
        self.bounds = c0.bounds.union(&c1.bounds);
        self.children = [c0.index, c1.index];
        self.split_axis = axis;
        self.n_primitives = 0;
    }
}

#[derive(Default, Copy, Clone)]
struct MortonPrimitive {
    primitive_index: usize,
    morton_code: u32,
}

#[derive(Default, Copy, Clone)]
struct LBVHTreelet {
    start_index: usize,
    n_primitive: usize,
    node_start: usize,
    root_index: usize,
}

impl LBVHTreelet {
    fn node<'a>(&self, arena: &'a Arena<BVHBuildNode>, offset: usize) -> &'a BVHBuildNode {
        arena.get(offset + self.node_start)
    }

    fn node_mut<'a>(
        &self,
        arena: &'a mut Arena<BVHBuildNode>,
        offset: usize,
    ) -> &'a mut BVHBuildNode {
        arena.get_mut(offset + self.node_start)
    }

    fn root_node<'a>(&self, arena: &'a Arena<BVHBuildNode>) -> &'a BVHBuildNode {
        arena.get(self.root_index + self.node_start)
    }
}

impl Indexed for BVHBuildNode {
    fn index(&self) -> usize {
        self.index
    }

    fn set_index(&mut self, index: usize) {
        self.index = index;
    }
}

#[derive(Default, Copy, Clone)]
struct LinearBVHNode {
    bounds: Bounds3f,
    primitive_or_second_child_offset: usize,
    n_primitives: usize,
    axis: usize,
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
    max_prims_in_node: usize,
    split_method: SplitMethod,
    primitives: Vec<PrimitiveDt>,
    nodes: Option<Vec<LinearBVHNode>>,
}

impl BVHAccel {
    pub fn new(
        p: Vec<PrimitiveDt>,
        max_prims_in_node: usize,
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
            let info = BVHPrimitiveInfo::new(i, accel.primitives[i].world_bound());
            primitive_infos.push(info);
        }

        let mut arena: Arena<BVHBuildNode> = Arena::with_capacity(1024 * 1024);
        let mut total_nodes = 0;
        let mut ordered_prims = Vec::with_capacity(accel.primitives.len());
        let root = if let SplitMethod::HLBVH = accel.split_method {
            accel.hlbvh_build(
                &mut arena,
                &primitive_infos,
                &mut total_nodes,
                &mut ordered_prims,
            )
        } else {
            accel.recursive_build(
                &mut arena,
                primitive_infos.as_mut_slice(),
                0,
                accel.primitives.len(),
                &mut total_nodes,
                &mut ordered_prims,
            )
        };

        accel.primitives = ordered_prims;
        accel.nodes = Some(vec![LinearBVHNode::default(); total_nodes]);
        let mut offset = 0;
        accel.flatten_bvh_tree(&arena, root, &mut offset);
        if total_nodes != offset {
            panic!("flattern_bvh_tree failed");
        }
        accel
    }

    fn recursive_build<'a>(
        &self,
        arena: &mut Arena<BVHBuildNode>,
        primitive_info: &mut [BVHPrimitiveInfo],
        start: usize,
        end: usize,
        total_nodes: &mut usize,
        ordered_prims: &mut Vec<PrimitiveDt>,
    ) -> usize {
        let (index, _) = arena.alloc(BVHBuildNode::default());
        *total_nodes += 1;
        let mut bounds = Bounds3f::default();
        for i in start..end {
            bounds = bounds.union(&primitive_info[i].bounds);
        }
        let n_primitives = end - start;
        if n_primitives == 1 {
            let first_prim_offset = ordered_prims.len();
            for i in start..end {
                let prim_num = primitive_info[i].primitive_number;
                ordered_prims.push(self.primitives[prim_num].clone());
            }
            arena
                .get_mut(index)
                .init_leaf(first_prim_offset, n_primitives, bounds);
            return index;
        } else {
            let centroid_bounds = Bounds3f::default();
            for i in start..end {
                centroid_bounds.union(&primitive_info[i].centroid);
            }
            let dim = centroid_bounds.maximum_extent();

            let mut mid = (start + end) / 2;
            if centroid_bounds.max[dim] == centroid_bounds.min[dim] {
                let first_prim_offset = ordered_prims.len();
                for i in start..end {
                    let prim_num = primitive_info[i].primitive_number;
                    ordered_prims.push(self.primitives[prim_num].clone());
                }
                arena
                    .get_mut(index)
                    .init_leaf(first_prim_offset, n_primitives, bounds);
                return index;
            } else {
                match self.split_method {
                    SplitMethod::Middle => {
                        let p_mid = (centroid_bounds.min[dim] + centroid_bounds.max[dim]) / 2.0;
                        mid = *&mut primitive_info[start..end + 1]
                            .iter_mut()
                            .partition_in_place(|&pi| pi.centroid[dim] < p_mid)
                            + start;
                        if mid == start || mid == end {
                            let mid = (start + end) / 2;
                            floydrivest::nth_element(primitive_info, mid, &mut |p1, p2| {
                                if p1.centroid[dim] < p2.centroid[dim] {
                                    Ordering::Less
                                } else if p1.centroid[dim] == p2.centroid[dim] {
                                    Ordering::Equal
                                } else {
                                    Ordering::Greater
                                }
                            });
                        }
                    }
                    SplitMethod::EqualCounts => {
                        let mid = (start + end) / 2;
                        floydrivest::nth_element(primitive_info, mid, &mut |p1, p2| {
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
                            floydrivest::nth_element(primitive_info, mid, &mut |p1, p2| {
                                if p1.centroid[dim] < p2.centroid[dim] {
                                    Ordering::Less
                                } else if p1.centroid[dim] == p2.centroid[dim] {
                                    Ordering::Equal
                                } else {
                                    Ordering::Greater
                                }
                            });
                        } else {
                            let n_buckets = 12;
                            let mut buckets = [BucketInfo::default(); 12];

                            for i in start..end {
                                let mut b = n_buckets
                                    * centroid_bounds.offset(&primitive_info[i].centroid)[dim]
                                        as usize;
                                if b == n_buckets {
                                    b = n_buckets - 1;
                                }

                                buckets[b].count += 1;
                                buckets[b].bounds.union(&primitive_info[i].bounds);
                            }

                            let mut cost = [0.0 as Float; 11];
                            for i in 0..n_buckets - 1 {
                                let b0 = Bounds3f::default();
                                let b1 = Bounds3f::default();
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
                            if n_primitives > self.max_prims_in_node || min_cost < leaf_cost {
                                mid = *&mut primitive_info[start..end + 1]
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
                                    let prim_num = primitive_info[i].primitive_number;
                                    ordered_prims.push(self.primitives[prim_num].clone());
                                }
                                arena.get_mut(index).init_leaf(
                                    first_prim_offset,
                                    n_primitives,
                                    bounds,
                                );
                                return index;
                            }
                        }
                    }
                }
                let c0 = self.recursive_build(
                    arena,
                    primitive_info,
                    start,
                    mid,
                    total_nodes,
                    ordered_prims,
                );
                let c1 = self.recursive_build(
                    arena,
                    primitive_info,
                    mid,
                    end,
                    total_nodes,
                    ordered_prims,
                );
                BVHBuildNode::static_init_interior(arena, dim, index, c0, c1);
            }
        }

        index
    }

    fn hlbvh_build(
        &self,
        arena: &mut Arena<BVHBuildNode>,
        primitive_info: &Vec<BVHPrimitiveInfo>,
        total_node: &mut usize,
        ordered_prims: &mut Vec<PrimitiveDt>,
    ) -> usize {
        let bounds = Bounds3f::default();
        for pi in primitive_info {
            bounds.union(&pi.centroid);
        }

        let mut morton_prims = vec![MortonPrimitive::default(); primitive_info.len()];

        //TODO parallel
        for i in 0..primitive_info.len() {
            let morton_bits = 10;
            let morton_scale = 1 << morton_bits;
            morton_prims[i].primitive_index = primitive_info[i].primitive_number;
            let centroid_offset = bounds.offset(&primitive_info[i].centroid);
            morton_prims[i].morton_code =
                encode_morton3(&(centroid_offset * morton_scale as Float));
        }

        radix_sort(&mut morton_prims);

        let mut treelets_to_build = Vec::new();
        let start = 0;
        for end in 1..morton_prims.len() + 1 {
            let mask = 0b00111111111111000000000000000000;
            if end == morton_prims.len()
                || (morton_prims[start].morton_code & mask)
                    != (morton_prims[end].morton_code & mask)
            {
                let n_primitives = end - start;
                let max_bvh_nodes = 2 * n_primitives;
                let node_start = arena
                    .alloc_extend(std::iter::repeat(BVHBuildNode::default()).take(max_bvh_nodes));
                treelets_to_build.push(LBVHTreelet {
                    start_index: start,
                    n_primitive: n_primitives,
                    node_start,
                    root_index: 0,
                });
            }
        }
        let atomic_total = AtomicUsize::new(0);
        let mut ordered_prim_offset = AtomicUsize::new(0);
        //TODO parallel
        let mut finished_treelets = Vec::with_capacity(treelets_to_build.len());
        for i in 0..treelets_to_build.len() {
            let mut nodes_created = 0;
            let first_bit_index = 29 - 12;
            let treelet = treelets_to_build[i];
            self.emit_lbvh(
                arena,
                &mut treelets_to_build[i],
                0,
                primitive_info,
                &morton_prims[treelet.start_index..],
                treelet.n_primitive,
                &mut nodes_created,
                ordered_prims,
                &mut ordered_prim_offset,
                first_bit_index,
            );
            let node = treelets_to_build[i].root_node(arena);
            finished_treelets.push(node.index);
            atomic_total.fetch_add(nodes_created, std::sync::atomic::Ordering::SeqCst);
        }
        *total_node = atomic_total.load(std::sync::atomic::Ordering::SeqCst);

        let length = finished_treelets.len();
        self.build_upper_sah(
            arena,
            finished_treelets.as_mut_slice(),
            0,
            length,
            total_node,
        )
    }

    fn emit_lbvh(
        &self,
        arena: &mut Arena<BVHBuildNode>,
        treelet: &mut LBVHTreelet,
        mut offset: usize,
        primitive_info: &Vec<BVHPrimitiveInfo>,
        morton_prims: &[MortonPrimitive],
        n_primitives: usize,
        total_nodes: &mut usize,
        ordered_prims: &mut Vec<PrimitiveDt>,
        ordered_prims_offset: &mut AtomicUsize,
        bit_index: i32,
    ) -> usize {
        if bit_index == -1 || n_primitives < self.max_prims_in_node {
            *total_nodes += 1;
            let node = treelet.node_mut(arena, offset);
            treelet.root_index = offset;
            let bounds = Bounds3f::default();
            let first_prim_offset =
                ordered_prims_offset.fetch_add(n_primitives, std::sync::atomic::Ordering::SeqCst);
            for i in 0..n_primitives {
                let primitive_index = morton_prims[i].primitive_index;
                ordered_prims[first_prim_offset + i] = self.primitives[primitive_index].clone();
                bounds.union(&primitive_info[primitive_index].bounds);
            }
            node.init_leaf(first_prim_offset, n_primitives, bounds);
            offset + 1
        } else {
            let mask = 1 << bit_index;
            if (morton_prims[0].morton_code & mask)
                == (morton_prims[n_primitives - 1].morton_code & mask)
            {
                return self.emit_lbvh(
                    arena,
                    treelet,
                    offset,
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
                if (morton_prims[search_start].morton_code & mask)
                    == (morton_prims[mid].morton_code & mask)
                {
                    search_start = mid;
                } else {
                    search_end = mid;
                }
            }

            let split_offset = search_end;
            *total_nodes += 1;
            let origin_offset = offset;
            offset = self.emit_lbvh(
                arena,
                treelet,
                offset + 1,
                primitive_info,
                morton_prims,
                split_offset,
                total_nodes,
                ordered_prims,
                ordered_prims_offset,
                bit_index - 1,
            );
            let c0 = treelet.root_node(arena).index;
            offset = self.emit_lbvh(
                arena,
                treelet,
                offset,
                primitive_info,
                &morton_prims[split_offset..],
                n_primitives - split_offset,
                total_nodes,
                ordered_prims,
                ordered_prims_offset,
                bit_index - 1,
            );
            let c1 = treelet.root_node(arena).index;

            let axis = (bit_index % 3) as usize;
            BVHBuildNode::static_init_interior(arena, axis, origin_offset, c0, c1);
            treelet.root_index = origin_offset;
            offset
        }
    }

    fn build_upper_sah(
        &self,
        arena: &mut Arena<BVHBuildNode>,
        treelet_roots: &mut [usize],
        start: usize,
        end: usize,
        total_nodes: &mut usize,
    ) -> usize {
        let n_nodes = end - start;
        if n_nodes == 1 {
            return treelet_roots[start];
        }
        *total_nodes += 1;
        let (index, _) = arena.alloc(BVHBuildNode::default());
        let bounds = Bounds3f::default();
        for i in start..end {
            bounds.union(&arena.get(treelet_roots[i]).bounds);
        }

        let centroid_bounds = Bounds3f::default();
        for i in start..end {
            let node = arena.get(treelet_roots[i]);
            let centroid = (node.bounds.min + node.bounds.max) * 0.5;
            centroid_bounds.union(&centroid);
        }
        let dim = centroid_bounds.maximum_extent();
        let n_buckets = 12;
        let mut buckets = [BucketInfo::default(); 12];

        for i in start..end {
            let node = arena.get(treelet_roots[i]);
            let centroid = (node.bounds.min[dim] + node.bounds.max[dim]) * 0.5;
            let mut b = n_buckets
                * ((centroid - centroid_bounds.min[dim])
                    / (centroid_bounds.max[dim] - centroid_bounds.min[dim]))
                    as usize;
            if b == n_buckets {
                b = n_buckets - 1;
            }
            buckets[b].count += 1;
            buckets[b].bounds.union(&node.bounds);
        }

        let mut cost = [0.0 as Float; 11];
        for i in 0..n_buckets - 1 {
            let b0 = Bounds3f::default();
            let b1 = Bounds3f::default();
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

        let (_, min_cost_split_bucket, _min_cost) =
            cost.iter()
                .fold((0, 0, Float::MAX), |(current, index, min_cost), cost| {
                    if *cost < min_cost {
                        (current + 1, current, *cost)
                    } else {
                        (current + 1, index, min_cost)
                    }
                });

        let mid = *&mut treelet_roots[start..end + 1]
            .iter_mut()
            .partition_in_place(|index| {
                let node = arena.get(*index);
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
        let c0 = self.build_upper_sah(arena, treelet_roots, start, mid, total_nodes);
        let c1 = self.build_upper_sah(arena, treelet_roots, mid, end, total_nodes);
        BVHBuildNode::static_init_interior(arena, dim, index, c0, c1);
        index
    }

    fn flatten_bvh_tree(
        &mut self,
        arena: &Arena<BVHBuildNode>,
        index: usize,
        offset: &mut usize,
    ) -> usize {
        let node = arena.get(index);
        let (ok, my_offset) = if let Some(nodes) = &mut self.nodes {
            let linear_node = &mut nodes[*offset as usize];
            linear_node.bounds = node.bounds;
            let my_offset = *offset;
            *offset += 1;
            if node.n_primitives > 0 {
                linear_node.primitive_or_second_child_offset = node.first_prim_offset;
                linear_node.n_primitives = node.n_primitives;
                (true, my_offset)
            } else {
                linear_node.axis = node.split_axis;
                linear_node.n_primitives = 0;
                (false, my_offset)
            }
        } else {
            (true, 0)
        };
        if ok {
            return my_offset;
        }

        self.flatten_bvh_tree(arena, node.children[0], offset);
        let primitive_or_second_child_offset =
            self.flatten_bvh_tree(arena, node.children[1], offset);
        if let Some(nodes) = &mut self.nodes {
            let linear_node = &mut nodes[my_offset];
            linear_node.primitive_or_second_child_offset = primitive_or_second_child_offset;
        }
        my_offset
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

    fn intersect(&self, ray: &mut Ray, si: &mut SurfaceInteraction) -> bool {
        if let Some(nodes) = &self.nodes {
            let mut hit = false;
            let inv_dir = Vector3f::new(1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z);
            let dir_is_neg = [
                if inv_dir.x < 0.0 { 1 } else { 0 } as usize,
                if inv_dir.y < 0.0 { 1 } else { 0 } as usize,
                if inv_dir.z < 0.0 { 1 } else { 0 } as usize,
            ];
            let mut to_visit_offset = 0;
            let mut current_node_index = 0;
            let mut nodes_to_vist = [0; 64];
            loop {
                let node = &nodes[current_node_index];
                if node.bounds.intersect_p((&*ray, &inv_dir, dir_is_neg)) {
                    if node.n_primitives > 0 {
                        for i in 0..node.n_primitives {
                            if self.primitives[node.primitive_or_second_child_offset + i]
                                .intersect(ray, si)
                            {
                                hit = true;
                            }
                        }
                        if to_visit_offset == 0 {
                            break;
                        }
                        to_visit_offset -= 1;
                        current_node_index = nodes_to_vist[to_visit_offset];
                    } else {
                        if dir_is_neg[node.axis] != 0 {
                            nodes_to_vist[to_visit_offset] = current_node_index + 1;
                            to_visit_offset += 1;
                            current_node_index = node.primitive_or_second_child_offset;
                        } else {
                            nodes_to_vist[to_visit_offset] = node.primitive_or_second_child_offset;
                            to_visit_offset += 1;
                            current_node_index = current_node_index + 1;
                        }
                    }
                } else {
                    if to_visit_offset == 0 {
                        break;
                    }
                    to_visit_offset -= 1;
                    current_node_index = nodes_to_vist[to_visit_offset];
                }
            }
            hit
        } else {
            false
        }
    }

    fn intersect_p(&self, ray: &Ray) -> bool {
        if let Some(nodes) = &self.nodes {
            let hit = false;
            let inv_dir = Vector3f::new(1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z);
            let dir_is_neg = [
                if inv_dir.x < 0.0 { 1 } else { 0 } as usize,
                if inv_dir.y < 0.0 { 1 } else { 0 } as usize,
                if inv_dir.z < 0.0 { 1 } else { 0 } as usize,
            ];
            let mut to_visit_offset = 0;
            let mut current_node_index = 0;
            let mut nodes_to_vist = [0; 64];
            loop {
                let node = &nodes[current_node_index];
                if node.bounds.intersect_p((&*ray, &inv_dir, dir_is_neg)) {
                    if node.n_primitives > 0 {
                        for i in 0..node.n_primitives {
                            if self.primitives[node.primitive_or_second_child_offset + i]
                                .intersect_p(ray)
                            {
                                return true;
                            }
                        }
                        if to_visit_offset == 0 {
                            break;
                        }
                        to_visit_offset -= 1;
                        current_node_index = nodes_to_vist[to_visit_offset];
                    } else {
                        if dir_is_neg[node.axis] != 0 {
                            nodes_to_vist[to_visit_offset] = current_node_index + 1;
                            to_visit_offset += 1;
                            current_node_index = node.primitive_or_second_child_offset;
                        } else {
                            nodes_to_vist[to_visit_offset] = node.primitive_or_second_child_offset;
                            to_visit_offset += 1;
                            current_node_index = current_node_index + 1;
                        }
                    }
                } else {
                    if to_visit_offset == 0 {
                        break;
                    }
                    to_visit_offset -= 1;
                    current_node_index = nodes_to_vist[to_visit_offset];
                }
            }
            hit
        } else {
            false
        }
    }

    fn get_area_light(&self) -> Option<AreaLightDt> {
        unimplemented!(
            "Aggregate does not support get_area_light method, use GeometricPrimitive instead"
        )
    }

    fn get_material(&self) -> Option<MaterialDt> {
        unimplemented!(
            "Aggregate does not support get_material method, use GeometricPrimitive instead"
        )
    }

    fn compute_scattering_functions(
        &self,
        _si: &mut SurfaceInteraction,
        _mode: TransportMode,
        _allow_multiple_lobes: bool,
    ) {
        unimplemented!("Aggregate does not support compute_scattering_function method, use GeometricPrimitive instead")
    }
}

#[derive(Default, Copy, Clone)]
struct BucketInfo {
    count: i32,
    bounds: Bounds3f,
}
