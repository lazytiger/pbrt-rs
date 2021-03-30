use crate::core::geometry::{Bounds3, Bounds3f, Point3f, Ray, Union, Vector3f};
use crate::core::interaction::SurfaceInteraction;
use crate::core::light::AreaLight;
use crate::core::material::{Material, TransportMode};
use crate::core::primitive::Primitive;
use crate::core::RealNum;
use crate::Float;
use num::traits::real::Real;
use std::any::Any;
use std::cmp::Ordering;
use std::mem::swap;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;

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
    children: Box<[BVHBuildNode; 2]>,
    split_axis: i32,
    first_prim_offset: i32,
    n_primitives: i32,
}

impl BVHBuildNode {
    fn init_leaf(&mut self, first: i32, n: i32, b: Bounds3f) {
        self.first_prim_offset = first;
        self.n_primitives = n;
        self.bounds = b;
    }

    fn init_interior(&mut self, axis: i32, c0: BVHBuildNode, c1: BVHBuildNode) {
        self.bounds = c0.bounds.union(&c1.bounds);
        self.children = Box::new([c0, c1]);
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
struct LBVHTreelet {
    start_index: i32,
    n_primitive: i32,
    build_nodes: Vec<BVHBuildNode>,
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
            let info = BVHPrimitiveInfo::new(i, accel.primitives[i].world_bound());
            primitive_infos.push(info);
        }

        let mut total_nodes = 0;
        let mut ordered_prims = Vec::with_capacity(accel.primitives.len());
        let root = if let SplitMethod::HLBVH = accel.split_method {
            accel.hlbvh_build(&primitive_infos, &mut total_nodes, &mut ordered_prims)
        } else {
            accel.recursive_build(
                primitive_infos.as_mut_slice(),
                0,
                accel.primitives.len() as i32,
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

    fn recursive_build(
        &self,
        primitive_info: &mut [BVHPrimitiveInfo],
        start: i32,
        end: i32,
        total_nodes: &mut i32,
        ordered_prims: &mut Vec<Arc<Box<Primitive>>>,
    ) -> BVHBuildNode {
        let mut node = BVHBuildNode::default();
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
                ordered_prims.push(self.primitives[prim_num].clone());
            }
            node.init_leaf(first_prim_offset as i32, n_primitives, bounds);
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
                    ordered_prims.push(self.primitives[prim_num].clone());
                }
                node.init_leaf(first_prim_offset as i32, n_primitives, bounds);
                return node;
            } else {
                match self.split_method {
                    SplitMethod::Middle => {
                        let p_mid = (centroid_bounds.min[dim] + centroid_bounds.max[dim]) / 2.0;
                        mid = *&mut primitive_info[start as usize..end as usize + 1]
                            .iter_mut()
                            .partition_in_place(|&pi| pi.centroid[dim] < p_mid)
                            as i32
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
                            if n_primitives > self.max_prims_in_node || min_cost < leaf_cost {
                                mid = *&mut primitive_info[start as usize..end as usize + 1]
                                    .iter_mut()
                                    .partition_in_place(|pi| {
                                        let mut b = n_buckets as Float
                                            * centroid_bounds.offset(&pi.centroid)[dim];
                                        if b == n_buckets as Float {
                                            b = (n_buckets - 1) as Float;
                                        }
                                        b < min_cost_split_bucket as Float
                                    }) as i32
                                    + start;
                            } else {
                                let first_prim_offset = ordered_prims.len();
                                for i in start..end {
                                    let prim_num = primitive_info[i as usize].primitive_number;
                                    ordered_prims.push(self.primitives[prim_num].clone());
                                }
                                node.init_leaf(first_prim_offset as i32, n_primitives, bounds);
                                return node;
                            }
                        }
                    }
                }
                node.init_interior(
                    dim as i32,
                    self.recursive_build(primitive_info, start, mid, total_nodes, ordered_prims),
                    self.recursive_build(primitive_info, mid, end, total_nodes, ordered_prims),
                );
            }
        }

        node
    }

    fn hlbvh_build(
        &self,
        primitive_info: &Vec<BVHPrimitiveInfo>,
        total_node: &mut i32,
        ordered_prims: &mut Vec<Arc<Box<Primitive>>>,
    ) -> BVHBuildNode {
        let mut node = BVHBuildNode::default();
        node
    }

    fn emit_lbvh(
        &self,
        build_nodes: Vec<BVHBuildNode>,
        primitive_info: &Vec<BVHPrimitiveInfo>,
        morton_prims: &[MortonPrimitive],
        total_nodes: &mut i32,
        ordered_prims: Vec<Arc<Box<Primitive>>>,
        ordered_prims_offset: &[AtomicI32],
        bit_index: i32,
    ) -> BVHBuildNode {
        let mut node = BVHBuildNode::default();
        node
    }

    fn build_upper_sah(
        &self,
        tree_let_roots: &Vec<&mut BVHBuildNode>,
        start: i32,
        end: i32,
        total_in_nodes: &[i32],
    ) -> BVHBuildNode {
        let mut node = BVHBuildNode::default();
        node
    }

    fn flatten_bvh_tree(&self, node: &BVHBuildNode, offset: &mut i32) -> i32 {
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
