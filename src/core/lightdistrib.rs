use crate::core::{
    geometry::{Bounds3f, Normal3, Normal3f, Point2f, Point3f, Point3i, Vector3f},
    integrator::compute_light_power_distribution,
    interaction::BaseInteraction,
    light::VisibilityTester,
    lowdiscrepancy::radical_inverse,
    medium::MediumInterface,
    pbrt::{clamp, Float},
    sampling::Distribution1D,
    scene::Scene,
};
use atom::AtomSetOnce;
use atomic::{Atomic, Ordering};
use std::{
    any::Any,
    sync::{atomic::Ordering::Acquire, Arc},
};

pub type LightDistributionDt = Arc<Box<dyn LightDistribution + Sync + Send>>;

pub trait LightDistribution {
    fn as_any(&self) -> &dyn Any;
    fn lookup(&self, p: &Point3f, _: Option<&Scene>) -> &Distribution1D;
}

pub struct UniformLightDistribution {
    distrib: Distribution1D,
}

impl UniformLightDistribution {
    pub fn new(scene: &Scene) -> Self {
        let prob = vec![1.0; scene.lights.len()];
        let distrib = Distribution1D::new(prob.as_slice());
        Self { distrib }
    }
}

impl LightDistribution for UniformLightDistribution {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lookup(&self, p: &Point3f, _: Option<&Scene>) -> &Distribution1D {
        &self.distrib
    }
}

pub struct PowerLightDistribution {
    distrib: Distribution1D,
}

impl PowerLightDistribution {
    pub fn new(scene: &Scene) -> Self {
        let distrib = compute_light_power_distribution(scene);
        Self {
            distrib: distrib.unwrap(),
        }
    }
}

impl LightDistribution for PowerLightDistribution {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lookup(&self, p: &Point3f, _: Option<&Scene>) -> &Distribution1D {
        &self.distrib
    }
}

struct HashEntry {
    packed_pos: Atomic<u64>,
    distribution: AtomSetOnce<Arc<Distribution1D>>,
}

pub struct SpatialLightDistribution {
    n_voxel: [usize; 3],
    hash_table_size: usize,
    hash_table: Vec<HashEntry>,
}

const INVALID_PACKED_POS: u64 = 0xffffffffffffffff;

impl SpatialLightDistribution {
    pub fn new(scene: &Scene, max_voxels: usize) -> Self {
        let b = scene.world_bound();
        let diag = b.diagonal();
        let b_max = diag[b.maximum_extent()];
        let mut n_voxel = [0; 3];
        for i in 0..3 {
            n_voxel[i] = std::cmp::max(1, (diag[i] / b_max * max_voxels as Float).round() as usize);
        }
        let hash_table_size = 4 * n_voxel[0] * n_voxel[1] * n_voxel[2];
        let mut hash_table = Vec::with_capacity(hash_table_size);
        for i in 0..hash_table_size {
            let entry = HashEntry {
                packed_pos: Atomic::new(INVALID_PACKED_POS),
                distribution: AtomSetOnce::empty(),
            };
            hash_table.push(entry);
        }
        Self {
            n_voxel,
            hash_table_size,
            hash_table,
        }
    }

    fn compute_distribution(&self, scene: &Scene, pi: Point3i) -> Distribution1D {
        let p0 = Point3f::new(
            pi[0] as Float / self.n_voxel[0] as Float,
            pi[1] as Float / self.n_voxel[1] as Float,
            pi[2] as Float / self.n_voxel[2] as Float,
        );

        let p1 = Point3f::new(
            pi[0] as Float + 1.0 / self.n_voxel[0] as Float,
            pi[1] as Float + 1.0 / self.n_voxel[1] as Float,
            pi[2] as Float + 1.0 / self.n_voxel[2] as Float,
        );

        let voxel_bounds =
            Bounds3f::from((scene.world_bound().lerp(&p0), scene.world_bound().lerp(&p1)));

        let n_samples = 128;
        let mut light_contrib = vec![0.0; scene.lights.len()];
        for i in 0..n_samples {
            let po = voxel_bounds.lerp(&Point3f::new(
                radical_inverse(0, i),
                radical_inverse(1, i),
                radical_inverse(2, i),
            ));
            let intr = BaseInteraction::new(
                po,
                Normal3f::default(),
                Vector3f::default(),
                Vector3f::new(1.0, 0.0, 0.0),
                0.0,
                MediumInterface::default(),
            );
            let u = Point2f::new(radical_inverse(3, i), radical_inverse(4, i));
            for j in 0..scene.lights.len() {
                let mut pdf = 0.0;
                let mut wi = Vector3f::default();
                let mut vis = VisibilityTester::default();
                let li = scene.lights[j].sample_li(&intr, &u, &mut wi, &mut pdf, &mut vis);
                if pdf > 0.0 {
                    light_contrib[j] += li.y_value() / pdf;
                }
            }
        }
        let sum_contrib: Float = light_contrib.iter().sum();
        let avg_contrib = sum_contrib / (n_samples as usize * light_contrib.len()) as Float;
        let min_contrib: Float = if avg_contrib > 0.0 {
            0.001 * avg_contrib
        } else {
            1.0
        };
        for i in 0..light_contrib.len() {
            light_contrib[i] = min_contrib.max(light_contrib[i]);
        }
        Distribution1D::new(light_contrib.as_slice())
    }
}

impl LightDistribution for SpatialLightDistribution {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lookup(&self, p: &Point3f, scene: Option<&Scene>) -> &Distribution1D {
        let scene = scene.unwrap();
        let offset = scene.world_bound().offset(p);
        let mut pi = Point3i::default();
        for i in 0..3 {
            pi[i] = clamp(
                (offset[i] * self.n_voxel[i] as Float) as i32,
                0,
                self.n_voxel[i] as i32 - 1,
            );
        }
        let packed_pos = (pi[0] as u64) << 40 | (pi[1] as u64) << 20 | (pi[2] as u64);
        let mut hash = packed_pos;
        hash ^= (hash >> 31);
        hash *= 0x7fb5d329728ea185;
        hash ^= (hash >> 27);
        hash *= 0x81dadef4bc2dd44d;
        hash ^= (hash >> 33);
        hash %= self.hash_table_size as u64;

        let mut step = 1;
        loop {
            let entry = &self.hash_table[hash as usize];
            let entry_packed_pos = entry.packed_pos.load(Ordering::Acquire);
            if entry_packed_pos == packed_pos {
                while let None = entry.distribution.get(Acquire) {}
                return entry.distribution.get(Ordering::Acquire).unwrap();
            } else if entry_packed_pos != INVALID_PACKED_POS {
                hash += step * step;
                if hash >= self.hash_table_size as u64 {
                    hash %= self.hash_table_size as u64;
                }
                step += 1;
            } else {
                let invalid = INVALID_PACKED_POS;
                if entry
                    .packed_pos
                    .compare_exchange_weak(invalid, packed_pos, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    let dist = self.compute_distribution(scene, pi);
                    entry
                        .distribution
                        .set_if_none(Arc::new(dist), Ordering::Release);
                    return entry.distribution.get(Ordering::SeqCst).unwrap();
                }
            }
        }
    }
}

pub fn create_light_sample_distribution(name: String, scene: &Scene) -> LightDistributionDt {
    if name == "uniform" || scene.lights.len() == 1 {
        Arc::new(Box::new(UniformLightDistribution::new(scene)))
    } else if name == "power" {
        Arc::new(Box::new(PowerLightDistribution::new(scene)))
    } else if name == "spatial" {
        Arc::new(Box::new(SpatialLightDistribution::new(scene, 64)))
    } else {
        panic!("unknown light sample distribution type {}", name);
    }
}
