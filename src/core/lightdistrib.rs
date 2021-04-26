use crate::core::{
    geometry::{Point3f, Point3i},
    integrator::compute_light_power_distribution,
    sampling::Distribution1D,
    scene::Scene,
};
use std::{any::Any, sync::Arc};

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

pub struct SpatialLightDistribution {
    scene: Scene,
    n_voxel: [usize; 3],
    hash_table_size: usize,
}

impl SpatialLightDistribution {
    pub fn new(scene: &Scene, max_voxels: usize) -> Self {
        todo!()
    }

    fn compute_distribution(&self, scene: &Scene, pi: Point3i) -> Distribution1D {
        todo!()
    }
}

impl LightDistribution for SpatialLightDistribution {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lookup(&self, p: &Point3f, _: Option<&Scene>) -> &Distribution1D {
        todo!()
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
