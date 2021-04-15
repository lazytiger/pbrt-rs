use crate::core::{geometry::Point3f, sampling::Distribution1D, scene::Scene};
use std::{any::Any, sync::Arc};

pub type LightDistributionDt = Arc<Box<dyn LightDistribution>>;

pub trait LightDistribution {
    fn as_any(&self) -> &dyn Any;
    fn lookup(&self, p: &Point3f) -> &Distribution1D;
}

pub struct UniformLightDistribution {
    distrib: Distribution1D,
}

impl UniformLightDistribution {
    pub fn new(scene: &Scene) -> Self {
        todo!()
    }
}

impl LightDistribution for UniformLightDistribution {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lookup(&self, p: &Point3f) -> &Distribution1D {
        todo!()
    }
}

pub struct PowerLightDistribution {
    distrib: Distribution1D,
}

impl PowerLightDistribution {
    pub fn new(scene: &Scene) -> Self {
        todo!()
    }
}

impl LightDistribution for PowerLightDistribution {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lookup(&self, p: &Point3f) -> &Distribution1D {
        todo!()
    }
}

pub fn create_light_sample_distribution(name: String, scene: &Scene) -> LightDistributionDt {
    todo!()
}
