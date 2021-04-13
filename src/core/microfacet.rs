use std::sync::Arc;

pub trait MicrofacetDistribution {}

pub type MicrofacetDistributionDt = Arc<Box<dyn MicrofacetDistribution>>;
