use crate::core::interaction::SurfaceInteraction;
use std::{
    any::Any,
    fmt::Debug,
    sync::{Arc, Mutex, RwLock},
};

#[derive(Clone, Copy)]
pub enum TransportMode {
    Radiance,
    Importance,
}

pub trait Material: Debug {
    fn as_any(&self) -> &dyn Any;
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    );
}

pub type MaterialDt = Arc<Box<dyn Material>>;
pub type MaterialDtMut = Arc<Mutex<Box<dyn Material>>>;
pub type MaterialDtRw = Arc<RwLock<Box<dyn Material>>>;
