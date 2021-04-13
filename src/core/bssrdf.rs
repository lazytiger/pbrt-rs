use crate::core::pbrt::Float;
use std::sync::{Arc, Mutex, RwLock};

pub fn fresnel_moment1(inv_eta: Float) -> Float {
    todo!()
}

pub fn fresnel_moment2(inv_eta_: Float) -> Float {
    todo!()
}

pub type BSSRDFDt = Arc<Box<dyn BSSRDF>>;
pub type BSSRDFDtMut = Arc<Mutex<Box<dyn BSSRDF>>>;
pub type BSSRDFDtRw = Arc<RwLock<Box<dyn BSSRDF>>>;

pub trait BSSRDF {}
