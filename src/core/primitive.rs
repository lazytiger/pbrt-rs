use crate::core::{
    geometry::{Bounds3f, Ray},
    interaction::SurfaceInteraction,
    light::LightDt,
    material::{Material, MaterialDt, TransportMode},
    medium::MediumInterface,
    pbrt::Float,
    shape::{Shape, ShapeDt},
    transform::AnimatedTransform,
};
use std::{
    any::Any,
    fmt::Debug,
    sync::{Arc, Mutex, RwLock},
};

pub trait Primitive: Debug {
    fn as_any(&self) -> &dyn Any;
    fn world_bound(&self) -> Bounds3f;
    fn intersect(&self, r: &mut Ray, si: &mut SurfaceInteraction) -> bool;
    fn intersect_p(&self, r: &Ray) -> bool;
    fn get_area_light(&self) -> Option<LightDt>;
    fn get_material(&self) -> Option<MaterialDt>;
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    );
}

#[derive(Debug)]
pub struct GeometricPrimitive {
    shape: ShapeDt,
    material: Option<MaterialDt>,
    area_light: Option<LightDt>,
    medium_interface: MediumInterface,
}

impl GeometricPrimitive {
    pub fn new(
        shape: ShapeDt,
        material: Option<MaterialDt>,
        area_light: Option<LightDt>,
        medium_interface: MediumInterface,
    ) -> GeometricPrimitive {
        GeometricPrimitive {
            shape,
            material,
            area_light,
            medium_interface,
        }
    }
}

impl Primitive for GeometricPrimitive {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn world_bound(&self) -> Bounds3f {
        self.shape.world_bound()
    }

    fn intersect(&self, r: &mut Ray, si: &mut SurfaceInteraction) -> bool {
        let mut hit: Float = 0.0;
        if !self.shape.intersect(r, &mut hit, si, true) {
            return false;
        }
        r.t_max = hit;
        //si.primitive = Some(self); todo in upper calling
        if self.medium_interface.is_medium_transition() {
            si.medium_interface = self.medium_interface.clone();
        } else {
            si.medium_interface = MediumInterface::new(r.medium.clone(), r.medium.clone());
        }
        true
    }

    fn intersect_p(&self, r: &Ray) -> bool {
        self.shape.intersect_p(r, true)
    }

    fn get_area_light(&self) -> Option<LightDt> {
        self.area_light.clone()
    }

    fn get_material(&self) -> Option<MaterialDt> {
        self.material.clone()
    }

    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    ) {
        if let Some(material) = &self.material {
            material.compute_scattering_functions(si, mode, allow_multiple_lobes);
        }
    }
}

#[derive(Debug)]
struct TransformedPrimitive {
    primitive: Option<PrimitiveDt>,
    primitive_to_world: AnimatedTransform,
}

impl TransformedPrimitive {
    pub fn new(
        primitive: Option<PrimitiveDt>,
        primitive_to_world: AnimatedTransform,
    ) -> TransformedPrimitive {
        TransformedPrimitive {
            primitive,
            primitive_to_world,
        }
    }
}

impl Primitive for TransformedPrimitive {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn world_bound(&self) -> Bounds3f {
        if let Some(primitive) = &self.primitive {
            self.primitive_to_world
                .motion_bounds(&primitive.world_bound())
        } else {
            Default::default()
        }
    }

    fn intersect(&self, r: &mut Ray, si: &mut SurfaceInteraction) -> bool {
        let interpolate_prim_to_world = self.primitive_to_world.interpolate(r.time);
        let mut ray = Ray::from((&interpolate_prim_to_world.inverse(), &*r));
        if let Some(primitive) = &self.primitive {
            if !primitive.intersect(&mut ray, si) {
                return false;
            }
            r.t_max = ray.t_max;
        }
        if interpolate_prim_to_world.is_identify() {
            *si = &interpolate_prim_to_world * &*si;
        }
        true
    }

    fn intersect_p(&self, r: &Ray) -> bool {
        let interpolate_prim_to_world = self.primitive_to_world.interpolate(r.time);
        let ray = Ray::from((&interpolate_prim_to_world.inverse(), &*r));
        if let Some(primitive) = &self.primitive {
            primitive.intersect_p(&ray)
        } else {
            false
        }
    }

    fn get_area_light(&self) -> Option<LightDt> {
        None
    }

    fn get_material(&self) -> Option<MaterialDt> {
        None
    }

    fn compute_scattering_functions(
        &self,
        _si: &mut SurfaceInteraction,
        _mode: TransportMode,
        _allow_multiple_lobes: bool,
    ) {
        unimplemented!()
    }
}

pub type PrimitiveDt = Arc<Box<dyn Primitive + Sync + Send>>;
pub type PrimitiveDtMut = Arc<Mutex<Box<dyn Primitive + Sync + Send>>>;
pub type PrimitiveDtRw = Arc<RwLock<Box<dyn Primitive + Sync + Send>>>;
