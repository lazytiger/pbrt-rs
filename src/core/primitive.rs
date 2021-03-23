use crate::core::geometry::{Bounds3f, Ray};
use crate::core::interaction::SurfaceInteraction;
use crate::core::light::AreaLight;
use crate::core::material::{Material, TransportMode};
use crate::core::medium::MediumInterface;
use crate::core::shape::Shape;
use std::any::Any;
use std::sync::Arc;

pub trait Primitive {
    fn as_any(&self) -> &dyn Any;
    fn world_bound(&self) -> Bounds3f;
    fn intersect(&self, r: &mut Ray) -> (bool, SurfaceInteraction);
    fn intersect_p(&self, r: &Ray) -> bool;
    fn get_area_light(&self) -> Option<&AreaLight>;
    fn get_material(&self) -> Option<&Material>;
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    );
}

pub struct GeometricPrimitive {
    shape: Arc<Box<Shape>>,
    material: Option<Arc<Box<Material>>>,
    area_light: Option<Arc<Box<AreaLight>>>,
    medium_interface: MediumInterface,
}

impl GeometricPrimitive {
    pub fn new(
        shape: Arc<Box<Shape>>,
        material: Option<Arc<Box<Material>>>,
        area_light: Option<Arc<Box<AreaLight>>>,
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

    fn intersect(&self, r: &mut Ray) -> (bool, SurfaceInteraction) {
        let (ok, hit, mut si) = self.shape.intersect(r, true);
        if !ok {
            return (false, si);
        }
        r.t_max = hit;
        si.primitive = Some(self);
    }

    fn intersect_p(&self, r: &Ray) -> bool {
        unimplemented!()
    }

    fn get_area_light(&self) -> Option<&dyn AreaLight> {
        unimplemented!()
    }

    fn get_material(&self) -> Option<&dyn Material> {
        unimplemented!()
    }

    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    ) {
        unimplemented!()
    }
}
