use crate::core::geometry::{Bounds3f, Ray};
use crate::core::interaction::SurfaceInteraction;
use crate::core::light::AreaLight;
use crate::core::material::{Material, TransportMode};
use crate::core::medium::MediumInterface;
use crate::core::shape::Shape;
use crate::core::transform::{AnimatedTransform, Transform};
use crate::Float;
use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

pub trait Primitive {
    fn as_any(&self) -> &dyn Any;
    fn world_bound(&self) -> Bounds3f;
    fn intersect(&self, r: &mut Ray, si: &mut SurfaceInteraction) -> bool;
    fn intersect_p(&self, r: &Ray) -> bool;
    fn get_area_light(&self) -> Option<Arc<Box<AreaLight>>>;
    fn get_material(&self) -> Option<Arc<Box<Material>>>;
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

    fn intersect(&self, r: &mut Ray, si: &mut SurfaceInteraction) -> (bool) {
        let mut hit: Float = 0.0;
        if !self.shape.intersect(r, &mut hit, si, true) {
            return false;
        }
        r.t_max = hit;
        si.primitive = Some(self);
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

    fn get_area_light(&self) -> Option<Arc<Box<AreaLight>>> {
        self.area_light.clone()
    }

    fn get_material(&self) -> Option<Arc<Box<Material>>> {
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

struct TransformedPrimitive {
    primitive: Option<Arc<Box<Primitive>>>,
    primitive_to_world: AnimatedTransform,
}

impl TransformedPrimitive {
    pub fn new(
        primitive: Option<Arc<Box<Primitive>>>,
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
        let mut ray = Ray::from((&interpolate_prim_to_world.inverse(), &*r));
        if let Some(primitive) = &self.primitive {
            primitive.intersect_p(&ray)
        } else {
            false
        }
    }

    fn get_area_light(&self) -> Option<Arc<Box<dyn AreaLight>>> {
        None
    }

    fn get_material(&self) -> Option<Arc<Box<dyn Material>>> {
        None
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
