use crate::core::geometry::{Normal3f, Vector3f};
use crate::core::transform::Transformf;

pub mod cone;
pub mod cylinder;
pub mod sphere;

pub(crate) struct BaseShape {
    object_to_world: Transformf,
    world_to_object: Transformf,
    reverse_orientation: bool,
    transform_swap_handedness: bool,
}

impl BaseShape {
    pub fn new(
        object_to_world: Transformf,
        world_to_object: Transformf,
        reverse_orientation: bool,
    ) -> BaseShape {
        BaseShape {
            object_to_world,
            world_to_object,
            reverse_orientation,
            transform_swap_handedness: false,
        }
    }
}

#[macro_export]
macro_rules! impl_base_shape {
    () => {
        fn object_to_world(&self) -> &Transformf {
            &self.base.object_to_world
        }

        fn world_to_object(&self) -> &Transformf {
            &self.base.world_to_object
        }

        fn reverse_orientation(&self) -> bool {
            self.base.reverse_orientation
        }

        fn transform_swap_handedness(&self) -> bool {
            self.base.transform_swap_handedness
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    };
}

pub fn compute_normal_differential(
    dpdu: &Vector3f,
    dpdv: &Vector3f,
    d2pduu: &Vector3f,
    d2pduv: &Vector3f,
    d2pdvv: &Vector3f,
) -> (Normal3f, Normal3f) {
    let e = dpdu.dot(&dpdu);
    let f = dpdu.dot(&dpdv);
    let g = dpdv.dot(&dpdv);
    let n = dpdu.cross(&dpdv).normalize();
    let ee = n.dot(&d2pduu);
    let ff = n.dot(&d2pduv);
    let gg = n.dot(&d2pdvv);

    let inv_egf2 = 1.0 / (e * g - f * f);
    let dndu = *dpdu * inv_egf2 * (ff * f - ee * g) + *dpdv * inv_egf2 * (ee * f - ff * e);
    let dndv = *dpdu * inv_egf2 * (gg * f - ff * g) + *dpdv * inv_egf2 * (ff * f - gg * e);
    (dndu, dndv)
}
