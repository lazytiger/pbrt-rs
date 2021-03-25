use crate::core::transform::Transformf;

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

        fn as_any(&self) -> &dyn Any {
            self
        }
    };
}
