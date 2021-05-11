use crate::{
    core::{
        geometry::{Normal3f, Point2f, Point3f, Ray, Vector3f},
        interaction::{BaseInteraction, Interaction},
        light::{BaseLight, Light, LightFlags, VisibilityTester},
        medium::MediumInterface,
        pbrt::{Float, PI},
        sampling::concentric_sample_disk,
        scene::Scene,
        spectrum::Spectrum,
        transform::{Transformf, Vector3Ref},
    },
    impl_base_light,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

#[derive(Debug, Deref, DerefMut)]
pub struct DistantLight {
    #[deref]
    #[deref_mut]
    base: BaseLight,
    l: Spectrum,
    w_light: Vector3f,
    world_center: Point3f,
    world_radius: Float,
}

impl DistantLight {
    pub fn new(light_to_world: Transformf, l: Spectrum, w: Vector3f) -> Self {
        let w_light = (&light_to_world * Vector3Ref(&w)).normalize();
        let base = BaseLight::new(
            LightFlags::DELTA_DIRECTION,
            light_to_world,
            MediumInterface::default(),
            1,
        );
        Self {
            base,
            w_light,
            l,
            world_radius: 0.0,
            world_center: Point3f::default(),
        }
    }
}

impl Light for DistantLight {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn sample_li(
        &self,
        iref: &dyn Interaction,
        u: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut f32,
        vis: &mut VisibilityTester,
    ) -> Spectrum {
        *wi = self.w_light;
        *pdf = 1.0;
        let p_outside = iref.as_base().p + self.w_light * (2.0 * self.world_radius);
        *vis = VisibilityTester::new(
            Arc::new(Box::new(iref.as_base().clone())),
            Arc::new(Box::new(BaseInteraction::from((
                p_outside,
                iref.as_base().time,
                self.medium_interface.clone(),
            )))),
        );
        self.l
    }

    fn power(&self) -> Spectrum {
        self.l * (PI * self.world_radius * self.world_radius)
    }

    fn pre_process(&mut self, scene: &Scene) {
        scene
            .world_bound()
            .bounding_sphere(&mut self.world_center, &mut self.world_radius);
    }

    fn pdf_li(&self, iref: &dyn Interaction, wi: &Vector3f) -> f32 {
        0.0
    }

    fn sample_le(
        &self,
        u1: &Point2f,
        u2: &Point2f,
        time: f32,
        ray: &mut Ray,
        n_light: &mut Normal3f,
        pdf_pos: &mut f32,
        pdf_dir: &mut f32,
    ) -> Spectrum {
        let (v1, v2) = self.w_light.coordinate_system();
        let cd = concentric_sample_disk(u1);
        let p_disk = self.world_center + (v1 * cd.x + v2 * cd.y) * self.world_radius;
        *ray = Ray::new(
            p_disk + self.w_light * self.world_radius,
            -self.w_light,
            Float::INFINITY,
            time,
            None,
        );
        *n_light = ray.d;
        *pdf_pos = 1.0 / (PI * self.world_radius * self.world_radius);
        *pdf_dir = 1.0;
        self.l
    }

    fn pdf_le(&self, ray: &Ray, n_light: &Vector3f, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        *pdf_pos = 1.0 / (PI * self.world_radius * self.world_radius);
        *pdf_dir = 0.0;
    }

    impl_base_light!();
}
