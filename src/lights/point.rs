use crate::{
    core::{
        geometry::{Normal3f, Point2f, Point3f, Ray, Vector3f},
        interaction::{BaseInteraction, Interaction, MediumInteraction},
        light::{BaseLight, Light, LightFlags, VisibilityTester},
        medium::MediumInterface,
        pbrt::PI,
        sampling::{uniform_sample_sphere, uniform_sphere_pdf},
        spectrum::Spectrum,
        transform::{Point3Ref, Transformf},
    },
    impl_base_light, Float,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

#[derive(Deref, DerefMut, Debug)]
pub struct PointLight {
    #[deref]
    #[deref_mut]
    base: BaseLight,
    p_light: Point3f,
    i: Spectrum,
}

impl PointLight {
    pub fn new(light_to_world: Transformf, medium_interface: MediumInterface, i: Spectrum) -> Self {
        let p_light = &light_to_world * Point3Ref(&Point3f::default());
        Self {
            base: BaseLight::new(
                LightFlags::DELTA_POSITION,
                light_to_world,
                medium_interface,
                1,
            ),
            p_light,
            i,
        }
    }
}

impl Light for PointLight {
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
        *wi = (self.p_light - iref.as_base().p).normalize();
        *pdf = 1.0;
        *vis = VisibilityTester::new(
            Arc::new(Box::new(iref.as_base().clone())),
            Arc::new(Box::new(BaseInteraction::from((
                self.p_light,
                iref.as_base().time,
                self.medium_interface.clone(),
            )))),
        );
        self.i / self.p_light.distance_square(&iref.as_base().p)
    }

    fn power(&self) -> Spectrum {
        self.i * (4.0 * PI)
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
        *ray = Ray::new(
            self.p_light,
            uniform_sample_sphere(u1),
            Float::INFINITY,
            time,
            self.medium_interface.inside.clone(),
        );
        *n_light = ray.d;
        *pdf_pos = 1.0;
        *pdf_dir = uniform_sphere_pdf();
        self.i
    }

    fn pdf_le(&self, ray: &Ray, n_light: &Vector3f, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        *pdf_pos = 0.0;
        *pdf_dir = uniform_sphere_pdf();
    }

    impl_base_light!();
}
