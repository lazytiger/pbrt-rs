use crate::{
    core::{
        geometry::{
            spherical_phi, spherical_theta, Normal3f, Point2f, Point2i, Point3f, Ray, Vector3f,
        },
        imageio::read_image,
        interaction::{BaseInteraction, Interaction},
        light::{BaseLight, Light, LightFlags, VisibilityTester},
        medium::MediumInterface,
        mipmap::{ImageWrap, MIPMap},
        pbrt::{INV_2_PI, INV_PI, PI},
        sampling::{uniform_sample_sphere, uniform_sphere_pdf},
        spectrum::{RGBSpectrum, Spectrum},
        transform::{Point3Ref, Transformf, Vector3Ref},
    },
    impl_base_light, Float,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

#[derive(Debug, Deref, DerefMut)]
pub struct GonioPhotometricLight {
    #[deref]
    #[deref_mut]
    base: BaseLight,
    p_light: Point3f,
    i: Spectrum,
    mipmap: Option<MIPMap<RGBSpectrum>>,
}

impl GonioPhotometricLight {
    pub fn new(
        light_to_world: Transformf,
        medium_interface: MediumInterface,
        i: Spectrum,
        texname: String,
    ) -> Self {
        let p_light = &light_to_world * Point3Ref(&Point3f::default());
        let base = BaseLight::new(
            LightFlags::DELTA_POSITION,
            light_to_world,
            medium_interface,
            1,
        );
        let mut resolution = Point2i::default();
        let texels = read_image(texname, &mut resolution);
        let mut mipmap = None;
        if let Some(texels) = texels {
            mipmap.replace(MIPMap::new(
                resolution,
                texels,
                false,
                8.0,
                ImageWrap::Repeat,
            ));
        }
        Self {
            base,
            p_light,
            i,
            mipmap,
        }
    }

    pub fn scale(&self, w: &Vector3f) -> Spectrum {
        let mut wp = (&self.world_to_light * Vector3Ref(w)).normalize();
        std::mem::swap(&mut wp.y, &mut wp.z);
        let theta = spherical_theta(&wp);
        let phi = spherical_phi(&wp);
        let st = Point2f::new(phi * INV_2_PI, theta * INV_PI);
        if let Some(mipmap) = &self.mipmap {
            mipmap.lookup(&st, 0.0)
        } else {
            1.0.into()
        }
    }
}

impl Light for GonioPhotometricLight {
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
        self.i * self.scale(&-*wi) / self.p_light.distance_square(&iref.as_base().p)
    }

    fn power(&self) -> Spectrum {
        self.i
            * if let Some(mipmap) = &self.mipmap {
                mipmap.lookup(&Point2f::new(0.5, 0.5), 0.5)
            } else {
                1.0.into()
            }
            * (4.0 * PI)
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
        self.i * self.scale(&ray.d)
    }

    fn pdf_le(&self, ray: &Ray, n_light: &Vector3f, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        *pdf_pos = 0.0;
        *pdf_dir = uniform_sphere_pdf();
    }

    impl_base_light!();
}
