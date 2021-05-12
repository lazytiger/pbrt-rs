use crate::{
    core::{
        geometry::{
            spherical_phi, spherical_theta, Normal3f, Point2f, Point2i, Point3f, Ray,
            RayDifferentials, Vector3, Vector3f,
        },
        imageio::read_image,
        interaction::{BaseInteraction, Interaction},
        light::{BaseLight, Light, LightFlags, VisibilityTester},
        medium::MediumInterface,
        mipmap::{ImageWrap, MIPMap},
        pbrt::{Float, INV_2_PI, INV_PI, PI},
        sampling::{concentric_sample_disk, Distribution2D},
        scene::Scene,
        spectrum::{RGBSpectrum, Spectrum},
        transform::{Transformf, Vector3Ref},
    },
    impl_base_light,
};
use derive_more::{Deref, DerefMut};
use std::{alloc::handle_alloc_error, any::Any, sync::Arc};

#[derive(Debug, Deref, DerefMut)]
pub struct InfiniteAreaLight {
    #[deref]
    #[deref_mut]
    base: BaseLight,
    l_map: MIPMap<RGBSpectrum>,
    world_center: Point3f,
    word_radius: Float,
    distribution: Distribution2D,
}

impl InfiniteAreaLight {
    pub fn new(
        light_to_world: Transformf,
        power: Spectrum,
        n_samples: usize,
        texmap: String,
    ) -> Self {
        let base = BaseLight::new(
            LightFlags::INFINITE,
            light_to_world,
            MediumInterface::default(),
            n_samples,
        );
        let mut resolution = Point2i::default();
        let texels = read_image(texmap, &mut resolution);
        let l_map = if let Some(mut texels) = texels {
            for i in 0..resolution.x * resolution.y {
                texels[i as usize] *= power.to_rgb_spectrum();
            }
            texels
        } else {
            resolution.x = 1;
            resolution.y = 1;
            vec![power.to_rgb_spectrum(); 1]
        };
        let l_map = MIPMap::new(resolution, l_map, false, 8.0, ImageWrap::Repeat);
        let width = 2 * l_map.width();
        let height = 2 * l_map.height();
        let mut img = vec![0.0; (width * height) as usize];
        let f_widht = 0.5 / std::cmp::min(width, height) as Float;
        for v in 0..height {
            let vp = (v as Float + 0.5) / height as Float;
            let sin_theta = (PI * vp).sin();
            for u in 0..width {
                let up = (u as Float + 0.5) / width as Float;
                img[(u + v * width) as usize] =
                    l_map.lookup(&Point2f::new(up, vp), f_widht).y_value();
                img[(u + v * width) as usize] *= sin_theta;
            }
        }
        let distribution = Distribution2D::new(img.as_slice(), width as usize, height as usize);
        Self {
            distribution,
            l_map,
            world_center: Default::default(),
            base,
            word_radius: 0.0,
        }
    }

    pub fn le(&self, ray: &RayDifferentials) -> Spectrum {
        let w = (&self.world_to_light * Vector3Ref(&ray.d)).normalize();
        let st = Point2f::new(spherical_phi(&w) * INV_2_PI, spherical_theta(&w) * INV_PI);
        self.l_map.lookup(&st, 0.0)
    }
}

impl Light for InfiniteAreaLight {
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
        let mut map_pdf = 0.0;
        let uv = self.distribution.sample_continuous(*u, Some(&mut map_pdf));
        if map_pdf == 0.0 {
            return 0.0.into();
        }

        let theta = uv[1] * PI;
        let phi = uv[0] * 2.0 * PI;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        *wi = &self.light_to_world
            * Vector3Ref(&Vector3f::new(
                sin_theta * cos_phi,
                sin_theta * sin_phi,
                cos_theta,
            ));
        *pdf = map_pdf / (2.0 * PI * PI * sin_theta);
        if sin_theta == 0.0 {
            *pdf = 0.0;
        }
        *vis = VisibilityTester::new(
            Arc::new(Box::new(iref.as_base().clone())),
            Arc::new(Box::new(BaseInteraction::from((
                iref.as_base().p + *wi * (2.0 * self.word_radius),
                iref.as_base().time,
                self.medium_interface.clone(),
            )))),
        );
        self.l_map.lookup(&uv, 0.0)
    }

    fn power(&self) -> Spectrum {
        self.l_map.lookup(&Point2f::new(0.5, 0.5), 0.5) * (PI * self.word_radius * self.word_radius)
    }

    fn pre_process(&mut self, scene: &Scene) {
        scene
            .world_bound()
            .bounding_sphere(&mut self.world_center, &mut self.word_radius);
    }

    fn pdf_li(&self, iref: &dyn Interaction, wi: &Vector3f) -> f32 {
        let wi = &self.world_to_light * Vector3Ref(wi);
        let theta = spherical_theta(&wi);
        let phi = spherical_phi(&wi);
        let sin_theta = theta.sin();
        if sin_theta == 0.0 {
            return 0.0;
        }
        self.distribution
            .pdf(&Point2f::new(phi * INV_2_PI, theta * INV_PI))
            / (2.0 * PI * PI * sin_theta)
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
        let u = *u1;
        let mut map_pdf = 0.0;
        let uv = self.distribution.sample_continuous(u, Some(&mut map_pdf));
        if map_pdf == 0.0 {
            return 0.0.into();
        }
        let theta = uv[1] * PI;
        let phi = uv[0] * 2.0 * PI;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let d = &self.light_to_world
            * Vector3Ref(&Vector3f::new(
                sin_theta * cos_phi,
                sin_theta * sin_phi,
                cos_theta,
            ));
        let (v1, v2) = d.coordinate_system();
        let d = -d;
        *n_light = d.normalize();
        let cd = concentric_sample_disk(u2);
        let p_disk = self.world_center + (v1 * cd.x + v2 * cd.y) * self.word_radius;
        *ray = Ray::new(
            p_disk + -d * self.word_radius,
            d,
            Float::INFINITY,
            time,
            None,
        );
        *pdf_dir = if sin_theta == 0.0 {
            0.0
        } else {
            map_pdf / (2.0 * PI * PI * sin_theta)
        };
        *pdf_pos = 1.0 / (PI * self.word_radius * self.word_radius);
        self.l_map.lookup(&uv, 0.0)
    }

    fn pdf_le(&self, ray: &Ray, n_light: &Vector3f, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        let d = &self.world_to_light * Vector3Ref(&ray.d);
        let d = -d;
        let theta = spherical_theta(&d);
        let phi = spherical_phi(&d);
        let uv = Point2f::new(phi * INV_2_PI, theta * INV_PI);
        let map_pdf = self.distribution.pdf(&uv);
        *pdf_dir = map_pdf / (2.0 * PI * PI * theta.sin());
        *pdf_pos = 1.0 / (PI * self.word_radius * self.word_radius);
    }

    impl_base_light!();
}
