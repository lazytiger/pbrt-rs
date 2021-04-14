use crate::core::{
    camera::CameraDt,
    geometry::{Bounds2i, RayDifferentials, Vector3f},
    integrator::{BaseSamplerIntegrator, Integrator},
    interaction::SurfaceInteraction,
    material::TransportMode,
    pbrt::Float,
    sampler::SamplerDtRw,
    sampling::{
        cosine_hemisphere_pdf, cosine_sample_hemisphere, uniform_hemisphere_pdf,
        uniform_sample_hemisphere,
    },
    scene::Scene,
    spectrum::Spectrum,
};
use derive_more::{Deref, DerefMut};
use std::any::Any;

#[derive(Deref, DerefMut)]
pub struct AOIntegrator {
    #[deref]
    #[deref_mut]
    base: BaseSamplerIntegrator,
    cos_sample: bool,
    n_samples: usize,
}

impl AOIntegrator {
    pub fn new(
        cos_sample: bool,
        n_samples: usize,
        camera: CameraDt,
        sampler: SamplerDtRw,
        pixel_bounds: Bounds2i,
    ) -> Self {
        let n_samples = sampler.read().unwrap().round_count(n_samples);
        sampler.write().unwrap().request_2d_array(n_samples);
        Self {
            base: BaseSamplerIntegrator::new(camera, sampler, pixel_bounds),
            cos_sample,
            n_samples,
        }
    }
}

impl Integrator for AOIntegrator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&self, scene: &Scene) {
        self.base.render(scene)
    }

    fn li(
        &self,
        r: &RayDifferentials,
        scene: &Scene,
        sampler: SamplerDtRw,
        depth: i32,
    ) -> Spectrum {
        let mut l = Spectrum::new(0.0);
        let mut ray = r.clone();
        let mut isect = SurfaceInteraction::default();

        loop {
            if scene.intersect(&mut ray.base, &mut isect) {
                isect.compute_scattering_functions(&ray, true, TransportMode::Radiance);
                if isect.bsdf.is_none() {
                    ray = isect.spawn_ray(&ray.d).into();
                    continue;
                }

                let n = isect.n.face_forward(-ray.d);
                let s = isect.dpdu.normalize();
                let t = isect.n.cross(&s);
                let u = sampler
                    .write()
                    .unwrap()
                    .get_2d_array(self.n_samples)
                    .unwrap();
                for i in 0..self.n_samples {
                    let (mut wi, pdf) = if self.cos_sample {
                        let wi = cosine_sample_hemisphere(&u[i]);
                        (wi, cosine_hemisphere_pdf(wi.z.abs()))
                    } else {
                        (uniform_sample_hemisphere(&u[i]), uniform_hemisphere_pdf())
                    };

                    wi = Vector3f::new(
                        s.x * wi.x + t.x * wi.y + n.x * wi.z,
                        s.y * wi.x + t.y * wi.y + n.y * wi.z,
                        s.z * wi.x + t.z * wi.y + n.z * wi.z,
                    );

                    if scene.intersect_p(&isect.spawn_ray(&wi)) {
                        l += Spectrum::new(wi.dot(&n) / (pdf * self.n_samples as Float));
                    }
                }
            }
            break;
        }
        l
    }
}
