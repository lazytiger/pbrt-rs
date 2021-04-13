use crate::{
    core::{
        camera::{BaseCamera, Camera, CameraSample},
        film::Film,
        geometry::{
            Bounds2f, Normal3f, Point2f, Point3f, Ray, RayDifferentials, Vector3, Vector3f,
        },
        interaction::{BaseInteraction, Interaction, InteractionDt},
        light::VisibilityTester,
        medium::{Medium, MediumDt, MediumInterface},
        pbrt::{lerp, Float, PI},
        sampling::concentric_sample_disk,
        spectrum::Spectrum,
        transform::{AnimatedTransform, Point3Ref, Transformf, Vector3Ref},
    },
    impl_base_camera,
};
use std::{any::Any, sync::Arc};

pub struct PerspectiveCamera {
    base: BaseCamera,
    dx_camera: Vector3f,
    dy_camera: Vector3f,
    camera_to_screen: Transformf,
    raster_to_camera: Transformf,
    screen_to_raster: Transformf,
    raster_to_screen: Transformf,
    lens_radius: Float,
    focal_distance: Float,
    a: Float,
}

impl PerspectiveCamera {
    pub fn new(
        camera_to_world: AnimatedTransform,
        screen_window: Bounds2f,
        shutter_open: Float,
        shutter_close: Float,
        lens_radius: Float,
        focal_distance: Float,
        fov: Float,
        film: Arc<Film>,
        medium: MediumDt,
    ) -> PerspectiveCamera {
        let camera_to_screen = Transformf::perspective(fov, 1e-2, 1000.0);
        let screen_to_raster = Transformf::scale(
            film.full_resolution.x as Float,
            film.full_resolution.y as Float,
            1.0,
        ) * Transformf::scale(
            1.0 / (screen_window.max.x - screen_window.min.x),
            1.0 / (screen_window.min.y - screen_window.max.y),
            1.0,
        ) * Transformf::translate(&Vector3f::new(
            -screen_window.min.x,
            -screen_window.max.y,
            0.0,
        ));
        let raster_to_screen = screen_to_raster.inverse();
        let raster_to_camera = camera_to_screen.inverse() * raster_to_screen;
        let res = film.full_resolution;
        let mut p_min = &raster_to_camera * Point3Ref(&Point3f::new(0.0, 0.0, 0.0));
        let mut p_max =
            &raster_to_camera * Point3Ref(&Point3f::new(res.x as Float, res.y as Float, 0.0));
        p_min /= p_min.z;
        p_max /= p_max.z;
        let a = ((p_max.x - p_min.x) * (p_max.y - p_min.y)).abs();
        PerspectiveCamera {
            base: BaseCamera::new(camera_to_world, shutter_open, shutter_close, film, medium),
            dx_camera: (&raster_to_camera * Point3Ref(&Vector3f::new(1.0, 0.0, 0.0))
                - &raster_to_camera * Point3Ref(&Point3f::new(0.0, 0.0, 0.0))),
            dy_camera: (&raster_to_camera * Point3Ref(&Vector3f::new(0.0, 1.0, 0.0))
                - &raster_to_camera * Point3Ref(&Point3f::new(0.0, 0.0, 0.0))),
            a,
            camera_to_screen,
            lens_radius,
            focal_distance,
            screen_to_raster,
            raster_to_camera,
            raster_to_screen: screen_to_raster.inverse(),
        }
    }
}

impl Camera for PerspectiveCamera {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> Float {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = &self.raster_to_camera * Point3Ref(&p_film);
        *ray = Ray::new(
            Vector3f::new(0.0, 0.0, 0.0),
            p_camera.normalize(),
            Float::INFINITY,
            0.0,
            None,
        );
        if self.lens_radius > 0.0 {
            let p_lens = concentric_sample_disk(&sample.p_lens) * self.lens_radius;
            let ft = self.focal_distance / ray.d.z;
            let p_focus = ray.point(ft);
            ray.o = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.d = (p_focus - ray.o).normalize();
        }

        ray.time = lerp(sample.time, self.shutter_open(), self.shutter_close());
        ray.medium = Some(self.medium());
        *ray = (self.camera_to_world(), &*ray).into();
        1.0
    }

    fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        ray: &mut RayDifferentials,
    ) -> Float {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = &self.raster_to_camera * Point3Ref(&p_film);
        *ray = RayDifferentials::new(
            Vector3f::new(0.0, 0.0, 0.0),
            p_camera.normalize(),
            Float::INFINITY,
            0.0,
            None,
        );
        if self.lens_radius > 0.0 {
            let p_lens = concentric_sample_disk(&sample.p_lens) * self.lens_radius;
            let ft = self.focal_distance / ray.d.z;
            let p_focus = ray.point(ft);
            ray.o = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.d = (p_focus - ray.o).normalize();
        }
        if self.lens_radius > 0.0 {
            let p_lens = concentric_sample_disk(&sample.p_lens) * self.lens_radius;

            let dx = (p_camera + self.dx_camera).normalize();
            let ft = self.focal_distance / dx.z;
            let p_focus = dx * ft;
            ray.rx_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.rx_direction = (p_focus - ray.rx_origin).normalize();

            let dy = (p_camera + self.dy_camera).normalize();
            let ft = self.focal_distance / dy.z;
            let p_focus = dy * ft;
            ray.ry_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.ry_direction = (p_focus - ray.ry_origin).normalize();
        } else {
            ray.rx_origin = ray.o;
            ray.ry_origin = ray.o;
            ray.rx_direction = (p_camera + self.dx_camera).normalize();
            ray.ry_direction = (p_camera + self.dy_camera).normalize();
        }

        ray.time = lerp(sample.time, self.shutter_open(), self.shutter_close());
        ray.medium = Some(self.medium());
        ray.has_differentials = true;
        *ray = (self.camera_to_world(), &*ray).into();
        1.0
    }

    fn we(&self, ray: &Ray, p_raster2: Option<&mut Point2f>) -> Spectrum {
        let c2w = self.camera_to_world().interpolate(ray.time);
        let cos_theta = ray
            .d
            .dot(&(&c2w * Vector3Ref(&Vector3f::new(0.0, 0.0, 1.0))));
        if cos_theta <= 0.0 {
            return Spectrum::new(0.0);
        }

        let p_focus = ray.point(
            if self.lens_radius > 0.0 {
                self.focal_distance
            } else {
                1.0
            } / cos_theta,
        );
        let p_raster = &(self.raster_to_camera.inverse() * (c2w.inverse())) * Point3Ref(&p_focus);
        if let Some(p_raster2) = p_raster2 {
            *p_raster2 = Point2f::new(p_raster.x, p_raster.y);
        }

        let sample_bounds = self.film().get_sample_bounds();
        if p_raster.x < sample_bounds.min.x as Float
            || p_raster.x >= sample_bounds.max.x as Float
            || p_raster.y < sample_bounds.min.y as Float
            || p_raster.y >= sample_bounds.max.y as Float
        {
            return Spectrum::new(0.0);
        }

        let lens_area = if self.lens_radius != 0.0 {
            PI * self.lens_radius * self.lens_radius
        } else {
            1.0
        };

        let cos2theta = cos_theta * cos_theta;
        Spectrum::new(1.0 / (self.a * lens_area * cos2theta * cos2theta))
    }

    fn pdf_we(&self, ray: &Ray, pdf_pos: &mut Float, pdf_dir: &mut Float) {
        let c2w = self.camera_to_world().interpolate(ray.time);
        let cos_theta = ray
            .d
            .dot(&(&c2w * Vector3Ref(&Vector3f::new(0.0, 0.0, 1.0))));
        if cos_theta <= 0.0 {
            *pdf_pos = 0.0;
            *pdf_dir = 0.0;
            return;
        }

        let p_focus = ray.point(
            if self.lens_radius > 0.0 {
                self.focal_distance
            } else {
                1.0
            } / cos_theta,
        );
        let p_raster = &(self.raster_to_camera.inverse() * (c2w.inverse())) * Point3Ref(&p_focus);

        let sample_bounds = self.film().get_sample_bounds();
        if p_raster.x < sample_bounds.min.x as Float
            || p_raster.x >= sample_bounds.max.x as Float
            || p_raster.y < sample_bounds.min.y as Float
            || p_raster.y >= sample_bounds.max.y as Float
        {
            *pdf_pos = 0.0;
            *pdf_dir = 0.0;
            return;
        }

        let lens_area = if self.lens_radius != 0.0 {
            PI * self.lens_radius * self.lens_radius
        } else {
            1.0
        };
        *pdf_pos = 1.0 / lens_area;
        *pdf_dir = 1.0 / (self.a * cos_theta * cos_theta * cos_theta);
    }

    fn sample_wi(
        &self,
        it: InteractionDt,
        u: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut f32,
        p_raster: Option<&mut Point2f>,
        vis: &mut VisibilityTester,
    ) -> Spectrum {
        let refer = it.as_base();
        let p_lens = concentric_sample_disk(u);
        let p_lens_world = Point3f::from((
            self.camera_to_world(),
            refer.time,
            Point3Ref(&Point3f::new(p_lens.x, p_lens.y, 0.0)),
        ));
        let mut lens_intr = BaseInteraction::from((
            p_lens_world,
            refer.time,
            MediumInterface::new(Some(self.medium()), Some(self.medium())),
        ));
        lens_intr.n = Vector3f::from((
            self.camera_to_world(),
            refer.time,
            Vector3Ref(&Vector3f::new(0.0, 0.0, 1.0)),
        ));

        *wi = lens_intr.p - refer.p;
        let dist = wi.length();
        *wi /= dist;

        let lens_area = if self.lens_radius != 0.0 {
            PI * self.lens_radius * self.lens_radius
        } else {
            1.0
        };
        *pdf = (dist * dist) / (lens_intr.n.abs_dot(wi) * lens_area);
        let ray = lens_intr.spawn_ray(&-*wi);
        *vis = VisibilityTester::new(it.clone(), Arc::new(Box::new(lens_intr)));
        self.we(&ray, p_raster)
    }

    impl_base_camera!();
}
