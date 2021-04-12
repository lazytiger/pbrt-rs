use crate::core::{
    filter::Filter,
    geometry::{Bounds2f, Bounds2i, Point2f, Point2i, Vector2f},
    pbrt::Float,
    spectrum::{xyz_to_rgb, Spectrum},
};
use std::sync::{Arc, Mutex};

#[derive(Default, Copy, Clone)]
struct Pixel {
    xyz: [Float; 3],
    filter_weight_sum: Float,
    splat_xyz: [Float; 3],
    pad: Float,
}
const FILTER_TABLE_WIDTH: usize = 16;

pub struct Film {
    pub full_resolution: Point2i,
    pub diagonal: Float,
    pub filter: Arc<Box<dyn Filter>>,
    pub filename: String,
    pub cropped_pixel_bounds: Bounds2i,
    pixels: Vec<Pixel>,
    filter_table: [Float; FILTER_TABLE_WIDTH * FILTER_TABLE_WIDTH],
    scale: Float,
    max_sample_luminance: Float,
}

impl Film {
    pub fn new(
        full_resolution: Point2i,
        crop_window: Bounds2f,
        filter: Arc<Box<dyn Filter>>,
        diagonal: Float,
        filename: String,
        scale: Float,
        max_sample_luminance: Float,
    ) -> Self {
        let diagonal = diagonal * 0.001;
        let cropped_pixel_bounds = Bounds2i::from((
            Point2i::new(
                (full_resolution.x as Float * crop_window.min.x).ceil() as i32,
                (full_resolution.y as Float * crop_window.min.y).ceil() as i32,
            ),
            Point2i::new(
                (full_resolution.x as Float * crop_window.max.x).ceil() as i32,
                (full_resolution.y as Float * crop_window.max.y).ceil() as i32,
            ),
        ));
        let pixels = vec![Pixel::default(); cropped_pixel_bounds.area() as usize];
        let mut filter_table = [0.0; FILTER_TABLE_WIDTH * FILTER_TABLE_WIDTH];
        let mut offset = 0;
        for y in 0..FILTER_TABLE_WIDTH {
            for x in 0..FILTER_TABLE_WIDTH {
                let p = Point2f::new(
                    (x as Float + 0.5) * filter.radius().x / FILTER_TABLE_WIDTH as Float,
                    (y as Float + 0.5) * filter.radius().y / FILTER_TABLE_WIDTH as Float,
                );
                filter_table[offset] = filter.evaluate(&p);
                offset += 1;
            }
        }
        Self {
            full_resolution,
            diagonal,
            filter,
            filename,
            scale,
            max_sample_luminance,
            cropped_pixel_bounds,
            pixels,
            filter_table,
        }
    }
    pub fn get_sample_bounds(&self) -> Bounds2i {
        let mut float_bounds = Bounds2f::new();
        float_bounds.min = Point2f::from(self.cropped_pixel_bounds.min).floor()
            + (Vector2f::new(0.5, 0.5) - *self.filter.radius());
        float_bounds.into()
    }

    pub fn get_physical_extent(&self) -> Bounds2f {
        let aspect = self.full_resolution.y as Float / self.full_resolution.x as Float;
        let x = (self.diagonal * self.diagonal).sqrt() / (1.0 + aspect * aspect);
        let y = aspect * x;
        Bounds2f::from((
            Point2f::new(-x / 2.0, -y / 2.0),
            Point2f::new(x / 2.0, y / 2.0),
        ))
    }

    pub fn get_film_tile(&self, sample_bounds: &Bounds2i) -> Arc<FilmTile> {
        let half_pixel = Vector2f::new(0.5, 0.5);
        let float_bounds: Bounds2f = (*sample_bounds).into();
        let p0: Point2i = (float_bounds.min - half_pixel - *self.filter.radius())
            .ceil()
            .into();
        let p1 = (float_bounds.max - half_pixel + *self.filter.radius())
            .floor()
            .into();
        let tile_pixel_bounds = Bounds2i::from((p0, p1)).intersect(&self.cropped_pixel_bounds);
        Arc::new(FilmTile::new(
            tile_pixel_bounds,
            *self.filter.radius(),
            &self.filter_table[..],
            FILTER_TABLE_WIDTH,
            self.max_sample_luminance,
        ))
    }

    pub fn merge_film_tile(&mut self, tile: Arc<FilmTile>) {
        for i in 0..2 {
            let pixel = tile.get_pixel_bounds()[i];
            let tile_pixel = tile.get_pixel(&pixel);
            let merge_pixel = self.get_pixel_mut(&pixel);
            let mut xyz = [0.0; 3];
            tile_pixel.contrib_sum.to_xyz(&mut xyz);

            for i in 0..3 {
                merge_pixel.xyz[i] += xyz[i];
            }
            merge_pixel.filter_weight_sum += tile_pixel.filter_weight_sum;
        }
    }

    pub fn set_image(&mut self, img: &[Spectrum]) {
        let n_pixels = self.cropped_pixel_bounds.area() as usize;
        for i in 0..n_pixels {
            let p = &mut self.pixels[i];
            img[i].to_xyz(&mut p.xyz);
            p.filter_weight_sum = 1.0;
            for i in 0..3 {
                p.splat_xyz[i] = 0.0;
            }
        }
    }

    pub fn add_splat(&mut self, p: &Point2f, mut v: Spectrum) {
        let pi: Point2i = p.floor().into();
        if self.cropped_pixel_bounds.inside(&pi) {
            return;
        }
        if v.y_value() > self.max_sample_luminance {
            v *= self.max_sample_luminance / v.y_value();
        }
        let mut xyz = [0.0; 3];
        v.to_xyz(&mut xyz);
        let pixel = self.get_pixel_mut(&pi);
        for i in 0..3 {
            pixel.splat_xyz[i] += xyz[i];
        }
    }

    pub fn write_image(&self, splat_scale: Float) {
        let mut rgb = vec![0.0; 3 * self.cropped_pixel_bounds.area() as usize];
        let mut offset = 0;
        for i in 0..2 {
            let p = self.cropped_pixel_bounds[i];
            let pixel = self.get_pixel(&p);
            xyz_to_rgb(&pixel.xyz, &mut rgb[3 * offset..]);
            let filter_weight_sum = pixel.filter_weight_sum;
            if filter_weight_sum != 0.0 {
                let inv_wt = 1.0 / filter_weight_sum;
                rgb[3 * offset] = (rgb[3 * offset] * inv_wt).max(0.0);
                rgb[3 * offset + 1] = (rgb[3 * offset + 1] * inv_wt).max(0.0);
                rgb[3 * offset + 2] = (rgb[3 * offset + 2] * inv_wt).max(0.0);
            }

            let mut splat_rgb = [0.0; 3];
            let splat_xyz = pixel.splat_xyz;
            xyz_to_rgb(&splat_xyz, &mut splat_rgb);
            rgb[3 * offset] += splat_scale * splat_rgb[0];
            rgb[3 * offset + 1] += splat_scale * splat_rgb[1];
            rgb[3 * offset + 2] += splat_scale * splat_rgb[2];

            rgb[3 * offset] *= self.scale;
            rgb[3 * offset + 1] *= self.scale;
            rgb[3 * offset + 2] *= self.scale;
            offset += 1;
        }
        todo!() //write_image
    }

    pub fn clear(&mut self) {
        for i in 0..2 {
            let p = self.cropped_pixel_bounds[i];
            let pixel = self.get_pixel_mut(&p);
            for c in 0..3 {
                pixel.splat_xyz[c] = 0.0;
                pixel.xyz[c] = 0.0;
            }
            pixel.filter_weight_sum = 0.0;
        }
    }

    fn get_pixel(&self, p: &Point2i) -> &Pixel {
        let width = self.cropped_pixel_bounds.max.x - self.cropped_pixel_bounds.min.x;
        let offset = (p.x - self.cropped_pixel_bounds.min.x)
            + (p.y - self.cropped_pixel_bounds.min.y) * width;
        &self.pixels[offset as usize]
    }

    fn get_pixel_mut(&mut self, p: &Point2i) -> &mut Pixel {
        let width = self.cropped_pixel_bounds.max.x - self.cropped_pixel_bounds.min.x;
        let offset = (p.x - self.cropped_pixel_bounds.min.x)
            + (p.y - self.cropped_pixel_bounds.min.y) * width;
        &mut self.pixels[offset as usize]
    }
}

#[derive(Clone)]
pub struct FilmTilePixel {
    contrib_sum: Spectrum,
    filter_weight_sum: Float,
}

impl Default for FilmTilePixel {
    fn default() -> Self {
        Self {
            contrib_sum: Spectrum::new(0.0),
            filter_weight_sum: 0.0,
        }
    }
}

pub struct FilmTile<'a> {
    pixel_bounds: Bounds2i,
    filter_radius: Vector2f,
    inv_filter_radius: Vector2f,
    filter_table: &'a [Float],
    filter_table_size: usize,
    pixels: Vec<FilmTilePixel>,
    max_sample_luminance: Float,
}

impl<'a> FilmTile<'a> {
    pub fn new(
        pixel_bounds: Bounds2i,
        filter_radius: Vector2f,
        filter_table: &'a [Float],
        filter_table_size: usize,
        max_sample_luminance: Float,
    ) -> Self {
        let inv_filter_radius = Point2f::new(1.0 / filter_radius.x, 1.0 / filter_radius.y);
        let pixels = vec![FilmTilePixel::default(); std::cmp::max(pixel_bounds.area(), 0) as usize];
        Self {
            pixel_bounds,
            filter_radius,
            inv_filter_radius,
            filter_table,
            filter_table_size,
            max_sample_luminance,
            pixels,
        }
    }

    pub fn add_sample(&mut self, p_film: &Point2f, mut l: Spectrum, sample_weight: Float) {
        if l.y_value() > self.max_sample_luminance {
            l *= self.max_sample_luminance / l.y_value();
        }

        let p_film_discrete = *p_film - Vector2f::new(0.5, 0.5);
        let p0 = (p_film_discrete - self.filter_radius).ceil();
        let p1 = (p_film_discrete + self.filter_radius).floor() + Point2f::new(1.0, 1.0);
        let p0 = p0.max(&self.pixel_bounds.min.into());
        let p1 = p1.max(&self.pixel_bounds.max.into());

        let mut ifx = vec![0; (p1.x - p0.x) as usize];
        for x in p0.x as usize..p1.x as usize {
            let fx = ((x as Float - p_film_discrete.x)
                * self.inv_filter_radius.x
                * self.filter_table_size as Float)
                .abs();
            ifx[x - p0.x as usize] = std::cmp::min(self.filter_table_size - 1, fx.floor() as usize);
        }
        let mut ify = vec![0; (p1.y - p0.y) as usize];
        for y in p0.y as usize..p1.y as usize {
            let fy = ((y as Float - p_film_discrete.y)
                * self.inv_filter_radius.y
                * self.filter_table_size as Float)
                .abs();
            ify[y - p0.y as usize] = std::cmp::min(self.filter_table_size - 1, fy.floor() as usize);
        }
        for y in p0.y as usize..p1.y as usize {
            for x in p0.x as usize..p1.x as usize {
                let offset =
                    ify[y - p0.y as usize] * self.filter_table_size + ifx[x - p0.x as usize];
                let filter_weight = self.filter_table[offset];
                let pixel = self.get_pixel_mut(&Point2i::new(x as i32, y as i32));
                pixel.contrib_sum += l * sample_weight * filter_weight;
                pixel.filter_weight_sum += filter_weight;
            }
        }
    }

    pub fn get_pixel(&self, p: &Point2i) -> &FilmTilePixel {
        let width = self.pixel_bounds.max.x - self.pixel_bounds.min.x;
        let offset = (p.x - self.pixel_bounds.min.x) + (p.y - self.pixel_bounds.min.y) * width;
        &self.pixels[offset as usize]
    }

    pub fn get_pixel_mut(&mut self, p: &Point2i) -> &mut FilmTilePixel {
        let width = self.pixel_bounds.max.x - self.pixel_bounds.min.x;
        let offset = (p.x - self.pixel_bounds.min.x) + (p.y - self.pixel_bounds.min.y) * width;
        &mut self.pixels[offset as usize]
    }

    pub fn get_pixel_bounds(&self) -> Bounds2i {
        self.pixel_bounds
    }
}
