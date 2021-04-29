use crate::{
    core::{
        geometry::{Point2f, Point2i, Vector2f},
        memory::BlockedArray,
        pbrt::{clamp, is_power_of_2, lerp, log_2_int_i32, mod_num, round_up_pow2_i32, Float},
        spectrum::{spectrum_lerp, Spectrum},
        texture::lanczos,
        RealNum,
    },
    shapes::curve::CurveType::Flat,
};
use std::ops::{Add, AddAssign, Div, Mul};

pub enum ImageWrap {
    Repeat,
    Black,
    Clamp,
}

#[derive(Default, Clone)]
struct ResampleWeight {
    first_texel: i32,
    weight: [Float; 4],
}

const WEIGHT_LUT_SIZE: usize = 128;
pub struct MIPMap<T> {
    do_trilinear: bool,
    max_anisotropy: Float,
    wrap_mode: ImageWrap,
    resolution: Point2i,
    pyramid: Vec<BlockedArray<T, 2>>,
    weight_lut: [Float; WEIGHT_LUT_SIZE],
    black: T,
}

pub trait ClampLerp {
    fn clamp(&self) -> Self;
    fn lerp(t: Float, s1: Self, s2: Self) -> Self;
}

impl ClampLerp for Float {
    fn clamp(&self) -> Float {
        clamp(*self, 0.0, Float::INFINITY)
    }

    fn lerp(t: f32, s1: Self, s2: Self) -> Self {
        lerp(t, s1, s2)
    }
}

impl ClampLerp for Spectrum {
    fn clamp(&self) -> Self {
        self.clamp(0.0, Float::INFINITY)
    }

    fn lerp(t: f32, s1: Self, s2: Self) -> Self {
        spectrum_lerp(t, s1, s2)
    }
}

impl<T> MIPMap<T>
where
    T: Default,
    T: Clone,
    T: Mul<Float, Output = T>,
    T: AddAssign,
    T: Add<Output = T>,
    T: Div<Float, Output = T>,
    T: From<Float>,
    T: ClampLerp,
{
    pub fn new(
        resolution: Point2i,
        img: Vec<T>,
        do_tri: bool,
        max_aniso: Float,
        wrap_mode: ImageWrap,
    ) -> Self {
        let mut mipmap = Self {
            resolution,
            max_anisotropy: max_aniso,
            wrap_mode,
            weight_lut: [0.0; WEIGHT_LUT_SIZE],
            do_trilinear: do_tri,
            pyramid: Vec::new(),
            black: 0.0.into(),
        };
        let mut resampled_image = None;
        if !is_power_of_2(resolution[0]) || !is_power_of_2(resolution[1]) {
            let res_pow2 = Point2i::new(
                round_up_pow2_i32(resolution[0]),
                round_up_pow2_i32(resolution[1]),
            );
            let s_weights = mipmap.resample_weights(resolution[0] as usize, res_pow2[0] as usize);
            resampled_image.replace(vec![T::default(); (res_pow2[0] * res_pow2[1]) as usize]);
            let resampled_image = resampled_image.as_mut().unwrap();
            for t in 0..mipmap.resolution[1] {
                for s in 0..res_pow2[0] {
                    resampled_image[(t * res_pow2[0] + s) as usize] = 0.0.into();
                    for j in 0..4 {
                        let mut orig_s = s_weights[s as usize].first_texel + j;
                        if let ImageWrap::Repeat = mipmap.wrap_mode {
                            orig_s = mod_num(orig_s, mipmap.resolution[0]);
                        } else if let ImageWrap::Clamp = mipmap.wrap_mode {
                            orig_s = clamp(orig_s, 0, mipmap.resolution[0] - 1);
                        }
                        if orig_s >= 0 && orig_s < mipmap.resolution[0] {
                            resampled_image[(t * res_pow2[0] + s) as usize] +=
                                img[(t * mipmap.resolution[0] + orig_s) as usize].clone()
                                    * s_weights[s as usize].weight[j as usize];
                        }
                    }
                }
            }

            let t_weights =
                mipmap.resample_weights(mipmap.resolution[1] as usize, res_pow2[1] as usize);
            for s in 0..res_pow2[0] {
                let mut work_data = vec![T::default(); res_pow2[1] as usize];
                for t in 0..res_pow2[1] {
                    work_data[t as usize] = 0.0.into();
                    for j in 0..4 {
                        let mut offset = t_weights[t as usize].first_texel + j;
                        if let ImageWrap::Repeat = mipmap.wrap_mode {
                            offset = mod_num(offset, mipmap.resolution[1]);
                        } else if let ImageWrap::Clamp = mipmap.wrap_mode {
                            offset = clamp(offset, 0, mipmap.resolution[1] - 1);
                        }
                        if offset >= 0 && offset < mipmap.resolution[1] {
                            work_data[t as usize] +=
                                resampled_image[(offset * res_pow2[0] + s) as usize].clone()
                                    * t_weights[t as usize].weight[j as usize];
                        }
                    }
                }
                for t in 0..res_pow2[1] {
                    resampled_image[(t * res_pow2[0] + s) as usize] = work_data[t as usize].clamp()
                }
            }

            mipmap.resolution = res_pow2;
        }

        let n_levels = 1 + log_2_int_i32(std::cmp::max(mipmap.resolution[0], mipmap.resolution[1]));
        mipmap
            .pyramid
            .resize(n_levels as usize, BlockedArray::default());
        mipmap.pyramid[0] = BlockedArray::new(
            mipmap.resolution[0] as usize,
            mipmap.resolution[1] as usize,
            resampled_image.as_ref(),
        );
        for i in 1..n_levels as usize {
            let s_res = std::cmp::max(1, mipmap.pyramid[i - 1].u_size() / 2);
            let t_res = std::cmp::max(1, mipmap.pyramid[i - 1].v_size() / 2);
            mipmap.pyramid[i] = BlockedArray::new(s_res, t_res, None);

            for t in 0..t_res {
                for s in 0..s_res {
                    mipmap.pyramid[i][(s, t)] = (mipmap.texel(i - 1, 2 * s, 2 * t)
                        + mipmap.texel(i - 1, 2 * s + 1, 2 * t)
                        + mipmap.texel(i - 1, 2 * s, 2 * t + 1)
                        + mipmap.texel(i - 1, 2 * s + 1, 2 * t + 1))
                        * 0.25;
                }
            }
        }

        if mipmap.weight_lut[0] == 0.0 {
            for i in 0..WEIGHT_LUT_SIZE {
                let mut alpha = 2.0;
                let r2 = i as Float / (WEIGHT_LUT_SIZE - 1) as Float;
                mipmap.weight_lut[i] = (-alpha * r2).exp() - (-alpha).exp();
            }
        }
        mipmap
    }

    pub fn width(&self) -> i32 {
        self.resolution[0]
    }

    pub fn height(&self) -> i32 {
        self.resolution[1]
    }

    pub fn levels(&self) -> usize {
        self.pyramid.len()
    }

    pub fn texel(&self, level: usize, mut s: usize, mut t: usize) -> T {
        let l = &self.pyramid[level];
        match self.wrap_mode {
            ImageWrap::Repeat => {
                s = mod_num(s, l.u_size());
                t = mod_num(t, l.v_size());
            }
            ImageWrap::Clamp => {
                s = clamp(s, 0, l.u_size() - 1);
                t = clamp(t, 0, l.v_size() - 1);
            }
            ImageWrap::Black => {
                if s < 0 || s >= l.u_size() || t < 0 || t >= l.v_size() {
                    return self.black.clone();
                }
            }
        }
        l[(s, t)].clone()
    }

    pub fn lookup(&self, st: &Point2f, width: Float) -> T {
        let level = self.levels() as Float - 1.0 + width.max(1e-8).log2();
        if level < 0.0 {
            self.triangle(0, st)
        } else if level > self.levels() as Float - 1.0 {
            self.texel(self.levels() - 1, 0, 0)
        } else {
            let ilevel = level.floor();
            let delta = level - ilevel;
            T::lerp(
                delta,
                self.triangle(ilevel as usize, st),
                self.triangle(ilevel as usize + 1, st),
            )
        }
    }

    pub fn lookup2(&self, st: &Point2f, mut dst0: Vector2f, mut dst1: Vector2f) -> T {
        if self.do_trilinear {
            let width = dst0[0].max(dst0[1]).max(dst1[0]).max(dst1[1]);
            return self.lookup(st, width);
        }

        if dst0.length_squared() < dst1.length_squared() {
            std::mem::swap(&mut dst0, &mut dst1);
        }
        let major_length = dst0.length();
        let mut minor_length = dst1.length();

        if minor_length * self.max_anisotropy < major_length && minor_length > 0.0 {
            let scale = major_length / (minor_length * self.max_anisotropy);
            dst1 *= scale;
            minor_length *= scale;
        }

        if minor_length == 0.0 {
            return self.triangle(0, st);
        }

        let lod = (self.levels() as Float - 1.0 + minor_length.log2()).max(0.0);
        let ilod = lod.floor() as usize;
        T::lerp(
            lod - ilod as Float,
            self.ewa(ilod, *st, dst0, dst1),
            self.ewa(ilod + 1, *st, dst0, dst1),
        )
    }

    fn resample_weights(&self, old_res: usize, new_res: usize) -> Vec<ResampleWeight> {
        let mut wt = vec![ResampleWeight::default(); new_res];
        let filter_width = 2.0;
        for i in 0..new_res {
            let center = (i as Float + 0.5) * old_res as Float / new_res as Float;
            wt[i].first_texel = (center - filter_width + 0.5).floor() as i32;
            for j in 0..4 {
                let pos = wt[i].first_texel as Float + j as Float + 0.5;
                wt[i].weight[j] = lanczos((pos - center) / filter_width, 2.0);
            }

            let inv_sum_wts =
                1.0 / (wt[i].weight[0] + wt[i].weight[1] + wt[i].weight[2] + wt[i].weight[3]);
            for j in 0..4 {
                wt[i].weight[j] *= inv_sum_wts;
            }
        }
        wt
    }

    fn triangle(&self, level: usize, st: &Point2f) -> T {
        let level = clamp(level, 0, self.levels() - 1);
        let s = st[0] * self.pyramid[level].u_size() as Float - 0.5;
        let t = st[1] * self.pyramid[level].v_size() as Float - 0.5;
        let s0 = s.floor() as usize;
        let t0 = t.floor() as usize;
        let ds = s - s0 as Float;
        let dt = t - t0 as Float;
        self.texel(level, s0, t0) * ((1.0 - ds) * (1.0 - dt))
            + self.texel(level, s0, t0 + 1) * ((1.0 - ds) * dt)
            + self.texel(level, s0 + 1, t0) * (ds * (1.0 - dt))
            + self.texel(level, s0 + 1, t0 + 1) * (ds * dt)
    }

    fn ewa(&self, level: usize, mut st: Point2f, mut dst0: Vector2f, mut dst1: Vector2f) -> T {
        if level >= self.levels() {
            return self.texel(self.levels() - 1, 0, 0);
        }
        st[0] = st[0] * self.pyramid[level].u_size() as Float - 0.5;
        st[1] = st[1] * self.pyramid[level].v_size() as Float - 0.5;
        dst0[0] *= self.pyramid[level].u_size() as Float;
        dst0[1] *= self.pyramid[level].v_size() as Float;
        dst1[0] *= self.pyramid[level].u_size() as Float;
        dst1[1] *= self.pyramid[level].v_size() as Float;

        let mut a = dst0[1] * dst0[1] + dst1[1] * dst1[1] + 1.0;
        let mut b = -2.0 * (dst0[0] * dst0[1] + dst1[0] * dst1[1]);
        let mut c = dst0[0] * dst0[0] + dst1[0] * dst1[0] + 1.0;
        let inv_f = 1.0 / (a * c - b * b * 0.25);
        a *= inv_f;
        b *= inv_f;
        c *= inv_f;

        let det = -b * b + 4.0 * a * c;
        let inv_det = 1.0 / det;
        let u_sqrt = (det * c).sqrt();
        let v_sqrt = (det * a).sqrt();
        let s0 = (st[0] - 2.0 * inv_det * u_sqrt).ceil() as i32;
        let s1 = (st[0] + 2.0 * inv_det * u_sqrt).floor() as i32;
        let t0 = (st[1] - 2.0 * inv_det * v_sqrt).ceil() as i32;
        let t1 = (st[1] + 2.0 * inv_det * v_sqrt).floor() as i32;

        let mut sum = T::from(0.0);
        let mut sum_wts = 0.0;
        for it in t0..t1 {
            let tt = it as Float - st[1];
            for is in s0..s1 {
                let ss = is as Float - st[0];
                let r2 = a * ss * ss + b * ss * tt + c * tt * tt;
                if r2 < 1.0 {
                    let index = std::cmp::min(
                        (r2 * WEIGHT_LUT_SIZE as Float) as usize,
                        WEIGHT_LUT_SIZE - 1,
                    );
                    let weight = self.weight_lut[index];
                    sum += self.texel(level, is as usize, it as usize) * weight;
                    sum_wts += weight;
                }
            }
        }
        sum / sum_wts
    }
}
