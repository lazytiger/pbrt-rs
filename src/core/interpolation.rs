use crate::core::pbrt::{find_interval, Float, INV_2_PI, PI};
use std::process::id;

pub fn catmull_rom(size: usize, nodes: &[Float], values: &[Float], x: Float) -> Float {
    if !(x >= nodes[0] && x <= nodes[size - 1]) {
        return 0.0;
    }

    let idx = find_interval(size, |i| nodes[i] <= x);
    let x0 = nodes[idx];
    let x1 = nodes[idx + 1];
    let f0 = values[idx];
    let f1 = values[idx + 1];
    let width = x1 - x0;
    let (mut d0, mut d1) = (0.0, 0.0);
    if idx > 0 {
        d0 = width * (f1 - values[idx - 1]) / (x1 - nodes[idx - 1]);
    } else {
        d0 = f1 - f0;
    }

    if idx + 2 < size {
        d1 = width * (values[idx + 2] - f0) / (nodes[idx + 2] - x0);
    } else {
        d1 = f1 - f0;
    }

    let t = (x - x0) / (x1 - x0);
    let t2 = t * t;
    let t3 = t2 * t;

    (2.0 * t3 - 3.0 * t2 + 1.0) * f0
        + (-2.0 * t3 + 3.0 * t2) * f1
        + (t3 - 2.0 * t2 + t) * d0
        + (t3 - t2) * d1
}

pub fn catmull_rom_weights(
    size: usize,
    nodes: &[Float],
    x: Float,
    offset: &mut usize,
    weights: &mut [Float],
) -> bool {
    if !(x >= nodes[0] && x <= nodes[size - 1]) {
        return false;
    }

    let idx = find_interval(size, |i| nodes[i] <= x);
    *offset = idx - 1;
    let x0 = nodes[idx];
    let x1 = nodes[idx + 1];
    let t = (x - x0) / (x1 - x0);
    let t2 = t * t;
    let t3 = t2 * t;
    weights[1] = 2.0 * t3 - 3.0 * t2 + 1.0;
    weights[2] = -2.0 * t3 + 3.0 * t2;
    if idx > 0 {
        let w0 = (t3 - 2.0 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        let w0 = t3 - 2.0 * t2 + t;
        weights[0] = 0.0;
        weights[1] -= w0;
        weights[2] += w0;
    }

    if idx + 2 < size {
        let w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
        weights[1] -= w3;
        weights[3] += w3;
    } else {
        let w3 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0.0;
    }
    true
}

pub fn sample_catmull_rom(
    n: usize,
    x: &[Float],
    f: &[Float],
    ff: &[Float],
    mut u: Float,
    fval: Option<&mut Float>,
    pdf: Option<&mut Float>,
) -> Float {
    u *= ff[n - 1];
    let i = find_interval(n, |i| ff[i] <= u);
    let x0 = x[i];
    let x1 = x[i + 1];
    let f0 = f[i];
    let f1 = f[i + 1];
    let width = x1 - x0;

    let mut d0 = 0.0;
    let mut d1 = 0.0;

    if i > 0 {
        d0 = width * (f1 - f[i - 1]) / (x1 - x[i - 1]);
    } else {
        d0 = f1 - f0;
    }

    if i + 2 < n {
        d1 = width * (f[i + 2] - f0) / (x[i + 2] - x0);
    } else {
        d1 = f1 - f0;
    }

    u = (u - ff[i]) / width;
    let mut t = 0.0;
    if f0 != f1 {
        t = (f0 - (f0 * f0 + 2.0 * u * (f1 - f0)).max(0.0).sqrt()) / (f0 - f1);
    } else {
        t = u / f0;
    }

    let mut a = 0.0;
    let mut b = 1.0;
    let mut ffhat = 0.0;
    let mut fhat = 0.0;

    loop {
        if !(t > a && t < b) {
            t = 0.5 * (a + b);
        }

        ffhat = t
            * (f0
                + t * (0.5 * d0
                    + t * ((1.0 / 3.0) * (-2.0 * d0 - d1) + f1 - f0
                        + t * (0.25 * (d0 + d1) + 0.5 * (f0 - f1)))));
        fhat = f0
            + t * (d0 + t * (-2.0 * d0 + d1 + 3.0 * (f1 - f0) + t * (d0 + d1 + 2.0 * (f0 - f1))));

        if (ffhat - u).abs() < 1e-6 || (b - a) < 1e-6 {
            break;
        }

        if ffhat - u < 0.0 {
            a = t;
        } else {
            b = t;
        }

        t -= (ffhat - u) / fhat;
    }

    if let Some(fval) = fval {
        *fval = fhat;
    }

    if let Some(pdf) = pdf {
        *pdf = fhat / ff[n - 1];
    }
    x0 * width * t
}

pub fn sample_catmull_rom_2d(
    size1: usize,
    size2: usize,
    nodes1: &[Float],
    nodes2: &[Float],
    values: &[Float],
    cdf: &[Float],
    alpha: Float,
    mut u: Float,
    fval: Option<&mut Float>,
    pdf: Option<&mut Float>,
) -> Float {
    let mut offset = 0;
    let mut weights = [0.0; 4];
    if !catmull_rom_weights(size1, nodes1, alpha, &mut offset, &mut weights) {
        return 0.0;
    }

    let interpolate = |array: &[Float], idx: usize| {
        let mut value = 0.0;
        for i in 0..4 {
            if weights[i] != 0.0 {
                value += array[(offset + i) * size2 + idx] * weights[i];
            }
        }
        value
    };

    let maximum = interpolate(cdf, size2 - 1);
    u *= maximum;
    let idx = find_interval(size2, |i| interpolate(cdf, i) <= u);

    let f0 = interpolate(values, idx);
    let f1 = interpolate(values, idx + 1);
    let x0 = nodes2[idx];
    let x1 = nodes2[idx + 1];
    let width = x1 - x0;
    let mut d0 = 0.0;
    let mut d1 = 0.0;

    u = (u - interpolate(cdf, idx)) / width;

    if idx > 0 {
        d0 = width * (f1 - interpolate(values, idx - 1)) / (x1 - nodes2[idx - 1]);
    } else {
        d0 = f1 - f0;
    }

    if idx + 2 < size2 {
        d1 = width * (interpolate(values, idx + 2) - f0) / (nodes2[idx + 2] - x0);
    } else {
        d1 = f1 - f0;
    }

    let mut t = 0.0;
    if f0 != f1 {
        t = (f0 - (f0 * f0 + 2.0 * u * (f1 - f0)).max(0.0).sqrt()) / (f0 - f1);
    } else {
        t = u / f0;
    }

    let mut a = 0.0;
    let mut b = 1.0;
    let mut ffhat = 0.0;
    let mut fhat = 0.0;

    loop {
        if !(t >= a && t <= b) {
            t = 0.5 * (a + b);
        }

        ffhat = t
            * (f0
                + t * (0.5 * d0
                    + t * ((1.0 / 3.0) * (-2.0 * d0 - d1) + f1 - f0
                        + t * (0.25 * (d0 + d1) + 0.5 * (f0 - f1)))));

        fhat = f0
            + t * (d0 + t * (-2.0 * d0 - d1 + 3.0 * (f1 - f0) + t * (d0 + d1 + 2.0 * (f0 - f1))));

        if (ffhat - u).abs() < 1e-6 || (b - a) < 1e-6 {
            break;
        }

        if ffhat - u < 0.0 {
            a = t;
        } else {
            b = t;
        }

        t -= (ffhat - u) / fhat;
    }

    if let Some(fval) = fval {
        *fval = fhat;
    }

    if let Some(pdf) = pdf {
        *pdf = fhat / maximum;
    }
    x0 + width * t
}

pub fn integrate_catmull_rom(n: usize, x: &[Float], values: &[Float], cdf: &mut [Float]) -> Float {
    let mut sum = 0.0;
    cdf[0] = 0.0;
    for i in 0..n - 1 {
        let (x0, x1, f0, f1) = (x[i], x[i + 1], values[i], values[i + 1]);
        let width = x1 - x0;

        let mut d0 = 0.0;
        let mut d1 = 0.0;
        if i > 0 {
            d0 = width * (f1 - values[i - 1]) / (x1 - x[i - 1]);
        } else {
            d0 = f1 - f0;
        }
        if i + 2 < n {
            d1 = width * (values[i + 2] - f0) / (x[i + 2] - x0);
        } else {
            d1 = f1 - f0;
        }

        sum += ((d0 - d1) * (1.0 / 12.0) + (f0 + f1) * 0.5) * width;
        cdf[i + 1] = sum;
    }
    sum
}

pub fn invert_catmull_rom(n: usize, x: &[Float], values: &[Float], u: Float) -> Float {
    if !(u > values[0]) {
        return x[0];
    } else if !(u < values[n - 1]) {
        return x[n - 1];
    }

    let i = find_interval(n, |i| values[i] <= u);
    let (x0, x1, f0, f1) = (x[i], x[i + 1], values[i], values[i + 1]);
    let width = x1 - x0;

    let mut d0 = 0.0;
    let mut d1 = 0.0;
    if i > 0 {
        d0 = width * (f1 - values[i - 1]) / (x1 - x[i - 1]);
    } else {
        d0 = f1 - f0;
    }
    if i + 2 < n {
        d1 = width * (values[i + 2] - f0) / (x[i + 2] - x0);
    } else {
        d1 = f1 - f0;
    }

    let mut a = 0.0;
    let mut b = 1.0;
    let mut t = 0.5;
    let mut ffhat = 0.0;
    let mut fhat = 0.0;

    loop {
        if !(t > a && t < b) {
            t = 0.5 * (a + b);
        }

        let t2 = t * t;
        let t3 = t2 * t;

        ffhat = (2.0 * t3 - 3.0 * t2 + 1.0) * f0
            + (-2.0 * t3 + 3.0 * t2) * f1
            + (t3 - 2.0 * t2 + t) * d0
            + (t3 - t2) * d1;

        // Set _fhat_ using Equation (not present)
        fhat = (6.0 * t2 - 6.0 * t) * f0
            + (-6.0 * t2 + 6.0 * t) * f1
            + (3.0 * t2 - 4.0 * t + 1.0) * d0
            + (3.0 * t2 - 2.0 * t) * d1;

        if (ffhat - u).abs() < 1e-6 || (b - a) < 1e-6 {
            break;
        }

        if ffhat - u < 0.0 {
            a = t;
        } else {
            b = t;
        }

        t -= (ffhat - u) / fhat;
    }
    x0 + t * width
}

pub fn fourier(a: &[Float], m: usize, cos_phi: f64) -> Float {
    let mut value: f64 = 0.0;
    let mut cos_k_minus_one_phi = cos_phi;
    let mut cos_k_phi: f64 = 1.0;
    for k in 0..m {
        value += a[k] as f64 * cos_k_phi;
        let cos_k_plus_one_phi = 2.0 * cos_phi * cos_k_phi - cos_k_minus_one_phi;
        cos_k_minus_one_phi = cos_k_phi;
        cos_k_phi = cos_k_plus_one_phi;
    }
    value as Float
}

pub fn sample_fourier(
    ak: &[Float],
    recip: &[Float],
    m: usize,
    mut u: Float,
    pdf: &mut Float,
    phi_ptr: &mut Float,
) -> Float {
    let flip = u >= 0.5;
    if flip {
        u = 1.0 - 2.0 * (u - 0.5);
    } else {
        u *= 2.0;
    }

    let mut a = 0.0;
    let mut b = PI;
    let mut phi = 0.5 * b;
    let mut ff = 0.0;
    let mut f = 0.0;
    loop {
        let cos_phi = phi.cos();
        let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
        let mut cos_phi_prev = cos_phi;
        let mut cos_phi_cur = 1.0;
        let mut sin_phi_prev = sin_phi;
        let mut sin_phi_cur = 0.0;

        ff += ak[0] * phi;
        f = ak[0];

        for k in 1..m {
            let sin_phi_next = 2.0 * cos_phi * sin_phi_cur - sin_phi_prev;
            let cos_phi_next = 2.0 * cos_phi * cos_phi_cur - cos_phi_prev;
            sin_phi_prev = sin_phi_cur;
            sin_phi_cur = sin_phi_next;
            cos_phi_prev = cos_phi_cur;
            cos_phi_cur = cos_phi_next;

            ff += ak[k] * recip[k] * sin_phi_next;
            f += ak[k] * cos_phi_next;
        }
        ff -= u * ak[0] * PI;

        if ff > 0.0 {
            b = phi;
        } else {
            a = phi;
        }

        if f.abs() < 1e-6 || (b - a) < 1e-6 {
            break;
        }

        phi -= ff / f;

        if !(phi > a && phi < b) {
            phi = 0.5 * (a + b);
        }
    }

    if flip {
        phi = 2.0 * PI - phi;
    }
    *pdf = INV_2_PI * f / ak[0];
    *phi_ptr = phi;
    f
}
