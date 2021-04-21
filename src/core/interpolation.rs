use crate::core::pbrt::Float;

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

pub fn sample_catmull_rom_2d(
    size1: usize,
    size2: usize,
    nodes1: &[Float],
    nodes2: &[Float],
    values: &[Float],
    cdf: &[Float],
    alpha: Float,
    u: Float,
    fval: Option<&mut Float>,
    pdf: Option<&mut Float>,
) -> Float {
    todo!()
}

pub fn sample_fourier(
    ak: &[Float],
    recip: &[Float],
    m: usize,
    u: Float,
    pdf: &mut Float,
    phi_ptr: &mut Float,
) -> Float {
    todo!()
}
