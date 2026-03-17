use num_traits::Float;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

pub trait Scalar: Float + AddAssign + SubAssign + MulAssign + DivAssign {
    fn default_abs_tol() -> Self;
    fn default_rel_tol() -> Self;
    fn default_chol_diag_tol() -> Self;

    fn approx_eq(a: Self, b: Self) -> bool {
        let diff = (a - b).abs();
        diff <= Self::default_abs_tol() + Self::default_rel_tol() * a.abs().max(b.abs())
    }
}

impl Scalar for f32 {
    fn default_abs_tol() -> Self {
        f32::epsilon() * 64f32
    }

    fn default_rel_tol() -> Self {
        f32::epsilon() * 1024f32
    }

    fn default_chol_diag_tol() -> Self {
        f32::epsilon() * 1024f32
    }
}
impl Scalar for f64 {
    fn default_abs_tol() -> Self {
        f64::epsilon() * 64f64
    }

    fn default_rel_tol() -> Self {
        f64::epsilon() * 1024f64
    }

    fn default_chol_diag_tol() -> Self {
        f64::epsilon() * 1024f64
    }
}
