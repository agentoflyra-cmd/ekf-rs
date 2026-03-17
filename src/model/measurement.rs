use std::ops::{AddAssign, MulAssign};

use crate::math::{matrix::Matrix, scalar_trait::Scalar};

#[allow(dead_code)]
pub trait MeasurementModel<T>
where
    T: Copy + Clone + AddAssign + MulAssign + Scalar,
{
    fn h(x: &Matrix<T>) -> Matrix<T>;
    fn jacobian(x: &Matrix<T>) -> Matrix<T>;
}
