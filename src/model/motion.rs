use std::ops::{AddAssign, MulAssign};

use crate::math::{matrix::Matrix, scalar_trait::Scalar};

#[allow(dead_code)]
pub trait MotionModel<T>
where
    T: Copy + Clone + AddAssign + MulAssign + Scalar,
{
    fn f(x: &Matrix<T>, u: &Matrix<T>) -> Matrix<T>;
    fn jacobian(x: &Matrix<T>, u: &Matrix<T>) -> Matrix<T>;
}
