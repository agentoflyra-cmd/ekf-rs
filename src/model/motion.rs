use crate::math::{matrix::Matrix, scalar_trait::Scalar};

#[allow(dead_code)]
pub trait MotionModel<T>
where
    T: Scalar,
{
    fn f(x: &Matrix<T>, u: &Matrix<T>) -> Matrix<T>;
    fn jacobian(x: &Matrix<T>, u: &Matrix<T>) -> Matrix<T>;
}

#[allow(dead_code)]
pub trait MotionNoiseModel<T>
where
    T: Scalar,
{
    // w is the linearization point of process noise, not real noise, usually zero
    fn f(x: &Matrix<T>, u: &Matrix<T>, w: &Matrix<T>) -> Matrix<T>;
    fn jacobian_x(x: &Matrix<T>, u: &Matrix<T>, w: &Matrix<T>) -> Matrix<T>;
    fn jacobian_w(x: &Matrix<T>, u: &Matrix<T>, w: &Matrix<T>) -> Matrix<T>;
}
