use crate::math::{matrix::Matrix, scalar_trait::Scalar};

#[allow(dead_code)]
pub trait MeasurementModel<T>
where
    T: Scalar,
{
    fn h(x: &Matrix<T>) -> Matrix<T>;
    fn jacobian(x: &Matrix<T>) -> Matrix<T>;
}

#[allow(dead_code)]
pub trait MeasurementNoiseModel<T>
where
    T: Scalar,
{
    // v is the linearization point of measurement noise, not real noise, usually zero
    fn h(x: &Matrix<T>, v: &Matrix<T>) -> Matrix<T>;
    fn jacobian_x(x: &Matrix<T>, v: &Matrix<T>) -> Matrix<T>;
    fn jacobain_v(x: &Matrix<T>, v: &Matrix<T>) -> Matrix<T>;
}
