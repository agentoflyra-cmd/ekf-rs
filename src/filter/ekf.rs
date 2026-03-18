use crate::{
    math::{errors::LinAlgError, matrix::Matrix, scalar_trait::Scalar},
    model::measurement::MeasurementModel,
    Backend, EkfState, MotionModel,
};

#[allow(dead_code)]
pub trait EkfSolver<T>
where
    T: Scalar,
{
    fn predict<M>(state: &mut EkfState<T>, u: &Matrix<T>) -> Result<(), LinAlgError<T>>
    where
        M: MotionModel<T>;
    fn update<H>(state: &mut EkfState<T>, z: &Matrix<T>) -> Result<(), LinAlgError<T>>
    where
        H: MeasurementModel<T>;
}

impl<T> EkfSolver<T> for EkfState<T>
where
    T: Scalar,
{
    #[allow(non_snake_case)]
    fn predict<M>(state: &mut EkfState<T>, u: &Matrix<T>) -> Result<(), LinAlgError<T>>
    where
        M: MotionModel<T>,
    {
        state.validate_base_shapes()?;
        let prev_x = state.state.clone();
        let f = M::jacobian(&prev_x, u);
        let predicted_x = M::f(&prev_x, u);
        state.validate_predict_shapes(&predicted_x, &f)?;
        let fp = f.matmul(&state.covariance)?;

        state.state = predicted_x;
        state.covariance = fp.matmul_transposed_rhs(&f)?.add(&state.process_noise)?;
        Ok(())
    }

    #[allow(non_snake_case)]
    fn update<H>(state: &mut EkfState<T>, z: &Matrix<T>) -> Result<(), LinAlgError<T>>
    where
        H: MeasurementModel<T>,
    {
        state.validate_base_shapes()?;
        let predicted_z = H::h(&state.state);
        let h = H::jacobian(&state.state);
        state.validate_update_shapes(z, &predicted_z, &h)?;
        let y = z.sub(&predicted_z)?;
        let hp = h.matmul(&state.covariance)?;
        let s = hp
            .matmul_transposed_rhs(&h)?
            .add(&state.measurement_noise)?
            .symmetrize()?;
        let k_t = s.solve_spd(&hp, "S")?;
        let k = k_t.transpose();

        state.state = state.state.add(&k.matmul(&y)?)?;
        let kh = k.matmul(&h)?;
        let n = kh.rows();
        let i_minus_kh = Matrix::eye(n).sub(&kh)?;
        let krk = k
            .matmul(&state.measurement_noise)?
            .matmul_transposed_rhs(&k)?;
        state.covariance = i_minus_kh
            .matmul(&state.covariance)?
            .matmul_transposed_rhs(&i_minus_kh)?
            .add(&krk)?
            .symmetrize()?;
        state.covariance.ensure_min_diagonal()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::EkfSolver;
    use crate::{model::measurement::MeasurementModel, EkfState, Matrix, MotionModel};
    use std::sync::Arc;

    struct LinearMotion1d;

    impl MotionModel<f64> for LinearMotion1d {
        fn f(x: &Matrix<f64>, u: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([x.storage[0] + u.storage[0]]),
                rows: 1,
                cols: 1,
            }
        }

        fn jacobian(_x: &Matrix<f64>, _u: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([1.0]),
                rows: 1,
                cols: 1,
            }
        }
    }

    struct IdentityMeasurement1d;

    impl MeasurementModel<f64> for IdentityMeasurement1d {
        fn h(x: &Matrix<f64>) -> Matrix<f64> {
            x.clone()
        }

        fn jacobian(_x: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([1.0]),
                rows: 1,
                cols: 1,
            }
        }
    }

    struct PositionOnlyMeasurement2d;

    impl MeasurementModel<f64> for PositionOnlyMeasurement2d {
        fn h(x: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([x.storage[0]]),
                rows: 1,
                cols: 1,
            }
        }

        fn jacobian(_x: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([1.0, 0.0]),
                rows: 1,
                cols: 2,
            }
        }
    }

    fn approx_eq(lhs: f64, rhs: f64) -> bool {
        (lhs - rhs).abs() < 1e-9
    }

    #[test]
    fn predict_and_update_match_linear_1d_gold_values() {
        let mut state = EkfState {
            state: Matrix {
                storage: Arc::from([1.0]),
                rows: 1,
                cols: 1,
            },
            covariance: Matrix {
                storage: Arc::from([2.0]),
                rows: 1,
                cols: 1,
            },
            process_noise: Matrix {
                storage: Arc::from([0.5]),
                rows: 1,
                cols: 1,
            },
            measurement_noise: Matrix {
                storage: Arc::from([1.0]),
                rows: 1,
                cols: 1,
            },
        };
        let u = Matrix {
            storage: Arc::from([0.5]),
            rows: 1,
            cols: 1,
        };
        let z = Matrix {
            storage: Arc::from([2.0]),
            rows: 1,
            cols: 1,
        };

        EkfState::predict::<LinearMotion1d>(&mut state, &u).expect("predict should succeed");
        assert!(approx_eq(state.state.storage[0], 1.5));
        assert!(approx_eq(state.covariance.storage[0], 2.5));

        EkfState::update::<IdentityMeasurement1d>(&mut state, &z).expect("update should succeed");
        assert!(approx_eq(state.state.storage[0], 1.8571428571428572));
        assert!(approx_eq(state.covariance.storage[0], 0.7142857142857144));
    }

    #[test]
    fn update_uses_h_transpose_for_gain_direction() {
        let mut state = EkfState {
            state: Matrix {
                storage: Arc::from([0.0, 10.0]),
                rows: 2,
                cols: 1,
            },
            covariance: Matrix {
                storage: Arc::from([4.0, 0.0, 0.0, 9.0]),
                rows: 2,
                cols: 2,
            },
            process_noise: Matrix {
                storage: Arc::from([0.0, 0.0, 0.0, 0.0]),
                rows: 2,
                cols: 2,
            },
            measurement_noise: Matrix {
                storage: Arc::from([1.0]),
                rows: 1,
                cols: 1,
            },
        };
        let z = Matrix {
            storage: Arc::from([2.0]),
            rows: 1,
            cols: 1,
        };

        EkfState::update::<PositionOnlyMeasurement2d>(&mut state, &z)
            .expect("update should succeed");

        assert!(approx_eq(state.state.storage[0], 1.6));
        assert!(approx_eq(state.state.storage[1], 10.0));
        assert!(approx_eq(state.covariance.storage[0], 0.8));
        assert!(approx_eq(state.covariance.storage[1], 0.0));
        assert!(approx_eq(state.covariance.storage[2], 0.0));
        assert!(approx_eq(state.covariance.storage[3], 9.0));
    }
}
