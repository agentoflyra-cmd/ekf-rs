use crate::{
    math::{errors::LinAlgError, matrix::Matrix},
    model::{measurement::MeasurementNoiseModel, motion::MotionNoiseModel},
    Backend, EkfState, Scalar,
};

pub trait EkfNoiseSolver<T>
where
    T: Scalar,
{
    fn predict<M>(
        state: &mut EkfState<T>,
        u: &Matrix<T>,
        w: &Matrix<T>,
    ) -> Result<(), LinAlgError<T>>
    where
        M: MotionNoiseModel<T>;
    fn update<H>(
        state: &mut EkfState<T>,
        z: &Matrix<T>,
        v: &Matrix<T>,
    ) -> Result<(), LinAlgError<T>>
    where
        H: MeasurementNoiseModel<T>;
}

impl<T> EkfNoiseSolver<T> for EkfState<T>
where
    T: Scalar,
{
    fn predict<M>(
        state: &mut EkfState<T>,
        u: &Matrix<T>,
        w: &Matrix<T>,
    ) -> Result<(), LinAlgError<T>>
    where
        M: MotionNoiseModel<T>,
    {
        state.validate_base_shapes()?;
        let prev_x = state.state.clone();
        let predict_x = M::f(&prev_x, u, w);
        let fx = M::jacobian_x(&prev_x, u, w);
        let fw = M::jacobian_w(&prev_x, u, w);
        state.validate_generalized_predict_shapes(&predict_x, &fx, &fw, w)?;
        let fpf = fx.matmul(&state.covariance)?.matmul_transposed_rhs(&fx)?;
        let fqf = fw
            .matmul(&state.process_noise)?
            .matmul_transposed_rhs(&fw)?;
        state.state = predict_x;
        state.covariance = fpf.add(&fqf)?;

        Ok(())
    }

    fn update<H>(
        state: &mut EkfState<T>,
        z: &Matrix<T>,
        v: &Matrix<T>,
    ) -> Result<(), LinAlgError<T>>
    where
        H: MeasurementNoiseModel<T>,
    {
        state.validate_base_shapes()?;
        let predict_z = H::h(&state.state, v);
        let y = z.sub(&predict_z)?;

        let hx = H::jacobian_x(&state.state, v);
        let hv = H::jacobain_v(&state.state, v);
        state.validate_generalized_update_shapes(z, &predict_z, &hx, &hv, v)?;
        // S = Hx P Hx^T + Hv R Hv^T

        let hp = hx.matmul(&state.covariance)?;
        let hph = hp.matmul_transposed_rhs(&hx)?;
        let hrh = hv
            .matmul(&state.measurement_noise)?
            .matmul_transposed_rhs(&hv)?;
        let s = hph.add(&hrh)?.symmetrize()?;
        let k_t = s.solve_spd(&hp, "S")?;
        let k = k_t.transpose();
        state.state = state.state.add(&k.matmul(&y)?)?;

        let kh = k.matmul(&hx)?;
        let n = kh.rows();
        let i_minus_kh = Matrix::eye(n).sub(&kh)?;
        let khrhk = k.matmul(&hrh)?.matmul_transposed_rhs(&k)?;
        state.covariance = i_minus_kh
            .matmul(&state.covariance)?
            .matmul_transposed_rhs(&i_minus_kh)?
            .add(&khrhk)?
            .symmetrize()?;
        state.covariance.ensure_min_diagonal()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::EkfNoiseSolver;
    use crate::{
        model::{measurement::MeasurementNoiseModel, motion::MotionNoiseModel},
        EkfState, Matrix,
        math::errors::LinAlgError,
    };
    use std::sync::Arc;

    struct ValidMotionNoise1d;
    struct BadFwMotionNoise1d;
    struct ValidMeasurementNoise1d;
    struct BadHvMeasurementNoise1d;

    impl MotionNoiseModel<f64> for ValidMotionNoise1d {
        fn f(x: &Matrix<f64>, u: &Matrix<f64>, w: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([x.storage[0] + u.storage[0] + w.storage[0]]),
                rows: 1,
                cols: 1,
            }
        }

        fn jacobian_x(_x: &Matrix<f64>, _u: &Matrix<f64>, _w: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([1.0]),
                rows: 1,
                cols: 1,
            }
        }

        fn jacobian_w(_x: &Matrix<f64>, _u: &Matrix<f64>, _w: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([1.0]),
                rows: 1,
                cols: 1,
            }
        }
    }

    impl MotionNoiseModel<f64> for BadFwMotionNoise1d {
        fn f(x: &Matrix<f64>, u: &Matrix<f64>, w: &Matrix<f64>) -> Matrix<f64> {
            ValidMotionNoise1d::f(x, u, w)
        }

        fn jacobian_x(x: &Matrix<f64>, u: &Matrix<f64>, w: &Matrix<f64>) -> Matrix<f64> {
            ValidMotionNoise1d::jacobian_x(x, u, w)
        }

        fn jacobian_w(_x: &Matrix<f64>, _u: &Matrix<f64>, _w: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([1.0, 0.0]),
                rows: 1,
                cols: 2,
            }
        }
    }

    impl MeasurementNoiseModel<f64> for ValidMeasurementNoise1d {
        fn h(x: &Matrix<f64>, v: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([x.storage[0] + v.storage[0]]),
                rows: 1,
                cols: 1,
            }
        }

        fn jacobian_x(_x: &Matrix<f64>, _v: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([1.0]),
                rows: 1,
                cols: 1,
            }
        }

        fn jacobain_v(_x: &Matrix<f64>, _v: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([1.0]),
                rows: 1,
                cols: 1,
            }
        }
    }

    impl MeasurementNoiseModel<f64> for BadHvMeasurementNoise1d {
        fn h(x: &Matrix<f64>, v: &Matrix<f64>) -> Matrix<f64> {
            ValidMeasurementNoise1d::h(x, v)
        }

        fn jacobian_x(x: &Matrix<f64>, v: &Matrix<f64>) -> Matrix<f64> {
            ValidMeasurementNoise1d::jacobian_x(x, v)
        }

        fn jacobain_v(_x: &Matrix<f64>, _v: &Matrix<f64>) -> Matrix<f64> {
            Matrix {
                storage: Arc::from([1.0, 0.0]),
                rows: 1,
                cols: 2,
            }
        }
    }

    fn scalar_state_with_noise_dims(process_noise_dim: usize, measurement_noise_dim: usize) -> EkfState<f64> {
        let process_noise = if process_noise_dim == 1 {
            Matrix {
                storage: Arc::from([0.25]),
                rows: 1,
                cols: 1,
            }
        } else {
            Matrix {
                storage: Arc::from([0.25, 0.0, 0.0, 0.25]),
                rows: 2,
                cols: 2,
            }
        };

        let measurement_noise = if measurement_noise_dim == 1 {
            Matrix {
                storage: Arc::from([0.5]),
                rows: 1,
                cols: 1,
            }
        } else {
            Matrix {
                storage: Arc::from([0.5, 0.0, 0.0, 0.5]),
                rows: 2,
                cols: 2,
            }
        };

        EkfState {
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
            process_noise,
            measurement_noise,
        }
    }

    #[test]
    fn generalized_predict_accepts_process_noise_dimension_distinct_from_state_dimension() {
        let mut state = scalar_state_with_noise_dims(1, 1);
        let u = Matrix {
            storage: Arc::from([0.5]),
            rows: 1,
            cols: 1,
        };
        let w = Matrix {
            storage: Arc::from([0.25]),
            rows: 1,
            cols: 1,
        };

        EkfState::predict::<ValidMotionNoise1d>(&mut state, &u, &w)
            .expect("generalized predict should accept Fw as n x q");
    }

    #[test]
    fn generalized_predict_rejects_bad_fw_shape() {
        let mut state = scalar_state_with_noise_dims(1, 1);
        let u = Matrix {
            storage: Arc::from([0.5]),
            rows: 1,
            cols: 1,
        };
        let w = Matrix {
            storage: Arc::from([0.25]),
            rows: 1,
            cols: 1,
        };

        let err = EkfState::predict::<BadFwMotionNoise1d>(&mut state, &u, &w)
            .expect_err("Fw shape mismatch should fail");

        assert!(matches!(
            err,
            LinAlgError::DimensionMismatch { op, lhs, rhs }
            if op == "Fw" && lhs == (1, 1) && rhs == (1, 2)
        ));
    }

    #[test]
    fn generalized_update_rejects_bad_hv_shape() {
        let mut state = scalar_state_with_noise_dims(1, 1);
        let z = Matrix {
            storage: Arc::from([2.0]),
            rows: 1,
            cols: 1,
        };
        let v = Matrix {
            storage: Arc::from([0.1]),
            rows: 1,
            cols: 1,
        };

        let err = EkfState::update::<BadHvMeasurementNoise1d>(&mut state, &z, &v)
            .expect_err("Hv shape mismatch should fail");

        assert!(matches!(
            err,
            LinAlgError::DimensionMismatch { op, lhs, rhs }
            if op == "Hv" && lhs == (1, 1) && rhs == (1, 2)
        ));
    }

    #[test]
    fn generalized_update_accepts_measurement_noise_dimension_distinct_from_state_dimension() {
        let mut state = scalar_state_with_noise_dims(1, 1);
        let z = Matrix {
            storage: Arc::from([2.0]),
            rows: 1,
            cols: 1,
        };
        let v = Matrix {
            storage: Arc::from([0.1]),
            rows: 1,
            cols: 1,
        };

        EkfState::update::<ValidMeasurementNoise1d>(&mut state, &z, &v)
            .expect("generalized update should accept Hv as m x r");
    }
}
