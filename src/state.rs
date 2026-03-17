use crate::math::{errors::LinAlgError, matrix::Matrix, scalar_trait::Scalar};

#[allow(dead_code)]
/// Configuration used to initialize an EKF state.
///
/// The fields correspond to the standard EKF quantities `x`, `P`, `Q`, and `R`.
pub struct EkfStateConfig<T>
where
    T: Scalar,
{
    /// State estimate vector `x`, shape `[n, 1]`.
    pub state: Matrix<T>,
    /// State covariance `P`, shape `[n, n]`.
    pub covariance: Matrix<T>,
    /// Process noise covariance `Q`, shape `[n, n]`.
    pub process_noise: Matrix<T>,
    /// Measurement noise covariance `R`, shape `[m, m]`.
    pub measurement_noise: Matrix<T>,
}

#[allow(dead_code)]
/// EKF state container.
///
/// The fields correspond to the standard EKF quantities `x`, `P`, `Q`, and `R`.
pub struct EkfState<T>
where
    T: Scalar,
{
    /// State estimate vector `x`, shape `[n, 1]`.
    pub state: Matrix<T>,
    /// State covariance `P`, shape `[n, n]`.
    pub covariance: Matrix<T>,
    /// Process noise covariance `Q`, shape `[n, n]`.
    pub process_noise: Matrix<T>,
    /// Measurement noise covariance `R`, shape `[m, m]`.
    pub measurement_noise: Matrix<T>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EkfShape {
    pub m: usize,
    pub n: usize,
}

impl<T> EkfStateConfig<T>
where
    T: Scalar,
{
    pub fn init(self) -> Result<EkfState<T>, LinAlgError<T>> {
        let state = EkfState {
            state: self.state,
            covariance: self.covariance,
            process_noise: self.process_noise,
            measurement_noise: self.measurement_noise,
        };
        state.validate_base_shapes()?;
        Ok(state)
    }
}

impl<T> EkfState<T>
where
    T: Scalar,
{
    pub fn new(
        x: Matrix<T>,
        p: Matrix<T>,
        q: Matrix<T>,
        r: Matrix<T>,
    ) -> Result<Self, LinAlgError<T>> {
        EkfStateConfig {
            state: x,
            covariance: p,
            process_noise: q,
            measurement_noise: r,
        }
        .init()
    }

    pub fn state_dim(&self) -> usize {
        self.state.rows()
    }

    pub fn measurement_dim(&self) -> usize {
        self.measurement_noise.rows()
    }

    pub(crate) fn validate_base_shapes(&self) -> Result<EkfShape, LinAlgError<T>> {
        if !self.state.is_col_vector() {
            return Err(LinAlgError::DimensionMismatch {
                op: "state",
                lhs: (self.state.rows(), 1),
                rhs: (self.state.rows(), self.state.cols()),
            });
        }

        let n = self.state.rows();
        self.covariance.assert_shape([n, n], "covariance")?;
        self.process_noise.assert_shape([n, n], "process_noise")?;
        self.measurement_noise.assert_square("measurement_noise")?;

        Ok(EkfShape {
            m: self.measurement_noise.rows(),
            n,
        })
    }

    pub(crate) fn validate_predict_shapes(
        &self,
        predicted_x: &Matrix<T>,
        f: &Matrix<T>,
    ) -> Result<(), LinAlgError<T>> {
        let shape = self.validate_base_shapes()?;
        predicted_x.assert_shape([shape.n, 1], "predicted_state")?;
        f.assert_shape([shape.n, shape.n], "F")?;
        Ok(())
    }

    pub(crate) fn validate_update_shapes(
        &self,
        z: &Matrix<T>,
        predicted_z: &Matrix<T>,
        h: &Matrix<T>,
    ) -> Result<(), LinAlgError<T>> {
        let shape = self.validate_base_shapes()?;

        z.assert_shape([shape.m, 1], "z")?;
        predicted_z.assert_shape([shape.m, 1], "predicted_measurement")?;
        h.assert_shape([shape.m, shape.n], "H")?;
        self.measurement_noise
            .assert_shape([shape.m, shape.m], "measurement_noise")?;
        Ok(())
    }
}
