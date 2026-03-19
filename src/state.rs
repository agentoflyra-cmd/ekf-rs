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
    pub n: usize,
    pub q: usize,
    pub r: usize,
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
                op: "state".to_string(),
                lhs: (self.state.rows(), 1),
                rhs: (self.state.rows(), self.state.cols()),
            });
        }

        let n = self.state.rows();
        self.covariance.assert_shape([n, n], "covariance")?;
        self.process_noise.assert_square("process_noise")?;
        self.measurement_noise.assert_square("measurement_noise")?;

        Ok(EkfShape {
            n,
            q: self.process_noise.rows(),
            r: self.measurement_noise.rows(),
        })
    }

    // pub(crate) fn validate_predict_shapes(
    //     &self,
    //     predicted_x: &Matrix<T>,
    //     f: &Matrix<T>,
    // ) -> Result<(), LinAlgError<T>> {
    //     let shape = self.validate_base_shapes()?;
    //     predicted_x.assert_shape([shape.n, 1], "predicted_state")?;
    //     f.assert_shape([shape.n, shape.n], "F")?;
    //     Ok(())
    // }

    pub(crate) fn validate_predict_shapes(
        &self,
        predicted_x: &Matrix<T>,
        matrices: &[(&str, &Matrix<T>)],
    ) -> Result<(), LinAlgError<T>> {
        let shape = self.validate_base_shapes()?;
        predicted_x.assert_shape([shape.n, 1], "predicted_state")?;
        self.process_noise.assert_shape([shape.n, shape.n], "process_noise")?;
        for &(matrix_name, f) in matrices {
            f.assert_shape([shape.n, shape.n], matrix_name)?;
        }
        Ok(())
    }

    pub(crate) fn validate_generalized_predict_shapes(
        &self,
        predicted_x: &Matrix<T>,
        fx: &Matrix<T>,
        fw: &Matrix<T>,
        w: &Matrix<T>,
    ) -> Result<(), LinAlgError<T>> {
        let shape = self.validate_base_shapes()?;
        predicted_x.assert_shape([shape.n, 1], "predicted_state")?;
        fx.assert_shape([shape.n, shape.n], "Fx")?;
        fw.assert_shape([shape.n, shape.q], "Fw")?;
        w.assert_shape([shape.q, 1], "w")?;
        self.process_noise
            .assert_shape([shape.q, shape.q], "process_noise")?;
        Ok(())
    }

    pub(crate) fn validate_update_shapes(
        &self,
        z: &Matrix<T>,
        predicted_z: &Matrix<T>,
        // h: &Matrix<T>,
        matrices: &[(&str, &Matrix<T>)],
    ) -> Result<(), LinAlgError<T>> {
        let shape = self.validate_base_shapes()?;
        let m = self.measurement_noise.rows();

        z.assert_shape([m, 1], "z")?;
        predicted_z.assert_shape([m, 1], "predicted_measurement")?;
        // h.assert_shape([shape.m, shape.n], "H")?;
        for &(matrix_name, h) in matrices {
            h.assert_shape([m, shape.n], matrix_name)?;
        }
        self.measurement_noise
            .assert_shape([m, m], "measurement_noise")?;
        Ok(())
    }

    pub(crate) fn validate_generalized_update_shapes(
        &self,
        z: &Matrix<T>,
        predicted_z: &Matrix<T>,
        hx: &Matrix<T>,
        hv: &Matrix<T>,
        v: &Matrix<T>,
    ) -> Result<(), LinAlgError<T>> {
        let shape = self.validate_base_shapes()?;
        if !z.is_col_vector() {
            return Err(LinAlgError::DimensionMismatch {
                op: "z".to_string(),
                lhs: (z.rows(), 1),
                rhs: (z.rows(), z.cols()),
            });
        }

        let m = z.rows();
        predicted_z.assert_shape([m, 1], "predicted_measurement")?;
        hx.assert_shape([m, shape.n], "Hx")?;
        hv.assert_shape([m, shape.r], "Hv")?;
        v.assert_shape([shape.r, 1], "v")?;
        self.measurement_noise
            .assert_shape([shape.r, shape.r], "measurement_noise")?;
        Ok(())
    }
}
