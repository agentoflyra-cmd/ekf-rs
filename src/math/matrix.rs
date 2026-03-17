use std::sync::Arc;

use crate::math::{errors::LinAlgError, scalar_trait::Scalar};

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Matrix<T>
where
    T: Scalar,
{
    // only impl for cpu
    pub(crate) storage: Arc<[T]>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl<T> Matrix<T>
where
    T: Scalar,
{
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            storage: Arc::from(vec![T::zero(); rows * cols]),
            rows,
            cols,
        }
    }

    pub fn eye(size: usize) -> Self {
        let mut out = vec![T::zero(); size * size];
        for i in 0..size {
            out[i * size + i] = T::one();
        }

        Self {
            storage: Arc::from(out),
            rows: size,
            cols: size,
        }
    }

    pub fn is_col_vector(&self) -> bool {
        self.cols == 1
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn is_phanox(&self) -> bool {
        self.rows == self.cols
    }

    pub fn transpose(&self) -> Self {
        let mut out = vec![T::zero(); self.rows * self.cols];

        for row in 0..self.rows {
            for col in 0..self.cols {
                out[col * self.rows + row] = self.storage[row * self.cols + col];
            }
        }

        Self {
            storage: Arc::from(out),
            rows: self.cols,
            cols: self.rows,
        }
    }

    pub fn symmetrize(&self) -> Result<Self, LinAlgError<T>> {
        self.assert_square("matrix")?;

        let transposed = self.transpose();
        let half = T::one() / (T::one() + T::one());
        let mut out = vec![T::zero(); self.rows * self.cols];

        for row in 0..self.rows {
            for col in 0..self.cols {
                let idx = row * self.cols + col;
                out[idx] = (self.storage[idx] + transposed.storage[idx]) * half;
            }
        }

        Ok(Self {
            storage: Arc::from(out),
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn assert_shape(
        &self,
        expected: [usize; 2],
        name: &'static str,
    ) -> Result<(), LinAlgError<T>> {
        if self.shape() == expected {
            Ok(())
        } else {
            Err(LinAlgError::DimensionMismatch {
                op: name,
                lhs: (expected[0], expected[1]),
                rhs: (self.rows, self.cols),
            })
        }
    }

    pub fn assert_square(&self, name: &'static str) -> Result<(), LinAlgError<T>> {
        if self.is_phanox() {
            Ok(())
        } else {
            Err(LinAlgError::DimensionMismatch {
                op: name,
                lhs: (self.rows, self.rows),
                rhs: (self.rows, self.cols),
            })
        }
    }
}
