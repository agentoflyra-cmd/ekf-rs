use std::{
    ops::{Index, IndexMut},
    sync::Arc,
};

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
    pub fn from_vec(rows: usize, cols: usize, storage: Vec<T>) -> Result<Self, LinAlgError<T>> {
        if storage.len() != rows * cols {
            return Err(LinAlgError::DimensionMismatch {
                op: "Matrix::from_vec",
                lhs: (rows, cols),
                rhs: (storage.len(), 1),
            });
        }

        Ok(Self {
            storage: Arc::from(storage),
            rows,
            cols,
        })
    }

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

    pub fn ensure_min_diagonal(&mut self) -> Result<(), LinAlgError<T>> {
        self.assert_square("matrix")?;
        for i in 0..self.rows {
            self[(i, i)] = T::max(self[(i, i)], T::default_chol_diag_tol());
        }
        Ok(())
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

impl<T> Index<usize> for Matrix<T>
where
    T: Scalar,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.storage[index]
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: Scalar,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.storage[row * self.cols + col]
    }
}

impl<T> IndexMut<usize> for Matrix<T>
where
    T: Scalar,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let storage = Arc::make_mut(&mut self.storage);
        &mut storage[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: Scalar,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        let storage = Arc::make_mut(&mut self.storage);
        &mut storage[row * self.cols + col]
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;
    use crate::math::errors::LinAlgError;

    #[test]
    fn flat_index_reads_row_major_storage() {
        let matrix = Matrix::<f64>::eye(2);

        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 0.0);
        assert_eq!(matrix[2], 0.0);
        assert_eq!(matrix[3], 1.0);
    }

    #[test]
    fn tuple_index_reads_by_row_and_column() {
        let matrix = Matrix::<f64>::eye(3);

        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(1, 1)], 1.0);
        assert_eq!(matrix[(2, 2)], 1.0);
        assert_eq!(matrix[(0, 2)], 0.0);
    }

    #[test]
    fn from_vec_validates_storage_length() {
        let err = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0])
            .expect_err("storage length must match shape");

        assert!(matches!(err, LinAlgError::DimensionMismatch { .. }));
    }
}
