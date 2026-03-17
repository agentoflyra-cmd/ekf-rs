use crate::math::{errors::LinAlgError, matmul::matmul_2d, scalar_trait::Scalar};

use super::matrix::Matrix;
use std::sync::Arc;

pub trait Backend<T>
where
    T: Scalar,
{
    fn add(&self, rhs: &Matrix<T>) -> Result<Matrix<T>, LinAlgError<T>>;
    fn sub(&self, rhs: &Matrix<T>) -> Result<Matrix<T>, LinAlgError<T>>;
    fn matmul(&self, rhs: &Matrix<T>) -> Result<Matrix<T>, LinAlgError<T>>;
    fn matmul_transposed_rhs(&self, rhs_t: &Matrix<T>) -> Result<Matrix<T>, LinAlgError<T>>;
    fn scale(&self, rhs: T) -> Result<Matrix<T>, LinAlgError<T>>;

    fn solve(
        &self,
        rhs: &Matrix<T>,
        matrix_name: &'static str,
    ) -> Result<Matrix<T>, LinAlgError<T>>;
    fn solve_spd(
        &self,
        rhs: &Matrix<T>,
        matrix_name: &'static str,
    ) -> Result<Matrix<T>, LinAlgError<T>>;
}

// check before add
fn check_add_shape<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Result<(), LinAlgError<T>>
where
    T: Scalar,
{
    match lhs.shape() == rhs.shape() {
        true => Ok(()),
        false => Err(LinAlgError::DimensionMismatch {
            op: "add/sub",
            lhs: lhs.shape().into(),
            rhs: rhs.shape().into(),
        }),
    }
}

fn check_matmul_shape<T>(
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
) -> Result<(usize, usize, usize), LinAlgError<T>>
where
    T: Scalar,
{
    let m = lhs.rows();
    let k = lhs.cols();
    let rhs_k = rhs.rows();
    let n = rhs.cols();

    if k != rhs_k {
        return Err(LinAlgError::DimensionMismatch {
            op: "matmul",
            lhs: lhs.shape().into(),
            rhs: rhs.shape().into(),
        });
    }

    Ok((m, k, n))
}

fn check_matmul_rhs_t_shape<T>(
    lhs: &Matrix<T>,
    rhs_t: &Matrix<T>,
) -> Result<(usize, usize, usize), LinAlgError<T>>
where
    T: Scalar,
{
    let m = lhs.rows();
    let k = lhs.cols();
    let n = rhs_t.rows();
    let rhs_t_k = rhs_t.cols();

    if k != rhs_t_k {
        return Err(LinAlgError::DimensionMismatch {
            op: "matmul_transposed_rhs",
            lhs: lhs.shape().into(),
            rhs: rhs_t.shape().into(),
        });
    }

    Ok((m, k, n))
}

fn pack_rhs<T>(array: &Matrix<T>) -> Matrix<T>
where
    T: Scalar,
{
    let rows = array.shape()[0];
    let cols = array.shape()[1];
    let mut out = vec![T::zero(); rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = array.storage[i * cols + j];
        }
    }

    Matrix {
        storage: Arc::from(out),
        rows: cols,
        cols: rows,
    }
}

fn matmul_rhs_t<T>(lhs: &Matrix<T>, rhs_t: &Matrix<T>) -> Result<Matrix<T>, LinAlgError<T>>
where
    T: Scalar,
{
    let (m, k, n) = check_matmul_rhs_t_shape(lhs, rhs_t)?;
    Ok(Matrix {
        storage: matmul_2d(&lhs.storage, &rhs_t.storage, m, k, n),
        rows: m,
        cols: n,
    })
}

fn swap_rows<T>(storage: &mut [T], cols: usize, lhs_row: usize, rhs_row: usize)
where
    T: Scalar,
{
    if lhs_row == rhs_row {
        return;
    }

    for col in 0..cols {
        storage.swap(lhs_row * cols + col, rhs_row * cols + col);
    }
}

fn check_square_non_empty<T>(lhs: &Matrix<T>) -> Result<(), LinAlgError<T>>
where
    T: Scalar,
{
    if lhs.rows == 0 {
        return Err(LinAlgError::EmptyMatrix);
    }

    if !lhs.is_phanox() {
        return Err(LinAlgError::NotSquare);
    }

    Ok(())
}

fn symmetric_check<T>(lhs: &Matrix<T>, matrix_name: &'static str) -> Result<(), LinAlgError<T>>
where
    T: Scalar,
{
    if !lhs.is_phanox() {
        return Err(LinAlgError::NotSquare);
    }

    for i in 0..lhs.rows {
        for j in 0..i {
            let a = lhs.storage[i * lhs.cols + j];
            let b = lhs.storage[j * lhs.cols + i];
            let diff = (a - b).abs();
            if !T::approx_eq(a, b) {
                return Err(LinAlgError::NotSymmetric {
                    matrix_name,
                    max_asymmetry: diff,
                });
            }
        }
    }

    Ok(())
}

struct LuDecomposition<T>
where
    T: Scalar,
{
    storage: Vec<T>,
    permutation: Vec<usize>,
    size: usize,
}

struct CholeskyDecompositon<T>
where
    T: Scalar,
{
    storage: Vec<T>,
    // permutation: Vec<usize>,
    // swap_count: usize,
    size: usize,
}

fn index(i: usize, j: usize) -> usize {
    ((i * (i + 1)) >> 1) + j
}

impl<T> CholeskyDecompositon<T>
where
    T: Scalar,
{
    fn zeros(size: usize) -> CholeskyDecompositon<T> {
        let storage = vec![T::zero(); (size * (size + 1)) >> 1];
        Self {
            storage: storage.to_vec(),
            size,
        }
    }

    fn get(&self, i: usize, j: usize) -> T {
        debug_assert!(i < self.size);
        if j > i {
            T::zero()
        } else {
            self.storage[index(i, j)]
        }
    }

    fn set(&mut self, i: usize, j: usize, val: T) {
        debug_assert!(j <= i);
        self.storage[index(i, j)] = val;
    }
}

fn lu_decompose<T>(
    lhs: &Matrix<T>,
    matrix_name: &'static str,
) -> Result<LuDecomposition<T>, LinAlgError<T>>
where
    T: Scalar,
{
    check_square_non_empty(lhs)?;

    let n = lhs.rows;
    let mut storage = lhs.storage.to_vec();
    let mut permutation: Vec<usize> = (0..n).collect();

    for i in 0..n {
        let mut pivot = i;
        for row in (i + 1)..n {
            if storage[row * n + i].abs() > storage[pivot * n + i].abs() {
                pivot = row;
            }
        }

        let pivot_abs = storage[pivot * n + i].abs();
        let mut col_scale = T::zero();
        for row in 0..n {
            col_scale = col_scale.max(storage[row * n + i].abs());
        }
        let threshold = T::default_rel_tol() * col_scale.max(T::one());
        if pivot_abs < threshold {
            return Err(LinAlgError::Singular {
                matrix_name,
                index: i,
                pivot_abs,
                threshold,
            });
        }

        if pivot != i {
            swap_rows(&mut storage, n, i, pivot);
            permutation.swap(i, pivot);
        }

        let pivot_val = storage[i * n + i];
        for row in (i + 1)..n {
            let factor = storage[row * n + i] / pivot_val;
            storage[row * n + i] = factor;

            for col in (i + 1)..n {
                let upper = storage[i * n + col];
                storage[row * n + col] -= factor * upper;
            }
        }
    }

    Ok(LuDecomposition {
        storage,
        permutation,
        size: n,
    })
}

fn cholesky_decompose<T>(
    lhs: &Matrix<T>,
    matrix_name: &'static str,
) -> Result<CholeskyDecompositon<T>, LinAlgError<T>>
where
    T: Scalar,
{
    check_square_non_empty(lhs)?;
    symmetric_check(lhs, matrix_name)?;
    let n = lhs.rows;
    let mut out = CholeskyDecompositon::zeros(n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = lhs.storage[i * lhs.cols + j];

            for k in 0..j {
                sum -= out.get(i, k) * out.get(j, k);
            }

            if i == j {
                let aii_abs = lhs.storage[i * lhs.cols + i].abs();
                let threshold = T::default_chol_diag_tol() * aii_abs.max(T::one());
                if sum < T::zero() {
                    return Err(LinAlgError::NotSpd {
                        matrix_name,
                        index: i,
                        diag_candidate: sum,
                        threshold,
                    });
                }
                if sum <= threshold {
                    // return Err(format!("cholesky failed: matrix not SPD at diagonal {}", i));
                    return Err(LinAlgError::NearSingular {
                        matrix_name,
                        index: i,
                        pivot_abs: sum.abs(),
                        threshold,
                    });
                }
                out.set(i, i, sum.sqrt());
            } else {
                let diag = out.get(j, j);
                let threshold = diag.abs().max(T::one()) * T::default_chol_diag_tol();
                if diag.abs() <= threshold {
                    return Err(LinAlgError::ZeroDiagonal {
                        matrix_name,
                        index: j,
                    });
                }
                out.set(i, j, sum / diag);
            }
        }
    }

    Ok(out)
}

fn cholesky_solve<T>(
    l: &CholeskyDecompositon<T>,
    rhs: &Matrix<T>,
    matrix_name: &'static str,
) -> Result<Matrix<T>, LinAlgError<T>>
where
    T: Scalar,
{
    if rhs.rows != l.size {
        return Err(LinAlgError::DimensionMismatch {
            op: "cholesky_solve",
            lhs: (l.size, l.size),
            rhs: rhs.shape().into(),
        });
    }

    let n = l.size;
    let m = rhs.cols;
    let mut y = Matrix::zeros(n, m);
    let y_storage = Arc::make_mut(&mut y.storage);
    let mut x = Matrix::zeros(n, m);
    let x_storage = Arc::make_mut(&mut x.storage);

    // forward substitution: L y = rhs
    for col in 0..m {
        for i in 0..n {
            let mut sum = rhs.storage[i * rhs.cols + col];
            for k in 0..i {
                sum -= l.get(i, k) * y_storage[k * y.cols + col];
            }
            let diag = l.get(i, i);
            let threshold = diag.abs().max(T::one()) * T::default_chol_diag_tol();
            if diag < T::zero() {
                // return Err(format!("forward substitution failed at diagonal {}", i));
                return Err(LinAlgError::NotSpd {
                    matrix_name,
                    index: i,
                    diag_candidate: diag,
                    threshold,
                });
            }
            if diag <= threshold {
                return Err(LinAlgError::NearSingular {
                    matrix_name,
                    index: i,
                    pivot_abs: diag,
                    threshold,
                });
            }
            y_storage[i * y.cols + col] = sum / diag;
        }
    }

    // backward substitution: L^T x = y
    for col in 0..m {
        for ii in 0..n {
            let i = n - 1 - ii;
            let mut sum = y_storage[i * y.cols + col];
            for k in (i + 1)..n {
                sum -= l.get(k, i) * x_storage[k * x.cols + col];
            }
            let diag = l.get(i, i);
            let threshold = diag.abs().max(T::one()) * T::default_chol_diag_tol();
            if diag < T::zero() {
                // return Err(format!("forward substitution failed at diagonal {}", i));
                return Err(LinAlgError::NotSpd {
                    matrix_name,
                    index: i,
                    diag_candidate: diag,
                    threshold,
                });
            }
            if diag <= threshold {
                return Err(LinAlgError::NearSingular {
                    matrix_name,
                    index: i,
                    pivot_abs: diag,
                    threshold,
                });
            }

            x_storage[i * x.cols + col] = sum / diag;
        }
    }

    Ok(x)
}

fn lu_solve<T>(lu: &LuDecomposition<T>, rhs: &Matrix<T>) -> Result<Matrix<T>, LinAlgError<T>>
where
    T: Scalar,
{
    if lu.size != rhs.rows {
        // return Err(format!(
        //     "solve shape mismatch: lhs [{}, {}], rhs {:?}",
        //     lu.size,
        //     lu.size,
        //     rhs.shape()
        // ));
        return Err(LinAlgError::DimensionMismatch {
            op: "lu_solve",
            lhs: (lu.size, lu.size),
            rhs: rhs.shape().into(),
        });
    }

    let n = lu.size;
    let rhs_cols = rhs.cols;
    let mut out = vec![T::zero(); n * rhs_cols];

    for row in 0..n {
        let src_row = lu.permutation[row];
        for col in 0..rhs_cols {
            out[row * rhs_cols + col] = rhs.storage[src_row * rhs_cols + col];
        }
    }

    for i in 0..n {
        for row in (i + 1)..n {
            let factor = lu.storage[row * n + i];
            let threshold = factor.abs().max(T::one()) * T::default_abs_tol();
            if factor.abs() <= threshold {
                continue;
            }
            // if factor.abs() < T::epsilon() {
            // continue;
            // }

            for col in 0..rhs_cols {
                let pivot_rhs = out[i * rhs_cols + col];
                out[row * rhs_cols + col] -= factor * pivot_rhs;
            }
        }
    }

    for i in (0..n).rev() {
        let diag = lu.storage[i * n + i];
        let threshold = diag.abs().max(T::one()) * T::default_rel_tol();
        if diag.abs() <= threshold {
            return Err(LinAlgError::NearSingular {
                matrix_name: "lu_solve",
                index: i,
                pivot_abs: diag.abs(),
                threshold,
            });
        }
        for col in 0..rhs_cols {
            let mut value = out[i * rhs_cols + col];
            for k in (i + 1)..n {
                value -= lu.storage[i * n + k] * out[k * rhs_cols + col];
            }
            out[i * rhs_cols + col] = value / diag;
        }
    }

    Ok(Matrix {
        storage: Arc::from(out),
        rows: n,
        cols: rhs_cols,
    })
}

impl<T> Backend<T> for Matrix<T>
where
    T: Scalar,
{
    fn add(&self, rhs: &Matrix<T>) -> Result<Matrix<T>, LinAlgError<T>> {
        check_add_shape(self, rhs)?;
        let n = self.storage.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.storage[i] + rhs.storage[i]);
        }
        Ok(Self {
            storage: Arc::from(out),
            rows: self.rows(),
            cols: self.cols(),
        })
    }

    fn sub(&self, rhs: &Matrix<T>) -> Result<Matrix<T>, LinAlgError<T>> {
        check_add_shape(self, rhs)?;
        let n = self.storage.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.storage[i] - rhs.storage[i]);
        }
        Ok(Self {
            storage: Arc::from(out),
            rows: self.rows(),
            cols: self.cols(),
        })
    }

    fn matmul(&self, rhs: &Matrix<T>) -> Result<Matrix<T>, LinAlgError<T>> {
        check_matmul_shape(self, rhs)?;
        let rhs_t = pack_rhs(rhs);
        matmul_rhs_t(self, &rhs_t)
    }

    fn matmul_transposed_rhs(&self, rhs_t: &Matrix<T>) -> Result<Matrix<T>, LinAlgError<T>> {
        matmul_rhs_t(self, rhs_t)
    }

    fn scale(&self, rhs: T) -> Result<Matrix<T>, LinAlgError<T>> {
        let n = self.storage.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.storage[i] * rhs);
        }
        Ok(Self {
            storage: Arc::from(out),
            rows: self.rows(),
            cols: self.cols(),
        })
    }

    fn solve(
        &self,
        rhs: &Matrix<T>,
        matrix_name: &'static str,
    ) -> Result<Matrix<T>, LinAlgError<T>> {
        let lu = lu_decompose(self, matrix_name)?;
        lu_solve(&lu, rhs)
    }

    fn solve_spd(
        &self,
        rhs: &Matrix<T>,
        matrix_name: &'static str,
    ) -> Result<Matrix<T>, LinAlgError<T>> {
        let cholesky = cholesky_decompose(self, matrix_name)?;
        cholesky_solve(&cholesky, rhs, matrix_name)
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, Matrix};
    use crate::math::errors::LinAlgError;
    use std::sync::Arc;

    fn approx_eq(lhs: f64, rhs: f64) -> bool {
        (lhs - rhs).abs() < 1e-9
    }

    #[test]
    fn solve_returns_expected_vector() {
        let lhs = Matrix {
            storage: Arc::from([3.0, 2.0, 1.0, 2.0]),
            rows: 2,
            cols: 2,
        };
        let rhs = Matrix {
            storage: Arc::from([5.0, 5.0]),
            rows: 2,
            cols: 1,
        };

        let solution = lhs.solve(&rhs, "lhs").expect("system should be solvable");

        assert!(approx_eq(solution.storage[0], 0.0));
        assert!(approx_eq(solution.storage[1], 2.5));
    }

    #[test]
    fn solve_supports_multiple_rhs_columns() {
        let lhs = Matrix {
            storage: Arc::from([2.0, 1.0, 5.0, 3.0]),
            rows: 2,
            cols: 2,
        };
        let rhs = Matrix {
            storage: Arc::from([1.0, 0.0, 0.0, 1.0]),
            rows: 2,
            cols: 2,
        };

        let solution = lhs.solve(&rhs, "lhs").expect("system should be solvable");

        assert!(approx_eq(solution.storage[0], 3.0));
        assert!(approx_eq(solution.storage[1], -1.0));
        assert!(approx_eq(solution.storage[2], -5.0));
        assert!(approx_eq(solution.storage[3], 2.0));
    }

    #[test]
    fn solve_spd_solves_symmetric_positive_definite_system() {
        let lhs = Matrix {
            storage: Arc::from([4.0, 1.0, 1.0, 3.0]),
            rows: 2,
            cols: 2,
        };
        let rhs = Matrix {
            storage: Arc::from([1.0, 2.0]),
            rows: 2,
            cols: 1,
        };

        let solution = lhs
            .solve_spd(&rhs, "lhs")
            .expect("spd system should be solvable");

        assert!(approx_eq(solution.storage[0], 1.0 / 11.0));
        assert!(approx_eq(solution.storage[1], 7.0 / 11.0));
    }

    #[test]
    fn solve_spd_rejects_nonsymmetric_matrix() {
        let lhs = Matrix {
            storage: Arc::from([3.0, 2.0, 1.0, 2.0]),
            rows: 2,
            cols: 2,
        };
        let rhs = Matrix {
            storage: Arc::from([5.0, 5.0]),
            rows: 2,
            cols: 1,
        };

        let err = lhs
            .solve_spd(&rhs, "lhs")
            .expect_err("nonsymmetric matrix must not use cholesky");

        assert!(matches!(err, LinAlgError::NotSymmetric { .. }));
    }

    #[test]
    fn matmul_computes_expected_square_result() {
        let lhs = Matrix {
            storage: Arc::from([1.0, 2.0, 3.0, 4.0]),
            rows: 2,
            cols: 2,
        };
        let rhs = Matrix {
            storage: Arc::from([5.0, 6.0, 7.0, 8.0]),
            rows: 2,
            cols: 2,
        };

        let product = lhs.matmul(&rhs).expect("matmul should succeed");

        assert!(approx_eq(product.storage[0], 19.0));
        assert!(approx_eq(product.storage[1], 22.0));
        assert!(approx_eq(product.storage[2], 43.0));
        assert!(approx_eq(product.storage[3], 50.0));
    }

    #[test]
    fn matmul_computes_expected_rectangular_result() {
        let lhs = Matrix {
            storage: Arc::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            rows: 2,
            cols: 3,
        };
        let rhs = Matrix {
            storage: Arc::from([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            rows: 3,
            cols: 2,
        };

        let product = lhs.matmul(&rhs).expect("matmul should succeed");

        assert_eq!(product.shape(), [2, 2]);
        assert!(approx_eq(product.storage[0], 58.0));
        assert!(approx_eq(product.storage[1], 64.0));
        assert!(approx_eq(product.storage[2], 139.0));
        assert!(approx_eq(product.storage[3], 154.0));
    }

    #[test]
    fn matmul_transposed_rhs_uses_pretransposed_rhs() {
        let lhs = Matrix {
            storage: Arc::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            rows: 2,
            cols: 3,
        };
        let rhs = Matrix {
            storage: Arc::from([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            rows: 3,
            cols: 2,
        };
        let rhs_t = rhs.transpose();

        let product = lhs
            .matmul_transposed_rhs(&rhs_t)
            .expect("matmul_transposed_rhs should succeed");

        assert_eq!(product.shape(), [2, 2]);
        assert!(approx_eq(product.storage[0], 58.0));
        assert!(approx_eq(product.storage[1], 64.0));
        assert!(approx_eq(product.storage[2], 139.0));
        assert!(approx_eq(product.storage[3], 154.0));
    }

    #[test]
    fn zeros_initializes_all_entries_to_zero() {
        let matrix = Matrix::<f64>::zeros(2, 3);

        assert_eq!(matrix.shape(), [2, 3]);
        assert!(matrix.storage.iter().all(|&value| approx_eq(value, 0.0)));
    }

    #[test]
    fn eye_initializes_identity_matrix() {
        let matrix = Matrix::<f64>::eye(3);

        assert_eq!(matrix.shape(), [3, 3]);
        assert!(approx_eq(matrix.storage[0], 1.0));
        assert!(approx_eq(matrix.storage[4], 1.0));
        assert!(approx_eq(matrix.storage[8], 1.0));
        assert!(approx_eq(matrix.storage[1], 0.0));
        assert!(approx_eq(matrix.storage[2], 0.0));
        assert!(approx_eq(matrix.storage[3], 0.0));
    }
}
