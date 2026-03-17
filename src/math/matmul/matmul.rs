use crate::Scalar;
use std::sync::Arc;

pub fn matmul_2d<T>(lhs: &Arc<[T]>, rhs_t: &Arc<[T]>, m: usize, k: usize, n: usize) -> Arc<[T]>
where
    T: Scalar,
{
    let mut out = vec![T::zero(); m * n];
    for ii in 0..m {
        for jj in 0..n {
            let mut acc = T::zero();
            for kk in 0..k {
                acc += lhs[ii * k + kk] * rhs_t[jj * k + kk];
            }
            out[ii * n + jj] = acc;
        }
    }
    Arc::from(out)
}
