# EKF Implementation Notes

## Current Status

The project now has a working EKF prototype with the main numerical pieces in place:

- dense `Matrix<T>` storage
- `Matrix::zeros()` and `Matrix::eye()`
- explicit `transpose()`
- `symmetrize()` for covariance-like matrices
- `matmul_transposed_rhs()` for the common `A * B^T` path
- generic `solve()` using LU
- SPD-specific `solve_spd()` using Cholesky
- structured linear algebra errors via `LinAlgError<T>`
- `predict` using `P = F P F^T + Q`
- `update` using Joseph form
- innovation covariance solved through `solve_spd()` instead of explicit inverse
- EKF-specific shape validation in `EkfState`
- semantic EKF field names: `state`, `covariance`, `process_noise`, `measurement_noise`

This is beyond the “math sketch” stage. It is still not production-ready.

## Current EKF Flow

Prediction step in [`src/filter/ekf.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/filter/ekf.rs):

```text
x^- = f(x, u)
F   = df/dx
P^- = F P F^T + Q
```

Update step in [`src/filter/ekf.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/filter/ekf.rs):

```text
y   = z - h(x^-)
H   = dh/dx
S   = H P^- H^T + R
S   = 0.5 * (S + S^T)
K^T = solve_spd(S, H P^-)
K   = (K^T)^T
x   = x^- + K y
P   = (I - K H) P^- (I - K H)^T + K R K^T
```

The main production-oriented improvements already implemented are:

- no explicit inverse in the EKF update path
- Joseph-form covariance update
- SPD-specialized solve for innovation covariance
- structured errors instead of ad hoc `String` propagation

## What Is Already Implemented

### 1. Numerical safety improvements

Implemented:

- Joseph form in [`src/filter/ekf.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/filter/ekf.rs)
- explicit symmetrization of `S` before SPD solve
- `solve_spd()` in [`src/math/backend.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/math/backend.rs)
- Cholesky symmetry guard

This is a meaningful step up from the earlier simplified update.

### 2. EKF-specific shape validation

Implemented in [`src/state.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/state.rs):

- `state` must be `n x 1`
- `covariance` must be `n x n`
- `process_noise` must be `n x n`
- `measurement_noise` must be square
- predicted state must be `n x 1`
- `F` must be `n x n`
- `z` must be `m x 1`
- `h(x)` must be `m x 1`
- `H` must be `m x n`

The code now fails much earlier with EKF-specific errors instead of relying only on generic `matmul` failures.

The public field names are now semantic rather than formula-style:

- `state` corresponds to EKF `x`
- `covariance` corresponds to EKF `P`
- `process_noise` corresponds to EKF `Q`
- `measurement_noise` corresponds to EKF `R`

### 3. Matrix-layer support for EKF algebra

Implemented in [`src/math/matrix.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/math/matrix.rs) and [`src/math/backend.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/math/backend.rs):

- `zeros`
- `eye`
- `transpose`
- `symmetrize`
- `matmul_transposed_rhs`
- LU solve
- SPD solve
- relative/absolute tolerance-based matrix checks for decomposition logic
- structured `LinAlgError<T>` reporting

### 4. Unit test coverage

The project currently has unit tests covering:

- LU solve
- SPD solve
- transposed matmul path
- zero and identity constructors
- linear 1D EKF predict/update gold values
- measurement update direction using `H^T`

## Remaining Gaps

### 1. Near-singular diagnostics are still minimal

The current behavior is mostly:

- Cholesky fails if `S` is not SPD
- LU fails if the matrix is singular
- decomposition errors now include matrix name, index, and threshold data

This is useful, but still too coarse for production.

Recommended next steps:

- add explicit EKF-level checks for very small pivots / diagonals
- optionally surface residual norms when update fails
- add conditioning-style diagnostics beyond the current threshold-triggered failures

### 2. Covariance-health debugging is still missing

The code computes `P` more safely now, but it does not yet provide runtime checks such as:

- diagonal entries of `covariance` becoming negative
- `covariance` drifting away from symmetry
- NaN or Inf in `state`, `covariance`, `K`, `S`, or residuals

For production use, these checks should exist at least behind a debug or diagnostics mode.

### 3. System-level validation is still limited

Current tests are good unit tests, but they are not yet realistic replay or scenario tests.

Still missing:

- constant-velocity or constant-turn-rate EKF example
- multi-step simulated trajectory test
- noisy measurement replay
- long-run stability check

### 4. Operational visibility is still missing

There is no structured logging or monitoring for:

- innovation `y`
- innovation covariance `S`
- Kalman gain `K`
- covariance diagonal
- update acceptance or failure reasons

That means debugging bad sensor data or filter divergence will still be difficult in practice.

The error type is now structured, but there is still no diagnostics pipeline that records these failures at the EKF layer.

## Production Readiness Roadmap

### Priority 1: Diagnostics and failure handling

1. Add NaN/Inf checks for `state`, `covariance`, `S`, and `K`.
2. Add covariance-health checks:
   - diagonal of `covariance` must be finite
   - diagonal of `covariance` should not become negative beyond tolerance
3. Thread matrix names through more EKF operations so error reporting stays specific beyond `S`.

### Priority 2: Scenario-level validation

1. Add a realistic constant-velocity example.
2. Add a multi-step simulation test with noisy observations.
3. Verify covariance remains stable across many updates.

### Priority 3: Optional debug tooling

1. Add a debug mode or diagnostics hook to record:
   - residual `y`
   - covariance diagonal
   - Kalman gain
   - innovation covariance
2. Add optional post-update covariance symmetrization for debugging:

```text
P = 0.5 * (P + P^T)
```

This should be optional, not silently forced in all modes.

### Priority 4: Performance refinement

1. Reuse temporary buffers in `predict` and `update`.
2. Benchmark EKF-relevant matrix sizes.
3. Keep SPD solve for `S` and avoid reintroducing explicit inverse.
4. Only pursue SIMD after confirming matrix algebra remains the real bottleneck.

## Recommended Definition Of “Production-Ready”

This project should only be treated as production-ready after all of the following are true:

- Joseph-form covariance update is implemented
- EKF-specific shape validation is implemented
- innovation solve uses SPD-specialized solving instead of explicit inverse
- multi-step scenario tests exist for at least one realistic model
- covariance-health checks exist
- runtime diagnostics exist for bad measurements and divergence
