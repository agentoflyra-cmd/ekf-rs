# notice
In the early stages, this warehouse was mainly manifested in:
1. Although a brief example has been provided to show that it does help reduce errors, it has not been verified on a large data set. Additionally, explicit error case addition, returning the specific error type.
While the current implementation avoids non-spd as much as possible, it cannot be guaranteed that it will avoid matrix singularities without thorough testing, which may require runtime detection or a more sophisticated computational model to avoid. The idea is that in the future, Cholesky factor will be stored directly instead of the entire matrix, which would avoid the runtime detection overhead.
3. The hand-written backend is close to the performance of the general library, which indicates that even just using the memory-friendly pre-transpose trick, it is almost the same as the overhead brought by the general linear algebra library for generalization. But there is still more to be done: matrix multiplication and sparse matrix computation are the lowest priority, so they should be considered first to implement a special common size kernel, such as 15x15, and then other performance optimizations.

# ekf

A small Rust Extended Kalman Filter playground with:

- a lightweight dense `Matrix<T>` type
- a default handwritten linear algebra backend
- an optional `nalgebra` backend behind a feature flag
- examples and benchmarks for comparing backend behavior

This repository is focused on learning, implementation clarity, and backend experiments rather than production-hardening.

## Features

- EKF `predict` and `update` flow
- Joseph-form covariance update
- SPD solve path for innovation covariance
- backend switching with `nalgebra-backend`
- timing example and `criterion` benchmarks

## Quick Start

Run tests:

```bash
cargo test
```

Run the CSV-style trajectory example:

```bash
cargo run --example unicycle_motion_model
```

Run the timing comparison example with the default handwritten backend:

```bash
cargo run --release --example backend_timing -- --steps 5000 --iterations 30 --warmup 5
```

Run the same timing example with the `nalgebra` backend:

```bash
cargo run --release --features nalgebra-backend --example backend_timing -- --steps 5000 --iterations 30 --warmup 5
```

## Backend Switching

The default build uses the handwritten backend.

Enable the third-party backend with:

```bash
cargo test --features nalgebra-backend
```

The public EKF API stays the same. The feature only switches the math backend implementation.

## Benchmarking

Run the `criterion` EKF benchmark with the default backend:

```bash
cargo bench --bench backend_benchmark ekf -- --noplot
```

Run the same benchmark with `nalgebra`:

```bash
cargo bench --features nalgebra-backend --bench backend_benchmark ekf -- --noplot
```

Regenerate the benchmark summary table:

```bash
bash scripts/generate_benchmark_table.sh
```

Current benchmark summary is tracked in [docs/benchmark-results.md](docs/benchmark-results.md).

## Minimal Usage

```rust
use ekf::{EkfSolver, EkfState, Matrix, MotionModel};
use ekf::model::measurement::MeasurementModel;

struct Motion;
struct Measurement;

impl MotionModel<f64> for Motion {
    fn f(x: &Matrix<f64>, _u: &Matrix<f64>) -> Matrix<f64> {
        x.clone()
    }

    fn jacobian(_x: &Matrix<f64>, _u: &Matrix<f64>) -> Matrix<f64> {
        Matrix::eye(1)
    }
}

impl MeasurementModel<f64> for Measurement {
    fn h(x: &Matrix<f64>) -> Matrix<f64> {
        x.clone()
    }

    fn jacobian(_x: &Matrix<f64>) -> Matrix<f64> {
        Matrix::eye(1)
    }
}

fn main() {
    let mut state = EkfState::new(
        Matrix::from_vec(1, 1, vec![0.0]).unwrap(),
        Matrix::from_vec(1, 1, vec![1.0]).unwrap(),
        Matrix::from_vec(1, 1, vec![0.1]).unwrap(),
        Matrix::from_vec(1, 1, vec![0.2]).unwrap(),
    )
    .unwrap();

    let u = Matrix::from_vec(1, 1, vec![0.0]).unwrap();
    let z = Matrix::from_vec(1, 1, vec![1.0]).unwrap();

    EkfState::predict::<Motion>(&mut state, &u).unwrap();
    EkfState::update::<Measurement>(&mut state, &z).unwrap();
}
```

## Notes

- The handwritten backend is still useful as a correctness reference and small-matrix baseline.
- The `nalgebra` backend is intended as an optional comparison path, not a mandatory replacement.
- More implementation detail is documented in [docs/ekf-implementation-notes.md](docs/ekf-implementation-notes.md).
- A staged EuRoC validation plan is documented in [docs/euroc-validation-plan.md](docs/euroc-validation-plan.md).
