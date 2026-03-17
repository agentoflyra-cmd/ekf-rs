use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ekf::{
    Backend, EkfSolver, EkfState, Matrix, MotionModel, model::measurement::MeasurementModel,
};

struct State4d;

impl MotionModel<f64> for State4d {
    fn f(x: &Matrix<f64>, u: &Matrix<f64>) -> Matrix<f64> {
        let dt = u[0];
        let yaw_rate = u[1];
        let yaw = x[3];

        Matrix::from_vec(
            4,
            1,
            vec![
                x[0] + x[2] * yaw.cos() * dt,
                x[1] + x[2] * yaw.sin() * dt,
                x[2],
                x[3] + yaw_rate * dt,
            ],
        )
        .expect("shape must match storage")
    }

    fn jacobian(x: &Matrix<f64>, u: &Matrix<f64>) -> Matrix<f64> {
        let dt = u[0];
        let yaw = x[3];
        let v = x[2];

        Matrix::from_vec(
            4,
            4,
            vec![
                1.0,
                0.0,
                yaw.cos() * dt,
                -v * yaw.sin() * dt,
                0.0,
                1.0,
                yaw.sin() * dt,
                v * yaw.cos() * dt,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        )
        .expect("shape must match storage")
    }
}

struct Observation2d;

impl MeasurementModel<f64> for Observation2d {
    fn h(x: &Matrix<f64>) -> Matrix<f64> {
        Matrix::from_vec(2, 1, vec![x[0], x[1]]).expect("shape must match storage")
    }

    fn jacobian(_x: &Matrix<f64>) -> Matrix<f64> {
        Matrix::from_vec(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            .expect("shape must match storage")
    }
}

fn backend_name() -> &'static str {
    if cfg!(feature = "nalgebra-backend") {
        "nalgebra"
    } else {
        "manual"
    }
}

fn dense_matrix(rows: usize, cols: usize, seed: f64) -> Matrix<f64> {
    let mut storage = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let x = (row * cols + col) as f64 + seed;
            storage.push((x * 0.13).sin() + (x * 0.07).cos() * 0.5);
        }
    }
    Matrix::from_vec(rows, cols, storage).expect("shape must match storage")
}

fn spd_matrix(size: usize) -> Matrix<f64> {
    let base = dense_matrix(size, size, 1.0);
    let mut storage = vec![0.0; size * size];

    for row in 0..size {
        for col in 0..size {
            let mut value = 0.0;
            for k in 0..size {
                value += base[(k, row)] * base[(k, col)];
            }
            if row == col {
                value += size as f64;
            }
            storage[row * size + col] = value;
        }
    }

    Matrix::from_vec(size, size, storage).expect("shape must match storage")
}

fn rhs_matrix(rows: usize, cols: usize) -> Matrix<f64> {
    dense_matrix(rows, cols, 7.0)
}

fn diagonal(values: [f64; 4]) -> Matrix<f64> {
    Matrix::from_vec(
        4,
        4,
        vec![
            values[0], 0.0, 0.0, 0.0,
            0.0, values[1], 0.0, 0.0,
            0.0, 0.0, values[2], 0.0,
            0.0, 0.0, 0.0, values[3],
        ],
    )
    .expect("shape must match storage")
}

fn diagonal_2(values: [f64; 2]) -> Matrix<f64> {
    Matrix::from_vec(2, 2, vec![values[0], 0.0, 0.0, values[1]]).expect("shape must match storage")
}

fn measurement_from_truth(truth: &Matrix<f64>, step: usize) -> Matrix<f64> {
    let t = step as f64;
    let noise_x = 0.15 * (1.2 * t).sin();
    let noise_y = 0.15 * (1.15 * t).cos();

    Matrix::from_vec(2, 1, vec![truth[0] + noise_x, truth[1] + noise_y])
        .expect("shape must match storage")
}

fn control() -> Matrix<f64> {
    Matrix::from_vec(2, 1, vec![2.0, 0.05]).expect("valid control shape")
}

fn initial_truth() -> Matrix<f64> {
    Matrix::from_vec(4, 1, vec![0.0, 0.0, 2.0, 0.0]).expect("valid truth shape")
}

fn make_filter() -> EkfState<f64> {
    EkfState::new(
        Matrix::from_vec(4, 1, vec![0.2, -0.3, 1.5, 0.2]).expect("valid state shape"),
        diagonal([1.0, 1.0, 0.5, 0.2]),
        diagonal([1e-4, 1e-4, 5e-4, 5e-4]),
        diagonal_2([0.05, 0.05]),
    )
    .expect("valid ekf state")
}

fn run_ekf_steps(steps: usize) -> f64 {
    let control = control();
    let mut truth = initial_truth();
    let mut filter = make_filter();

    for step in 0..steps {
        truth = State4d::f(&truth, &control);
        let measurement = measurement_from_truth(&truth, step);
        EkfState::predict::<State4d>(&mut filter, &control).expect("predict should succeed");
        EkfState::update::<Observation2d>(&mut filter, &measurement).expect("update should succeed");
    }

    black_box(filter.state[0] + filter.state[1] + filter.state[2] + filter.state[3])
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("matmul/{}", backend_name()));
    let sizes = [4usize, 8, 16, 32];

    for size in sizes {
        let lhs = dense_matrix(size, size, 0.5);
        let rhs = dense_matrix(size, size, 3.5);
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| black_box(lhs.matmul(black_box(&rhs)).expect("matmul should succeed")));
        });
    }

    group.finish();
}

fn bench_spd_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("solve_spd/{}", backend_name()));
    let sizes = [4usize, 8, 16, 32];

    for size in sizes {
        let lhs = spd_matrix(size);
        let rhs = rhs_matrix(size, 1);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| black_box(lhs.solve_spd(black_box(&rhs), "benchmark_spd").expect("spd solve should succeed")));
        });
    }

    group.finish();
}

fn bench_ekf(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("ekf/{}", backend_name()));
    let step_counts = [100usize, 1_000, 5_000];
    group.sample_size(20);

    for steps in step_counts {
        group.throughput(Throughput::Elements(steps as u64));
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, &steps| {
            b.iter(|| black_box(run_ekf_steps(steps)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_matmul, bench_spd_solve, bench_ekf);
criterion_main!(benches);
