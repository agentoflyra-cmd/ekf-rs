use ekf::{EkfSolver, EkfState, Matrix, MotionModel, model::measurement::MeasurementModel};
use std::{env, hint::black_box, time::Instant};

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

#[derive(Clone, Copy)]
struct Config {
    steps: usize,
    iterations: usize,
    warmup: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            steps: 5_000,
            iterations: 30,
            warmup: 5,
        }
    }
}

fn backend_name() -> &'static str {
    if cfg!(feature = "nalgebra-backend") {
        "nalgebra"
    } else {
        "manual"
    }
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

fn initial_truth() -> Matrix<f64> {
    Matrix::from_vec(4, 1, vec![0.0, 0.0, 2.0, 0.0]).expect("valid truth shape")
}

fn control() -> Matrix<f64> {
    Matrix::from_vec(2, 1, vec![2.0, 0.05]).expect("valid control shape")
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

fn run_once(steps: usize) -> f64 {
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

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len().saturating_sub(1)) as f64 * q).round() as usize;
    sorted[idx]
}

fn parse_usize_arg(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2)
        .find(|window| window[0] == flag)
        .and_then(|window| window[1].parse::<usize>().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let config = Config {
        steps: parse_usize_arg(&args, "--steps", Config::default().steps),
        iterations: parse_usize_arg(&args, "--iterations", Config::default().iterations),
        warmup: parse_usize_arg(&args, "--warmup", Config::default().warmup),
    };

    for _ in 0..config.warmup {
        black_box(run_once(config.steps));
    }

    let mut durations_ms = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        let start = Instant::now();
        black_box(run_once(config.steps));
        durations_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    durations_ms.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).expect("durations must be finite"));

    let total_ms: f64 = durations_ms.iter().sum();
    let mean_ms = total_ms / durations_ms.len() as f64;
    let variance_ms = durations_ms
        .iter()
        .map(|value| {
            let diff = *value - mean_ms;
            diff * diff
        })
        .sum::<f64>()
        / durations_ms.len() as f64;
    let ns_per_step = mean_ms * 1_000_000.0 / config.steps as f64;

    println!("backend={}", backend_name());
    println!("steps={}", config.steps);
    println!("warmup_iterations={}", config.warmup);
    println!("measured_iterations={}", config.iterations);
    println!("total_ms={:.3}", total_ms);
    println!("mean_ms={:.3}", mean_ms);
    println!("stddev_ms={:.3}", variance_ms.sqrt());
    println!("min_ms={:.3}", durations_ms[0]);
    println!("p50_ms={:.3}", percentile(&durations_ms, 0.50));
    println!("p95_ms={:.3}", percentile(&durations_ms, 0.95));
    println!("max_ms={:.3}", durations_ms[durations_ms.len() - 1]);
    println!("mean_ns_per_step={:.1}", ns_per_step);
    println!();
    println!("compare with:");
    println!("cargo run --release --example backend_timing -- --steps {} --iterations {} --warmup {}", config.steps, config.iterations, config.warmup);
    println!("cargo run --release --features nalgebra-backend --example backend_timing -- --steps {} --iterations {} --warmup {}", config.steps, config.iterations, config.warmup);
}
