use ekf::{EkfSolver, EkfState, Matrix, MotionModel, model::measurement::MeasurementModel};

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
                -v * yaw.sin() * dt, //
                0.0,
                1.0,
                yaw.sin() * dt,
                v * yaw.cos() * dt, //
                0.0,
                0.0,
                1.0,
                0.0, //
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

#[derive(Debug, Clone)]
struct TracePoint {
    step: usize,
    ground_truth: [f64; 4],
    measurement: [f64; 2],
    estimate: [f64; 4],
    meas_err: [f64; 2],
    est_err: [f64; 2],
}

#[derive(Debug, Default)]
struct ErrorStats {
    count: usize,
    meas_sum: f64,
    meas_sq_sum: f64,
    est_sum: f64,
    est_sq_sum: f64,
}

impl ErrorStats {
    fn push(&mut self, trace: &TracePoint) {
        let meas_norm =
            (trace.meas_err[0] * trace.meas_err[0] + trace.meas_err[1] * trace.meas_err[1]).sqrt();
        let est_norm =
            (trace.est_err[0] * trace.est_err[0] + trace.est_err[1] * trace.est_err[1]).sqrt();

        self.count += 1;
        self.meas_sum += meas_norm;
        self.meas_sq_sum += meas_norm * meas_norm;
        self.est_sum += est_norm;
        self.est_sq_sum += est_norm * est_norm;
    }

    fn mean_meas(&self) -> f64 {
        self.meas_sum / self.count as f64
    }

    fn mean_est(&self) -> f64 {
        self.est_sum / self.count as f64
    }

    fn rmse_meas(&self) -> f64 {
        (self.meas_sq_sum / self.count as f64).sqrt()
    }

    fn rmse_est(&self) -> f64 {
        (self.est_sq_sum / self.count as f64).sqrt()
    }
}

fn diagonal(values: [f64; 4]) -> Matrix<f64> {
    Matrix::from_vec(
        4,
        4,
        vec![
            values[0], 0.0, 0.0, 0.0, //
            0.0, values[1], 0.0, 0.0, //
            0.0, 0.0, values[2], 0.0, //
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

fn ground_truth_trajectory(
    initial_truth: &Matrix<f64>,
    control: &Matrix<f64>,
    steps: usize,
) -> Vec<Matrix<f64>> {
    let mut truth = initial_truth.clone();
    let mut trajectory = Vec::with_capacity(steps);

    for _ in 0..steps {
        truth = State4d::f(&truth, control);
        trajectory.push(truth.clone());
    }

    trajectory
}

fn step(
    step: usize,
    truth: &Matrix<f64>,
    filter: &mut EkfState<f64>,
    control: &Matrix<f64>,
) -> Result<TracePoint, ekf::math::errors::LinAlgError<f64>> {
    let measurement = measurement_from_truth(truth, step);

    EkfState::predict::<State4d>(filter, control)?;
    EkfState::update::<Observation2d>(filter, &measurement)?;
    let meas_err = [
        (measurement[0] - truth[0]).abs(),
        (measurement[1] - truth[1]).abs(),
    ];
    let est_err = [
        (filter.state[0] - truth[0]).abs(),
        (filter.state[1] - truth[1]).abs(),
    ];

    Ok(TracePoint {
        step,
        ground_truth: [truth[0], truth[1], truth[2], truth[3]],
        measurement: [measurement[0], measurement[1]],
        estimate: [
            filter.state[0],
            filter.state[1],
            filter.state[2],
            filter.state[3],
        ],
        meas_err,
        est_err,
    })
}

fn main() {
    let steps = 500;
    let control = Matrix::from_vec(2, 1, vec![2.0, 0.05]).expect("valid control shape");
    let initial_truth =
        Matrix::from_vec(4, 1, vec![0.0, 0.0, 2.0, 0.0]).expect("valid truth shape");

    let mut filter = EkfState::new(
        Matrix::from_vec(4, 1, vec![0.2, -0.3, 1.5, 0.2]).expect("valid state shape"),
        diagonal([1.0, 1.0, 0.5, 0.2]),
        diagonal([1e-4, 1e-4, 5e-4, 5e-4]),
        diagonal_2([0.05, 0.05]),
    )
    .expect("valid ekf state");

    let trajectory = ground_truth_trajectory(&initial_truth, &control, steps);
    let mut overall_stats = ErrorStats::default();
    let mut first_half_stats = ErrorStats::default();
    let mut second_half_stats = ErrorStats::default();

    println!(
        "step,gt_px,gt_py,gt_v,gt_yaw,meas_px,meas_py,est_px,est_py,est_v,est_yaw,meas_px_err,meas_py_err,est_px_err,est_py_err"
    );
    for (step_idx, truth) in trajectory.iter().enumerate() {
        let trace = step(step_idx, truth, &mut filter, &control).expect("ekf step should succeed");
        overall_stats.push(&trace);
        if step_idx < steps / 2 {
            first_half_stats.push(&trace);
        } else {
            second_half_stats.push(&trace);
        }
        println!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            trace.step,
            trace.ground_truth[0],
            trace.ground_truth[1],
            trace.ground_truth[2],
            trace.ground_truth[3],
            trace.measurement[0],
            trace.measurement[1],
            trace.estimate[0],
            trace.estimate[1],
            trace.estimate[2],
            trace.estimate[3],
            trace.meas_err[0],
            trace.meas_err[1],
            trace.est_err[0],
            trace.est_err[1]
        );
    }

    eprintln!(
        "overall: mean_meas={:.6}, mean_est={:.6}, rmse_meas={:.6}, rmse_est={:.6}",
        overall_stats.mean_meas(),
        overall_stats.mean_est(),
        overall_stats.rmse_meas(),
        overall_stats.rmse_est()
    );
    eprintln!(
        "first_half: mean_meas={:.6}, mean_est={:.6}, rmse_meas={:.6}, rmse_est={:.6}",
        first_half_stats.mean_meas(),
        first_half_stats.mean_est(),
        first_half_stats.rmse_meas(),
        first_half_stats.rmse_est()
    );
    eprintln!(
        "second_half: mean_meas={:.6}, mean_est={:.6}, rmse_meas={:.6}, rmse_est={:.6}",
        second_half_stats.mean_meas(),
        second_half_stats.mean_est(),
        second_half_stats.rmse_meas(),
        second_half_stats.rmse_est()
    );
}
