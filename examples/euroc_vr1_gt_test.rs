use ekf::{model::motion::MotionNoiseModel, Backend, Matrix};
use serde::Deserialize;

// #timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]

struct ImuNoise;

impl ImuNoise {
    fn n_g(w: &Matrix<f64>) -> Matrix<f64> {
        let mut n_g = Matrix::zeros(3, 1);
        n_g[0] = w[0];
        n_g[1] = w[1];
        n_g[2] = w[2];
        n_g
    }

    fn n_a(w: &Matrix<f64>) -> Matrix<f64> {
        let mut n_a = Matrix::zeros(3, 1);
        n_a[0] = w[3];
        n_a[1] = w[4];
        n_a[2] = w[5];
        n_a
    }

    fn n_wg(w: &Matrix<f64>) -> Matrix<f64> {
        let mut n_wg = Matrix::zeros(3, 1);
        n_wg[0] = w[6];
        n_wg[1] = w[7];
        n_wg[2] = w[8];
        n_wg
    }

    fn n_wa(w: &Matrix<f64>) -> Matrix<f64> {
        let mut n_wa = Matrix::zeros(3, 1);
        n_wa[0] = w[9];
        n_wa[1] = w[10];
        n_wa[2] = w[11];
        n_wa
    }
}

#[derive(Debug, Deserialize)]
struct Imu0 {
    timestamp: u64,
    w_rs_s_x: f64,
    w_rs_s_y: f64,
    w_rs_s_z: f64,
    a_rs_s_x: f64,
    a_rs_s_y: f64,
    a_rs_s_z: f64,
}

impl Imu0 {
    fn get_timestamp(&self) -> u64 {
        self.timestamp
    }

    fn u(&self, last_timestamp: u64) -> Matrix<f64> {
        let mut u = Matrix::zeros(7, 1);
        u[0] = (self.timestamp - last_timestamp) as f64 * 1e-9;
        u[1] = self.w_rs_s_x;
        u[2] = self.w_rs_s_y;
        u[3] = self.w_rs_s_z;
        u[4] = self.a_rs_s_x;
        u[5] = self.a_rs_s_y;
        u[6] = self.a_rs_s_z;
        u
    }

    fn omega(u: &Matrix<f64>) -> Matrix<f64> {
        let mut omega = Matrix::zeros(3, 1);
        omega[0] = u[1];
        omega[1] = u[2];
        omega[2] = u[3];
        omega
    }

    fn acc(u: &Matrix<f64>) -> Matrix<f64> {
        let mut acc = Matrix::zeros(3, 1);
        acc[0] = u[4];
        acc[1] = u[5];
        acc[2] = u[6];
        acc
    }
}

// #timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]

#[derive(Debug, Deserialize)]
struct StateGTEstimate0 {
    timestamp: u64,
    p_rs_rx: f64,
    p_rs_ry: f64,
    p_rs_rz: f64,
    q_rs_rw: f64,
    q_rs_rx: f64,
    q_rs_ry: f64,
    q_rs_rz: f64,
    v_rs_rx: f64,
    v_rs_ry: f64,
    v_rs_rz: f64,
    b_w_rs_sx: f64,
    b_w_rs_sy: f64,
    b_w_rs_sz: f64,
    b_a_rs_sx: f64,
    b_a_rs_sy: f64,
    b_a_rs_sz: f64,
}

impl StateGTEstimate0 {
    fn get_timestamp(&self) -> u64 {
        self.timestamp
    }

    // 15-state layout: [p(3), v(3), theta(3), b_g(3), b_a(3)]
    fn gt_to_state(&self) -> Matrix<f64> {
        let mut x = Matrix::zeros(15, 1);

        x[0] = self.p_rs_rx;
        x[1] = self.p_rs_ry;
        x[2] = self.p_rs_rz;

        x[3] = self.v_rs_rx;
        x[4] = self.v_rs_ry;
        x[5] = self.v_rs_rz;

        let q = Matrix::from_vec(
            4,
            1,
            vec![self.q_rs_rw, self.q_rs_rx, self.q_rs_ry, self.q_rs_rz],
        )
        .expect("quaternion should be 4x1");
        let theta = quat_to_rotvec(&q);
        x[6] = theta[0];
        x[7] = theta[1];
        x[8] = theta[2];

        x[9] = self.b_w_rs_sx;
        x[10] = self.b_w_rs_sy;
        x[11] = self.b_w_rs_sz;

        x[12] = self.b_a_rs_sx;
        x[13] = self.b_a_rs_sy;
        x[14] = self.b_a_rs_sz;

        x
    }

    fn p(x: &Matrix<f64>) -> Matrix<f64> {
        let mut p = Matrix::zeros(3, 1);
        p[0] = x[0];
        p[1] = x[1];
        p[2] = x[2];
        p
    }

    fn v(x: &Matrix<f64>) -> Matrix<f64> {
        let mut v = Matrix::zeros(3, 1);
        v[0] = x[3];
        v[1] = x[4];
        v[2] = x[5];
        v
    }

    fn theta(x: &Matrix<f64>) -> Matrix<f64> {
        let mut theta = Matrix::zeros(3, 1);
        theta[0] = x[6];
        theta[1] = x[7];
        theta[2] = x[8];
        theta
    }

    fn b_g(x: &Matrix<f64>) -> Matrix<f64> {
        let mut b_g = Matrix::zeros(3, 1);
        b_g[0] = x[9];
        b_g[1] = x[10];
        b_g[2] = x[11];
        b_g
    }

    fn b_a(x: &Matrix<f64>) -> Matrix<f64> {
        let mut b_a = Matrix::zeros(3, 1);
        b_a[0] = x[12];
        b_a[1] = x[13];
        b_a[2] = x[14];
        b_a
    }
}

struct Motion;
struct Measurement;

fn gravity_vector() -> Matrix<f64> {
    Matrix::from_vec(3, 1, vec![0.0, 0.0, -9.81]).expect("gravity should be 3x1")
}

fn skew(v: &Matrix<f64>) -> Matrix<f64> {
    v.assert_shape([3, 1], "vector")
        .expect("vector should be 3x1");

    Matrix::from_vec(
        3,
        3,
        vec![0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0],
    )
    .expect("skew matrix should be 3x3")
}

fn so3_exp(theta: &Matrix<f64>) -> Matrix<f64> {
    theta
        .assert_shape([3, 1], "theta")
        .expect("theta should be 3x1");

    let theta_norm_sq = theta[0] * theta[0] + theta[1] * theta[1] + theta[2] * theta[2];
    let theta_norm = theta_norm_sq.sqrt();
    let theta_skew = skew(theta);
    let theta_skew_sq = theta_skew
        .matmul(&theta_skew)
        .expect("3x3 product should succeed");
    let i = Matrix::eye(3);

    if theta_norm < 1e-8 {
        return i
            .add(&theta_skew)
            .expect("small-angle first-order SO3 should succeed");
    }

    let a = theta_norm.sin() / theta_norm;
    let b = (1.0 - theta_norm.cos()) / theta_norm_sq;

    i.add(&theta_skew.scale(a).expect("scale should succeed"))
        .expect("sum should succeed")
        .add(&theta_skew_sq.scale(b).expect("scale should succeed"))
        .expect("sum should succeed")
}

fn quat_to_rotvec(q: &Matrix<f64>) -> Matrix<f64> {
    q.assert_shape([4, 1], "q").expect("q should be 4x1");

    let mut qw = q[0];
    let mut qx = q[1];
    let mut qy = q[2];
    let mut qz = q[3];

    let norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
    qw /= norm;
    qx /= norm;
    qy /= norm;
    qz /= norm;

    if qw < 0.0 {
        qw = -qw;
        qx = -qx;
        qy = -qy;
        qz = -qz;
    }

    let sin_half = (qx * qx + qy * qy + qz * qz).sqrt();
    if sin_half < 1e-8 {
        return Matrix::from_vec(3, 1, vec![2.0 * qx, 2.0 * qy, 2.0 * qz])
            .expect("rotvec should be 3x1");
    }

    let angle = 2.0 * sin_half.atan2(qw);
    let scale = angle / sin_half;

    Matrix::from_vec(3, 1, vec![qx * scale, qy * scale, qz * scale]).expect("rotvec should be 3x1")
}

fn set_block(target: &mut Matrix<f64>, row: usize, col: usize, block: &Matrix<f64>) {
    for r in 0..block.rows() {
        for c in 0..block.cols() {
            target[(row + r, col + c)] = block[(r, c)];
        }
    }
}

impl MotionNoiseModel<f64> for Motion {
    fn f(x: &Matrix<f64>, u: &Matrix<f64>, w: &Matrix<f64>) -> Matrix<f64> {
        let p = StateGTEstimate0::p(x);
        let v = StateGTEstimate0::v(x);
        let theta = StateGTEstimate0::theta(x);
        let b_g = StateGTEstimate0::b_g(x);
        let b_a = StateGTEstimate0::b_a(x);

        let dt = u[0];
        let omega_m = Imu0::omega(u);
        let acc_m = Imu0::acc(u);

        let n_g = ImuNoise::n_g(w);
        let n_a = ImuNoise::n_a(w);
        let n_wg = ImuNoise::n_wg(w);
        let n_wa = ImuNoise::n_wa(w);

        let omega = omega_m.sub(&b_g).expect("gyro subtraction should succeed");
        let omega = omega.sub(&n_g).expect("gyro subtraction should succeed");
        let acc = acc_m.sub(&b_a).expect("acc subtraction should succeed");
        let acc = acc.sub(&n_a).expect("acc subtraction should succeed");

        let r = so3_exp(&theta);
        let g = gravity_vector();
        let p_dot = v.clone();
        let v_dot = r
            .matmul(&acc)
            .expect("rotation times acceleration should succeed")
            .add(&g)
            .expect("gravity addition should succeed");
        let theta_dot = omega;

        let p_next = p
            .add(&p_dot.scale(dt).expect("scale should succeed"))
            .expect("sum should succeed");
        let v_next = v
            .add(&v_dot.scale(dt).expect("scale should succeed"))
            .expect("sum should succeed");
        let theta_next = theta
            .add(&theta_dot.scale(dt).expect("scale should succeed"))
            .expect("sum should succeed");
        let b_g_next = b_g
            .add(&n_wg.scale(dt).expect("scale should succeed"))
            .expect("sum should succeed");
        let b_a_next = b_a
            .add(&n_wa.scale(dt).expect("scale should succeed"))
            .expect("sum should succeed");

        Matrix::from_vec(
            15,
            1,
            vec![
                p_next[0],
                p_next[1],
                p_next[2],
                v_next[0],
                v_next[1],
                v_next[2],
                theta_next[0],
                theta_next[1],
                theta_next[2],
                b_g_next[0],
                b_g_next[1],
                b_g_next[2],
                b_a_next[0],
                b_a_next[1],
                b_a_next[2],
            ],
        )
        .expect("state should be 15x1")
    }

    fn jacobian_x(x: &Matrix<f64>, u: &Matrix<f64>, w: &Matrix<f64>) -> Matrix<f64> {
        let theta = StateGTEstimate0::theta(x);
        let b_a = StateGTEstimate0::b_a(x);

        let dt = u[0];
        let acc_m = Imu0::acc(u);
        let n_a = ImuNoise::n_a(w);

        let r = so3_exp(&theta);
        let acc = acc_m.sub(&b_a).expect("acc subtraction should succeed");
        let acc = acc.sub(&n_a).expect("acc subtraction should succeed");

        let mut f = Matrix::zeros(15, 15);

        set_block(&mut f, 0, 0, &Matrix::eye(3));
        set_block(
            &mut f,
            0,
            3,
            &Matrix::eye(3).scale(dt).expect("scale should succeed"),
        );

        set_block(&mut f, 3, 3, &Matrix::eye(3));
        set_block(
            &mut f,
            3,
            6,
            &r.matmul(&skew(&acc))
                .expect("3x3 product should succeed")
                .scale(-dt)
                .expect("scale should succeed"),
        );
        set_block(&mut f, 3, 12, &r.scale(-dt).expect("scale should succeed"));

        set_block(&mut f, 6, 6, &Matrix::eye(3));
        set_block(
            &mut f,
            6,
            9,
            &Matrix::eye(3).scale(-dt).expect("scale should succeed"),
        );

        set_block(&mut f, 9, 9, &Matrix::eye(3));
        set_block(&mut f, 12, 12, &Matrix::eye(3));

        f
    }

    fn jacobian_w(x: &Matrix<f64>, u: &Matrix<f64>, _w: &Matrix<f64>) -> Matrix<f64> {
        let theta = StateGTEstimate0::theta(x);
        let dt = u[0];
        let r = so3_exp(&theta);

        let mut fw = Matrix::zeros(15, 12);

        set_block(&mut fw, 3, 3, &r.scale(-dt).expect("scale should succeed"));
        set_block(
            &mut fw,
            6,
            0,
            &Matrix::eye(3).scale(-dt).expect("scale should succeed"),
        );
        set_block(
            &mut fw,
            9,
            6,
            &Matrix::eye(3).scale(dt).expect("scale should succeed"),
        );
        set_block(
            &mut fw,
            12,
            9,
            &Matrix::eye(3).scale(dt).expect("scale should succeed"),
        );

        fw
    }
}

fn main() {
    let _ = Measurement;
}
