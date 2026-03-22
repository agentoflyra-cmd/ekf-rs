# EuRoC MAV Vicon Room 1 真实观测验证方案

## 目标

使用 `EuRoC MAV Vicon Room 1` 数据集，直接验证当前仓库在真实传感器输入下的 EKF 能力：

- `predict` 使用真实 IMU
- `update` 使用真实视觉观测
- ground truth 只用于离线评估，不作为滤波器观测输入

当前仓库已经具备 EKF 主链路和广义噪声接口，相关入口见：

- [`src/filter/ekf.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/filter/ekf.rs)
- [`src/filter/generalized_ekf.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/filter/generalized_ekf.rs)
- [`src/model/motion.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/model/motion.rs)
- [`src/model/measurement.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/model/measurement.rs)
- [`src/state.rs`](/home/lyra/git_project/rust-ai-learning/ekf/src/state.rs)

结论先说清楚：`GT 伪观测` 对这个仓库不是必需路线。既然目标就是验证真实数据下的系统能力，那就应当直接用真实 IMU 和真实视觉观测，把 GT 降级成评估真值。

## 为什么不把 GT 当伪观测

把 GT 塞进滤波器，能验证的只有：

- 代码链路能不能跑
- 维度和矩阵有没有明显错误
- 某些数值问题会不会直接爆掉

但它不能回答你真正关心的问题：

- 真实视觉观测是否足够稳定
- 时间同步是否正确
- 前端观测噪声对 EKF 的影响如何
- 观测中断时系统会不会发散

更实际地说，GT 伪观测会掩盖两类关键问题：

- 前端观测质量问题
- 真实 `R` 建模问题

所以这份方案改成：

- GT 只做评估
- 真实观测直接进滤波器

## 数据输入

对当前仓库，建议直接使用下面这些原始数据：

- `mav0/imu0/data.csv`
- `mav0/cam0/data.csv`
- `mav0/cam1/data.csv`
- `mav0/cam0/data/*.png`
- `mav0/cam1/data/*.png`
- `mav0/imu0/sensor.yaml`
- `mav0/cam0/sensor.yaml`
- `mav0/cam1/sensor.yaml`

只用于评估的数据：

- `mav0/state_groundtruth_estimate0/data.csv`

建议把内部数据结构分成四类：

1. `ImuSample`
2. `StereoFrame`
3. `VisualObservation`
4. `GroundTruthSample`

其中 `GroundTruthSample` 不进入 `update`，只在评估模块里使用。

## 总体架构

建议拆成 4 层，避免把数据解析、视觉前端、滤波和评估耦合在一起：

1. 数据层
   读取 EuRoC CSV、图像和 YAML。
2. 前端层
   从双目图像生成真实视觉观测。
3. 滤波层
   用 IMU 传播、用视觉观测更新。
4. 评估层
   用 GT 计算 ATE、RPE、NIS 和数值健康指标。

建议新增 example：

- `examples/euroc_vr1_real_obs.rs`

如果前端代码比较多，也可以后续再抽：

- `src/dataset/euroc.rs`
- `src/eval/metrics.rs`

## 运动模型

建议直接按 15 维误差状态 IMU-EKF 设计。

名义状态：

```text
x = [p, v, q, b_g, b_a]
```

其中：

- `p ∈ R^3`：位置
- `v ∈ R^3`：速度
- `q`：姿态四元数
- `b_g ∈ R^3`：gyro bias
- `b_a ∈ R^3`：acc bias

误差状态：

```text
δx = [δp, δv, δθ, δb_g, δb_a]
```

总维度为 `15`。

### 连续时间模型

设 IMU 测量为：

```text
ω_m = ω + b_g + n_g
a_m = a + b_a + n_a
```

则名义状态传播为：

```text
p_dot = v
v_dot = R(q) (a_m - b_a) + g
q_dot = 1/2 * Omega(ω_m - b_g) * q
b_g_dot = 0
b_a_dot = 0
```

过程噪声通过误差状态进入：

```text
w = [n_g, n_a, n_wg, n_wa]
```

### 离散传播

对相邻 IMU 时刻 `k -> k+1`，令：

```text
dt = (timestamp_k - timestamp_{k-1}) * 1e-9
```

则一阶离散传播可写为：

```text
ω_hat_k = ω_m_k - b_g_k
a_hat_k = a_m_k - b_a_k

p_{k+1} = p_k + v_k dt + 1/2 (R_k a_hat_k + g) dt^2
v_{k+1} = v_k + (R_k a_hat_k + g) dt
q_{k+1} = q_k ⊗ Exp(ω_hat_k dt)
b_{g,k+1} = b_{g,k}
b_{a,k+1} = b_{a,k}
```

这里：

- `R_k = R(q_k)`
- `Exp(·)` 是 `SO(3)` 指数映射

## 传播 Jacobian

当前仓库更适合走广义形式：

```text
x_{k+1} = f(x_k, u_k, w_k)
P_{k+1} = F_{x,k} P_k F_{x,k}^T + F_{w,k} Q_k F_{w,k}^T
```

其中：

- `u_k = [dt, ω_m, a_m]`
- `w_k` 为过程噪声线性化点，通常取零
- `F_x = ∂f/∂δx`
- `F_w = ∂f/∂w`

若采用 15 维误差状态，一阶近似块结构可写为：

```text
F_x =
[ I   dt I    0                0         0
  0    I    -R[a_hat]x dt      0      -R dt
  0    0     I-[ω_hat]x dt   -I dt      0
  0    0       0               I         0
  0    0       0               0         I ]
```

其中：

- `[·]x` 表示反对称矩阵
- `R = R(q_k)`
- `a_hat = a_m - b_a`
- `ω_hat = ω_m - b_g`

对应过程噪声 Jacobian 可取：

```text
F_w =
[ 0      0      0      0
  0    -R dt    0      0
 -I dt   0    I dt     0
  0      0     I dt    0
  0      0      0    I dt ]
```

如果你的 `w` 排列为：

```text
w = [n_g, n_a, n_wg, n_wa]
```

那 `F_w` 的列块维度分别是 `3, 3, 3, 3`。

## 真实观测定义

关键问题不是“要不要真实观测”，而是“真实观测在这个仓库里先定义成什么”。

不建议一开始直接做特征级 EKF 更新。对当前仓库，更现实的路线是：

1. 双目前端先输出位姿或相对位姿
2. EKF 将其作为位置/位姿观测吸收

也就是说，先做：

- `IMU propagation`
- `stereo VO front-end`
- `pose update`

而不是一开始就做：

- 特征级观测
- 地标状态扩维
- MSCKF / 滑窗

### 推荐观测类型

建议按下面顺序推进。

#### 路线 A：位姿观测

由双目前端输出：

```text
z_k = [p_vo, q_vo]
```

观测模型：

```text
h(x_k) = [p_k, q_k]
```

这条路线最直接，适合先把系统跑通。

#### 路线 B：位置观测

如果前端暂时只能稳定给出平移：

```text
z_k = p_vo
h(x_k) = p_k
```

这比位姿观测弱，但实现最简单。

#### 路线 C：相对位姿观测

如果前端给的是相邻关键帧之间的相对运动：

```text
z_k = delta T_{k-1,k}
```

这种方式可以做，但会把观测模型复杂度明显抬高，不建议作为第一步。

## 时间同步

真实观测路线里，时间同步比 GT 伪观测更重要。

建议规则：

- IMU 是主时钟，逐条执行 `predict`
- 图像时间戳触发视觉前端
- 当前端产出观测时，在对应图像时刻执行 `update`
- 若图像时刻落在两个 IMU 之间，需要先推进到该图像时刻

GT 只在评估时做对齐，不进入滤波器。

## 噪声建模建议

直接上真实观测后，`Q` 和 `R` 就必须按传感器意义建。

### 过程噪声 `Q`

建议至少包含：

- gyro measurement noise
- acc measurement noise
- gyro bias random walk
- acc bias random walk

### 观测噪声 `R`

如果观测是外部视觉前端输出的位姿：

- 平移噪声
- 旋转噪声

如果观测是位置：

- 位置噪声

不要把 `R` 设得过小。真实视觉观测一定有前端误差、匹配退化和短时跳变。

## 指标

既然直接做真实观测，指标就要围绕“真实系统表现”来定。

建议至少输出：

- ATE RMSE
- RPE
- 位置 RMSE
- 姿态误差 RMSE
- 每步创新范数 `||y||`
- NIS
- update 成功率
- 连续无更新时长
- `P` 最小对角线
- `S` 求解失败次数
- `NaN/Inf` 检查

其中最关键的是：

- ATE
- RPE
- NIS
- update 成功率

## 最小可行实现

建议按下面顺序推进：

1. 读取 `imu0/data.csv`
2. 读取 `cam0/cam1` 时间戳和图像
3. 建立双目视觉前端，先输出位姿或位置观测
4. 用 IMU 做 `predict`
5. 用视觉观测做 `update`
6. 用 GT 只做离线评估

建议最小产物为：

1. `examples/euroc_vr1_real_obs.rs`
2. 一份结果 CSV
3. 一份评估 markdown

结果 CSV 建议包含：

- `timestamp`
- `px, py, pz`
- `qx, qy, qz, qw`
- `gt_px, gt_py, gt_pz`
- `residual_norm`
- `nis`
- `cov_min_diag`
- `update_ok`

## 完成标准

满足下面条件后，可以认为这条真实观测验证链路建立完成：

1. 使用真实 IMU 和真实视觉观测，Vicon Room 1 能完整跑通。
2. 当前端短时丢失观测时，系统不会立刻数值崩溃。
3. ATE / RPE 可稳定复现。
4. 失败时能区分是前端问题、同步问题还是滤波器问题。

## 不建议的做法

以下做法不建议再作为主线：

- 把 GT 当滤波器观测输入
- 用 GT 先把系统“调顺”再接真实观测
- 没有前端质量评估就直接调 `R`
- 一开始就上特征级 EKF 更新

## 结论

对当前仓库，更合理的 EuRoC 验证路线是：

- 直接使用真实 IMU 和真实视觉观测
- 把 GT 仅作为离线评估数据
- 用 15 维误差状态做 IMU 传播
- 先把视觉前端输出定义成位姿或位置观测

这条路线更接近你真正要验证的系统能力，也不会被 GT 伪观测掩盖真实问题。
