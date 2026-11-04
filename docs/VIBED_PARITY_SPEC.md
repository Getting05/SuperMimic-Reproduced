# Vibed-Parity FSQ：直接替换式对齐开发规格（供 Vibecoding 执行）

> 源分支：`Getting05/SuperMimic-Reproduced:dev/FSQ`  
> 新分支：`dev/vibed-parity-fsq`  
> 目标代码参照：`zita-ch/supermimic_vibed:main`  
> 对齐原则：**直接将算法数据接口、网络、训练及评测实现改成 Vibed 的定义，不实现两套算法的配置切换。保留 P2、FSQ 和原项目工程基础。**  
> 重要：本文件是一份实施任务说明，不代表仓库代码已被修改。

## 0. 执行任务与不可更改约束

1. 从远程最新 `origin/dev/FSQ` 创建 `dev/vibed-parity-fsq`；先记录源 commit SHA，检查干净工作区，不覆盖现有未提交改动。分阶段提交；最终推送新分支。
2. **直接替换现有算法语义**。禁止新增 `legacy/vibed`、`parity_mode`、`wm_mode`、`goal_mode`、`obs_mode`、`use_vibed_xxx` 等分支或兼容路径。统一执行 Vibed 语义。
3. 保留 Astro P2 V3、29 DoF 的物理资产、MJCF、关节顺序、默认动作尺度及控制周期 50 Hz（物理步长 0.005 s × 4）。保留 `Policy` 的 **40D FSQ、每维 9 levels**，不能变为 Vibed 的连续 32D latent。策略 MLP 宽度沿用 `(2048,1024,512,256)`；WM 隐藏宽度 `(2048,2048,2048)`；Critic `(256,128)`。
4. **接触求解设置完全不改**：`nconmax=35`、`njmax=250`、`MujocoCfg(iterations=10, ls_iterations=20)`；不改变 integrator、contact backend、collision geom、foot collision、接触参数等与 Solver 相关的配置。
5. **保留现有配置入口和已有参数**：例如 `--env.robot`、`--policy.fsq`、`--env.no-p2-advanced-actuators`、`--env.sampling-mode`、SGS 全套参数、CPU/GPU motion-library、DDP、多卡启动、W&B、resume、checkpoints、日志。可以为匹配 Vibed 更新原参数默认值或重新定义其训练计数含义；**不要为新旧算法额外增加配置开关**。已被新算法淘汰但需要保留 schema 的字段应注明 deprecated，并对非默认使用给出明确告警，不要暗中执行旧算法。
6. 当前 `dev/FSQ` 已经实现标准可选 SGS，`p2_advanced_actuators` 默认 `False`，还有 CPU/GPU 动作库选择；不要重复开发，也不要在本分支删除 SGS。
7. 对 G1 支持进行基本兼容测试；优先保证 P2 训练。不同机器人必须保留不同的 MJCF、关节控制和允许的 body mapping；“Vibed 对齐”指算法语义与张量布局对齐，不是假装 P2 与 G1 具有相同物理参数。
8. 新状态/模型/归一化接口将破坏旧 checkpoint 的形状及语义兼容性；**严禁静默加载旧 checkpoint 继续训练**。新 checkpoint 包含明确格式版本；在读取旧版时给出错误提示。只要求新分支 checkpoint 能正确 resume；如需要迁移旧权重，应另行开发独立、显式的转换脚本，不能埋在正常训练路径。
9. 不要对仓库做与任务无关的大规模重构。维持原有模块边界，避免复制 `supermimic_vibed` 的整套 IsaacLab 场景或另建平行实现。

## 1. 全局张量契约：唯一真实定义

### 1.1 Simulator 原始数据与面向模型的状态

- **MuJoCo 物理内部不动**：`qpos=(root_pos_world[3], root_quat_wxyz[4], joint_q[29])` 为 36D；`qvel=(root_lin_vel_world[3], root_ang_vel_body[3], joint_qd[29])` 为 35D。
- **面向 Replay / WM 的 `S_t` 改为 Vibed 的 71D 顺序**：

```
S_t = [
  root_pos_world[3],       # 0:3
  root_quat_wxyz[4],       # 3:7
  root_lin_vel_world[3],   # 7:10
  root_ang_vel_world[3],   # 10:13
  joint_pos[29],          # 13:42
  joint_vel[29],          # 42:71
] 
```

- P2 pelvis = floating-base root。所有模型/Replay 侧与参考 motion 库按此 71D 顺序统一。`root_ang_vel_world = R(q) @ root_ang_vel_body`；写回 MuJoCo 时做逆变换。位置减 `scene.env_origins`，与 MotionLibrary reference 同一坐标系。
- `StateIO.read/write` 实施明确映射并提供往返测试。`world_model/state_spec.py`、`world_model/heads.py`、`data/recorder.py`、`imagination/*` 中所有旧版 `s[...,7:36]` 等切片同步修改。**禁止仅改注释或局部 reshape。**
- 分别检查 root body/link 的 pose 与 velocity，在 MuJoCo 与 GPU Forward Kinematics 比对。仿真器底层的 `qpos | qvel` 仍是 71 个数，但不代表其排列等于面向模型的 71D `S_t`。

### 1.2 参考动作状态

- Motion library 的每帧 reference 统一映射为相同的 71D `S_ref`：`[p3, quat4, v_world3, w_world3, q29, qd29]`。
- P2 NPZ 转换（PyRoki retarget -> 50Hz）保留；读取时按名称校验 29 个关节顺序。FK 继续来自同一 P2 MJCF；不可直接使用 G1 的 FK 常数。
- 参考旋转 quaternion 都约定 wxyz；转换为旋转矩阵时使用同一工具函数。需对比静态姿态与带速度 FK 的 14 个 tracked bodies。

### 1.3 Policy Observation：单帧严格 99D

```
o_t = [
  joint_pos - default_joint_pos - encoder_bias,           # 29D
  joint_vel,                                              # 29D
  R_robot^T @ root_ang_vel_world,                         # 3D
  R_robot^T @ [0,0,-1],                                  # 3D
  a_prev_executed,                                        # 29D
  rot6d(R_robot^T @ R_ref_pelvis_at_f),                   # 6D
] = 99D
```

- 历史 10 帧，形状 `(B,10,99)`，**按时间优先 time-major** flatten 为 `(B,990)`，与 Vibed 一致。不要继续使用 Reproduced 的 term-major 930D 布局。
- 6D relative orientation 来自**当前机器人 pelvis 与当前 reference pelvis**，不是 world-yaw、不是 torso/waist anchor。平移的 root-tracking feedback 不额外注入默认 99D（Vibed 本身没有此反馈）。
- Simulator Observation 与 WM Imagination 通过同一套纯函数/共享计算验证结果一致；可在 manager 中包一层新 observation function，但不能让仿真器采用 `joint_pos_rel(biased=True)` 的旧语义而 imagination 使用 `-xi_bias`。这是重点检查项目。
- 噪声按 Vibed 的单帧 99D 项：`joint_pos±0.01`、`joint_vel±0.5`、`ang_vel±0.2`、`projected_gravity±0.05`、`a_prev 0`、`relative_root_orientation±0.05`，分量独立均匀噪声。
- Episode reset 时历史在 `(10,99)` 上用第一帧 Observation 重复填充，不能混进上一 episode 的历史。
- 保留原有 `anchor_obs` 配置入口（属于旧代码已有开关），默认关闭；如果继续允许开启，它只能作为额外的 9D 研究输入（并注明不再属于严格 Vibed parity）。**不得因此产生两套 93D/99D 主 Observation 的分支实现**。

### 1.4 Future Reference：每帧 73D、10 帧 730D

定义 `f` 为当前参考帧，`H_f = yaw_mat(R_ref[f])`。Goal 使用 `f+1,...,f+10`：

```
g_{f+k} = [
  H_f^T @ (p_ref[f+k] - p_ref[f]),        # 3D：相对当前reference heading位移
  rot6d(H_f^T @ R_ref[f+k]),              # 6D：相对当前reference heading姿态
  R_ref[f+k]^T @ v_world[f+k],            # 3D：未来root线速度（body frame）
  R_ref[f+k]^T @ w_world[f+k],            # 3D：未来root角速度（body frame）
  joint_q[f+k],                           # 29D：绝对关节角（不减 q0）
  0.1 * joint_qd[f+k],                    # 29D：速度缩放
] = 73D;
G_f = concat(g_{f+1},...,g_{f+10}) = 730D
```

- 未来索引越过 clip 结尾时，姿态/关节位置 clamp 到最后有效帧；**仅超出原始记录时**将速度、关节速度设为零。落在真实最后一帧时保留原始数据速度。
- 需要同步模拟器命令、Policy forward、WM Imagination、Critic、Encoder 和评估。
- 默认 `goal_dim=730`；若原有 `anchor_obs=True`，只允许在 730 后附加旧研究用 9D，默认训练基线不使用。
- 在 3 帧手工参考轨迹上对照 Vibed 的 `MotionLib.goal()`，用数值测试锁定字段顺序、旋转约定、zero-padding 行为。

## 2. Policy、归一化、激活函数、Critic

### 2.1 Policy + FSQ

- Goal Encoder：`730 -> [2048,1024,512,256] -> 40`，隐藏层 **ELU**（按 Vibed）；尾部 `tanh` 后沿用 Reproduced 原有 FSQ STE 量化，40 个元素各取九个 centered levels `{-4,-3,...,4}`；保留原有 40D 接口，**不可改为连续 32D**。
- Policy Decoder：`(40 + 10*99) = 1030 -> [2048,1024,512,256] -> 29`，隐藏层 **ELU**，尾层小增益初始化沿用 Vibed 的 `weight *= 0.01; bias=0` 语义，不添加输出 `tanh` 或隐藏的新缩放。
- `Policy.obs_norm = RunningNorm(99)`：收集真实仿真数据时按单帧 99D 更新；forward 时在 `(B,10,99)` 上逐帧应用相同 normalizer 后 **time-major flatten**。
- `Policy.goal_norm = RunningNorm(730)`：收集时按完整 730D goal 更新；在 Encoder 输入前应用。
- **移除旧固定 `obs_term_scale`/`goal_term_scale` 在前向计算中的作用**，不要固定尺度与 RunningNorm 叠加两次。旧配置字段可保留作 deprecated schema，明确日志告警。
- Normalizer 在 evaluation 不更新；Imagination 使用 collect 得到的冻结统计，不通过 imagined states 更新。
- `RunningNorm` 对齐 Vibed（累计 `mean`、`var`、`count`，默认 `max_count=2e8`，`eps=1e-2`），将其状态纳入 checkpoint。

### 2.2 Critic：914D + ELU

```
critic_features = [
  invariant_state[71],
  ref_joint_q - robot_joint_q[29],
  reference_body_pos_reanchored - robot_body_pos, 转到robot heading并flatten[14*3=42],
  H_robot^T @ (ref_root_pos - robot_root_pos)[3],
  rot6d(R_robot^T @ R_ref_root)[6],
  normalize_xi(xi)[33],
] = 184D
Critic_input = concat(critic_features184, goal730) = 914D
Critic: 914 -> 256 -> 128 -> 1，隐藏层ELU + RunningNorm(914)
```

- Ref body 的 reanchor 规则与 Vibed 的 pelvis 跟踪 reward 完全相同，不可混用腰部 anchor；`14` 个 tracked body 中 P2 不存在的 G1 专有 torso 命名继续使用**当前已可用的 P2 body mapping**，但 pelvis 始终是 anchor/root。
- Critic 可以使用特权 reference/body errors；Actor 的默认 99D 不额外注入这些全局误差。
- `Critic.norm` 从 imagined rollout 的 `H+1` 个 state-feature 统计更新，按 Vibed 时机更新并参与 checkpoint。

## 3. World Model：133D 输入、70D 直接状态增量

### 3.1 invariant state 与输入

```
invariant_features(S_t) = [
  root_height[1],
  rot6d(H_robot^T @ R_robot)[6],
  R_robot^T @ root_lin_vel_world[3],
  R_robot^T @ root_ang_vel_world[3],
  joint_pos[29],
  joint_vel[29],
] = 71D

WM_input = concat(invariant71, executed_action29, normalize_xi33) = 133D
```

- `normalize_xi(xi)`：摩擦 `(mu-0.75)/0.45`、29D encoder bias `/0.01`、3D COM offset `/0.05`。`xi` 必须反映真正施加到 Simulator 的参数；不允许一边有传感器偏置、一边 action target 不按 Vibed 处理（详见 §5）。
- 不继续使用旧固定 `StateSpec.in_scale` 作为 WM 网络输入归一化，替换为 `WM.in_norm = RunningNorm(133)`，真实 collect 时更新。

### 3.2 Dynamics 网络与输出

- WM 主干：`133 -> 2048 -> 2048 -> 2048 -> 70`；每个隐藏层 **Linear + LayerNorm + SiLU**，末层 Linear 零初始化。
- **删除旧显式恒速积分先验的 active forward**：不再有 `p_new=p+dt*v_new+residual` 或 `q_new=q+dt*qd+residual`；完全按 Vibed 70D 预测的 frame-local state increments 积分。
- 70D 预测顺序与语义必须是：

```
delta70 = [
  dp_heading[3],      # (H_robot^T @ (p_next - p))
  dtheta_body[3],     # log(R_robot^T @ R_next)
  dv_body[3],        # R_robot^T @ (v_next - v)
  domega_body[3],    # R_robot^T @ (w_next_world - w_world)
  dq[29],            # q_next - q
  dqd[29],           # qd_next - qd
]
```

积分：`p_next=p + H_robot @ dp`，`R_next=R_robot @ Exp(dtheta)`，`v_next=v + R_robot @ dv`，`w_next=w + R_robot @ domega`，`q_next=q+dq`，`qd_next=qd+dqd`。模型端 state 仍 flatten 为 71D Vibed 布局。
- `WM.out_norm = RunningNorm(70,eps=1e-4)`，从真实相邻状态提取的 `delta70` 更新；神经网络输出为 normalized delta，实际 `delta=pred * out_norm.std() + out_norm.mean`。
- **仅 MLP matmul 使用 bf16 autocast（如硬件支持）**；状态更新、旋转、FK、reward、loss 保持 float32，不能让 bf16 污染 state integration。

### 3.3 WM 损失与终止

- 与 Vibed 一致，取 k=8 的自回归预测轨迹；每个预测时刻与真实 `s1` 比较 **70D frame-local error**。
- 归一化误差分组最小标准差 `ERR_FLOOR` 分别：`dp3:0.003`、`dtheta3:0.005`、`dv3:0.02`、`dw3:0.05`、`dq29:0.003`、`dqd29:0.05`。
- 对每步先对 70 维求 mean(squared_error)，再乘 Valid Mask；沿 k 步求和，按 batch size 求 mean，**不要把用于优化的 WM loss 无意再除以 8**（Vibed 的日志显示项才除以 8）。
- 不再训练 Reproduced 的独立 learned termination MLP/BCE head：对齐 Vibed 用纯解析式终止条件（Simulator hard termination、Imagination soft termination）。保留旧 term-related config 字段但标为废弃、不参与计算。
- Imagination 解析式 soft termination 以 pelvis 高度误差阈值 0.25 m、倾斜误差阈值 0.8 为标准；softness `tz=0.02`、`tt=0.04`，按 Vibed 的 sigmoid survival 公式计算。

## 4. Reward 与终止：全部以 pelvis 为 root/anchor

- 将 **`anchor_body_name='pelvis'`** 作为 P2 和 G1 唯一 active 跟踪 anchor；当前 P2 的 `waist_roll_link` 可以继续作为 tracked body 和 COM-offset 物理位置，但**不得再作为 reward anchor**。
- 6 个 tracking terms 与 Vibed 逐项相同：
  1. pelvis 全球位置 `0.5 * exp(-e_root_pos / 0.3^2)`；
  2. pelvis 全球姿态 `0.5 * exp(-e_root_ori / 0.4^2)`；
  3. 相对 pelvis 的 14-body 位置 `1.0 * exp(-mean(e_body_pos) / 0.3^2)`；
  4. 相对 pelvis 的 14-body 姿态 `1.0 * exp(-mean(e_body_ori) / 0.4^2)`；
  5. 14-body world linear velocity `1.0 * exp(-mean(e_lin) / 1.0^2)`；
  6. 14-body world angular velocity `1.0 * exp(-mean(e_ang) / 3.14^2)`。
- 惩罚项：`action_rate_l2 weight=-0.1`；`joint_limit weight=-2.0`；总 reward 统一乘 `dt=0.02`，每步理论最大值 `0.1`。
- Reference bodies 按 Vibed 重新锚定：参考 bodies 的 **XY 与 yaw** 对齐到当前机器人 pelvis，参考 body 的 Z/roll/pitch 保持参考定义。逐项比较 Simulator manager 的奖励与 WM analytic reward，同一 `S_{t+1}, f_{t+1}, a_t, a_{t-1}` 应数值一致。
- hard termination 按 **pelvis** 的 ref height difference `>0.25 m`、projected-gravity z difference `>0.8`；自然 clip-end / 10 秒 time-out 仍为 truncation，不能当训练 failure；Imagination 使用上节软终止。
- **时间对齐强制测试**：动作 `a_t` 从 `f_t` condition 产生；物理推进得到 `S_{t+1}`；奖励及 termination 与 `f_{t+1}` 比较；新观测中的 relative reference orientation 使用 `f_{t+1}`；下一个目标用 `goal(f_{t+1})`。不能继续在 Imagine 内用旧 `f_t` 参考计算下一状态奖励。

## 5. Action、Encoder Bias 与环境噪声注入

- **对齐 Vibed 的噪声注入位置：在 Collect/Trainer 中加到输出动作后、调用 `env.step()` 之前**：

```
a_policy = policy(o_hist, goal)
a_exec = clamp(a_policy + sigma * randn_like(a_policy), -20, 20)
env.step(a_exec)
```

- 保留旧 `--env.action-noise-std` 配置名，默认 sigma=0.1，但 `NoisyJointPositionAction` 内部**不得再次加噪**，即把原内部 noise 功能关闭/移除 active 逻辑；Replay、Simulator、`last_action` Observation、Reward action-rate 统一使用**执行后 a_exec**，像 Vibed 一样。
- WM Replay 保存 `a_exec`；Actor Imagination 没有 Gaussian execution noise（仅有规定的 observation noise）；测试 `action_noise_std=0` 与 `=0.1`，验证噪声精确加一次且惩罚项、观测、Replay 同一含义。
- **Encoder Bias 语义必须成对调整**：Vibed 的 `q_obs = q - q0 - bias`，同时执行 position target 为 `q_target = q0 + bias + alpha*a_exec`。当前 mjlab 的 `encoder_bias` 事件/`joint_pos_rel(biased=True)` 可能不是这种符号及 actuator offset 组合；不要只改 analytic observation。需要在 Simulator manager 中准确实现并加 assertion，避免重复偏置或漏偏置。
- Domain randomization 配比保持 Vibed：COM offset 在 startup 采样、friction `U[0.3,1.2]` 在 startup 采样（不同于 Reproduced 以前的 reset 重采样）、encoder_bias 每个 episode reset；评测 friction=1.0、全噪声关闭。P2 COM 的物理作用 body 使用真实可用的 waist/torso mapping，并确认 `SimContext` 读取的是同一处 body 的实际偏移。
- **保留 P2 Advanced Actuator 配置及默认关闭**；如果旧开关打开，物理系统包含额外 command delay 内部状态，与不包含 delay 状态的 71D WM 不完全满足 Markov 条件。不要为了对齐 Vibed 删除开关，但对该配置给出显式实验风险说明；训练 parity 基线使用 standard PD。

## 6. Replay Buffer：直接改为 Vibed 的时序与字段

- 保留 GPU 环形存储能力，但修改为 **time-major `[T, N, ...]`**，`T=1000`、`N=4096/rank`。字段定义严格为：

```
s:      (T,N,71) float32  # step前
s1:     (T,N,71) float32  # step后、reset前
 a:     (T,N,29) float32  # 执行动作（已加噪）
a_prev: (T,N,29) float32  # 上一步执行动作
xi:     (T,N,33) float32
r:      (T,N) float32
f:      (T,N) int64       # 当前reference frame f_t
ep_step:(T,N) int32
done:   (T,N) bool        # terminated OR truncated，供窗口mask
```

- 每次完整的 vectorized `env.step` **仅向 buffer 写入一行所有 N 个环境的 transition**；不再像 Reproduced 旧 Buffer 那样按 env 保存独立 ring、追加 terminal row、单独存 clean/executed 两个版本的动作或显式 episode id。
- **这是改动难点**：mjlab 自带自动 reset。必须在 `record_pre_reset(env_ids)` 捕获 **终止前的真实 `s_{t+1}`**，而在 `record_post_step()` 合并已终止和未终止环境的数据，才向 time-major ReplayBuffer 写完整一行。不能在 `env.step` 返回后直接读状态当作终止 transition 的 `s1`，因为那可能已经是下一 episode 的 reset state。
- 追踪 `s_t`、`a_exec`、`a_prev`、`xi`、`f_t`、`ep_step`、`reward`、`done`；对 reset env 保存原 episode pending transition，reset 后写下一帧的状态作为下次 `s_t`；确保 recorder hooks 不会被重复调用或多写。
- `sample_windows(B,k)` 与 Vibed 一致：随机 age+env，抽取 k 连续 vectorized 行，沿 done 生成 mask；`s1` 使用最后一步真实 next-state；若中间终止，后面误差无效。mask 的终止语义要与 `done` 一致。
- `sample_starts(B,hist)` 与 Vibed 一致：通过 `ep_step` 和 buffer age 裁剪 10 帧历史，越界回填该 episode 第一帧；不允许跨 episode；构造 `a_prev`、`xi`、`f` 的历史。
- 采样时保证 **完整的 32-step Imagination reference**：`f + horizon <= clip_end[clip_id[f]]`。采用批量 oversampling 过滤，不足时重采，若数据集根本不存在合法长度，应显式抛错而非无限循环。不能在 BPTT 末尾无限重复最后一帧参考。
- 检查 Replay Buffer 内存：1000×4096 个 transition，各字段 float32，总量约 4 GB/GPU（未计梯度/临时采样），不能导致 P2 大库或多卡 OOM。CPU Motion Library 模式仍需正常工作。
- 已有 SGS 只控制 simulator episode reset 位置，不改变 ReplayBuffer 的随机 transition/window 采样语义；保留其 rank-local checkpoint、统计及 resume。

## 7. 训练循环：严格采用 Vibed 的更新计数定义

- Reproduced 过去的 `schedule.total_iterations=320000` 是 **vectorized env steps**；Vibed 的 `iterations=320000` 是 **WM 更新轮次**，不是同一个训练量。新分支在保留 `total_iterations` 配置名的同时，**统一把它解释为 WM 更新轮次**。
- 保留旧 `updates_per_env_step=0.4`（UTD），由 update loop 的 accumulator 每一个 WM 更新轮次收集平均 `1/0.4=2.5` 次 vectorized env steps。记录 `env_step_count`、`wm_update_count` 和 `sim_transitions = env_step_count * num_envs * world_size`（实际统计优先）。
- Buffer warmup：保留已有 `schedule.warmup_iterations` 配置字段，但以 **64 vectorized env steps** 为新的默认值（更新注释为 buffer warmup env steps）。完成最初 collect 才进入更新循环。
- **WM-only warmup：在训练起始前 500 个 WM 更新轮次，仅执行 collect + 10 个 WM minibatches；Policy / Critic 暂不更新。** 不新增 `wm_only` 选择开关；在核心 trainer 中使用单一明确常量 `WM_ONLY_UPDATES=500`，累计 WM 更新数达到 500 后开始 Policy/ Critic 更新。继续每轮 10 个 WM minibatches、2048 条 8-step windows、一次 4096×32 imagination、Critic 5 epochs×4 minibatches。
- 学习率从 WM update iteration 计数，按 250000 更新衰减到原值 0.03 倍，其后保持常数。多 GPU 的 `sqrt(world_size)` 学习率缩放沿用原工程定义（作为特定于 Reproduced 的多卡实现）。如果做对照实验，记录该缩放及 world size。
- Policy objective、survival accumulation、GAE(λ=.95)、γ=.99、H=32、critic bootstrap、梯度剪裁按 Vibed 对齐：Policy norm 1、value 0.1，Critic norm 1，World Model norm 1；对 Policy 梯度非有限值要跳过 optimizer step 并记录。
- `torch.compile` 缓存静态形状，保留已有合理的编译/日志/多卡功能；修改输出后对 Imagination 和 WM compile 重新 profile。各 rank 的 running normalization 统计必须同步或以数学等价的全局统计合并，不能仅广播权重而让 `obs_norm/goal_norm/wm_norm` 在不同 GPU 漂移；不能引入 DDP collective 死锁。
- Checkpoint 包括：`wm / policy / critic`、全部 RunningNorm buffers、optimizer、`wm_update_count`、`env_step_count`、训练相位、sampler 状态（包括 SGS rank-local）、RNG 状态、配置快照、格式版本。恢复时应继续正确阶段和学习率位置。旧 `dev/FSQ` checkpoint 必须拒绝直接 resume。

## 8. Evaluation：以 Vibed 官方算法语义为主

- 每个 clip 从第一帧开始，一次 deterministic rollout；全部噪声和 DR 关闭、foot friction=1.0；物理仿真 P2 与当前分支保持一致；失败为训练用的 hard termination（pelvis height/tilt），正常 Clip End 不是失败。
- **主指标统一使用完整 clip 长度分母**：`sum_executed_rewards / [(reference_clip_len-1) * max_step_reward]`。Reproduced 旧评估按实际执行步数作分母，可以作为旁支诊断列输出，但不应再是主 KPI。
- 同时输出 `per_clip norm_return`、完整步长完成率、failed 数、pelvis world position/orientation error、body position/orientation error（分别提供 world 和 pelvis-reanchored 视角）、joint position/velocity error、训练/评估终止原因。按每 clip 平均值与按帧加权平均值分别汇总。
- `eval/benchmark.py`、`scripts/eval_benchmark.py`、`diag_drift.py` 等使用新 99D+730D、time-major history 与 World/Sim state adapter。不能发生评估仍按旧 680D goal 运行。

## 9. 建议逐文件实施（保留原工程模块，不建立第二套系统）

| 文件 | 必做内容 |
|---|---|
| `src/supermimic/envs/state_io.py` | 原生 MuJoCo `(qpos,qvel)` ↔ Vibed 71D `S` 的双向转换；world angular velocity、env origin |
| `src/supermimic/envs/mdp/motion_library.py` | Reference 71D 顺序、73D/frame goal、未来 `f+1..f+10`、clip-end padding |
| `src/supermimic/envs/mdp/observations.py` | 99D observation、pelvis 相对朝向；复用 shared pure functions |
| `src/supermimic/envs/tracking_env_cfg.py` | pelvis anchor、99D group、标准 PD 默认、friction startup、noise仅collect、Reward/Termination配置；contact不动 |
| `src/supermimic/envs/mdp/commands.py` | pelvis anchor，14 body mapping，SGS/return采样保留，准确 f/f+1，reference cache |
| `src/supermimic/envs/actions.py` | setpoint 包括 encoder_bias，取消内部额外动作噪声；确保 clean/executed 不混用 |
| `src/supermimic/envs/sim_context.py` | ξ33 与 simulator 实际物理摩擦、bias、COM 对齐 |
| `src/supermimic/world_model/state_spec.py` | Vibed 71D invariant、133D输入、70D直接增量定义、frame-local errors、转换校验 |
| `src/supermimic/world_model/dynamics.py` | 133-2048-2048-2048-70、LayerNorm+SiLU、zero final |
| `src/supermimic/world_model/world_model.py` | RunningNorm(133/70)、直接增量更新、8-step state loss、移除active learned termination head |
| `src/supermimic/world_model/wm_trainer.py` | Vibed 8-step masked-NSE/RunningNorm及日志、梯度规模一致 |
| `src/supermimic/world_model/heads.py` | 99D analytic obs、pelvis reward 与 hard/soft termination、Critic features统一 |
| `src/supermimic/policy/nets.py` | ELU、RunningNorm(99/730)、time-major 990、FSQ40、Decoder 1030、Critic914 |
| `src/supermimic/data/replay_buffer.py` | time-major 1000×N、Vibed字段、window mask、history回填 |
| `src/supermimic/data/recorder.py` | mjlab reset 前缓存 s1 + step 后整行写buffer（无terminal row） |
| `src/supermimic/imagination/compiled.py` | `f_t -> f_{t+1}` 与 reward/obs/soft-term 同步，无旧 termination logits |
| `src/supermimic/imagination/rollout.py` | 99D `(B,10,99)` 时序、730D goal、H32 合法起点、pelvis reference f+1 |
| `src/supermimic/imagination/policy_trainer.py` | sample_valid_starts、Vibed actor/critic 914D、RunningNorm timing与GAE |
| `src/supermimic/training/loop.py` | 在Trainer中一次加动作噪声、Vibed 更新轮次、buffer warmup64、WM-only500、resume与多卡处理 |
| `src/supermimic/config.py` | 原有参数保留、不加新旧模式开关；调整必要默认值和注释，废弃字段显式标注 |
| `src/supermimic/eval/benchmark.py` 与 `scripts/eval_benchmark.py` | Vibed 完整 clip return、99/730/71新接口、Root/body metrics、评估无噪声 |
| `tests/` | 新增/修改逐项契约/双实现数值对照/训练 smoke tests |

**注意**：如果实际文件结构或函数已变，以最新 `origin/dev/FSQ` 为准；先用代码搜索找出全部 `71/93/99/680/730/930/990/788/914` 硬编码和状态切片；不能仅根据上表逐文件生硬替换。

## 10. 逐阶段提交，禁止一次大改无法调试

1. `chore: branch baseline + contract tests`：固定源码 SHA、记录基线配置、添加契约测试，确认原始单卡训练可运行。
2. `refactor: unify simulator and replay state layout`：71D state 读写、FK、坐标系、reference 状态一致；通过 roundtrip tests。
3. `feat: align goal and proprio observation with Vibed`：73D future、99D current obs、10 帧 time-major、bias 验证。
4. `feat: align pelvis reward and timestep semantics`：pelvis reference、reanchor、hard/soft term、f+1 时间检查；contact不动。
5. `refactor: move action noise to collector`：单次噪声、setpoint/bias、a_prev/replay/reward一致。
6. `refactor: replace replay recorder and buffer with time-major transitions`：reset前s1、done-mask、history tests、SGS不回归。
7. `feat: align running normalization, policy activation and critic`：ELU、RunningNorm、FSQ接口、914D critic。
8. `feat: align world model dynamics and 8-step loss`：70D直接增量、LayerNorm+SiLU、WM input/output running stats、解析终止。
9. `feat: align imagination and training schedule`：合法32步起点、64步buffer warmup、500 WM-only、320k更新语义与DDP。
10. `test: unify parity evaluation and resume`：完整clip主指标、定量消融、checkpoint版本、多卡/SGS/regression tests。

每个 commit 必须有测试结果；如果修改中发现先前假设与 mjlab hooks 不符，应先写打印/最小复现测试验证 hooks 顺序再重构 Recorder，不能依靠猜测。

## 11. 验收测试（必须全部通过）

**张量/数学：**
- `StateIO` MuJoCo ↔ Vibed71D 往返、root_ang_vel body/world 正反变换、quaternion normalize；相机/场景 env origin 不混入运动学。
- `Observation` shape `(N,99)`，history `(N,10,99)`，flatten `(N,990)`，各维数值与 `analytic_observation` 一致；初始 history 重复首帧。
- `Future Goal` shape `(N,730)`、精确字段顺序、`f+1..f+10`、跨 clip end 的 pose hold/速度零。
- `Policy`: `730 -> FSQ40 -> [40+990] -> 29`，FSQ 值始终落在 centered nine levels，STE 保有有限梯度，ELU；Normalizer 统计更新正确。
- `WM`: `S71 + a29 + xi33 -> 70 -> S1(71)`，LayerNorm+SiLU；70D target 与积分逆算能在构造轨迹上闭环。
- `Critic`: 184+730=914，特征顺序可由单元测试核对。
- `Reference reward`: 对指定 `(S_{t+1}, f_{t+1}, a_t, a_prev)`，simulator 与 imagine reward/termination 数值匹配；pelvis/waist明确不混。

**数据/训练：**
- 动作加噪唯一一次，`a_prev`、setpoint、reward、Replay 中的 `a` 一致；encoder bias同时影响 obs和setpoint，符号与Vibed一致。
- Time-major Replay 每次 env step 写一整行；终止后的 `s1` 为 reset 前状态；window done-mask 不穿 episode；history padding 不串episode；SGS/return reset 都可用。
- 32步 imagination start 总满足 `f+32<=clip_end`；短 clip 错误提示，不死循环。
- 正确计数：buffer warmup64次vectorized steps；WM 更新轮次0..499只训练WM；第500次之后开始policy；UTD0.4；320000轮对应约800000次vectorized env steps。
- WM 8-step 与 Policy 32-step no NaN / Inf；WM真实多步误差按1/8 step、8/8 step报告。
- 1卡和多卡 DDP smoke；多卡梯度与 RunningNorm 统计同步，无 collective deadlock；`--env.sampling-mode sgs` 无回归。
- 新 checkpoint 完整保存/恢复 WM-only阶段、normalizer、optimizer、SGS rank状态；旧 `dev/FSQ` checkpoint 被明确拒绝。
- Contact solver 参数通过静态配置测试验证保持未变化。

**评测：**
- 每 clip 从第0帧起评价，最多到参考结束；失败提前终止时用完整长度作分母。无随机扰动、脚部摩擦固定1.0；同时输出每动作、pelvis root、relative body、world body、joint误差。
- 与旧 `dev/FSQ` 基线在相同 P2 数据、相同真实 simulator transitions、同一新主指标下对比；记录FSQ离散策略与连续32D Vibed的**唯一有意差异**，不要将两机器人结果直接视为算法消融。

## 12. 开发完成后必须报告

请输出：
1. 新分支名、源分支 commit SHA、开发完成 commit SHA；
2. 修改文件列表与关键 diff；
3. 71D / 99D / 730D / 133D / 70D / 914D 实际打印维度及对应索引映射；
4. 改动/保留/废弃的配置字段；
5. 单卡、多卡、SGS、短 clip、resume 和 eval 测试命令及测试结果；
6. 与 Vibed 对齐的各项审计表；
7. 仍未与 Vibed 相同的地方（必须包含 P2 物理资产/电机、FSQ40×9、Contact Solver 未动、原工程DDP与SGS）；
8. 最小可复现实验与 A/B benchmark 指令。

### 对照实现链接

- Reproduced 分支：https://github.com/Getting05/SuperMimic-Reproduced/tree/dev/FSQ
- Vibed 主分支：https://github.com/zita-ch/supermimic_vibed
- Vibed 实现定义：https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/mdp.py
- Vibed 网络与归一化：https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/networks.py
- Vibed 参考轨迹：https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/motion.py
- Vibed Replay Buffer：https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/buffer.py
- Vibed 训练逻辑：https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/train.py
- Vibed Simulator：https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/env.py
