# Oracle-MPC Capacity Gate 实验协议

**Last updated:** 2026-05-12

本文档记录 MuJoCo Oracle-MPC capacity gate 的实验协议、参数定义、成功标准和下一步 obstacle gate。

---

## 1. 当前阶段定位

当前处于 **第 1 关：MuJoCo Oracle-MPC capacity gate**。

目标：在 MuJoCo 真实动力学下，用 Oracle-MPC（CEM + 真实 rollout）验证 push-to-pose 任务可解性。

**尚未进入：**
- dataset generation
- learned dynamics / world model 训练
- flat / object / causal representation 对比
- OOD final evaluation
- SO-101 real robot validation

---

## 2. 主配置定义

### config23 / c23_precise（主 baseline）

| 参数 | 值 |
|------|-----|
| horizon | 80 |
| execute_steps | 20 |
| max_mpc_steps | 25 |
| num_samples | 1024 |
| num_elites | 96 |
| num_iterations | 5 |
| 总执行预算 | 25 × 20 = 500 env steps |

### c23_strict600（confirmatory eval 用）

| 参数 | 值 |
|------|-----|
| horizon | 80 |
| execute_steps | 20 |
| max_mpc_steps | 30 |
| num_samples | 1024 |
| num_elites | 96 |
| num_iterations | 5 |
| 总执行预算 | 30 × 20 = 600 env steps |

### c25_fast（保守备用）

| 参数 | 值 |
|------|-----|
| horizon | 80 |
| execute_steps | 30 |
| max_mpc_steps | 20 |
| num_samples | 512 |
| num_elites | 48 |
| num_iterations | 5 |
| 总执行预算 | 20 × 30 = 600 env steps |

---

## 3. 两种运行模式

### A. Boundary search（探索能力边界）

- `disable_early_stop=True`
- 跑满预算（max_mpc_steps × execute_steps）
- 用于探索 best/final 能力边界
- 主要指标：`mean_best_pos_error`、`mean_final_pos_error`、`mean_final_theta_error_deg`

### B. Confirmatory eval（正式验证）

- `strict_pose_stop=True`
- `stop_pos_threshold = 0.0015`（1.5mm）
- `stop_theta_threshold_deg = 3.0`
- 必须同时满足：`pos_error <= 0.0015 AND theta_error_deg <= 3.0`
- 用于避免"已经完成但继续推坏"

---

## 4. 成功标准

### Paper 主 success 定义

**primary_success = success_pose_1cm_5deg**
即 `final_pos_error <= 0.01m AND final_theta_error_deg <= 5.0deg`

详见 `docs/success_rate_revise.md`。

### 当前 open_space / mild_offset 阶段

| 指标 | 目标 |
|------|------|
| primary_success_rate (success_pose_1cm_5deg) | ≥ 0.8 |
| success_pos_0p5cm_rate | ≥ 0.6 |
| success_pose_0p5cm_5deg_rate | ≥ 0.6 |
| mean_final_pos_error | ≤ 5mm |
| mean_final_theta_error_deg | ≤ 5° |

### obstacle gate 阶段（待定）

- 需要在 blocking / narrow_passage / edge_goal 模板上通过
- 通过标准：MuJoCo 场景里真实出现 obstacle geom，pusher/object 与 obstacle 有合理碰撞/避让，视频可视化确认

---

## 5. Smoke test 命令（人工执行）

```bash
cd ~/my_robot_project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# strict pose stop smoke test（1 template）
PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode mujoco_oracle_mpc_closed_loop \
  --split train_sim_id \
  --max-templates 1 \
  --horizon 80 \
  --execute-steps 20 \
  --max-mpc-steps 30 \
  --num-samples 1024 \
  --num-elites 96 \
  --num-iterations 5 \
  --strict-pose-stop \
  --stop-pos-threshold 0.0015 \
  --stop-theta-threshold-deg 3.0 \
  --out runs/debug/c23_strict600_smoke.json
```

检查重点：
- 不能在 5cm 处提前停止
- 只有 `pos<=1.5mm 且 theta<=3°` 才 STRICT POSE EARLY STOP
- `strict_pose_stop_step / strict_pose_stop_pos_error / strict_pose_stop_theta_error_deg` 要正常记录

---

## 6. Confirmatory eval 命令（smoke test 通过后执行）

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/run_c23_strictstop_eval.py
```

**注意：**
- `run_c23_strictstop_eval.py` 当前选择的是 3 open_space + 3 mild_offset
- mild_offset 只能算 weak constrained，不是真正 obstacle
- 不能把这个结果写成真实 obstacle performance

---

## 7. Obstacle gate 通过后的局部 sweep

在真实 obstacle 接入后，围绕 config23 做小范围 sweep：

```
horizon: [80, 100]
execute_steps: [15, 20]
max_mpc_steps: [30, 35]
num_samples: [1024]
num_elites: [96]
num_iterations: [5, 7]
总共 16 configs
```

目标：判断 obstacle 下是需要更长 horizon、更频繁 replanning，还是更多总预算。

---

## 8. 后续关卡路线

1. **Oracle-MPC capacity gate**（当前，接近完成，差真实 obstacle gate）
2. **Dataset generation gate**（生成 episode：state_t, action_t, state_t+1, goal, template_id, split, success/failure, contact, obstacle metadata）
3. **Learned dynamics / representation gate**（训练 flat / object / causal，验证 learned rollout + fixed CEM-MPC 在 ID 上工作）
4. **OOD evaluation gate**（train_sim_id / test_sim_id / test_sim_layout_ood / test_sim_shape_ood，比较三种 encoder）
5. **SO-101 real validation gate**（小规模 real-ID adapted OOD）
