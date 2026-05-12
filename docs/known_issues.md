# Known Issues & Risk Register

**Last updated:** 2026-05-12

本文档记录当前已知风险、不要误判的地方、以及尚未解决的问题。

---

## 1. 【高风险】真实 MuJoCo obstacle 尚未接入

**状态：** 未解决

**问题：**
- `MujocoPushEnv` 当前不从 reset template 中读取 `obstacles` 字段
- MuJoCo XML/model 中没有动态生成 obstacle geom
- blocking / narrow_passage / edge_goal 等 layout 家族无法真正测试
- 当前所有 capacity 结果均来自 open_space 或 mild_offset（弱约束）

**不要误判：**
- ❌ 不要把 mild_offset 说成真实 obstacle
- ❌ 不要把当前 capacity 结果说成 layout OOD 能力
- ❌ 不要在 obstacle gate 通过前进入 learned model 阶段

**需要验证的 checklist：**
1. `reset_templates_v0.json` 是否包含 blocking / narrow_passage / edge_goal 模板
2. `MujocoPushEnv` 是否读取 `template["obstacles"]`
3. MuJoCo XML/model 是否 instantiate obstacle geom
4. obstacle 是否能参与 collision
5. render 视频里是否能肉眼看到 obstacle
6. cost 是否对 obstacle collision / avoidance 产生影响
7. config23_strict600 在 2-3 个真实 obstacle template 上是否不崩、不穿模、不乱飞

---

## 2. 【已修复】strict pose stop 被 legacy 5cm 截断

**状态：** 已修复（2026-05-12），待 smoke test 确认

**旧问题：**
- legacy 5cm success 设置 `success=True`
- chunk end 的 `if success and not disable_early_stop: break` 导致 strict pose stop 开启时仍被 5cm 截断
- 表现：strict pose stop 开启但系统在 5cm 处提前停止

**修复内容（`src/metrics/mujoco_oracle_capacity.py`）：**
- 引入 `should_stop`、`legacy_success_reached`、`strict_pose_stop_active`
- strict pose active 时，legacy 5cm 只写 `legacy_success_reached=True`，不设 `should_stop`
- chunk end 改为 `if should_stop: break`
- strict 阈值判断改为 `<=`

**验证命令（人工执行）：**
```bash
PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode mujoco_oracle_mpc_closed_loop \
  --split train_sim_id --max-templates 1 \
  --horizon 80 --execute-steps 20 --max-mpc-steps 30 \
  --num-samples 1024 --num-elites 96 --num-iterations 5 \
  --strict-pose-stop --stop-pos-threshold 0.0015 \
  --stop-theta-threshold-deg 3.0 \
  --out runs/debug/c23_strict600_smoke.json
```

检查重点：不能在 5cm 处提前停止；只有 `pos<=1.5mm 且 theta<=3°` 才触发 STRICT POSE EARLY STOP。

---

## 3. 【注意】5cm early stop 不适合作为能力边界判断

**状态：** 已确认结论，文档记录

**问题：**
- 旧的 5cm early stop 让系统看起来只能到 4.8cm 左右
- no-early-stop full budget 后，Oracle-MPC 可达到毫米级误差
- 5cm 只能作为粗成功统计，不应作为正式 pose-to-goal 完成标准

**正确做法：**
- Boundary search：`disable_early_stop=True`，跑满预算
- Confirmatory eval：`strict_pose_stop=True`，达到 1.5mm + 3° 后停止

---

## 4. 【注意】mild_offset 不是真实 obstacle

**状态：** 持续风险

**问题：**
- `mild_offset` layout 家族只是目标位置偏移，不包含真实物理障碍物
- 在 mild_offset 上的好结果不能等同于 obstacle avoidance 能力
- 当前 confirmatory eval（3 open_space + 3 mild_offset）只能说明 open/weak-constrained 能力

**不要误判：**
- ❌ 不要把 mild_offset 结果写成 obstacle performance
- ❌ 不要在 paper 中声称 layout OOD 已解决

---

## 5. 【注意】当前 capacity 结果仅来自 train_sim_id

**状态：** 持续风险

**问题：**
- 所有 boundary search 和 confirmatory eval 均在 `train_sim_id` split 上
- 尚未在 `test_sim_layout_ood` 或 `test_sim_shape_ood` 上测试
- 当前结果只能说明 in-distribution 能力

---

## 6. 【注意】planner 参数不能在 encoder 对比时分别调优

**状态：** 持续风险

**规则：**
- flat / object / causal 三种 encoder 必须使用完全相同的 planner 配置
- 不能为某个 encoder 单独调 horizon / num_samples / cost weights
- 违反此规则会使 OOD gap 分析无效
