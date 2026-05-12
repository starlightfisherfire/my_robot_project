# 文件拓扑关系图

本文档描述 Paper 1 实验系统中关键文件的拓扑连接关系。

---

## 完整拓扑层次结构

```
┌─────────────────────────────────────────────────────────────────┐
│                      数据定义与生成层                              │
├─────────────────────────────────────────────────────────────────┤
│ src/interventions/shape_families.py                             │
│ src/interventions/layout_families.py                            │
│ src/interventions/sampling_rules.py                             │
│         ↓                                                        │
│ scripts/generate_reset_templates.py                             │
│         ↓                                                        │
│ data/sim/metadata/reset_templates_v0.json                       │
│         ↓                                                        │
│ src/interventions/reset_template_loader.py                      │
│ src/data/metadata_schema.py                                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      配置层 (Config)                              │
├─────────────────────────────────────────────────────────────────┤
│ configs/object_specs.yaml        → ObjectShapeFactory           │
│ configs/reset_templates.yaml     → generate_reset_templates.py  │
│ configs/splits.yaml              → train/OOD split 定义         │
│ configs/planner/cem_mpc_capacity.yaml → capacity sweep 参数     │
│ configs/planner/cem_mpc.yaml     → CEM-MPC 默认参数             │
│ configs/planner/cost_weights.yaml → rollout_cost 权重            │
│ configs/train/causality_aware.yaml    → CausalityAwareEncoder   │
│ configs/train/flat_high_level.yaml    → FlatEncoder             │
│ configs/train/object_centric_noncausal.yaml → ObjectCentricEncoder│
│ configs/eval/eval_default.yaml   → 评估配置                     │
│ configs/baselines.yaml           → baseline 参数                │
│ configs/failure_codes.yaml       → 失败码定义                   │
│ configs/analysis/representation_analysis.yaml → 分析配置        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      环境层 (Dynamics)                            │
├─────────────────────────────────────────────────────────────────┤
│ src/envs/toy_push_env.py          (toy dynamics, 接口测试)       │
│ src/envs/object_shape_factory.py  (MuJoCo compound geom 工厂)    │
│   └── 读取 configs/object_specs.yaml                            │
│   └── 生成 T/L/cross/bar/square/cylinder 的 geom XML            │
│ src/envs/mujoco_push_env.py       (MuJoCo dynamics, 主实验)      │
│   └── 调用 ObjectShapeFactory 构建 MuJoCo XML                   │
│         ↓                                                        │
│ scripts/debug_mujoco_env.py       (环境接口验证)                  │
│ scripts/debug_shape_factory.py    (形状工厂验证)                  │
│ scripts/debug_mujoco_shape_render.py (形状渲染验证)               │
│ scripts/preview_object_shapes.py  (形状预览)                     │
│ scripts/preview_reset_templates.py (模板预览)                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      规划器层 (Planning)                          │
├─────────────────────────────────────────────────────────────────┤
│ src/planners/cost_functions.py    (cost计算核心)                 │
│         ↓                                                        │
│ src/planners/oracle_rollout.py    (toy oracle rollout)          │
│ src/planners/mujoco_oracle_rollout.py  (MuJoCo oracle rollout)  │
│         ↓                                                        │
│ scripts/debug_oracle_rollout.py                                 │
│ scripts/debug_mujoco_oracle_rollout.py                          │
│         ↓                                                        │
│ src/planners/cem_mpc.py            (CEM优化器，固定不变)          │
│ src/planners/heuristic_baselines.py (启发式 baseline)           │
│         ↓                                                        │
│ scripts/debug_cem_mpc_toy.py                                    │
│ scripts/render_closed_loop_rollout.py (闭环 rollout 渲染)        │
│ scripts/render_pusher_capacity.py  (pusher capacity 渲染)        │
│ scripts/debug_pusher_capacity.py  (pusher capacity 调试)         │
│ scripts/visualize_mpc_rollout.py  (MPC rollout 可视化)           │
│ scripts/diagnose_mpc_rollout.py   (MPC rollout 诊断)             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Oracle-MPC 容量验证层                           │
├─────────────────────────────────────────────────────────────────┤
│ src/metrics/planner_capacity.py   (通用 capacity 抽象)           │
│ src/metrics/toy_oracle_capacity.py                              │
│ src/metrics/mujoco_oracle_capacity.py                           │
│         ↓                                                        │
│ scripts/check_mpc_capacity.py     (统一入口)                     │
│   --mode state_sanity                                           │
│   --mode toy_oracle_mpc                                         │
│   --mode mujoco_oracle_mpc                                      │
│   --mode mujoco_oracle_mpc_closed_loop                          │
│         ↓                                                        │
│ scripts/run_wide_mpc_sweep_with_best_video.py (wide sweep)     │
│ scripts/run_mpc_eval.py           (MPC 评估)                    │
│ scripts/run_closed_loop_sweep.py  (闭环 sweep)                  │
│         ↓                                                        │
│ scripts/analyze_results.py        (结果分析)                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Metrics 与评估层                                │
├─────────────────────────────────────────────────────────────────┤
│ src/metrics/success_metrics.py    (成功率计算)                   │
│ src/metrics/failure_analysis.py   (失败分析)                     │
│ src/metrics/ood_gap.py            (OOD 泛化差距)                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   数据归一化层 (Critical!)                        │
├─────────────────────────────────────────────────────────────────┤
│ src/data/state_normalizer.py                                    │
│   ⚠️  MUST fit only on train/adaptation splits                  │
│   ⚠️  NEVER fit on test/OOD splits                              │
│         ↓                                                        │
│ scripts/debug_state_normalizer.py                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   模型层 (Learned Models)                         │
├─────────────────────────────────────────────────────────────────┤
│ src/models/encoders.py                                          │
│   - FlatEncoder                                                 │
│   - ObjectCentricEncoder                                        │
│   - CausalityAwareEncoder                                       │
│         ↓                                                        │
│ src/models/heads.py                                             │
│   - DynamicsHead                                                │
│   - SubgoalHead                                                 │
│         ↓                                                        │
│ src/models/rig_world.py            (统一接口)                     │
│         ↓                                                        │
│ scripts/debug_encoder_variants.py                               │
│ scripts/debug_rig_world_model.py                                │
│         ↓                                                        │
│ src/models/losses.py               (训练loss)                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Learned Rollout 层                             │
├─────────────────────────────────────────────────────────────────┤
│ src/planners/rollout_model.py                                   │
│   (使用learned model进行rollout预测)                             │
│         ↓                                                        │
│ 与 cem_mpc.py 结合                                               │
│         ↓                                                        │
│ Learned Model + Fixed CEM-MPC                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   工具层 (Utils)                                  │
├─────────────────────────────────────────────────────────────────┤
│ src/utils/io.py                   (I/O 工具)                    │
│ src/utils/logging_utils.py        (日志工具)                     │
│ src/utils/seed.py                 (随机种子)                     │
│ src/data/episode_loader.py        (episode 加载)                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   评估与对比层                                    │
├─────────────────────────────────────────────────────────────────┤
│ 对比三种encoder在相同planner下的OOD泛化性能                        │
│   - ID test                                                     │
│   - Layout OOD (blocking, narrow_passage, edge_goal)            │
│   - Shape OOD (L-shape)                                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   基准文档层 (Benchmark)                          │
├─────────────────────────────────────────────────────────────────┤
│ benchmark/benchmark_card.md       (基准卡片)                     │
│ benchmark/baselines.md            (baseline 定义)               │
│ benchmark/tasks.md                (任务定义)                     │
│ benchmark/splits.md               (split 定义)                  │
│ benchmark/metrics.md              (metrics 定义)                │
│ benchmark/dataset_card.md         (数据集卡片)                   │
│ benchmark/leaderboard_template.csv (排行榜模板)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 关键依赖关系详解

### 1. 数据流向

```
reset_templates_v0.json
    → reset_template_loader.load_reset_templates()
    → MujocoPushEnv.reset_from_template()
    → 环境初始状态
```

### 2. Oracle-MPC 验证流向

```
reset_template
    → MujocoPushEnv.reset_from_template()
    → mujoco_oracle_rollout.rollout_action_sequence_mujoco()
    → cost_functions.rollout_cost()
    → CEMMPC.plan(cost_fn)
    → mujoco_oracle_capacity.evaluate_one_template_mujoco_oracle_mpc()
    → check_mpc_capacity.py --mode mujoco_oracle_mpc
```

### 3. ObjectShapeFactory 依赖链

```
configs/object_specs.yaml
    → ObjectShapeFactory.__init__()
    → ObjectShapeFactory.get_object_geoms_xml()
    → ObjectShapeFactory.get_goal_ghost_geoms_xml()
    → mujoco_push_env._build_xml_with_shape()
    → MuJoCo XML (T/L/cross/bar/square/cylinder 复合几何体)
```

### 4. Learned Model 训练流向 (后续)

```
reset_templates
    → 数据收集 (oracle policy / random policy)
    → StateNormalizer.fit(train_data)  ⚠️ 只在train上fit
    → 训练 RIGWorldModel (flat / object_centric / causality_aware)
    → 保存 checkpoint
```

### 5. Learned Model 评估流向 (后续)

```
reset_template
    → MujocoPushEnv.reset_from_template()
    → StateNormalizer.transform(state)  ⚠️ 使用train上fit的normalizer
    → RIGWorldModel.forward(normalized_state, action)
    → rollout_model.rollout_with_learned_model()
    → CEMMPC.plan(learned_cost_fn)
    → 评估成功率 (ID / Layout OOD / Shape OOD)
```

---

## 关键文件重要性分级

### 🔴 核心基础设施 (必须先验证)

1. **mujoco_push_env.py** - MuJoCo环境封装
2. **object_shape_factory.py** - MuJoCo compound geom 工厂，被 mujoco_push_env 调用
3. **cost_functions.py** - cost计算核心
4. **cem_mpc.py** - 固定规划器
5. **mujoco_oracle_rollout.py** - Oracle rollout接口
6. **mujoco_oracle_capacity.py** - Oracle-MPC容量测试

### 🟡 数据与归一化 (训练前必须)

7. **reset_template_loader.py** - 加载reset templates
8. **metadata_schema.py** - 数据schema定义
9. **state_normalizer.py** - ⚠️ 数据归一化 (严格split规则)
10. **layout_families.py** - Layout OOD定义
11. **shape_families.py** - Shape OOD定义
12. **sampling_rules.py** - 采样规则验证

### 🟢 模型层 (训练与评估)

13. **encoders.py** - 三种encoder实现
14. **heads.py** - 预测头
15. **rig_world.py** - 统一模型接口
16. **losses.py** - 训练loss
17. **rollout_model.py** - Learned rollout接口

### 🔵 脚本与工具

18. **check_mpc_capacity.py** - 容量验证统一入口
19. **run_wide_mpc_sweep_with_best_video.py** - Wide sweep 脚本
20. **generate_reset_templates.py** - 生成reset templates
21. **render_closed_loop_rollout.py** - 闭环 rollout 渲染
22. **analyze_results.py** - 结果分析
23. **debug_*.py** - 各模块烟雾测试

### ⚪ 配置层

24. **configs/object_specs.yaml** - 物体形状规格 (被 ObjectShapeFactory 读取)
25. **configs/reset_templates.yaml** - Reset template 配置
26. **configs/splits.yaml** - 数据分割定义
27. **configs/planner/cem_mpc_capacity.yaml** - CEM-MPC capacity sweep 参数
28. **configs/planner/cost_weights.yaml** - Cost 权重配置

### 🟣 基准文档层

29. **benchmark/benchmark_card.md** - 基准卡片
30. **benchmark/baselines.md** - Baseline 定义
31. **benchmark/tasks.md** - 任务定义
32. **benchmark/splits.md** - Split 定义
33. **benchmark/metrics.md** - Metrics 定义
34. **benchmark/dataset_card.md** - 数据集卡片

---

## 当前验证状态

### ✅ 已验证
- mujoco_push_env.py (基础 reset/step/contact)
- object_shape_factory.py (compound geom 生成)
- mujoco_oracle_rollout.py
- debug_mujoco_oracle_rollout.py
- debug_shape_factory.py
- debug_mujoco_shape_render.py
- check_mpc_capacity.py --mode state_sanity
- check_mpc_capacity.py --mode toy_oracle_mpc

### ✅ 正在进行 (Wide Sweep)
- mujoco_oracle_capacity.py
- check_mpc_capacity.py --mode mujoco_oracle_mpc_closed_loop
- run_wide_mpc_sweep_with_best_video.py (~20 parallel processes executed)

### ✅ 已完成
- render_closed_loop_rollout.py (单个 config 视频渲染, config25)
- strict-pose-stop 敏感参数 (pos 0.0015m + theta 3.0°)

### 📋 后续需要
- state_normalizer.py (训练前验证)
- encoders.py, heads.py, rig_world.py (训练时)
- rollout_model.py (评估时)

---

## SO101 真机配置参数参考

### Pusher (推杆) 默认几何参数 (MuJoCo 模拟)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pusher_radius` | 0.010 m (1.0 cm) | 推杆半径 |
| `pusher_halfheight` | 0.014 m (1.4 cm) | 推杆半高 |
| `pusher_z` | 0.016 m (1.6 cm) | 推杆z轴位置 (底面高度) |

这些参数可通过 `--pusher-radius`、`--pusher-halfheight`、`--pusher-z` 覆盖。

---

## 实验流程检查清单

### Phase 1: Oracle-MPC 容量验证 (当前)
- [x] 验证 mujoco_push_env.py
- [x] 验证 object_shape_factory.py
- [x] 验证 mujoco_oracle_rollout.py
- [x] 运行 wide sweep (~20 parallel processes)
- [x] 完成 wide sweep 全部配置
- [ ] 运行 analyze_results.py 分析 sweep 结果
- [x] 渲染 boundary config 闭环视频 (config25)
- [x] 确认 Oracle-MPC 在 train_sim_id 上成功率边界
- [ ] 确认 Oracle-MPC 在 layout OOD 上的降级程度

### Phase 2: 数据收集
- [ ] 使用 Oracle-MPC 收集 train_sim_id 数据
- [ ] 验证 StateNormalizer 只在 train 上 fit
- [ ] 检查数据split无泄漏

### Phase 3: 模型训练
- [ ] 训练 FlatEncoder + Heads
- [ ] 训练 ObjectCentricEncoder + Heads
- [ ] 训练 CausalityAwareEncoder + Heads
- [ ] 相同训练数据、相同超参数

### Phase 4: 模型评估
- [ ] Learned Model + Fixed CEM-MPC
- [ ] 评估 ID test
- [ ] 评估 Layout OOD
- [ ] 评估 Shape OOD
- [ ] 对比三种encoder的OOD降级程度

---

## 重要规则提醒

1. **CEM-MPC 固定**: 所有encoder变体使用相同的planner参数
2. **StateNormalizer 规则**: 只在train/adaptation上fit，test上只transform
3. **Split 隔离**: Layout OOD / Shape OOD 不能出现在训练集
4. **Toy vs MuJoCo**: ToyPushEnv 只用于接口测试，不能作为实验结果
5. **Oracle-MPC 先行**: 必须先验证Oracle-MPC容量，才能归因learned model失败
6. **Wide sweep 不进 early-stop**: wide_overnight_v2 sweep 为 no-early-stop 模式
7. **ObjectShapeFactory 已集成**: T/L/cross/bar/square/cylinder 复合几何体通过 YAML 配置驱动
8. **`--max-templates` 选择逻辑**: 按 JSON 文件原始顺序取前 N 个 (不 shuffle)，`train_sim_id` 的第一个始终是 `open_space + T_shape + 0 障碍物`

注意，file_topology_map.md 是文件结构索引，不是当前实验状态真相。
当前实验状态以 docs/current_sprint.md 和 docs/code_audit.md 为主要标准，并且根据用户实际需求调整。