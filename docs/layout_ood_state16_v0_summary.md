# layout_ood_state16_v0 — 实验聚合报告
## 给 ChatGPT 分析的完整上下文

生成时间: 2026-05-17 16:00 CST
生成者: Paper1 助手 (Agent main, lobster1 bot)

---

## 一、实验背景与目标

### Paper 1 核心问题
比较三种高层表示（flat / object-centric / causality-aware）在固定 CEM-MPC 下
对 structural layout OOD 的预测泛化能力。

### 本轮实验 (layout_ood_state16_v0)
- 训练: 随机生成的单障碍物 blocking + open 模板
- 测试: 固定模板的 dual-obstacle passage（compositional OOD）
- 仅做 open-loop dynamics prediction，不做 closed-loop MPC
- 使用原始 16 维 structured state token schema，不引入视觉

---

## 二、设计决策记录（Paper1 助手的思路）

### 2.1 随机模板 vs 固定模板
- 训练数据: **无限随机采样**，不用固定模板
  - 理由: dynamics model 需要 state-action 覆盖多样性
  - 每次 episode 随机采样障碍物尺寸(width 3~12cm, length 4~15cm)、位置(在 obj→goal 路径上 ±4cm jitter)、旋转(±0.6 rad)
  - 约束: 障碍物不覆盖 object 或 goal，障碍物 X 必须在 (obj_x+4cm, goal_x-4cm) 内
  
- 测试评估: 用固定模板文件 reset_templates_obstacle_10family_v0.json
  - 理由: 可复现、可对比

### 2.2 MPPI + 噪声混合策略
- 50% episode: 纯 MPPI (noise_std=0)
- 30% episode: MPPI + 低噪声 (std=0.05)  
- 20% episode: MPPI + 高噪声 (std=0.15)
- 目的: 覆盖最优路径 + 偏离路径 + 恢复路径，避免 dynamics model 只学到最优路径附近的动力学

### 2.3 先采后选，不过滤
- 采集时: 所有 episode 都保存（成功 + 失败）
- 选择时: 训练脚本按 metadata (family/success/contact_count/collision_count) 灵活过滤
- 不预设"好数据"标准

### 2.4 不做的事情
- 不改 16 维 schema
- 不改 MuJoCo 物理
- 不改 cost/planner 核心
- 不引入视觉
- 不做 learned-MPC closed-loop

---

## 三、已创建/修改的文件

### 新增文件

| 文件 | 用途 |
|------|------|
| `configs/experiments/layout_ood_state16_v0.yaml` | 实验配置（采集/训练/评估参数） |
| `src/data/template_generator.py` | 随机模板生成器 (open/blocking/passage) |
| `scripts/collect_layout_ood_state16.py` | 数据采集脚本 (MPPI + 噪声 + EpisodeWriter) |
| `scripts/preview_random_obstacle_templates.py` | 模板可视化预览 |
| `artifacts/random_template_preview/preview_grid.png` | 13 个随机模板的俯视图 |

### 数据文件

| 文件 | 说明 |
|------|------|
| `data/sim/layout_ood_state16_v0/episodes/*.npz` (10个) | 每个 21-28KB，250 transitions × 6tokens × 16dim |
| `data/sim/layout_ood_state16_v0/metadata/episodes.jsonl` | 每行一个 episode 的元信息 |

### 已有的核心模块（本轮未修改，仅使用）

| 模块 | 文件 | 状态 |
|------|------|------|
| MuJoCo env + obstacles | `src/envs/mujoco_push_env.py` | ✅ 已实现，支持编译时 obstacle |
| MPPI planner | `src/planners/mppi.py` | ✅ |
| MuJoCo oracle rollout | `src/planners/mujoco_oracle_rollout.py` | ✅ |
| Cost functions | `src/planners/cost_functions.py` | ✅ |
| Episode writer | `src/data/episode_writer.py` | ✅ |
| State normalizer | `src/data/state_normalizer.py` | ✅ |
| FlatEncoder | `src/models/encoders.py` | ✅ |
| ObjectCentricEncoder | `src/models/encoders.py` | ✅ |
| CausalityAwareEncoder | `src/models/encoders.py` | ✅ |
| DynamicsHead + SubgoalHead | `src/models/heads.py` | ✅ |
| RIGWorldModel | `src/models/rig_world.py` | ✅ |
| Losses | `src/models/losses.py` | ✅ |

### 空文件（尚未实现）

| 文件 | 说明 |
|------|------|
| `scripts/collect_sim_data.py` | 空文件 |
| `scripts/train_high_level.py` | 空文件 |
| `scripts/eval_policy.py` | 空文件 |
| `src/data/episode_loader.py` | 空文件 |

---

## 四、10 个 episode 采集结果

### 运行参数
```
MPPI: horizon=8, num_samples=128, num_iterations=2
       ↑ 注意：这是弱参数，仅为快速验证数据格式
       正式采集应用 horizon=12, num_samples=512, num_iterations=3
```

### Episode 明细

| # | Family | 成功 | 最终距离 | 接触步 | 噪声 | ID |
|---|--------|------|---------|--------|------|-----|
| 1 | passage | ❌ | 0.274m | 12 | 0.05 | 8a9856e7-882 |
| 2 | open | ❌ | 0.220m | 25 | 0.05 | 718f7994-3b5 |
| 3 | passage | ❌ | 0.284m | 65 | 0.05 | 40cca682-4d2 |
| 4 | blocking | ❌ | 0.268m | 38 | 0.15 | 1970b99c-8ac |
| 5 | blocking | ❌ | 0.344m | 48 | 0.15 | c360f73a-321 |
| 6 | blocking | ❌ | 0.284m | 51 | 0.0 | 7e946b2f-008 |
| 7 | passage | ❌ | 0.191m | 27 | 0.05 | eb96b727-df6 |
| 8 | blocking | ❌ | 0.271m | 26 | 0.0 | f0eddbd3-497 |
| 9 | blocking | ❌ | 0.158m | 34 | 0.0 | db174b90-785 |
| 10 | blocking | ❌ | 0.141m | 38 | 0.05 | 4f51e569-dad |

### 统计
- 0/10 成功 (MPPI 参数太弱)
- 2500 个 transitions
- 总耗时 705 秒 (~12 分钟)
- 总数据 276KB

### 数据分布
- blocking: 6 episodes
- open: 1 episode
- passage: 3 episodes

---

## 五、数据格式验证

每个 npz 文件包含:
```
states:              (T, 6, 16) float32   ← 核心
actions_norm:        (T, 2)   float32
actions_physical:    (T, 2)   float32
next_states:         (T, 6, 16) float32
object_poses:        (T, 3)   float32
next_object_poses:   (T, 3)   float32
ee_positions:        (T, 2)   float32
next_ee_positions:   (T, 2)   float32
contact_flags:       (T,)     bool
collision_flags:     (T,)     bool
actual_ee_velocities:(T, 2)   float32
actual_object_velocities:(T, 3) float32
goal_pose:           (3,)     float32
obstacle_features:   (18,)    float32
```

Token 顺序: [0:EE, 1:Object, 2:Goal, 3:Obs1, 4:Obs2, 5:Obs3(填充零)]
Token 维度: [x, y, sinθ, cosθ, vx, vy, ω, size_x, size_y, shape_T, shape_L, shape_other, mass, friction, contact, valid]

格式与 `docs/model_design.md` 完全一致 ✅

---

## 六、16 维 State Schema 评估

### 当前 schema
```
0:  x
1:  y
2:  sin(theta)
3:  cos(theta)
4:  vx
5:  vy
6:  omega
7:  size_x
8:  size_y
9:  shape_T
10: shape_L
11: shape_other
12: mass_norm
13: friction_norm
14: contact_flag
15: valid_flag
```

### 已知局限
1. 缺少相对量 (ee→object 距离、object→goal 距离)
2. 缺少障碍物影响力表征 (blocking_ratio、clearance)
3. 缺少测量置信度字段 (sim-to-real 桥)
4. valid_flag 作为 padding 标记过于粗糙

### Paper1 助手的建议
- Paper 1 sim 实验: 16 维够用
- 后续 sim-to-real: 扩展到 24 维 (加 rel_goal、confidence、clearance 等)
- 等 Paper 2 感知桥就绪后再升级

---

## 七、已发现问题

### 问题 1: MPPI 参数
- 当前: horizon=8, 128 samples, 2 iterations → 全部失败
- 之前 heavy_pusher 成功实验: horizon=12~20, 512~1024 samples, 3~5 iterations
- 解决方案: 正式采集时用 horizon=12, 512 samples, 3 iterations

### 问题 2: 三个核心脚本为空
- collect_sim_data.py, train_high_level.py, eval_policy.py 全是空文件
- 训练和评估脚本尚未实现

### 问题 3: passage_bypass 模板不足
- reset_templates_obstacle_10family_v0.json 中 bypass 只有各 1 个模板
- 无法做统计比较

### 问题 4: 数据采集速度
- 单核 MPPI oracle rollout 每 episode 2-3 分钟
- 2000 episode 需要约 80 小时 (单核)

---

## 八、下一步行动 (待决策)

1. **调整 MPPI 参数后正式采集** (horizon=12, 512 samples, 3 iterations)
2. **实现 train_layout_ood_state16.py** (训练 flat baseline)
3. **实现 eval_layout_ood_state16.py** (open-loop rollout error 评估)
4. **并行化数据采集** (多进程，利用多核)
5. **考虑用 CEM 替代 MPPI** 作采集策略（CEM 在单障碍物场景可能更快）

---

## 附录 A: 实验配置文件

\`\`\`yaml
$(cat /home/brucewu/my_robot_project/configs/experiments/layout_ood_state16_v0.yaml)
\`\`\`

---

## 附录 B: 随机模板生成器

\`\`\`python
$(cat /home/brucewu/my_robot_project/src/data/template_generator.py)
\`\`\`

---

## 附录 C: 数据采集脚本

\`\`\`python
$(cat /home/brucewu/my_robot_project/scripts/collect_layout_ood_state16.py)
\`\`\`

---

## 附录 D: 已有 RIGWorldModel 接口

\`\`\`python
$(cat /home/brucewu/my_robot_project/src/models/rig_world.py)
\`\`\`

---

## 附录 E: 已有 EpisodeWriter 格式

\`\`\`python
$(cat /home/brucewu/my_robot_project/src/data/episode_writer.py)
\`\`\`

---

## 附录 F: 10-family 模板文件结构

$(cd /home/brucewu/my_robot_project && ~/miniconda3/envs/lerobot/bin/python -c "
import json
d = json.load(open('data/sim/metadata/reset_templates_obstacle_10family_v0.json'))
from collections import Counter
families = Counter()
for t in d:
    families[t['family']] += 1
print('| Family | Count | Obstacles | Gap |')
print('|--------|-------|-----------|-----|')
for f in sorted(families):
    ts = [t for t in d if t['family']==f]
    obs_counts = set(len(t['obstacles']) for t in ts)
    gaps = set(t.get('passage_gap','-') for t in ts)
    print(f'| {f} | {families[f]} | {sorted(obs_counts)} | {sorted(gaps)} |')
")

---

## 附录 G: 现有 split 协议 (简化)

- train_sim_id: open_space + mild_offset (T-shape only)
- test_sim_layout_ood_blocking: blocking_easy/medium/hard
- test_sim_layout_ood_passage: passage_direct_wide/medium/narrow + bypass
- test_sim_shape_ood: L-shape
- Real robot splits: 已定义，数据目录为空

---

## 附录 H: 已知不完整项

| 项目 | 状态 |
|------|------|
| MuJoCo env + obstacles | ✅ 可用 |
| 3种 encoder | ✅ py_compile 通过 |
| RIGWorldModel 统一接口 | ✅ |
| StateNormalizer | ✅ |
| 数据采集脚本 | ⚠️ 可用但 MPPI 参数需调强 |
| 训练脚本 | ❌ 空文件 |
| 评估脚本 | ❌ 空文件 |
| Episode loader | ❌ 空文件 |
| Real robot 数据管线 | ❌ 空目录 |

