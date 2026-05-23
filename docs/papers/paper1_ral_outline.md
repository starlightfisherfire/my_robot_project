# Paper 1: Causal-Aware Object-Level Representations for OOD Generalization in Robot Pushing

> **目标期刊：** IEEE Robotics and Automation Letters (RA-L) / CoRL
> **论文类型：** 6+2 pages (RA-L) or 8 pages (CoRL)
> **当前状态：** 🔄 实验中期，oracle MPC gate 已通过

---

## Abstract（约 200 词）

**TODO after all experiments complete.**

```
Keywords: object-centric representation, OOD generalization, model predictive control, 
robot pushing, causal structure, MuJoCo simulation
```

---

## I. Introduction

### A. Motivation
- 当前机器人操控系统在分布外（OOD）场景下泛化能力不足
- 大多数方法依赖大规模数据 + VLA/LLM，缺少对物理因果结构的显式建模
- **核心论点：** 真正的 OOD 泛化需要理解"物体是什么、在哪、如何互动"——因果表征而非统计关联

### B. Research Question
在给定 manipulation 任务、给定 CEM-MPC planner、给定 OOD family 下，什么样的最小 action-relevant structured object state 足以支撑短时动力学预测、规划和 OOD 泛化？

### C. Approach
1. MuJoCo + SO-101 推放任务平台
2. Oracle MPC 作为 planning upper bound
3. Layout OOD (主) + Shape OOD (次)
4. 因果感知物体级表征（object identity, pose, contact, obstacle relation）

### D. Contributions
1. Benchmark: structured OOD evaluation protocol for object pushing
2. Oracle MPC capability boundary on obstacle/passage scenarios
3. Phase transition analysis: speed-budget product determines navigability
4. Cost function design for collision-aware planning
5. — (learned model results to be added)

---

## II. Related Work

### A. Object-Centric Representations in Robotics
- Slot attention, object-centric learning
- Key insight: most prior work evaluates on reconstruction, not OOD control

### B. Model Predictive Control for Manipulation
- CEM-MPC, MPPI, sampling-based planning
- Most use privileged state; this work explicitly measures the gap

### C. OOD Generalization in Robot Learning
- Domain randomization, data augmentation → don't address structural OOD
- Causal representation learning → promising but rarely tested on manipulation

### D. Non-Ergodic / Obstacle-Rich Pushing
- PushT benchmark (Diffusion Policy)
- Key difference: our benchmark includes structured obstacles + OOD splits

---

## III. Problem Formulation

### A. Task Definition
- Planar pushing: T/L-shaped objects to goal pose
- State: object pose (x, y, θ) + obstacle layout + pusher state
- Action: 2D pusher displacement

### B. OOD Splits
| Split | OOD Type | Description |
|-------|----------|-------------|
| train_sim_id | ID | Training template layouts |
| test_sim_id | Layout OOD | Unseen obstacle configurations |
| test_shape_id | Shape OOD | Unseen object geometries |

### C. Template Families
- blocking_easy / blocking_medium / blocking_hard
- passage_direct_wide / passage_direct_medium / passage_direct_narrow
- Each: initial object pose + goal pose + obstacle configuration

---

## IV. Method: Oracle MPC with Collision-Aware Cost

### A. MuJoCo Simulation Environment
- SO-101 arm, planar pushing
- Compile-time obstacles (wall-style barriers, 0.05m height)
- True rigid-body collision dynamics

### B. CEM-MPC Planner
- Horizon, samples, elites, iterations
- Planner parameters fixed across all experiments

### C. Cost Function Design
```
total_cost = w_pos * pos_error² + w_theta * theta_error²
           + w_collision * collision_any + w_collision_step * collision_count
           + w_reach * reach_cost + w_no_contact * no_contact_cost
           + w_action * action_norm + w_smooth * action_smoothness
```

### D. Strict Pose Stop Criterion
- pos_error ≤ 0.0015m AND theta_error ≤ 3.0°
- Early termination when both conditions met

---

## V. Experiments

### A. Experiment 1: Planner Capacity Gate（Open Space）
- **Question:** Can oracle MPC reach sub-cm precision on open_space?
- **Result:** Yes, 5cm success 100%, 0.5cm success rate depends on config
- **Config matrix:** horizon × budget × samples

### B. Experiment 2: Obstacle Gate（Full Sixpack）
- **Question:** Can oracle MPC handle obstacle/passage scenarios?
- **Config:** Speed × Budget sweep (5 speeds, 3 budgets, 6 templates, 66 videos)
- **Key result:** speed020_b1000 achieves 83% overall, uniquely cracks blocking_hard

### C. Experiment 3: Speed-Budget Phase Transition
- **Question:** Why does blocking_hard require (speed ≥ 0.020 AND budget ≥ 1000)?
- **Result:** Phase transition: search space must exceed geometric occlusion radius
- **Table:** Full speed × budget × template matrix

### D. Experiment 4: Physical Feasibility Boundary
- **Question:** Why does passage_direct_narrow fail universally?
- **Result:** Geometric analysis → pusher+object cross-section > passage gap
- **Claim:** Oracle MPC upper bound at physical feasibility boundary

### E. Experiment 5: Collision Cost Ablation（TODO）
- **Question:** Does w_collision_step improve obstacle avoidance?
- **Design:** Compare max-only vs max+sum collision cost

### F. Experiment 6: Speed Ablation（Higher Speeds）
- **Question:** Does increasing speed to 0.15/0.20 improve results?
- **Result:** 0.020 > 0.015 > 0.10 > 0.075 in success rate
- **Detail:** speed020_b1000 best at 83% (5/6)

### G. Experiment 7: Parallel CEM Validation
- **Question:** Does parallel CEM produce identical costs to serial?
- **Design:** compare-only mode, serial vs parallel
- **Result:** Max abs diff ≤ 1e-6

---

## VI. Results

### A. Main Result Tables
- [TABLE 1] Full success matrix (speed × budget × template)
- [TABLE 2] Config ranking by composite score
- [TABLE 3] Failure analysis per template family

### B. Key Figures
- [FIG 1] Speed-budget phase transition diagram (blocking_hard)
- [FIG 2] Passage width vs success rate (physical feasibility boundary)
- [FIG 3] Collision count vs success rate (CEM exploration signal)
- [FIG 4] Cost decomposition for successful vs failed trajectories
- [FIG 5] Video frame sequence: blocking_hard success vs failure

### C. Analysis
1. **Phase transition:** speed × budget > threshold → navigable
2. **Physical boundary:** passage_direct_narrow universally infeasible
3. **Collision as signal:** higher collision count ≠ worse; can indicate active exploration
4. **Cost function: max+sum distinguishes stuck from grazing**

---

## VII. Discussion

### A. What Oracle MPC Tells Us
- Oracle MPC with true dynamics achieves 83% on obstacle-rich tasks
- Remaining failures are geometric, not planner artifacts
- This defines the upper bound for learned models

### B. Implications for Learned Models
- Learned dynamics must match oracle within certain tolerance to avoid "planning blind spots"
- Speed-budget phase transition suggests learned models need sufficient predictive accuracy
- Collision cost design matters: single collision flag insufficient for contact-rich manipulation

### C. Limitations
- Fixed CEM-MPC parameters (not optimal per-template)
- Single object type per experiment
- No sim-to-real transfer
- No multi-object scenes

---

## VIII. Conclusion
- Oracle MPC with collision-aware cost solves 83% of OOD obstacle pushing tasks
- Phase transition identified: critical speed-budget product for obstacle navigation
- Physical feasibility boundary: passage width < pusher+object cross-section
- Benchmark + protocol designed for evaluating learned models

**Future work:** Replace oracle dynamics with learned model, measure OOD gap.

---

## Appendix

### A. Full Experiment Logs
- Available in `runs/debug/`

### B. Cost Function Derivation
- `src/planners/cost_functions.py`

### C. Benchmark Infrastructure
- `benchmark/`

### D. Template System
- `data/sim/metadata/reset_templates_obstacle_sixpack_v0.json`

---

**数据待填入的占位符：**
- [ ] Abstract final numbers
- [ ] All tables with exact values
- [ ] All figures
- [ ] Statistical significance tests
- [ ] Learned model results (Paper 1 Stage 2)
