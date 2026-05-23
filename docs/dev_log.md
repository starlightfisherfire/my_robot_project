# Dev Log — Paper 1

## 2026-05-23: Learned Rollout Stack Self-Check PASS

**时间：** 2026-05-23 23:00

**里程碑：** learned rollout stack self-check 全部通过

**报告路径：** `runs/self_check/learned_rollout_stack_self_check.json`

**检查结果：**

| Check | Status |
|-------|--------|
| repo_import_check | ✅ PASS |
| py_compile_check | ✅ PASS |
| dummy_interface_check | ✅ PASS |
| cost_fn_check | ✅ PASS |
| dataset_check | ✅ PASS (36 episodes) |
| checkpoint_check | ✅ PASS |
| learned_mpc_smoke_check | ✅ PASS (smoke_pass=true, planned_cost < zero_cost) |

**overall_status: PASS**

**关键指标：**
- zero_cost_finite: true
- planned_cost_finite: true
- smoke_pass: true (planned_cost < zero_cost in at least one episode)

**含义：**
- learned rollout 路径端到端可跑
- CEM 能找到比零动作更好的规划
- 但这是 internal smoke，不是真实 MuJoCo closed-loop

**注意事项：**
- run_learned_mpc_eval.py 使用 learned rollout 做状态更新，不是 MuJoCo 物理仿真
- success_rate 不能作为论文主结果
- 需要扩展到更多 episode 验证稳定性
- 需要数据质量审计决定是否重采数据

**下一步：**
1. 扩展 learned MPC smoke 到 3-10 episodes
2. 数据质量审计
3. 决定是否训练三个模型或重采数据
