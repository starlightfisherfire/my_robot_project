# ChatGPT Analysis Package V2

**项目:** Paper1 - Causal Representation for Push Manipulation  
**日期:** 2026-05-24  
**目的:** V2 闭环实验诊断，寻求改进建议

---

## 快速开始

1. **先读 `SUMMARY_FOR_CHATGPT.md`** — 完整进度和问题
2. **再读 `docs/closed_loop_action_planner_fix_report.md`** — V2 实验报告
3. **查看 `results/offline_eval_summary.json`** — 离线评估数据
4. **查看 `core_files/`** — 核心代码实现

---

## 核心发现

| 模型 | 离线 10-step RMSE | 闭环改善 | 闭环成功 |
|------|-------------------|----------|----------|
| flat | 3.10 mm | ✅ 1.2cm | ❌ |
| object_centric | 2.04 mm | ❌ | ❌ |
| causality_aware | **0.59 mm** | ❌ | ❌ |

**矛盾:** Causality_aware 离线最好但闭环最差。

---

## 关键问题

1. 如何让 learned model 引导 EE 接近物体？
2. Causality-aware 为什么闭环最差？
3. Oracle MPPI 92% vs Learned CEM ~0%，差距来自哪里？
4. 下一步最有 ROI 的方向是什么？

---

## 目录结构

```
chatgpt_analysis_v2/
├── README.md                      # 本文件
├── SUMMARY_FOR_CHATGPT.md         # 完整总结
├── core_files/                    # 核心代码（12个文件）
├── configs/                       # 训练配置（4个文件）
├── results/                       # 评估结果（6个文件）
└── docs/                          # 文档（4个文件）
```

---

**最后更新:** 2026-05-24 13:00 GMT+8
