# 文献监控 Pipeline 设计

> 为 Agentic WM Position Paper 服务  
> 目标: 持续监控相关文献，自动摘要，对比我们的框架，发现 novelty threat

## 架构

```
arXiv API (每2天查询)
    ↓
关键词过滤 → 候选论文列表
    ↓
AI 摘要: 提取 (what, how, gap vs us)
    ↓
存储到 literature/ 目录
    ↓
差异分析: 与 Agentic WM 框架对比
    ↓
Alert: 如果高度重叠 → 通知
```

## 关键词查询

```
查询组 A: World Model + Robotics
  "world model" AND (robot OR manipulation OR embodiment)
  
查询组 B: Video Foundation Model + Action
  ("video prediction" OR "video foundation model") AND (action OR control)
  
查询组 C: Agent / Tool Use + Embodiment
  ("tool use" OR "function calling" OR "agent") AND (robot OR world model)
  
查询组 D: Pretrain + Fine-tune + Video + Robot
  ("pretrain" OR "foundation model") AND "video" AND "robot"
```

## 摘要模板

对每篇候选论文，AI 自动生成:

```
## [Title]
- arXiv: [id]
- Date: [date]
- Authors: [authors]

### What they did
[一句话核心贡献]

### Architecture
[简化架构描述]

### Key difference from Agentic WM
[如果做了解耦: 说明] / [如果没做: 指出差异]

### Relevance score: ★★★★★ (0-5)
[对我们的 relevance 判断]
```

## 实现方案

### 方案 A: OpenClaw Cron Job (推荐)

```bash
# 每2天自动运行
cron job → isolated session → 
  1. query arxiv API
  2. fetch new papers
  3. AI summarize
  4. write to literature/inbox/
  5. compare with our framework
  6. alert if high overlap
```

### 方案 B: 手动触发

你或我主动运行一个脚本，做增量查询。

## 产出物

```
literature/
├── README.md              # 监控概览
├── pipeline.py            # 查询+摘要脚本
├── queries.txt            # 关键词列表
├── inbox/                 # 新论文摘要 (AI生成)
│   ├── 2026-05-20_vjepa3.md
│   ├── 2026-05-22_deltaworld2.md
│   └── ...
├── processed/             # 已阅读/已引用的
│   ├── cjepa.md
│   ├── dreamerv3.md
│   └── ...
└── matrix.md              # 对比矩阵 (持续更新)
```

## 对比矩阵 (核心产物)

| Paper | Frame-as-Token? | Decoupled Action? | Tool Use? | Two-Layer Pred? | Overlap |
|-------|----------------|-------------------|-----------|-----------------|---------|
| C-JEPA | ❌ (slots) | ❌ | ❌ | ❌ | Low |
| V-JEPA2 | ❌ (patch) | ❌ | ❌ | ❌ | Low |
| DreamerV3 | ❌ (RSSM) | ❌ | ❌ | ❌ | Low |
| DeltaWorld | ✅ (delta) | ❌ | ❌ | ❌ | Medium |
| GR-1 | ❌ (patch) | ❌ | ❌ | ❌ | Low |
| RT-2 | N/A (VLM) | ❌ | ✅ (LLM-based) | ❌ | Low |
| ... | | | | | |
```

这个矩阵直接放进 position paper 的 Related Work 章节。

---

要我直接开始搭建这个 pipeline 吗？先从手动脚本开始，验证可行性后再加 cron 自动化。
