# Agentic WM + Diffusion Policy — MVP 架构设计

> 日期: 2026-05-22 00:30  
> 作者: brucewu + Paper1 助手  
> 定位: Agentic World Model + Diffusion Action Decoder 最小可跑实现  
> 前置: FTWM 最终架构 (2026-05-20), Agentic WM Position Paper (2026-05-20)  
> 状态: ✅ Draft v0.2 — Added Related Work (DreamZero + VLA/WM comparison)

---

## 0. 前置：π0 数据可用性分析

### π0 训练数据构成

| 数据源 | 规模 | 是否开源 |
|--------|------|----------|
| **PI 专有数据** | 10,000+ 小时灵巧操作，7 种机器人，68 个任务 | ❌ **不公开** |
| **OXE (Open X-Embodiment)** | 100 万+ trajectory，22 种机器人，500+ 技能 | ✅ 开源 |
| **Bridge v2** | 宽数据集，桌面操作 | ✅ 开源 |
| **DROID** | 大规模灵巧操作 | ✅ 开源 |
| **互联网图文预训练** | SigLIP 预训练数据 | ✅ 部分开源 |

### 结论：基线对比策略

```
可以做：
✅ 使用 OXE + Bridge v2 公开数据训练 Agentic WM
✅ 使用 openpi 仓库的 π0 checkpoint（仅权重）做直接推理对比
✅ 在相同 OXE 子集上 fair comparison（都用公开数据微调）

不能做：
❌ 复现 π0 完整的 10,000 小时训练（专有数据不公开）
❌ 直接声称"在相同数据量下优于 π0"

论文策略：
- 使用 OXE/Bridge 公开数据，在 controlled setting 下对比
- 标明 π0 完整训练用了额外专有数据
- 强调 Agentic WM 的数据优势：Base Model 用视频预训练（不需要动作标注）
```

---

## 1. MVP 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                      Agentic WM + Diffusion                        │
│                                                                    │
│  ┌─────────────┐                                                  │
│  │ Frame_t      │                                                 │
│  │ (H×W×3)      │                                                 │
│  └──────┬───────┘                                                 │
│         │                                                          │
│  ┌──────▼───────┐                                                  │
│  │ FrameEncoder │  ← DINOv2 / SigLIP (冻结)                       │
│  │              │     每帧 → 1 个全局向量 v_t ∈ ℝ^D               │
│  └──────┬───────┘                                                  │
│         │                                                          │
│  ┌──────▼──────────────────────────────────────────────┐          │
│  │           Causal Transformer (L 层)                   │          │
│  │                                                       │          │
│  │  [v_0][v_1][v_2][v_3][v_4][txt]                      │          │
│  │    │    │    │    │    │    │                         │          │
│  │    ▼    ▼    ▼    ▼    ▼    ▼                         │          │
│  │  │  self-attention (Prefix-LM 模式) │                 │          │
│  │  │  视觉 token 之间: 双向 attention  │                 │          │
│  │  │  文本 token: causal, 可见全部视觉│                  │          │
│  │                                                       │          │
│  │  输出: h_t (全局隐状态)                                │          │
│  └──────┬──────────────────────────────────────────────┘          │
│         │                                                          │
│    ┌────┴────────────┐                                            │
│    │                  │                                            │
│  ┌─▼──────────┐  ┌───▼──────────────────┐                         │
│  │ Base Head  │  │ Action-Cond Head     │                         │
│  │            │  │                      │                         │
│  │ h_t → MLP  │  │ (h_t, a_lift) → MLP  │                         │
│  │   ↓        │  │   ↓                  │                         │
│  │ pred_v     │  │ pred_State_{t+1}     │                         │
│  │ _{t+1}     │  │                      │                         │
│  │            │  │ L_action =            │                         │
│  │ L_base =   │  │ MSE(pred, actual)     │                         │
│  │ MSE(pred,  │  │                      │                         │
│  │ actual)    │  │ Δ = pred_v - pred_S   │                         │
│  └────────────┘  │  (因果效应)           │                         │
│                  └──────────┬───────────┘                         │
│                             │                                      │
│                  ┌──────────▼───────────┐                         │
│                  │   Tool Interface      │                         │
│                  │                      │                         │
│                  │  条件于:              │                         │
│                  │  · h_t (WM 隐状态)    │                         │
│                  │  · Δ (因果效应)       │                         │
│                  │  · State_t^action     │                         │
│                  │                      │                         │
│                  │  ┌──────────────────┐ │                         │
│                  │  │ Diffusion Policy │ │  ← 精细连续操作        │
│                  │  │ (DDPM/DDIM)      │ │                         │
│                  │  │ 在动作空间去噪    │ │                         │
│                  │  └────────┬─────────┘ │                         │
│                  │           ▼           │                         │
│                  │  [Δx,Δy,Δz,Δr,Δp,    │                         │
│                  │   Δy,gripper]        │                         │
│                  └──────────────────────┘                         │
│                             │                                      │
│                             ▼                                      │
│                      Robot Execution                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据流详解

### 2.1 前向传播 (推理)

```
Step 1: 帧编码
  Frame_t → DINOv2 → v_t ∈ ℝ^768
  (N 帧历史 → N 个向量)

Step 2: 序列处理
  [v_0, v_1, ..., v_t] + PosEnc
      ↓ Causal Transformer (L=4, d_model=512)
  h_t ∈ ℝ^512  (全局隐状态)

Step 3: 双重预测
  Base Head:     h_t → MLP → pred_v_{t+1}     (自然演化)
  Action Head:   (h_t, A_{t-1}) → MLP → pred_State_{t+1}  (干预后)

Step 4: 动作生成 (Diffusion Policy)
  Condition: concat(h_t, Δ) ∈ ℝ^(512 + 512) = ℝ^1024
      ↓
  Diffusion: 从噪声 x_K 逐步去噪 K 步
      ↓
  x_0 = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]  ∈ ℝ^7

Step 5: 执行
  Robot.execute(x_0)
  World → Frame_{t+1} → goto Step 1
```

### 2.2 训练时

```
训练分两个阶段：

Stage 1: Base Model 预训练 (纯视频，无动作)
  - 输入: 视频序列 {Frame_0, ..., Frame_T}
  - 目标: 预测下一帧编码
  - Loss: L_base = MSE(pred_v_{t+1}, Encoder(Frame_{t+1}))
  - 梯度回传: Causal Transformer + Base Head
  - Encoder 冻结

Stage 2: 联合训练 (机器人数据)
  - 输入: 视频序列 + 动作序列 + 文本指令
  - 目标:
    a) Action Head: MSE(pred_State_{t+1}, actual_State_{t+1})
    b) Diffusion Policy: 动作去噪 loss (standard diffusion loss)
  - 梯度回传: Causal Transformer + Action Head + Diffusion
  - Base Head 可选择性冻结或微调
```

---

## 3. 关键模块规格

### 3.1 Frame Encoder

```
选择: DINOv2 ViT-B/14 (推荐) 或 SigLIP
原因: 
  - 已在大量视觉数据上预训练，泛化好
  - 输出 768-d (ViT-B)，计算量适中
  - π0 用 SigLIP，两者性能接近，选 DINOv2 便于对比差异化

压缩方式:
  每帧 → ViT → CLS token 或 mean pool → v_t ∈ ℝ^768
  
  (不保留 patch tokens，彻底压缩为 1 个向量)
  → 这就是"Frame-as-Token"的核心设计选择

冻结/微调:
  Stage 1: 冻结 (只训 Transformer)
  Stage 2: 可选微调 (lr=1e-6, 极低学习率)
```

### 3.2 Causal Transformer

```
参数:
  d_model:     512
  n_heads:     8
  n_layers:    4 (MVP 阶段，可扩展)
  dim_feedforward: 2048
  max_seq_len: 16 (5 帧历史 + 1 文本 token)
  dropout:     0.1

Attention 模式: Prefix-LM
  ┌─────────────────────────────────────┐
  │          v_0  v_1  v_2  v_3  v_4  txt│
  │  v_0    [●    ●    ●    ●    ●  │ ✗ ]│
  │  v_1    [●    ●    ●    ●    ●  │ ✗ ]│
  │  v_2    [●    ●    ●    ●    ●  │ ✗ ]│  视觉 = Prefix
  │  v_3    [●    ●    ●    ●    ●  │ ✗ ]│  (双向)
  │  v_4    [●    ●    ●    ●    ●  │ ✗ ]│
  │        ─────────────────────────┼────│
  │  txt    [●    ●    ●    ●    ●  │ ● ]│  文本 = Suffix
  └─────────────────────────────────────┘  (causal, 可见全部视觉)

理由:
  - 视觉 token 之间双向 → 充分融合 N 帧时空信息
  - 文本 causal 但能看到全部视觉 → 任务条件于完整的世界理解
  - 从 π0 借鉴，已被验证有效
```

### 3.3 Base Head (自然演化预测)

```
输入:  h_t (Transformer 最后位置输出)
结构:  Linear(512, 256) → ReLU → Linear(256, 768)
输出:  pred_v_{t+1} ∈ ℝ^768
Loss:  L_base = MSE(pred_v_{t+1}, actual_v_{t+1})

含义: 预测"如果我不干预，世界下一步会变成什么样"
```

### 3.4 Action-Cond Head (干预预测)

```
输入:  (h_t, A_{t-1})
      其中 h_t ∈ ℝ^512, A_{t-1} ∈ ℝ^256 (lifted action)

结构:  Linear(768, 256) → ReLU → Linear(256, 512)
输出:  pred_State_{t+1} ∈ ℝ^512
Loss:  L_action = MSE(pred_State_{t+1}, actual_State_{t+1})

含义: 预测"如果我执行了这个动作，世界会变成什么样"

因果效应: Δ = pred_v_{t+1} - pred_State_{t+1}  (都在 512/768 维空间共享)
         → 通过投影矩阵对齐后计算差值
```

### 3.5 Action Lift (动作升维)

```
目的: 将低维动作投影到与 State_t 同一语义空间

输入:  a_t = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip] ∈ ℝ^7

结构:  可学习的 Embedding
      Linear(7, 128) → ReLU → Linear(128, 256)
输出:  A_t ∈ ℝ^256

选项:
  - Simple MLP (MVP 阶段)
  - FiLM conditioning (后续，更 expressive)
  - Cross-attention with env context (后续，更 rich)

类比: 这就是 LLM 中 word embedding 的等价物
      离散词 → embedding 向量，连续动作 → lifted 向量
```

### 3.6 Diffusion Action Decoder

```
类型: DDPM (Denoising Diffusion Probabilistic Model)
      或 DDIM (更快推理)

动作空间维度: d_a = 7
  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
  每维归一化到 [-1, 1]

去噪步数: K = 100 (训练时), K = 10-20 (DDIM 推理加速)

Conditioning (条件于 WM 输出):
  c = concat(h_t, Δ_proj) ∈ ℝ^(512 + 512) = ℝ^1024
  其中 Δ_proj = Linear(512/768, 512)(Δ) 确保维度一致

去噪网络: UNet1D 风格
  ┌────────────────────────────────────────┐
  │  x_k ∈ ℝ^7  (当前噪声动作)             │
  │  step k ∈ ℝ  (时间步编码)              │
  │  c ∈ ℝ^1024 (conditioning)            │
  │         │                               │
  │  ┌──────▼──────────────────────┐        │
  │  │  FiLM-conditioned MLP blocks │        │
  │  │                              │        │
  │  │  Block i:                    │        │
  │  │    Linear(7→64) → FiLM(c)   │        │
  │  │    → ReLU → Linear(64→64)   │        │
  │  │    → FiLM(c) → ReLU         │        │
  │  │                              │        │
  │  │  共 4 个 block               │        │
  │  └──────────────────────────────┘        │
  │         │                               │
  │         ▼                               │
  │  预测的噪声 ε_θ(x_k, k, c) ∈ ℝ^7        │
  └────────────────────────────────────────┘

Loss: L_diff = ||ε - ε_θ(x_k, k, c)||^2
     其中 x_k = √(ᾱ_k)·x_0 + √(1-ᾱ_k)·ε
     ε ~ N(0, I), x_0 = 真实动作

为什么在动作空间做 Diffusion 可行？
  动作空间只有 7 维，远小于像素空间 (百万维)
  → 去噪步数可以很少 (K=10-20)
  → 推理速度快 (毫秒级)
  → 这正是 π0 成功的关键原因
```

---

## 4. 训练策略

### 4.1 Phase 0: 数据准备

```
【选项 A】使用公开数据 (论文 baseline)

数据源:
  1. OXE 子集 (Bridge v2, Fractal, Kuka) — ~50K trajectories
  2. DROID (如果有访问) — 大规模操作数据
  3. MuJoCo 仿真渲染视频 — 不受限，可以生成任意多

预处理:
  - 帧: resize → 224×224 → DINOv2 → v_t ∈ ℝ^768
  - 动作: 归一化到 [-1, 1]
  - 文本: tokenize → 1 个 pooling token 或简短序列
  - 序列长度: N=5 帧历史窗口

【选项 B】使用 MuJoCo 仿真 (快速验证)

优势:
  - 无限数据生成
  - 完全可控环境
  - 适合做 ablation study
  - OOD 测试容易设计

建议路径: 
  先用 MuJoCo 验证架构 → 再用 OXE 做公开数据对比
```

### 4.2 Phase 1: Base Model 视频预训练

```
目标: 无动作标注，纯视频学习物理演化

数据: 
  - MuJoCo 渲染视频 (球滚动、碰撞、堆叠等场景)
  - 或 OXE 视频数据 (忽略动作标注)
  - 或 YouTube 物体交互视频

超参数:
  - batch_size: 64
  - seq_len: 16 (每次取 16 帧窗口)
  - lr: 1e-4 (Transformer), Encoder 冻结
  - optimizer: AdamW
  - epochs: 100-200 (取决于数据量)

训练循环:
  for batch in video_loader:
      frames = batch['frames']  # [B, T+1, 3, 224, 224]
      v_0_T = encoder(frames[:, :T])    # [B, T, 768]
      v_target = encoder(frames[:, T])  # [B, 768]
      
      h_T = causal_transformer(v_0_T)   # [B, 512]
      pred_v = base_head(h_T)           # [B, 768]
      
      loss = MSE(pred_v, v_target)
      loss.backward()
      
  只更新: Causal Transformer + Base Head
  不更新: DINOv2 Encoder

验证指标:
  - next-frame encoding prediction error
  - 可视化: 用解码器重建 pred_v + 与实际帧对比
```

### 4.3 Phase 2: 联合训练 (加入 Action + Diffusion)

```
目标: 在机器人数据上训练动作生成

数据: MuJoCo 或 OXE + 动作标注

超参数:
  - batch_size: 32
  - lr_transformer: 1e-5 (极低，保护预训练知识)
  - lr_diffusion: 1e-4 (从头训练)
  - optimizer: AdamW
  - diffusion_steps: K=100
  - epochs: 50-100

训练循环:
  for batch in robot_loader:
      frames = batch['frames']         # [B, T+1, 3, 224, 224]
      actions = batch['actions']       # [B, T, 7]
      texts = batch['instructions']    # [B, 1] (任务文本)
      
      # Step 1: WM 前向
      v_0_T = encoder(frames[:, :T])   # [B, T, 768]
      v_target = encoder(frames[:, T]) # [B, 768]
      
      h_T = causal_transformer(v_0_T + [text_tokens])  # [B, 512]
      
      # Step 2: Base prediction
      pred_v = base_head(h_T)          # [B, 768]
      loss_base = MSE(pred_v, v_target)
      
      # Step 3: Action prediction
      A_prev = action_lift(actions[:, -1])   # [B, 256]
      pred_state = action_head(h_T, A_prev)  # [B, 512]
      actual_state = h_T  # 当前 state，或用下一帧的 h 做 target
      loss_action = MSE(pred_state, actual_state)
      
      # Step 4: Diffusion action generation
      a_target = actions[:, -1]        # [B, 7] (当前步的真实动作)
      Δ_proj = project(Δ)             # 因果效应 cond
      c = concat(h_T, Δ_proj)         # [B, 1024]
      
      # DDPM training
      k ~ Uniform(1, K)
      ε ~ N(0, I)
      x_k = sqrt(alpha_bar_k) * a_target + sqrt(1-alpha_bar_k) * ε
      ε_pred = diffusion_net(x_k, k, c)
      loss_diff = MSE(ε, ε_pred)
      
      # Total loss
      loss = loss_base + loss_action + loss_diff
      loss.backward()
```

### 4.4 Phase 3: 推理闭环 (可选)

```
真实闭环测试:
  1. Camera → Frame_t
  2. Encoder → v_t
  3. [v_0...v_t, txt] → Transformer → h_t
  4. Base Head → pred_v_{t+1}
  5. Action Head → pred_State_{t+1}
  6. Δ = pred_v - pred_State
  7. Diffusion(c=h_t, Δ) → a_t
  8. Robot.execute(a_t)
  9. 等待下一帧 → goto 1

评估: task success rate, trajectory smoothness, OOD generalization
```

---

## 5. 与 π0 的系统对比

| 维度 | π0 (Physical Intelligence) | Agentic WM + Diffusion (本文) |
|------|---------------------------|------------------------------|
| **视觉 Encoder** | SigLIP (保留 patch tokens) | DINOv2 (压缩为 1 个向量/帧) |
| **Backbone** | Gemma-27B LLM | Causal Transformer (L=4, d=512) |
| **压缩率** | 256 tokens/帧 (patch 级) | **1 token/帧** (帧本质) |
| **物理建模** | LLM attention 隐式捕获帧间变化 | **显式预测** State_{t+1} (base + action-cond) |
| **动作生成** | Flow Matching (单路 condition) | Diffusion Policy (双路: h_t + Δ) |
| **因果效应** | 无显式 Δ | **Δ = pred_v - pred_State** |
| **文本集成** | Prefix-LM, 文本在后 | Prefix-LM, 文本在后 (同 π0) |
| **训练数据** | 10K hrs 专有 + OXE | OXE/Bridge 公开，或 MuJoCo 仿真 |
| **预训练** | LLM 图文预训练 → SigLIP | **Base Model 视频预训练 (无动作)** |
| **数据效率** | 需要大量动作标注 | Base 不用动作标注，只需 Action 阶段用 |
| **实施例无关** | 否 (LLM 专有) | **是** (换机器人只换 Tool 定义) |

---

## 6. Related Work

### 6.1 DreamZero / World Action Model (NVIDIA, 2025-2026)

DreamZero 是当前最接近本文思路的工作。它是一个 14B 参数的视频 Diffusion Transformer，在预训练视频扩散模型基础上联合预测未来视频帧和动作序列，实现了 zero-shot 泛化能力（是传统 VLA 的 2 倍以上）。

**核心相似点：**
- 共享"用视频预训练学物理，不依赖 LLM 语义空间"的哲学
- 动作和视觉联合建模
- 强调"理解世界再行动"

**关键差异：**

| 维度 | DreamZero | Agentic WM (本文) |
|------|-----------|-------------------|
| **预测空间** | 像素空间 (视频 Diffusion) | 压缩嵌入空间 (Frame-as-Token) |
| **参数规模** | 14B (巨大) | ~数十M (轻量) |
| **动作与模型的关系** | 动作在 Diffusion 内部联合生成 | **Tool Interface 解耦** |
| **因果建模** | 无显式因果效应 | **Δ = pred_v - pred_state** |
| **实施例独立性** | 动作空间绑定在模型内 | 换机器人只换 Tool 定义 |
| **训练哲学** | 端到端大模型 | **三阶段 LLM 路径** (pretrain→tool-use→agentic) |

DreamZero 的存在对本工作是一个重要验证：NVIDIA 级别的团队也在押注"扔掉 LLM、用视频做物理理解"这条路。但 DreamZero 选择了超大模型 + 像素空间生成的重路径，而本文选择轻量压缩 + Tool Interface 解耦的轻路径——两种策略的互补性值得深入比较。

### 6.2 视频世界模型 + 机器人控制

| 工作 | 思路 | 与本文差异 |
|------|------|-----------|
| **Ctrl-World** (2025) | 可控世界模型生成操作视频 | 目标是视频渲染，不含动作执行 |
| **Genie Envisioner** (2025) | 视频生成 + 动作解码器 | 没有 Frame-as-Token 压缩，无显式 Δ |
| **Human2Robot** (2025) | 视频预测引导动作解码 | 预测与执行分离，无统一因果框架 |
| **Video Prediction Policy** (2025) | 用视频预测表征条件化策略 | 缺少 Tool Interface 和 Layer 2 预测 |
| **Amplify** (2025) | 无动作视频学习运动先验 | 缺动作生成和执行模块 |

### 6.3 VLA 模型

| 工作 | 思路 | 与本文差异 |
|------|------|-----------|
| **π0** (Physical Intelligence, 2024-2025) | LLM backbone + Flow Matching Action Expert | LLM 语义空间，非物理空间；无 Tool Interface |
| **RT-2** (Google, 2024) | VLM → 离散动作 token | 动作离散化，无 Diffusion |
| **OpenVLA** (2024-2025) | 开源 VLA，fine-tune 范式 | 纯模仿学习，无世界模型 |
| **Diffusion-VLA** (ICML 2025) | 统一 Diffusion + 自回归 | 仍依赖 LLM backbone |

### 6.4 本文定位

```
                        VLA 路线                    WM 路线
                        ────────                    ───────
  RT-2 ──→ OpenVLA ──→ π0                     Dreamer ──→ IRIS ──→ C-JEPA
                │                                      │
                └──────── 本文 ────────────────────────┘
                      Agentic WM + Diffusion
                      
              融合 VLA 的 Diffusion 动作生成
               + WM 的显式状态预测
               + 独创的 Tool Interface 解耦
               + LLM 发展路径作为架构蓝图
```

本文的核心贡献不是单一技术创新，而是**融合两个领域各自最佳实践**——VLA 的 Diffusion 动作生成 + WM 的显式物理预测——并用 **Tool Interface** 和 **LLM 发展路径哲学** 将它们统一为一个可扩展框架。

---

## 7. 实验设计 (消融与对比)

### 7.1 核心实验

| 实验 | 条件 | 目的 |
|------|------|------|
| **E1: Base Only** | h_t → pred_v_{t+1} | 验证视频预训练学到的物理知识 |
| **E2: Single Cond** | h_t → Diffusion → action | 作为 baseline (对标 π0 单路) |
| **E3: Dual Cond** | (h_t, Δ) → Diffusion → action | **主要假设**：双路条件优于单路 |
| **E4: Triple Cond** | (h_t, Δ, pred_State) → Diffusion | 探索信息上限 |
| **E5: Action Lift 消融** | 有无 Action Lift 模块 | Lift 的作用 |
| **E6: Encoder 冻结 vs 微调** | DINOv2 freeze/finetune | 视觉特征对下游的影响 |
| **E7: Transformer 深度** | L=2,4,8 | Scaling 行为 |

### 7.2 Baseline 对比

```
1. π0 (openpi checkpoint, OXE fine-tuned) — 最强 baseline
2. Diffusion Policy (无 WM, 纯 CNN + Diffusion) — 消融 WM 必要性
3. ACT (Action Chunk Transformer, 无 Diffusion) — 消融 Diffusion
4. RT-2 风格 (离散动作 token) — 动作表示消融
```

### 7.3 评估指标

```
- 动作预测误差: MSE(â_t, a_t)
- 任务成功率: success_rate (%)
- OOD 泛化: 新布局/新物体配置下的成功率
- 轨迹平滑度: jerk = Σ|a_{t+1} - a_t|
- 推理速度: fps (包括 Diffusion 去噪)
```

---

## 8. 代码结构 (建议)

```
project_root/
├── configs/
│   ├── base_pretrain.yaml
│   ├── joint_finetune.yaml
│   └── inference.yaml
├── src/
│   ├── encoders/
│   │   └── dino_encoder.py          # DINOv2 帧编码器
│   ├── transformer/
│   │   ├── prefix_lm.py             # Prefix-LM attention
│   │   └── causal_wm.py             # Causal World Transformer
│   ├── heads/
│   │   ├── base_head.py             # 自然演化预测
│   │   ├── action_head.py           # 干预预测
│   │   └── action_lift.py           # Action Embedding
│   ├── diffusion/
│   │   ├── unet_1d.py               # 1D UNet 去噪网络
│   │   ├── diffusion_process.py     # DDPM 前向/反向过程
│   │   └── diffusion_policy.py      # Diffusion 策略封装
│   ├── data/
│   │   ├── oxe_dataset.py           # OXE 数据加载
│   │   ├── mujoco_dataset.py        # MuJoCo 仿真数据
│   │   └── video_dataset.py         # 纯视频数据 (Base 预训练)
│   └── training/
│       ├── train_base.py            # Phase 1: Base Model 预训练
│       ├── train_joint.py           # Phase 2: 联合训练
│       └── eval.py                  # 推理 + 评估
├── experiments/
│   └── mujoco_push_t/               # MuJoCo Push-T 实验
├── checkpoints/
└── logs/
```

---

## 9. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| **Δ 信号太弱** | 中 | 增大 loss 权重，或用 cosine similarity 替代 MSE |
| **Diffusion 推理太慢** | 低 | 动作空间 7 维，DDIM 10 步即可，预估 < 5ms |
| **视频预训练数据不够** | 中 | MuJoCo 仿真可无限生成；YouTube 采集视频 |
| **Base Model 学不到物理** | 低 | next-frame prediction 是 well-established objective |
| **与 π0 公平对比困难** | 中 | 在相同 OXE 子集上对比，标注数据量差异 |
| **Prefix-LM 实现复杂** | 低 | PyTorch 自定义 attention mask 即可实现 |

---

## 10. 里程碑路线图

```
Week 1-2: Phase 0 — 数据准备 + 代码骨架
  [ ] DINOv2 encoder 封装
  [ ] MuJoCo Push-T 环境搭建 + 数据生成
  [ ] Prefix-LM Transformer 实现
  [ ] Base Head + Action Head 实现

Week 3-4: Phase 1 — Base Model 预训练
  [ ] 纯视频数据加载 pipeline
  [ ] Base Model 训练 (next-frame prediction)
  [ ] 验证: loss curve, next-frame 预测质量

Week 5-6: Phase 2 — 联合训练 + Diffusion
  [ ] Action Lift 模块
  [ ] 1D UNet Diffusion 网络
  [ ] DDPM training loop
  [ ] 联合训练 (Base + Action + Diffusion)

Week 7-8: Phase 3 — 验证 + 消融
  [ ] Simulation 闭环测试
  [ ] 核心消融实验 (E1-E7)
  [ ] Baseline 对比 (Diffusion Policy, ACT)
  [ ] OOD 泛化测试

Week 9+: 论文写作 + 公开数据实验
  [ ] OXE 数据实验
  [ ] π0 对比 (openpi checkpoint)
  [ ] 论文初稿
```

---

## A. 附录：动作空间定义

```
动作向量 a_t ∈ ℝ^7:

  末端执行器位姿变化 (相对于当前位姿):
  [0] Δx      ∈ [-0.05, 0.05] m    前后
  [1] Δy      ∈ [-0.05, 0.05] m    左右
  [2] Δz      ∈ [-0.05, 0.05] m    上下
  [3] Δroll   ∈ [-π/8, π/8] rad   绕 x 轴旋转
  [4] Δpitch  ∈ [-π/8, π/8] rad   绕 y 轴旋转
  [5] Δyaw    ∈ [-π/2, π/2] rad   绕 z 轴旋转
  [6] gripper ∈ {0, 1}             夹爪 (0=闭合, 1=张开)

归一化: 每维缩放到 [-1, 1]
  Δx_norm = Δx / 0.05
  Δyaw_norm = Δyaw / (π/2)
  gripper: {0, 1} 保持不变或缩放到 [-1, 1]

可以选择的其他动作空间:
  - Joint-space: [θ_1, ..., θ_7] (7 个关节角度)
  - Velocity control: [v_1, ..., v_7] (关节速度)
  - 绝对位姿: [x, y, z, r, p, y, g] (世界坐标系)
  
推荐 MVP 使用 delta-pose，因为:
  - 与 π0 一致，便于对比
  - 相对运动更容易学习
  - Diffusion 在紧凑动作空间工作最好
```

---

## B. 附录：Diffusion 噪声调度

```
使用 cosine schedule (Nichol & Dhariwal, 2021):

  ᾱ_k = cos((k/K + s) / (1 + s) · π/2)^2

  其中 s = 0.008 (小偏移防止 ᾱ_0 过小)

DDIM 推理加速:
  K=100 (训练) → K_infer=10 (推理)
  加速比 ≈ 10x，质量几乎不降 (动作空间足够小)

采样过程 (DDIM, K_infer=10):
  for k from K_infer down to 1:
    x_{k-1} = √(ᾱ_{k-1}) · (x_k - √(1-ᾱ_k)·ε_θ(x_k, k, c)) / √(ᾱ_k)
            + √(1-ᾱ_{k-1}) · ε_θ(x_k, k, c)
  output x_0
```

---

*Draft v0.1 — 2026-05-22 00:30 — 等待 brucewu 审阅确认*
