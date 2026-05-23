# ACWM — Action-Centric World Model 架构图

> 工作名称: ACWM | 别名候选: OCAM / CAWM / ART / GATOR  
> 日期: 2026-05-19 | 作者: brucewu

---

## 总览

```mermaid
flowchart TB
    subgraph INPUT["输入层"]
        direction LR
        CAM["📷 视觉感知器<br/>(Paper2: 从RGB/D学习<br/>Paper1: 结构化状态直接给)"]
        S1["[obj_R, obj_A, obj_B, ...] × T帧"]
        GE["🎯 Goal Encoder<br/>(自然语言指令 → 目标状态向量)"]
        G1["goal_embedding"]
    end

    subgraph TRANSFORMER["🧠 Object-Centric Causal Transformer"]
        direction TB
        PE["Positional + Object + Temporal Embedding"]
        
        subgraph BLOCK1["Transformer Block 1..L"]
            SA["Multi-Head Self-Attention<br/>━━━━━━━━━━━━━━━━<br/>• 跨时间: 历史帧 ↔ 当前帧 (隐式时序预测)<br/>• 跨物体: obj_i ↔ obj_j (隐式因果交互)<br/>• 差距感知: state_rep ↔ goal_rep<br/>━━━━━━━━━━━━━━━━<br/>📊 Causal-Aware: attention 权重涌现因果结构"]
        end
        
        FFN["Feed-Forward Network<br/>(隐式状态预测 + 动作后果评估)"]
        SC["自包含表征<br/>Self-Contained Representation<br/>━━━━━━━━━━━━<br/>融合了所有物体、所有时刻、<br/>目标差距的因果交互信息"]
    end

    subgraph OUTPUT["输出层 — 自回归动作生成"]
        direction TB
        AH["Action Head (MLP)"]
        A0["a₀ (大步)"]
        A1["a₁ (中步)"]
        A2["a₂ (微调)"]
        STOP["<STOP>"]
        SEQ["变长动作序列: [a₀, a₁, a₂, <STOP>]"]
    end

    subgraph ENV["🌍 环境闭环"]
        EXEC["⚡ 执行器<br/>(仿真/真实机械臂)"]
        WORLD["物理世界状态更新<br/>s_{t+1} = T(s_t, a_t)"]
        PERC["感知器编码新状态"]
    end

    INPUT --> TRANSFORMER
    TRANSFORMER --> OUTPUT
    OUTPUT --> ENV
    ENV -->|"新状态帧追加到序列"| INPUT

    style INPUT fill:#e1f5fe
    style TRANSFORMER fill:#fff3e0
    style OUTPUT fill:#e8f5e9
    style ENV fill:#fce4ec
```

---

## 核心闭环 (Action-Perception Loop)

```mermaid
sequenceDiagram
    participant W as 🌍 物理世界
    participant P as 📷 感知器
    participant WM as 🧠 ACWM
    participant E as ⚡ 执行器

    Note over W,E: 初始状态
    P->>P: 编码当前世界 → Object-Centric tokens
    P->>WM: [obj_R(t-k:t), obj_A..., obj_B...] + goal
    
    rect rgb(255, 243, 224)
        Note over WM: Transformer 内部隐式处理
        WM->>WM: ① Attention: 物体间因果交互
        WM->>WM: ② 隐式预测: "执行a会对各物体产生什么后果"
        WM->>WM: ③ 差距计算: "各物体离goal还有多远"
        WM->>WM: ④ 决策: "需要多少步? 每步做什么?"
    end
    
    WM->>E: 动作序列 [a₀, a₁, a₂, <STOP>]
    E->>W: 执行 a₀
    W->>W: 物理演化
    P->>P: 编码新状态
    P->>WM: [obj_R(t-k+1:t+1), obj_A..., obj_B...] + goal
    
    Note over WM: 闭环: 重新决策, 可修正或继续
    
    WM->>E: [a₁', a₂'] (可能修正)
    E->>W: 执行 a₁'
    
    Note over W,WM: ... 循环直到 STOP 或 goal 达成 ...
    
    W->>W: 🎯 所有物体到达 goal!
```

---

## Object-Centric Token 结构

```mermaid
flowchart LR
    subgraph FRAME_T["帧 t"]
        O1["obj_R<br/>位置,姿态,关节角"]
        O2["obj_A<br/>位置,速度,形状"]
        O3["obj_B<br/>位置,速度,形状"]
        O4["obj_C<br/>位置,速度,形状"]
    end
    
    subgraph FRAME_T1["帧 t-1"]
        O1p["obj_R"]
        O2p["obj_A"]
        O3p["obj_B"]
        O4p["obj_C"]
    end
    
    subgraph FRAME_TK["帧 t-k"]
        O1k["obj_R"]
        O2k["obj_A"]
        O3k["obj_B"]
        O4k["obj_C"]
    end

    FRAME_TK --> FRAME_T1 --> FRAME_T

    style O1 fill:#ffcdd2
    style O2 fill:#c8e6c9
    style O3 fill:#bbdefb
    style O4 fill:#fff9c4
```

---

## Causal-Aware Attention 示意

```mermaid
flowchart TB
    subgraph ATTENTION["Attention Matrix Q·K^T"]
        direction LR
        subgraph COL["Query (当前)"]
            Q_R["obj_R(t)"]
            Q_A["obj_A(t)"]
            Q_B["obj_B(t)"]
        end
        subgraph ROW["Key (历史)"]
            K_Rp["obj_R(t-1)"]
            K_Ap["obj_A(t-1)"]
            K_Bp["obj_B(t-1)"]
            K_ACT["action(t-1)"]
        end
    end

    Q_A -->|"🔥 HIGH<br/>直接被推"| K_ACT
    Q_A -->|"🔥 HIGH<br/>自身动量"| K_Ap
    Q_B -->|"🟡 MID<br/>接触因果"| K_Ap
    Q_B -->|"🟢 LOW<br/>无因果"| K_ACT
    Q_R -->|"🟡 MID<br/>执行器状态"| K_Rp

    NOTE["因果结构从 interventional data 中涌现<br/>无需显式因果图模块"]
```

---

## 自适应步数机制

```mermaid
flowchart TD
    START["输入: (states, goal)"] --> TRANS["Transformer 编码"]
    TRANS --> PRED["隐式预测动作后果"]
    PRED --> OUTPUT["输出: aₖ"]
    OUTPUT --> CHECK{"各物体距离 goal<br/>≤ 阈值?"}
    CHECK -->|"❌ 否"| FEEDBACK["将 aₖ 追加到内部序列<br/>(内部 rollout, 未真实执行)"]
    FEEDBACK --> TRANS
    CHECK -->|"✅ 是"| STOP_O["输出 <STOP>"]
    STOP_O --> EXEC_ALL["一次性执行 [a₀...aₙ]<br/>或逐步执行+re-plan"]

    style CHECK fill:#fff3e0
    style STOP_O fill:#e8f5e9
```

---

## Diffusion 类比视角

```mermaid
flowchart LR
    subgraph DIFF["Diffusion 模型"]
        NT["x_T (纯噪声)"] -->|"去噪步 1"| D1["x_{T-1}"]
        D1 -->|"去噪步 2"| D2["x_{T-2}"]
        D2 -->|"..."| D3["x_0 (清晰图像)"]
    end

    subgraph ACW["ACWM"]
        ST["s_current (远离goal)"] -->|"动作 a₀ (大步)"| S1["s_1"]
        S1 -->|"动作 a₁ (中步)"| S2["s_2"]
        S2 -->|"动作 a₂ (微调)"| SG["s_n ≈ s_goal"]
    end

    DIFF -.->|"类比: 逐步减小噪声≡逐步缩小状态差距"| ACW
```

---

## 训练范式

```mermaid
flowchart TB
    subgraph TRAIN["训练阶段 (Teacher Forcing)"]
        DATA["专家轨迹 (仿真MPC生成)<br/>(s₀, a*₀, s₁, a*₁, ..., sₙ=g)"]
        LOSS["Loss = CrossEntropy(STOP) + MSE(a_pred, a*)"]
    end

    subgraph INFER["推理阶段 (闭环自回归)"]
        CLOSED["WM 自回归输出动作<br/>→ 执行 → 感知 → 再输出<br/>(无需访问 ground truth action)"]
    end

    TRAIN --> INFER
```

---

## 与 CJEPA 的核心差异

| 维度 | CJEPA | ACWM (本设计) |
|------|-------|--------------|
| **任务目标** | 预测 latent 状态 | 输出动作 |
| **表征方式** | 全局 latent | Object-Centric tokens |
| **因果推理** | 无显式设计 | Attention 涌现因果结构 |
| **输出** | next latent state | action sequence + STOP |
| **误差处理** | 预测误差累积 | 闭环执行→纠错 |
| **步数** | 固定 horizon | 自适应 STOP |
| **泛化关键** | 通用 latent 空间 | Object-level 组合泛化 |
| **哲学** | "先理解世界，再行动" | "理解和行动一体" |

---

## 消融实验设计

| 消融项 | 预期效果 |
|--------|---------|
| – Object-Centric (改为扁平 state vector) | 物体数量 OOD 退化 |
| – STOP 机制 (固定 horizon) | 长 horizon 任务退化 |
| – Causal-Aware (随机 shuffle attention) | 因果场景退化, 伪相关增加 |
| – 闭环 (改为 open-loop) | 物理参数 OOD 退化 |
| – 自适应步长 (固定 step size) | 效率下降, 微调能力退化 |

---

*Last updated: 2026-05-19 | Next: 细化各模块的实现细节*
