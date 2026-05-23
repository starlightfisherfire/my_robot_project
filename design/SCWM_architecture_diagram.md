# SCWM 架构图

> 清晰版 | 2026-05-20

---

## 一、总览: 闭环架构

```mermaid
flowchart TB
    subgraph WORLD["🌍 物理世界"]
        PHYS["真实/仿真环境<br/>s_{t+1} = T(s_t, a_t)"]
    end

    subgraph PERCEPTION["📷 感知编码"]
        RAW["obs_t<br/>(结构化状态 / RGB)"]
        ENC["Observation Encoder<br/>结构化: MLP | 视觉: ViT+Slot"]
        Z_OBS["z_obs_t ∈ ℝ^128"]
    end

    subgraph CORE["🧠 SCWM 核心: Self-Contained World State"]
        direction TB
        
        S_PREV["State_{t-1} ∈ ℝ^256<br/>(来自上一时刻)"]
        
        subgraph UPDATE_BLOCK["State Update (Causal Transformer)"]
            QUERY["Query: State_{t-1}<br/>\"我对世界的当前理解\""]
            KV["Key/Value: z_obs_t + act_{t-1}<br/>\"新观测 + 上一步动作\""]
            CROSS["Cross-Attention<br/>━━━━━━━━━━<br/>State_{t-1} 查询新信息<br/>→ 更新为 State_t"]
        end
        
        S_NEW["State_t ∈ ℝ^256<br/>\"更新后的世界理解\"<br/>━━━━━━━━━━<br/>自包含 · 持久 · 可累积"]
    end

    subgraph HEADS["📊 预测头"]
        NL["Next-Latent Head<br/>State_t → ẑ_obs_{t+1}"]
        DYN["Dynamics Head<br/>(State_t, act_t) → Δpose"]
        SG["Subgoal Head<br/>State_t → subgoal"]
    end

    subgraph PLANNER["🎯 Planner (Paper1: CEM-MPC)"]
        ROLLOUT["rollout_model.py<br/>用 Dynamics Head 做前向模拟"]
        MPC["CEM-MPC 优化动作序列"]
        ACT["输出: a_t"]
    end

    WORLD --> RAW
    RAW --> ENC
    ENC --> Z_OBS
    Z_OBS --> KV
    S_PREV --> QUERY
    QUERY --> CROSS
    KV --> CROSS
    CROSS --> S_NEW
    
    S_NEW --> NL
    S_NEW --> DYN
    S_NEW --> SG
    
    DYN --> ROLLOUT
    SG --> MPC
    ROLLOUT --> MPC
    MPC --> ACT
    ACT --> WORLD

    style CORE fill:#fff3e0,stroke:#ff9800
    style S_NEW fill:#ffe0b2,stroke:#ff9800,stroke-width:3px
    style NL fill:#e8f5e9
    style DYN fill:#e8f5e9
    style SG fill:#e8f5e9
```

---

## 二、核心: State Update 机制 (展开)

这是整个架构最重要的一张图。

```mermaid
flowchart LR
    subgraph INPUT["输入 (时刻 t)"]
        OBS_t["z_obs_t<br/>当前观测 latent<br/>128 dim"]
        ACT_t1["act_{t-1}<br/>上一步动作<br/>action_dim"]
        S_t1["State_{t-1}<br/>上一时刻的世界状态<br/>256 dim"]
    end

    subgraph FUSE["信息融合"]
        SEQ["构造输入序列<br/>━━━━━━━━━━<br/>[State_{t-1}, z_obs_t, act_{t-1}]<br/>→ 3 tokens"]
    end

    subgraph TRANSFORMER["Causal Transformer (L 层)"]
        direction TB
        SA1["Self-Attention Layer 1<br/>━━━━━━━━━━<br/>State_{t-1} 关注 z_obs_t<br/>\"新观测告诉我什么?\""]
        FFN1["FFN"]
        SA2["Self-Attention Layer 2<br/>━━━━━━━━━━<br/>融合信息, 抽象物理规律"]
        FFN2["FFN"]
    end

    subgraph OUTPUT["输出"]
        S_t["State_t<br/>━━ 提取 State_{t-1} 位置的输出<br/>256 dim<br/>━━<br/>自包含的压缩世界表征<br/>编码了 0:t 时刻的全部信息"]
    end

    OBS_t --> SEQ
    ACT_t1 --> SEQ
    S_t1 --> SEQ
    SEQ --> SA1
    SA1 --> FFN1
    FFN1 --> SA2
    SA2 --> FFN2
    FFN2 --> S_t

    style TRANSFORMER fill:#e3f2fd
    style S_t fill:#ffe0b2,stroke:#ff9800,stroke-width:3px
```

---

## 三、训练流程

```mermaid
flowchart TB
    subgraph TRAJ["输入: 完整轨迹 (Teacher Forcing)"]
        O0["obs_0"] --> A0["act_0"] --> O1["obs_1"] --> A1["act_1"] --> O2["obs_2"] --> DOTS["..."] --> OT["obs_T"]
    end

    subgraph ENCODE["编码所有帧 (一次性)"]
        ENC_ALL["Encoder(obs_0...T)<br/>→ z_obs_{0:T}"]
    end

    subgraph LOOP["State 自回归更新 (训练时 teacher forcing)"]
        INIT["State_0 = InitState()"]
        T0["Update(State_0, z_obs_0, act_0)<br/>→ State_1 → 预测 z_obs_1"]
        T1["Update(State_1, z_obs_1, act_1)<br/>→ State_2 → 预测 z_obs_2"]
        TT["... → State_T → 预测 z_obs_T"]
    end

    subgraph LOSS["Loss 计算"]
        L_PRED["L_pred = Σ MSE(ẑ_obs_t, z_obs_t)<br/>━━ 自监督主目标<br/>(next-latent prediction)"]
        L_DYN["L_dyn = MSE(Δpred, Δtrue)<br/>━━ 监督辅助<br/>(对接 Paper1 planner)"]
        TOTAL["L_total = L_pred + λ·L_dyn"]
    end

    TRAJ --> ENCODE
    ENCODE --> LOOP
    LOOP --> LOSS

    style L_PRED fill:#c8e6c9
    style L_DYN fill:#fff9c4
    style TOTAL fill:#ffcdd2
```

---

## 四、长上下文: Sliding Window + State 压缩

```mermaid
flowchart TB
    subgraph FULL["完整历史 (100 帧) — 不直接全喂入"]
        H0["obs_0"] -.-> H10["... obs_10 ..."] -.-> H50["... obs_50 ..."] -.-> H90["obs_90"] -.-> H99["obs_99"]
    end

    subgraph WINDOW_A["Window A: t=0..9"]
        WA_OBS["obs_0 ... obs_9"]
        WA_ACT["act_0 ... act_8"]
        WA_CORE["SCWM Core<br/>━━━━━━━━<br/>完整注意力 over 10 帧"]
        WA_STATE["State_9<br/>━━ 压缩了 0..9 的全部信息"]
    end

    subgraph WINDOW_B["Window B: t=10..19"]
        WB_INPUT["State_9 (压缩历史) + obs_10..obs_19"]
        WB_CORE["SCWM Core<br/>━━━━━━━━<br/>State_9 作为 prefix token<br/>+ 10 帧精细注意力"]
        WB_STATE["State_19<br/>━━ 压缩了 0..19 的全部信息"]
    end

    subgraph WINDOW_LAST["... 最终 Window"]
        WL_STATE["State_99<br/>━━ 压缩了完整 100 帧的全部信息"]
    end

    H0 --> WA_OBS
    WA_OBS --> WA_CORE
    WA_ACT --> WA_CORE
    WA_CORE --> WA_STATE
    WA_STATE --> WB_INPUT
    WB_INPUT --> WB_CORE
    WB_CORE --> WB_STATE
    WB_STATE -.-> WL_STATE

    style WA_STATE fill:#ffe0b2,stroke:#ff9800
    style WB_STATE fill:#ffe0b2,stroke:#ff9800
    style WL_STATE fill:#ff9800,stroke:#e65100,stroke-width:3px
```

---

## 五、推理闭环

```mermaid
sequenceDiagram
    participant ENV as 🌍 环境
    participant ENC as 📷 Encoder
    participant SCWM as 🧠 SCWM Core
    participant HEADS as 📊 Heads
    participant MPC as 🎯 CEM-MPC

    Note over ENV,MPC: 初始状态
    
    ENV->>ENC: obs_0
    ENC->>SCWM: z_obs_0
    SCWM->>SCWM: State_0 = InitState()
    
    loop 闭环控制
        SCWM->>SCWM: State_t = Update(State_{t-1}, z_obs_t, act_{t-1})
        SCWM->>HEADS: State_t
        HEADS->>MPC: Dynamics + Subgoal
        MPC->>MPC: Rollout + 优化
        MPC->>ENV: a_t (执行)
        ENV->>ENV: 物理演化 s_{t+1}
        ENV->>ENC: obs_{t+1}
        ENC->>SCWM: z_obs_{t+1}
    end
    
    Note over ENV,MPC: 任务完成 ✅
```

---

## 六、对比: SCWM vs C-JEPA

```mermaid
flowchart LR
    subgraph CJEPA["C-JEPA (基线)"]
        direction TB
        C_CTX["context(t-4:t)<br/>前5帧编码"]
        C_PRED["Predictor"]
        C_TGT["target(t+1)<br/>下一帧编码<br/>━━━━━━<br/>❌ 无持久状态<br/>❌ 每次从零编码<br/>❌ 受限于 context window"]
        C_CTX --> C_PRED --> C_TGT
    end

    subgraph SCWM_V["SCWM (本设计)"]
        direction TB
        S_INIT["State_{t-1}<br/>(压缩了全部历史)"]
        S_UPD["State Update<br/>(Transformer)"]
        S_NEW2["State_t<br/>━━━━━━<br/>✅ 持久状态<br/>✅ 增量更新<br/>✅ 长程依赖"]
        S_INIT --> S_UPD --> S_NEW2
    end

    CJEPA -.->|"对比: 谁在 OOD 下更鲁棒?"| SCWM_V

    style CJEPA fill:#fce4ec
    style SCWM_V fill:#e8f5e9
```

---

## 七、与 Paper1 的继承关系

```mermaid
flowchart TB
    subgraph PAPER1["Paper 1: 机制探索"]
        P1_FLAT["Flat Encoder"]
        P1_OBJ["Object-Centric Encoder"]
        P1_CAUSAL["Causality-Aware Encoder"]
        P1_MPC["固定 CEM-MPC"]
    end

    subgraph PAPER2["Paper 2: 基座世界模型"]
        P2_CJEPA["C-JEPA (基线)"]
        P2_SCWM["SCWM (本设计)"]
    end

    P1_FLAT -->|"对比维度"| P2_CJEPA
    P1_OBJ -->|"Token 结构继承"| P2_SCWM
    P1_CAUSAL -->|"因果归纳偏置继承"| P2_SCWM
    P1_MPC -->|"相同的 Planner"| P2_CJEPA
    P1_MPC -->|"相同的 Planner"| P2_SCWM

    style PAPER1 fill:#e1f5fe
    style PAPER2 fill:#fff3e0
    style P2_SCWM fill:#ffe0b2,stroke:#ff9800,stroke-width:3px
```

---

*架构图版本: v1.0 | 2026-05-20*
