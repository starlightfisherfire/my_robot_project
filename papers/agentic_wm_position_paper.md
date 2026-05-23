# Agentic World Models: Replicating the Large Language Model Development Path for Physical Intelligence

**Authors:** Bruce Wu, Paper1 Assistant  
**Date:** 2026-05-20 (Draft v0.1)  
**Status:** Position Paper — Claiming the Idea  

---

## Abstract

The development of Large Language Models (LLMs) followed a clear, replicable trajectory: pretrain on massive passive data for understanding, then add tool-use capabilities for agency. This path—from pure comprehension to tool-augmented action—proved remarkably general, enabling the same base model to write code, search the web, and control external systems. Yet in robot learning and world models, the dominant paradigm remains fundamentally different: actions are integrated into the model architecture from the start, treated as internal outputs rather than external tools. We argue that this divergence is a mistake. The LLM development path is not merely an analogy for building world models—it is the correct architectural blueprint. We propose **Agentic World Models (AWM)**: a framework in which a video-based world model is first trained purely to understand and predict world states (pretraining), and then equipped with a tool interface that allows it to call upon a robot body (tool-use). Action is not an output of the model; action is a tool the model learns to use. This decoupling yields a universal architecture that generalizes across embodiments, data modalities, and tasks. We formalize the framework, specify the architecture, contrast it with existing world models (C-JEPA, Dreamer, IRIS, DeltaWorld), and outline a research roadmap toward physical foundation models.

---

## 1. Introduction

Something remarkable happened between 2022 and 2026. Large Language Models evolved from pure text predictors into general-purpose agents. The path was not planned in advance—it emerged—but in retrospect it is strikingly clean:

1. **Pretrain** on vast amounts of passive text (next-token prediction) → language understanding.
2. **Fine-tune** on instruction-following data → task execution.
3. **Add tool use** (function calling, browsing, code execution) → agency in the world.
4. **Scale** and integrate → emergent reasoning, multi-step planning, self-correction.

The same GPT-4 base model that completes sentences can also search the web, call APIs, generate and execute code, and control software interfaces. The critical architectural insight is that **the model never needed to know about tools during pretraining**. Tool use is a capability *added later*, through a structured interface—`function_call(name, parameters)`—that the model learns to emit when needed. The tool's execution and its consequences are handled externally, not by the model itself.

This paper argues that the exact same trajectory should be deliberately followed for building physical world models for robotics.

The current landscape of robot world models—from Dreamer [Hafner et al., 2023] to C-JEPA [Anonymous, 2024] to IRIS [Micheli et al., 2023]—takes the opposite approach. Actions are integrated into the model's core architecture: the world model receives `(observation, action)` pairs and learns to predict next observations *conditioned on actions*. The action is an internal variable, baked into the dynamics model from day one.

We believe this is the fundamental architectural mistake that prevents world models from scaling like LLMs. If actions are inside the model, then:

- Every new robot embodiment requires retraining the world model.
- Training data *must* contain action labels.
- The model cannot benefit from the vast amount of actionless video data (YouTube, egocentric video, simulation renders).
- There is no clean separation between "understanding the world" and "acting in the world."

Our proposal: **separate world understanding from tool use, exactly as LLMs do.**

---

## 2. The LLM Development Path as Blueprint

### 2.1 Why LLMs Succeeded

The LLM development path succeeded not because of any single architectural trick, but because it respected a fundamental separation of concerns:

| Stage | What is learned | Data required |
|-------|----------------|---------------|
| **Pretraining** | Language structure, world knowledge, reasoning patterns | Raw text (abundant, passive) |
| **Instruction tuning** | Following commands, task execution | Instruction-output pairs (moderate cost) |
| **Tool use** | When and how to use external capabilities | Tool-augmented demonstrations (moderate cost) |

The key property: **each stage builds on the previous without requiring changes to the base model's core architecture**. A pretrained LLM can be instruction-tuned; an instruction-tuned LLM can learn tool use. The hidden state—the model's internal representation—remains a single unified vector that encodes everything: the conversation history, the current task, and the decision to call a tool.

### 2.2 The Blueprint for World Models

We propose that the same three-stage blueprint applies directly to physical world models:

| LLM Stage | AWM Stage | What is learned | Data |
|-----------|-----------|----------------|------|
| Pretrain on text | **Pretrain on video** | Physical dynamics, object permanence, spatial relations, contact mechanics | Any video (YouTube, sim, egocentric, human demos) |
| Instruction tuning | **Task-conditioned fine-tuning** (optional) | Goal-directed behavior understanding | Task-labeled video |
| Tool use | **Tool Interface** | How to use a specific robot body to achieve goals | Robot-specific demonstrations |

The architectural isomorphism is precise:

```
LLM:                    AWM:
token_t → Embedding     Frame_t → Encoder → v_t
[h_0,...,h_t] → TF      [v_0,...,v_t] → Causal TF
hidden_t                State_t
hidden_t → next_token   State_t → pred_v_{t+1}
hidden_t → func_call    State_t → tool_call
```

---

## 3. Existing World Models: A Systematic Limitation

### 3.1 The Dominant Paradigm: Action-Integrated World Models

Nearly all existing world models for robotics embed actions as an internal component of the dynamics model. We survey the major families:

**Dreamer Family (DreamerV1-V3) [Hafner et al., 2019-2023].** The RSSM (Recurrent State-Space Model) receives `(state_t, action_t)` and predicts `state_{t+1}` and `reward_t`. The action is an integral input to the recurrent state update. While powerful for RL, this tightly couples the world model to a specific action space.

**Transformer-based World Models (TWM, IRIS) [Robine et al., 2023; Micheli et al., 2023].** These use transformers to model environment dynamics, but actions are embedded alongside observations in the input sequence. IRIS tokenizes observations with VQ-VAE and interleaves them with action tokens. Again, the action space is part of the model architecture.

**C-JEPA (Causal Joint Embedding Predictive Architecture) [Anonymous, 2024].** Uses slot attention to produce object-centric representations. While innovative in its use of object-level masking for causal inductive bias, C-JEPA still requires actions during training for the masking mechanism. The slots compete to explain the input, but there is no persistent global state.

**DeltaWorld [Anonymous, 2024].** Encodes frame-to-frame differences as single "delta tokens" processed by a transformer. This is the closest existing work to our frame-as-token approach, but DeltaWorld is designed for video generation, not robotic control, and has no tool interface.

**V-JEPA / V-JEPA 2 [Bardes et al., 2024; Meta, 2025].** Video-pretrained models using joint embedding predictive architecture. V-JEPA 2 demonstrates zero-shot robot control, but the training objective and architecture do not separate world understanding from action.

**GR-1 / GR-2 [Cheang et al., 2024].** GPT-style autoregressive models pretrained on video and fine-tuned for robot manipulation. Actions are predicted as part of the autoregressive sequence, not decoupled.

**RT-2 / Gemini Robotics [Google, 2024-2025].** Vision-Language-Action models that use LLM backbones with tool-use capabilities *for reasoning*, but the robot actions are still model outputs, not external tools in our sense.

### 3.2 The Common Limitation

All these models share one property: **action is inside the architecture**. This means:

1. **Embodiment-locked**: The world model cannot transfer to a different robot without architectural changes.
2. **Data-inefficient**: Training requires action-labeled data, which is expensive to collect.
3. **No pure world understanding**: The model learns "what happens when I act" but never purely "what happens in the world."
4. **No causal contrast**: There is no natural way to separate natural evolution from intervention effects.

---

## 4. Agentic World Models: The Framework

### 4.1 Core Principle

> A world model should understand the world *before* it learns to act. Action is a tool the model learns to use, not a variable baked into its architecture.

### 4.2 Three-Stage Development

**Stage 1: Video Pretraining (World Understanding)**

```
Input:   Arbitrary video sequences (no action labels)
Model:   Frame_t → Encoder → v_t (single vector per frame)
         [v_0, ..., v_t] → Causal Transformer → State_t (global vector)
Output:  pred_v_{t+1}  (next-frame prediction)
Loss:    L = MSE(pred_v_{t+1}, actual_v_{t+1})
```

At this stage, the model learns:
- Physical dynamics (gravity, inertia, contact, collision)
- Object persistence and identity across frames
- Spatial relations and scene geometry
- Natural evolution of the world without intervention

The key architectural choices:
- **One vector per frame** (not multiple slots): Each frame is compressed into a single "frame essence" vector, analogous to a word embedding in LLMs.
- **Causal Transformer**: Frames interact across time through self-attention, with each position attending only to past frames.
- **Single global State_t**: The hidden state at position t encodes all information from frames 0 to t. This is the "self-contained world state" — a compressed, persistent representation analogous to an LLM's KV-cache.
- **Residual updates**: State_t = State_{t-1} + Δ(State_{t-1}, v_t), ensuring information is preserved by default unless the observation demands change.

**Stage 2: Tool Interface (Learning to Act)**

```
Input:   State_t (from pretrained base model)
Output:  tool_call = (tool_id, parameters)
Example: tool_call = (move_joints, {targets: [θ_1, ..., θ_6]})
Training: Supervised on robot demonstration data
```

The tool interface is a simple classification + regression head on top of State_t. It does not modify the base model's architecture. The model learns:
- Which tool to use in which world state
- What parameters to pass
- Multi-step tool-use sequences (planning)

**Action Lift — Bridging Semantic Spaces:**

A critical design element is the **Action Lift** module. Low-dimensional robot actions (e.g., 6-DOF joint targets) exist in a different semantic space than the world state. We lift actions into the same high-dimensional semantic space as State_t:

```
a_t (low-dim, 6d) → ActionLift(a_t, env_context, historical_actions, historical_states) → A_t (256d)
```

This lifted action vector A_t can then participate in the world model's semantic operations—it can be attended to, added to State_t, or used for counterfactual prediction. This is isomorphic to how discrete word tokens are lifted into continuous embedding space in LLMs.

**Stage 3: Agentic (Multi-Step Reasoning + Tool Chaining)**

```
Loop:
  1. State_t → tool_call_1 → execute → new frame
  2. State_{t+1} → tool_call_2 → execute → new frame
  ...
  N. State_{t+N} → task complete
```

This mirrors LLM agent workflows: the model observes, decides which tool to call, observes the result, and continues. Complex tasks decompose into sequences of tool calls, with the world model's persistent state accumulating context across steps.

### 4.3 Two-Layer Causal Framework

AWM enables a natural causal interpretation:

```
State_t ───→ pred_v_{t+1}        (Natural evolution: what happens without me)
  │
  └──→ a_t → pred_State_{t+1}    (Intervention: what happens if I act)

Difference = causal effect of action a_t
```

This directly instantiates the counterfactual framework of causal inference [Pearl, 2009] within the world model. The base model predicts natural evolution; the action-conditioned prediction estimates intervention effects. Their difference quantifies the causal impact of the action—a property no existing world model explicitly provides.

---

## 5. Detailed Architecture

### 5.1 Frame Encoder

```
Frame_t (H × W × 3) → Encoder → v_t ∈ ℝ^D

Options:
- Structured state (Paper 1): MLP on object-state tokens → v_t
- Visual (Paper 2): Pretrained ViT (DINOv2, MAE) → v_t  
- Future: End-to-end trained encoder
```

The encoder can be frozen (MVP), fine-tuned, or trained end-to-end. The key invariant: each frame maps to exactly one vector.

### 5.2 Causal World Transformer

```
Input: [v_0, v_1, ..., v_t] + PositionalEncoding(time_step)
Architecture: L-layer Causal Transformer
  - d_model = 256
  - Causal attention mask (attend only to past)
  - Residual connections (preserve information by default)
Output: h_t at position t → State_t = h_t
```

### 5.3 Long Context: Sliding Window + State Compression

```
Full history: [obs_0, ..., obs_99]  (100 frames)
     ↓ Compress
State_{t-window} (256d)  ← encodes everything before the window
     + 
Fine-grained attention over recent window: [obs_{t-window:t}]
     ↓
State_t  ← fuses compressed history + recent details
```

This is analogous to Transformer-XL's segment-level recurrence. The State vector acts as a compressed memory of all past frames, enabling arbitrarily long context without quadratic attention cost.

### 5.4 Training Objectives

**Base Model (Stage 1):**
```
L_base = MSE(pred_v_{t+1}, actual_v_{t+1})   [self-supervised]
       + λ_reg * L_regularization            [optional]
```

**Tool Interface (Stage 2):**
```
L_tool = CrossEntropy(tool_id_pred, tool_id_true)          [classification]
       + MSE(params_pred, params_true)                    [regression]
       + MSE(pred_State_{t+1}, actual_State_{t+1})        [consequence prediction]
```

### 5.5 Optional Slot Decoding (Diagnostics Only)

```
State_t → SlotDecoder → [s_0, s_1, ..., s_K]  (per-object representations)
```

This is an optional downstream decoding, useful for interpretability and debugging. It is not part of the core architecture. This inverts the C-JEPA design: in C-JEPA, slots are the *primary* encoding; in AWM, slots are an *optional* decoding from the global state.

---

## 6. Why This Framework Is General

### 6.1 Embodiment Independence

The base model (Stage 1) has no concept of "robot" or "action." It only understands frames and how they evolve. This means:

- **Same base model for any robot**: A robotic arm, a drone, a humanoid hand—all use the same video understanding backbone.
- **Tool definitions are external**: Changing the robot only changes the tool schema (e.g., `move_joints` vs `fly_to`), not the base model.

### 6.2 Data Flexibility

| Training data | Base Model | Tool Interface |
|--------------|------------|----------------|
| YouTube videos | ✅ | ❌ |
| Simulation renders | ✅ | ✅ (if action-labeled) |
| Human egocentric video | ✅ | ❌ |
| Robot teleoperation | ✅ | ✅ |
| 3D renderings | ✅ | ✅ |
| Depth maps | ✅ | ✅ |

### 6.3 Bidirectional Fine-tuning

Tool-use data can *improve* the base model's world understanding. When the model learns that "move_joints causes this specific visual change," this knowledge can back-propagate to refine the base model's understanding of physical causation. A robot's specific capabilities and constraints become part of its world model.

---

## 7. Key Claims and Novelty

This paper makes the following claims:

1. **Meta-Claim**: The LLM development path—pretrain for understanding, then add tool use for agency—is not merely an analogy for world model development; it is the correct architectural blueprint that should be deliberately followed.

2. **Architectural Claim**: World models should be built as Frame-as-Token Causal Transformers with a single global persistent State vector, not as slot-based or action-integrated architectures.

3. **Decoupling Claim**: Action should be treated as external tool use, not as an internal model output. This enables embodiment independence and massive data scaling.

4. **Causal Claim**: The two-layer prediction framework (natural evolution vs. intervention) provides a natural causal interpretation that no existing world model offers.

5. **Generality Claim**: This framework unifies 2D video, 3D rendering, and structured state under the same architecture, requiring only a change in the encoder.

To our knowledge, **no prior work has proposed the complete combination of these claims.** Individual components exist in isolation (frame-as-token in DeltaWorld, video pretraining in V-JEPA 2, tool use in Gemini Robotics), but the systematic framework—treating the LLM development trajectory as an explicit blueprint for physical world models—is new.

---

## 8. Preliminary Evidence and Research Roadmap

### 8.1 Immediate Validation (Paper 1 Environment)

We propose validating the core architecture in a simplified setting before scaling to video:

- **Environment**: Structured object-state push-to-pose (Paper 1)
- **Encoder**: MLP on structured state tokens
- **Base Model**: Causal Transformer with global State_t, trained on next-state prediction
- **Tool Interface**: Simple action head producing delta-pose actions
- **Baseline**: C-JEPA style slot-based world model (no persistent state)
- **Metrics**: OOD generalization (layout OOD, shape OOD), prediction error, control success rate

**Hypothesis**: AWM's persistent global state outperforms C-JEPA's stateless slot encoding on OOD generalization, particularly for tasks requiring long-range physical reasoning.

### 8.2 Visual Scaling (Paper 2)

- Replace structured-state encoder with pretrained ViT
- Train on MuJoCo-rendered video
- Evaluate OOD control: AWM vs. C-JEPA visual world model
- Extend to multi-task and cross-embodiment evaluation

### 8.3 Real Robot Deployment (SO-101)

- Collect teleoperation data on SO-101 robotic arm
- Fine-tune tool interface on real robot data
- Compare AWM + Diffusion Policy vs. end-to-end visuomotor policy on OOD layouts

### 8.4 Long-term Vision

- Pretrain base model on internet-scale video
- Fine-tune tool interfaces for multiple robot embodiments
- Demonstrate cross-embodiment transfer (same base model, different tools)
- Explore emergent planning capabilities from tool-use training

---

## 9. Related Work

We position our work relative to several research threads:

**World Models for Control.** Dreamer [Hafner et al., 2019-2023], IRIS [Micheli et al., 2023], TWM [Robine et al., 2023], and STORM [Zhang et al., 2024] all integrate actions into the world model architecture. Our work is the first to propose complete decoupling.

**Slot-based World Models.** C-JEPA [Anonymous, 2024] and Slot Attention [Locatello et al., 2020] use competitive slot mechanisms for object-centric representations. We invert this: slots are optional *outputs* decoded from a global state, not the primary encoding.

**Video Foundation Models.** V-JEPA [Bardes et al., 2024], Sora [OpenAI, 2024], DeltaWorld [Anonymous, 2024], and Cosmos [NVIDIA, 2025] demonstrate powerful video understanding and generation. None propose the explicit two-stage (understanding → tool-use) framework for robotic control.

**LLM-based Robot Control.** RT-2 [Google, 2024], Gemini Robotics [Google, 2025], and Octo [Team, 2024] use LLM/VLM backbones for robot control. These use LLMs *as* world models, whereas we propose building dedicated world models *using the LLM development paradigm.*

**Tool Use and Agents.** Function calling in GPT-4 [OpenAI, 2023], tool use in Claude [Anthropic, 2024], and the MCP protocol [2025] establish the tool-use paradigm for LLMs. Our contribution is extending this paradigm to physical world models.

---

## 10. Discussion and Open Problems

**When is tool use enough?** For simple manipulation tasks, a tool interface may be sufficient. For tasks requiring continuous high-frequency control (e.g., dynamic manipulation), the tool interface may need to operate at higher temporal resolution.

**Scaling laws for world models?** Do world models exhibit scaling laws analogous to LLMs? Does more video data → better physical understanding → better OOD generalization? This is an important empirical question.

**What is the right tool granularity?** Should tools be low-level (`move_joints`) or high-level (`push_object_to_pose`)? This likely depends on the task complexity and data availability.

**Safety and alignment.** An agentic world model that can call tools in the physical world raises safety concerns. The tool interface provides a natural safety boundary: tools can be restricted, monitored, and sandboxed.

---

## 11. Conclusion

The development of Large Language Models revealed a clean trajectory: from passive understanding to tool-augmented agency. We argue that this trajectory is not specific to language—it is the correct blueprint for building physical world models. We proposed Agentic World Models (AWM): a framework that separates world understanding (video pretraining with next-frame prediction) from action (tool interface), maintains a single global persistent state, treats frames as tokens in a causal transformer, and provides a natural causal interpretation through two-layer prediction. This framework is embodiment-independent, data-flexible, and architecturally aligned with the most successful AI paradigm of our time. We invite the community to join us in exploring this path toward physical foundation models.

---

## References

*(To be completed with full citations)*

- Hafner et al. "DreamerV3: Mastering Diverse Domains through World Models." 2023.
- Micheli et al. "Transformers are Sample-Efficient World Models." ICLR 2023.
- Robine et al. "Transformer-based World Models." 2023.
- Bardes et al. "V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video." 2024.
- Anonymous. "C-JEPA: Causal Joint Embedding Predictive Architecture." 2024.
- Anonymous. "DeltaWorld / DeltaTok." 2024.
- OpenAI. "GPT-4 Technical Report." 2023.
- Pearl. "Causality." Cambridge University Press, 2009.
- LeCun. "A Path Towards Autonomous Machine Intelligence." 2022.
- Team et al. "Octo: An Open-Source Generalist Robot Policy." 2024.
- Google DeepMind. "RT-2: Vision-Language-Action Models." 2024.
- NVIDIA. "Cosmos World Foundation Model Platform." 2025.
- Meta AI. "V-JEPA 2." 2025.
- Cheang et al. "GR-1: Unleashing Large-Scale Video Pretraining." 2024.
- Locatello et al. "Object-Centric Learning with Slot Attention." NeurIPS 2020.

---

*Draft v0.1 | 2026-05-20 | For discussion and feedback*
