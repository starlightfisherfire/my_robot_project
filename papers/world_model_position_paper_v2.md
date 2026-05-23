# World Models Should Learn to Understand Before They Learn to Act: A Position on Decoupled Physical Intelligence

**Author:** Bruce Wu  
**Date:** 2026-05-20 (Draft v2.0)  
**Status:** Position Paper — Revised and Extended

---

## Abstract

The dominant paradigm in robot world models tightly couples perception, dynamics, and action into a single architecture: the model receives `(observation, action)` pairs and learns to predict next observations *conditioned on actions*. We argue that this coupling is the fundamental bottleneck preventing world models from scaling. Drawing on the development trajectory of Large Language Models—from passive pretraining to tool-augmented agency—we propose that world models should follow the same path: first learn to *understand* the physical world through pure observation prediction, then learn to *act* through a decoupled tool interface. This position rests on five interconnected claims: (1) world states are compressible, but the compression structure matters more than the compression ratio; (2) representations and their consumers are inseparable—what matters is not representation quality in isolation, but how easily downstream planners can consume it; (3) causal decomposition of world states serves primarily as a *consumer-facing* optimization, not an intrinsic representation improvement; (4) self-contained representations—encoding not just "what is" but "what will happen" and "what I can do"—are necessary for effective planning; and (5) the LLM development path (pretrain → tool-use → agentic) is not merely an analogy for physical world models—it is the correct architectural blueprint. We formalize this framework, connect it to existing work, and outline a research program for validation.

---

## 1. Introduction: The Question Behind the Question

The field of robot learning has produced impressive results in recent years. Vision-Language-Action models (RT-2, Gemini Robotics) can follow language instructions. World models (DreamerV3, IRIS, C-JEPA) can predict future states. Diffusion policies can generate complex manipulation trajectories. Yet a fundamental question remains unanswered:

> **What is the right way to represent the physical world for planning and control?**

This is not an engineering question. It is a conceptual one. The answer determines:
- How we train world models (what data, what objectives)
- How we connect world models to action (what architecture, what interface)
- How we evaluate world models (what metrics, what generalization)
- Whether world models can scale like LLMs, or remain confined to narrow domains

This paper is our position on this question. It is the result of a systematic architectural exploration—from action-centric designs to state-centric designs to agentic designs—combined with a careful analysis of why LLMs succeeded where world models have not.

---

## 2. Five Claims

### Claim 1: World States Are Compressible, but Compression Structure Matters

The physical world is high-dimensional: every frame contains millions of pixels, every object has position, velocity, shape, material, and affordances. Yet for the purpose of planning and control, most of this information is irrelevant. A robot pushing a T-shaped block to a target position does not need to know the block's texture or the background color.

This observation is not new. Representation learning—from autoencoders to VAEs to JEPA—is built on it. But we make a stronger claim:

> **The structure of compression determines the utility of the representation, independent of the compression ratio.**

Consider three ways to compress a scene with a robot arm, a target block, and two obstacles:

1. **Flat compression**: Encode the entire scene into a single vector. The downstream dynamics model must learn from scratch which components interact with which.

2. **Object-centric compression**: Decompose the scene into per-object slots. The dynamics model knows that information is grouped by object, but must still learn inter-object physical relationships.

3. **Causal decomposition**: Decompose the scene into functionally coherent groups: `z_stable` (permanent properties), `z_dynamics` (time-varying kinematics), `z_affordance` (action-relevant features), `z_nuisance` (irrelevant variation). The dynamics model receives information pre-organized by physical mechanism.

The claim is not that causal decomposition produces "better" representations in some abstract sense. It produces representations that are *easier for downstream consumers to use*. This distinction—between representation quality and consumer accessibility—is central to our position.

**Evidence from our experiments.** In our Paper 1 environment (structured-state push-to-pose), we compared Flat, Object-Centric, and Causality-Aware encoders with a fixed CEM-MPC planner. The key finding: the three encoders produce similar in-distribution performance, but diverge on out-of-distribution generalization. The causal encoder's advantage is not that it captures more information—it is that it presents information in a structure that the planner can exploit without having to learn the physical grouping itself.

### Claim 2: Representations and Consumers Are Inseparable

A world model does not act. It produces representations that are consumed by downstream systems—RL policies, MPC planners, MCTS search, diffusion models. The world model's value is determined not by the quality of its representations in isolation, but by the *total system performance* of world model + consumer.

This insight is underappreciated in the world model literature. Most papers evaluate world models on prediction error (MSE of next-state prediction) or reconstruction quality (FID of generated frames). But these metrics do not capture the representation's *consumability*—how easily a downstream planner can extract actionable information.

Consider the analogy to word embeddings. Word2Vec and GloVe produce similar embedding quality on intrinsic metrics (similarity, analogy). But for downstream NLP tasks, the choice of embedding can significantly affect performance—not because one embedding is "better," but because different embeddings present information in structures that different consumers find more or less accessible.

**The consumer landscape.** We can characterize existing world model architectures by their implicit consumer:

| World Model | Implicit Consumer | Coupling |
|-------------|-------------------|----------|
| DreamerV3 | RL policy (inside the loop) | Tight |
| TD-MPC2 | MPC planner (external) | Medium |
| MuZero | MCTS search (external) | Medium |
| C-JEPA | Not specified (representation only) | Loose |
| V-JEPA 2 | Robot policy (fine-tuned) | Tight |

The critical observation: **none of these architectures explicitly design the representation for consumer accessibility.** The representation is trained with a self-supervised objective (next-state prediction, JEPA loss, reconstruction), and the consumer is expected to adapt.

We argue for the opposite design principle: **the representation format should be chosen to minimize consumer complexity.** If the consumer is a simple linear planner, the representation should encode planning-relevant structure. If the consumer is a powerful RL policy, the representation can be less structured. The right question is not "what is the best representation?" but "what is the best (representation, consumer) pair?"

### Claim 3: Causal Decomposition Serves the Consumer

Building on Claims 1 and 2, we make a specific claim about causal decomposition:

> **The value of causal decomposition (stable/dynamics/affordance/nuisance) is not that it produces better representations, but that it reduces the learning burden on the consumer.**

In a flat encoding, the dynamics model must learn:
- Which components of the state are permanent (stable)
- Which components change over time (dynamics)
- Which components are relevant to action (affordance)
- Which components are irrelevant (nuisance)

In a causally decomposed encoding, this grouping is provided *a priori*. The dynamics model receives `z_dynamics` and knows "these are the kinematic variables." It receives `z_affordance` and knows "these are the action-relevant features." The physical mechanism is partially encoded in the representation structure, not fully learned by the dynamics model.

This is analogous to how structured programming reduces cognitive load: a programmer who receives `data.velocity` and `data.position` has an easier task than one who receives `data[0:6]` and must learn which indices correspond to which physical quantities.

**Implications for evaluation.** If causal decomposition serves the consumer, then the right evaluation metric is not "representation quality" but "consumer performance"—planning success rate, OOD generalization, sample efficiency. We should compare encoders not by their reconstruction error, but by how well their downstream consumers perform.

### Claim 4: Self-Contained Representations Are Necessary

Language is self-contained. A word embedding encodes not just "what the word means" but "how it relates to other words," "what it implies about the context," and "what it predicts about the next word." This self-containment is what enables language models to perform reasoning, planning, and generation from a single hidden state.

We argue that world model representations should be similarly self-contained. A world state representation should encode:

1. **What is** (current object properties, positions, velocities)
2. **What will happen** (predicted natural evolution, physical tendencies)
3. **What I can do** (affordances, action consequences)
4. **What has happened** (temporal context, interaction history)

Most existing world models encode only (1). Some implicitly encode (2) through next-state prediction. Very few encode (3) or (4).

**The temporal dimension.** A key insight from our analysis: structured object states (positions, velocities) are "clean but dead"—they capture the current snapshot but lack temporal context. A ball at position (0.5, 0.3) with velocity (0.1, 0) is a complete state description, but it does not encode "this ball has been moving toward the obstacle for the past 5 frames" or "this ball was stationary until the robot pushed it."

We propose that self-containment requires temporal mixing: the state at time t should be a function of the current observation *and* a window of recent observations. This is not simply stacking frames (as in standard RL). It is a learned compression of temporal context into the state vector, analogous to how an LLM's hidden state compresses the conversation history.

**Evidence from ablation.** In our Paper 1 experiments, we observe that including historical context (H=6 frames) in the state representation significantly improves OOD generalization compared to single-frame states (H=1). The temporal context allows the dynamics model to distinguish between "the object is stationary because no force was applied" and "the object is stationary because it is in contact with a wall"—a distinction that is invisible from a single frame.

### Claim 5: The LLM Development Path Is the Blueprint

Large Language Models followed a clear, reproducible trajectory:

1. **Pretrain** on massive passive data (next-token prediction) → language understanding
2. **Fine-tune** on instruction data → task execution
3. **Add tool use** (function calling, browsing, code execution) → agency
4. **Scale** → emergent reasoning, multi-step planning, self-correction

The critical architectural insight: **the model never needed to know about tools during pretraining.** Tool use is a capability *added later*, through a structured interface (`function_call(name, parameters)`) that the model learns to emit when needed. The tool's execution and consequences are handled externally.

We claim that this trajectory should be *deliberately replicated* for physical world models:

| LLM Stage | World Model Stage | Learning Objective | Data |
|-----------|-------------------|-------------------|------|
| Pretrain on text | **Pretrain on video** | Physical dynamics, object permanence, spatial relations | Any video (YouTube, sim, egocentric) |
| Instruction tuning | **Task-conditioned fine-tuning** | Goal-directed understanding | Task-labeled video |
| Tool use | **Tool Interface** | How to use a specific robot body | Robot-specific demonstrations |

**Why this is not just an analogy.** The LLM path succeeded because it respected a fundamental separation of concerns:
- Pretraining learns *world knowledge* from abundant, passive data
- Tool use learns *agency* from scarce, active data
- The two are connected through a shared representation (the hidden state)

If we bake action into the world model from the start (as in Dreamer, IRIS, C-JEPA), we lose this separation. The model can only learn from action-labeled data, which is expensive. It cannot benefit from the vast amount of actionless video. And it cannot transfer to a new embodiment without retraining.

---

## 3. The Agentic World Model Framework

Based on these five claims, we propose the **Agentic World Model (AWM)** framework.

### 3.1 Architecture Overview

```
Stage 1: Video Pretraining (World Understanding)
  Frame_t → Encoder → v_t (one vector per frame)
  [v_0, ..., v_t] → Causal Transformer → State_t
  Loss: MSE(pred_v_{t+1}, actual_v_{t+1})

Stage 2: Tool Interface (Learning to Act)
  State_t → ToolHead → tool_call = (tool_id, parameters)
  Loss: Supervised on robot demonstrations

Stage 3: Agentic Loop (Multi-Step Reasoning)
  Observe → State → tool_call → execute → observe → ...
```

### 3.2 Key Design Decisions

**Frame-as-Token.** Each frame is compressed into a single vector `v_t ∈ ℝ^D`, analogous to a word embedding in LLMs. This is not object-centric (multiple slots per frame) or pixel-level (tokenized patches). It is a single "frame essence" vector that captures everything relevant about the frame.

**Causal Transformer with Persistent State.** The transformer processes the sequence of frame vectors with causal attention (each position attends only to past positions). The hidden state at position t, `State_t`, is a compressed representation of all frames from 0 to t. This is the world model's "memory"—analogous to an LLM's KV-cache.

**Residual State Updates.** `State_t = State_{t-1} + Δ(State_{t-1}, v_t)`. Information is preserved by default unless the observation demands change. This ensures stability and prevents catastrophic forgetting of past context.

**Decoupled Action via Tool Interface.** Actions are not model outputs. They are tool calls emitted by a lightweight head on top of State_t. The tool interface is trained separately from the base model, and can be swapped for different embodiments without modifying the base model.

**Action Lift Module.** Low-dimensional robot actions (e.g., 6-DOF joint targets) are lifted into the same high-dimensional semantic space as State_t:

```
a_t (6-dim) → ActionLift(a_t, context) → A_t (256-dim)
```

This is analogous to how discrete word tokens are lifted into continuous embeddings in LLMs. The lifted action vector can then participate in the world model's semantic operations.

### 3.3 Two-Layer Causal Framework

AWM enables a natural causal interpretation:

```
State_t ──→ pred_v_{t+1}          (Natural evolution: what happens without intervention)
State_t ──→ a_t ──→ pred_v_{t+1}' (Intervention: what happens if I act)

Causal effect = pred_v_{t+1}' - pred_v_{t+1}
```

This directly instantiates the counterfactual framework of causal inference [Pearl, 2009]. The base model predicts natural evolution; the action-conditioned prediction estimates intervention effects. Their difference quantifies the causal impact of the action—a property no existing world model explicitly provides.

---

## 4. Relation to Existing Work

### 4.1 Action-Integrated World Models

**Dreamer Family.** RSSM receives `(state_t, action_t)` and predicts `state_{t+1}`. Action is baked into the recurrent update. This couples the world model to a specific action space and prevents transfer across embodiments.

**IRIS / TWM.** Transformer-based, but actions are interleaved with observations in the input sequence. The model cannot process actionless video.

**C-JEPA.** Slot attention with object-level masking for causal inductive bias. Innovative, but no persistent state—each prediction is from a fresh context window. The slots compete to explain the input, but there is no memory across time.

**Common limitation.** All these models require action-labeled data for training. They cannot benefit from the vast amount of actionless video data (YouTube, egocentric video, simulation renders without action annotations).

### 4.2 Video Foundation Models

**V-JEPA / V-JEPA 2.** Video-pretrained using JEPA. V-JEPA 2 demonstrates zero-shot robot control, but the architecture does not separate world understanding from action. The model learns "what happens in video" but not "what happens when I act."

**Sora / Cosmos.** Video generation models with implicit world knowledge. Impressive generation quality, but not designed for planning or control. No tool interface, no causal framework.

**DeltaWorld.** Frame-to-frame differences as tokens. Close to our frame-as-token approach, but designed for video generation, not control. No persistent state, no tool interface.

### 4.3 LLM-based Robot Control

**RT-2 / Gemini Robotics.** Use LLM/VLM backbones for robot control. These use LLMs *as* world models—we propose building dedicated world models *using the LLM development paradigm.* The distinction is important: LLMs understand language, not physics. A dedicated world model can learn physical dynamics that LLMs cannot.

### 4.4 The Consumer Perspective

Most related work focuses on the world model itself. We emphasize the *consumer*—the downstream system that uses the world model's representations for planning. This perspective connects to:

**Model Predictive Control (MPC).** MPC uses a dynamics model for planning. The quality of the dynamics model directly determines planning performance. AWM's tool interface provides a clean API for MPC.

**RL + World Models.** DreamerV3 uses RL as the consumer of its world model. We argue that RL + MPPI may be a stronger consumer: RL provides coarse plans, MPPI refines them through sampling-based optimization.

**Hierarchical Planning.** AWM's two-layer framework (natural evolution + intervention) naturally supports hierarchical planning: the base model predicts high-level state evolution, the tool interface selects low-level actions.

---

## 5. The Role of Causal Decomposition: A Deeper Analysis

### 5.1 Three Encoder Paradigms

In our Paper 1 experiments, we compare three encoder paradigms:

| Encoder | State Structure | Consumer Burden |
|---------|----------------|-----------------|
| **Flat** | Single vector, no structure | Dynamics model must learn all physical groupings |
| **Object-Centric** | Per-object slots | Dynamics model knows object boundaries, must learn inter-object physics |
| **Causality-Aware** | z_stable, z_dynamics, z_affordance, z_nuisance | Dynamics model receives pre-grouped physical mechanisms |

The key insight: **the three encoders differ not in what they encode, but in how they organize what they encode.** All three can, in principle, capture the same information. But they present it to the consumer in different structures, which dramatically affects the consumer's learning burden.

### 5.2 Why Causal Decomposition Works (When It Works)

Causal decomposition works when:
1. The consumer is relatively simple (e.g., linear dynamics head, MPC planner)
2. The task involves clear physical mechanisms (e.g., stable object properties vs. dynamic interactions)
3. The test distribution shifts along causal boundaries (e.g., new object shapes but same physical laws)

Causal decomposition may not help when:
1. The consumer is very powerful (e.g., large RL policy that can learn any function)
2. The causal structure is ambiguous or overlapping
3. The test distribution does not respect causal boundaries

### 5.3 Causal Decomposition as Consumer Optimization

We reframe causal decomposition not as a representation quality improvement, but as a **consumer-facing optimization**:

> Causal decomposition is a way of "pre-computing" the physical groupings that the consumer would otherwise need to learn. It shifts learning from the consumer to the encoder.

This framing has important implications:
- The right evaluation metric is consumer performance, not encoder reconstruction error
- The right encoder depends on the consumer: simple consumers benefit more from causal decomposition
- Causal decomposition is most valuable when data is limited (the consumer cannot learn the groupings from data alone)

---

## 6. The Self-Contained State: From "Clean but Dead" to "Alive"

### 6.1 The Problem with Structured States

Structured object states (positions, velocities, sizes) are the standard representation in robotics. They are clean, interpretable, and easy to work with. But they have a fundamental limitation:

> **Structured states are "clean but dead"—they capture the current snapshot but lack temporal context, causal history, and affordance information.**

A ball at position (0.5, 0.3) with velocity (0.1, 0) is a complete state description. But it does not encode:
- "This ball has been moving toward the obstacle for 5 frames" (temporal trend)
- "The robot pushed this ball 3 frames ago" (causal history)
- "This ball is within reach of the robot" (affordance)
- "This ball's trajectory was modified by contact with the wall" (interaction history)

### 6.2 Self-Contained Representation

We define a self-contained representation as one that encodes:

1. **What is** — current object properties (position, velocity, shape)
2. **What has happened** — temporal context (recent trajectory, interaction history)
3. **What will happen** — predicted evolution (physical tendencies, momentum)
4. **What I can do** — affordances (reachable, manipulable, relevant)

This is analogous to how language embeddings work: a word vector encodes not just the word's definition, but its relationships, contexts, and implications.

### 6.3 Temporal Mixing as the Key Mechanism

The mechanism for achieving self-containment is **temporal mixing**: the state at time t is a function of the current observation *and* a window of recent observations.

```
State_t = f(obs_t, obs_{t-1}, ..., obs_{t-H})
```

This is not simply stacking frames (as in standard RL). It is a learned compression of temporal context into the state vector. The compression must:
- Preserve relevant temporal patterns (trends, cycles, interactions)
- Discard irrelevant temporal variation (noise, lighting changes)
- Maintain causal directionality (past influences future, not vice versa)

**Ablation evidence.** In our experiments, H=1 (single frame) produces states that are "clean but dead." H=6 (six historical frames) produces states that capture temporal trends and interaction history. The improvement in OOD generalization is significant: the dynamics model can distinguish between "stationary because no force" and "stationary because in contact with wall"—a distinction invisible from a single frame.

---

## 7. Implications and Open Questions

### 7.1 For Architecture Design

- **Frame-as-Token is the right granularity.** Object-centric representations (multiple slots per frame) add complexity without clear benefit when the consumer is powerful. A single frame-essence vector, processed by a causal transformer, provides a cleaner interface.

- **Persistent state is essential.** Without persistent state, the world model cannot accumulate temporal context or maintain object permanence across occlusions. The state vector is the world model's "memory."

- **Action should be decoupled.** Baking action into the world model architecture prevents scaling to new embodiments and prevents learning from actionless data. The tool interface is the right abstraction.

### 7.2 For Training

- **Pretrain on any video.** The base model should learn from YouTube, simulation renders, egocentric video—any source of visual data, with or without action labels. Action-labeled data is only needed for the tool interface.

- **Self-supervised objectives are sufficient.** Next-frame prediction (or next-state prediction) provides enough signal for learning physical dynamics. No reward, no language, no human annotation is needed for the base model.

- **Scale matters.** Like LLMs, world models need to be large enough to store the vast knowledge required for physical understanding. The compression ratio is less important than the model's capacity.

### 7.3 For Evaluation

- **OOD generalization is the right metric.** In-distribution performance measures memorization. OOD generalization measures understanding. The right test is: can the model handle new object shapes, new layouts, new physical configurations?

- **Consumer performance, not encoder quality.** Evaluate the world model by how well its downstream consumer (MPC, RL, planner) performs, not by reconstruction error or prediction MSE.

- **Causal contrast is the diagnostic.** The difference between natural-evolution prediction and intervention prediction measures the model's causal understanding. If this difference is zero, the model does not understand causation.

### 7.4 Open Questions

1. **Scaling laws for world models.** Do world models exhibit scaling laws analogous to LLMs? Does more video data → better physical understanding → better OOD generalization?

2. **What is the right tool granularity?** Low-level (`move_joints`) vs. high-level (`push_object_to_pose`)? This likely depends on task complexity and data availability.

3. **When is causal decomposition necessary?** If the consumer is powerful enough (e.g., a large transformer-based dynamics model), does causal decomposition still help? Or is it only beneficial for simple consumers?

4. **How to handle continuous dynamics?** LLMs operate on discrete tokens. Physical dynamics are continuous. How should the frame encoder handle continuous state transitions?

5. **Cross-embodiment transfer.** Can the same base model, with different tool interfaces, control a robotic arm, a drone, and a humanoid hand? This is the ultimate test of the decoupling hypothesis.

---

## 8. Research Program

We propose a systematic research program to validate these claims:

### Paper 1: Causal Decomposition for Structured States
- **Question:** Does causal decomposition improve OOD generalization for simple consumers?
- **Method:** Compare Flat, Object-Centric, Causality-Aware encoders with fixed CEM-MPC
- **Status:** In progress (MPPI Stage 2C sweep running)

### Paper 2: JEPA for Automatic Causal Discovery
- **Question:** Can JEPA learn causal decomposition from visual input without supervision?
- **Method:** C-JEPA with OOD evaluation
- **Connection:** Paper 1 → Paper 2 changes input modality (structured → visual)

### Paper 3: World Model to Executor Interface
- **Question:** Can learned representations be directly consumed by MPC/RL? Can they transfer across hardware?
- **Method:** JEPA representation → MPC/RL → real robot (SO-101)
- **Connection:** Paper 2 → Paper 3 adds the consumer and real-world transfer

### Paper 4: Agentic World Model
- **Question:** Does the full LLM-style pretrain → tool-use → agentic path work for physical world models?
- **Method:** Video pretraining + tool interface + multi-step planning
- **Connection:** Paper 3 → Paper 4 generalizes to the full framework

---

## 9. Conclusion

The development of Large Language Models revealed a clean trajectory: from passive understanding to tool-augmented agency. We argue that this trajectory is not specific to language—it is the correct blueprint for building physical world models.

Our position rests on five interconnected claims:
1. World states are compressible, but compression structure matters more than ratio
2. Representations and consumers are inseparable—evaluate the pair, not the representation alone
3. Causal decomposition serves the consumer, not the representation
4. Self-contained representations (encoding what is, what was, what will be, what can be done) are necessary
5. The LLM development path (pretrain → tool-use → agentic) is the correct architectural blueprint

These claims are not independent. They form a coherent worldview: the right world model is one that compresses physical reality into self-contained representations, organized by causal structure, designed for easy consumption, and developed through the same staged approach that made LLMs successful.

We do not claim that this is the only path. But we claim that it is the most promising path toward scalable, generalizable physical intelligence. The alternative—baking action into the architecture, training on narrow action-labeled data, evaluating on in-distribution benchmarks—has produced impressive demos but has not produced understanding.

The question is not whether world models can be built. The question is whether they can be built to *understand*. We believe the answer is yes, and the path is clear.

---

## References

- Hafner et al. "DreamerV3: Mastering Diverse Domains through World Models." 2023.
- Micheli et al. "Transformers are Sample-Efficient World Models." ICLR 2023.
- Bardes et al. "V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video." 2024.
- Meta AI. "V-JEPA 2." 2025.
- OpenAI. "GPT-4 Technical Report." 2023.
- Pearl. "Causality." Cambridge University Press, 2009.
- LeCun. "A Path Towards Autonomous Machine Intelligence." 2022.
- Team et al. "Octo: An Open-Source Generalist Robot Policy." 2024.
- Google DeepMind. "RT-2: Vision-Language-Action Models." 2024.
- NVIDIA. "Cosmos World Foundation Model Platform." 2025.
- Cheang et al. "GR-1: Unleashing Large-Scale Video Pretraining." 2024.
- Locatello et al. "Object-Centric Learning with Slot Attention." NeurIPS 2020.
- Robine et al. "Transformer-based World Models." 2023.
- Zhang et al. "STORM: Efficient Stochastic Transformer-based World Models for Reinforcement Learning." 2024.

---

*Draft v2.0 | 2026-05-20 | Based on systematic architectural exploration and multi-day discussion*
