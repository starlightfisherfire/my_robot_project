# Model Design: Paper 1 High-Level Representation Models

Status: v0.1 implementation draft  
Scope: Paper 1  
Task: structured-state planar push-to-pose / Push-T-style pushing  
Planner: fixed CEM-MPC  

---

## 1. Purpose

This document defines the model design for Paper 1.

Paper 1 compares three high-level representation models under the same:

- structured object-state input;
- train / validation / test splits;
- state normalization rule;
- prediction heads;
- CEM-MPC planner;
- cost function;
- evaluation metrics.

The core comparison is:

```text
Flat high-level representation
vs
Object-centric non-causal high-level representation
vs
Causality-aware object-level high-level representation
```

The purpose is not to build a full visual world model in Paper 1. Instead, Paper 1 tests whether representation structure improves structural OOD generalization under a fixed planner.

The current scientific question is:

> Given privileged structured object-state input, do object-centric and causality-aware high-level representations reduce OOD degradation compared with a flat representation?

---

## 2. Overall Pipeline

The full Paper 1 pipeline is:

```text
raw structured object state
→ state normalizer
→ high-level representation model
    → Flat encoder
    → Object-centric encoder
    → Causality-aware encoder
→ temporal aggregation
→ prediction heads
    → dynamics head
    → subgoal head
→ rollout_model.py
→ fixed CEM-MPC planner
→ MuJoCo / SO-101 execution
→ success rate / OOD gap / failure analysis
```

The main comparison is the high-level representation model. The planner is fixed.

---

## 3. Input Format

Each sample is a short history of structured object states.

Default input shape:

```text
[B, H, N, D_raw] = [B, 6, 6, 16]
```

where:

- `B`: batch size;
- `H = 6`: history length;
- `N = 6`: number of tokens per frame;
- `D_raw = 16`: raw token feature dimension.

Each frame contains:

```text
1 end-effector token
1 manipulated object token
1 goal token
3 obstacle tokens
```

So each time step has:

```text
N_tokens = 6
```

Each token uses the same 16-dimensional schema:

```text
x
y
sin(theta)
cos(theta)
vx
vy
omega
size_x
size_y
shape_T
shape_L
shape_other
mass_norm
friction_norm
contact_flag
valid_flag
```

Some fields may be zero-filled when they are not meaningful for a token type. For example, the goal token may have zero velocity, and padded obstacle tokens should use `valid_flag = 0`.

---

## 4. State Normalizer

The state normalizer is a preprocessing module, not a neural network.

It handles:

- coordinate alignment;
- angle conversion;
- continuous feature normalization;
- obstacle padding and valid masks;
- shape one-hot fields;
- sim / real coordinate alignment if needed.

Important leakage rule:

```text
The normalizer can only be fitted on train or adaptation splits.
It must never be fitted on OOD test splits.
```

Allowed fitting sets include:

```text
train_sim_id
adapt_real_id
```

Forbidden fitting sets include:

```text
test_sim_id
test_sim_layout_ood
test_sim_shape_ood
test_real_id
test_real_layout_ood
test_real_shape_ood
```

Target file:

```text
src/data/state_normalizer.py
```

Expected minimal class:

```python
class StateNormalizer:
    def fit(self, episodes):
        ...

    def transform(self, state):
        ...

    def inverse_transform(self, normalized_state):
        ...

    def save(self, path):
        ...

    @classmethod
    def load(cls, path):
        ...
```

---

## 5. Encoder Variants

All three encoders receive the same normalized input:

```text
[B, 6, 6, 16]
```

All three encoders output a planner-facing representation:

```text
z: [B, 256]
```

This shared output size is a fairness constraint.

---

## 6. Flat Encoder

The flat encoder is the simplest baseline.

It ignores explicit object structure after input formatting. At each time step, all object tokens are flattened into one vector.

Per-frame input:

```text
[6, 16] → [96]
```

Architecture:

```text
[96] → MLP → [128]
history of 6 frame embeddings → GRU → [256]
```

Expected output:

```text
z_flat: [B, 256]
```

Target class:

```text
FlatEncoder
```

Target file:

```text
src/models/encoders.py
```

The flat encoder answers:

> If the model receives all structured state variables but no explicit object-level inductive bias, how well can it generalize?

---

## 7. Object-Centric Encoder

The object-centric encoder preserves token structure.

Each token is first projected into a shared embedding space:

```text
16 → 128
```

For each frame:

```text
[6, 16] → [6, 128]
```

A small Transformer encoder processes the object token set:

```text
d_model = 128
num_layers = 2
nhead = 4
ffn_dim = 256
```

Then token embeddings are pooled into a frame embedding.

Initial pooling choice:

```text
masked mean pooling
```

Temporal aggregation:

```text
history of 6 frame embeddings → GRU → [256]
```

Expected output:

```text
z_object: [B, 256]
```

Target class:

```text
ObjectCentricEncoder
```

Target file:

```text
src/models/encoders.py
```

The object-centric encoder answers:

> Is ordinary object-centric processing already enough for structural OOD generalization?

---

## 8. Causality-Aware Encoder

The causality-aware encoder builds on object-centric processing and adds an explicit factorized representation.

The intended slots are:

```text
z_stable:      [B, 32]
z_dynamics:    [B, 32]
z_affordance:  [B, 32]
z_nuisance:    [B, 32]
```

These slots are concatenated:

```text
[32, 32, 32, 32] → [128]
```

Then projected to the planner-facing representation:

```text
[128] → [256]
```

Expected output:

```text
z_causal: [B, 256]
```

The model should optionally return both the final representation and the slots:

```python
{
    "z": z,
    "z_stable": z_stable,
    "z_dynamics": z_dynamics,
    "z_affordance": z_affordance,
    "z_nuisance": z_nuisance,
}
```

Target class:

```text
CausalityAwareEncoder
```

Target file:

```text
src/models/encoders.py
```

The causality-aware encoder answers:

> Do mechanism-aware / causality-aware representation constraints provide additional robustness beyond ordinary object-centric processing?

Important wording rule:

```text
Use "causality-aware" or "mechanism-aware".
Do not claim that the model fully discovers true causal variables.
```

---

## 9. Prediction Heads

Prediction heads translate high-level representations into planner-facing quantities.

The heads are not the main contribution. They are the interface between representation learning and CEM-MPC.

Target file:

```text
src/models/heads.py
```

---

### 9.1 Dynamics Head

The dynamics head predicts object pose change.

Input:

```text
z:      [B, 256]
action: [B, action_dim]
```

Output:

```text
pred_delta_object_pose: [B, 3]
```

The three output dimensions are:

```text
delta_x
delta_y
delta_theta
```

This head is used by the learned rollout model.

---

### 9.2 Subgoal Head

The subgoal head predicts an intermediate target relative to the current object state or final goal.

Input:

```text
z: [B, 256]
```

Output:

```text
pred_subgoal_delta: [B, 3]
```

The three output dimensions are:

```text
subgoal_dx
subgoal_dy
subgoal_dtheta
```

For Paper 1 v0.1, the main heads are:

```text
dynamics head
subgoal head
```

The affordance head is deferred to v0.2.

---

### 9.3 Deferred Affordance Head

The affordance head is not required for the first runnable model.

Possible future outputs:

```text
contact side
push direction
rotate-then-translate flag
```

This head may be useful for shape OOD analysis, especially T-shape to L-shape generalization.

---

## 10. Losses

Target file:

```text
src/models/losses.py
```

The minimum v0.1 supervised loss is:

```text
L_total = L_dynamics + lambda_subgoal * L_subgoal
```

where:

```text
L_dynamics = MSE(pred_delta_object_pose, target_delta_object_pose)
L_subgoal  = MSE(pred_subgoal_delta, target_subgoal_delta)
```

For pose-like vectors, angle differences should be wrapped safely:

```text
dtheta → atan2(sin(dtheta), cos(dtheta))
```

Implementation rule:

```text
Avoid in-place tensor modification in loss functions.
```

For example, avoid:

```python
diff[:, 2] = wrapped_angle
```

Use non-in-place operations such as `torch.cat`.

Future v0.2 losses may include:

```text
affordance loss
slot regularization
invariance loss
causal / mechanism-aware regularization
```

These are not required for the first runnable model.

---

## 11. Rollout Interface to CEM-MPC

The learned model is used by:

```text
src/planners/rollout_model.py
```

The rollout interface receives:

```text
current state
candidate action sequence
learned model
normalizer
```

and returns:

```text
predicted future states
```

Conceptually:

```text
state_t + action_t
→ learned dynamics prediction
→ state_{t+1}
→ repeat over horizon
```

Important distinction:

```text
rollout_model.py = learned forward prediction interface
cem_mpc.py       = planner / optimizer
```

---

## 12. Planner Capacity Protocol

Before comparing learned representations, Paper 1 needs a planner capacity check.

The planner capacity check uses Oracle-MPC:

```text
current state
+ candidate action sequence
+ MuJoCo simulator rollout
→ predicted future state
→ cost
→ best action
```

Here, "oracle dynamics" means:

```text
MuJoCo simulator ground-truth dynamics inside the simulation environment.
```

It does not mean real-world true physics.

Purpose:

> If CEM-MPC fails even with MuJoCo oracle dynamics, then learned model failures cannot be attributed to representation quality alone.

Target document:

```text
docs/planner_capacity_protocol.md
```

Target config:

```text
configs/planner/cem_mpc_capacity.yaml
```

Target script:

```text
scripts/check_mpc_capacity.py
```

---

## 13. Fair Comparison Rules

The representation comparison must control the following:

- same task;
- same input format;
- same state normalizer rule;
- same train / validation / test splits;
- same prediction heads;
- same CEM-MPC planner;
- same cost function;
- same success metrics;
- same final representation dimension `[B, 256]`.

The planner should not be tuned separately for different encoders.

If planner parameters change, the change should apply to all model variants.

---

## 14. Evaluation Metrics

Main task metrics:

```text
success rate
final position error
final orientation error
OOD gap
collision rate
steps to success
failure code distribution
```

OOD gap is defined as:

```text
OOD gap = ID performance - OOD performance
```

For success rate:

```text
gap_success = success_ID - success_OOD
```

A smaller OOD gap indicates better robustness.

Target metric files:

```text
src/metrics/success_metrics.py
src/metrics/ood_gap.py
src/metrics/failure_analysis.py
```

---

## 15. Representation Diagnostics

Representation diagnostics are an analysis module, not the first runnable training objective.

Purpose:

> Analyze whether object-centric and causality-aware encoders reorganize the representation space into a more mechanism-aligned and OOD-stable geometry.

Initial diagnostics:

```text
embedding export
PCA / UMAP visualization
ID → OOD linear probe
representation shift
slot-specific probe for causal model
```

Deferred diagnostics:

```text
intrinsic dimension
persistent homology
topology shift
```

Target config:

```text
configs/analysis/representation_analysis.yaml
```

Target future files:

```text
src/analysis/embedding_export.py
src/analysis/representation_geometry.py
src/analysis/probes.py
src/analysis/intrinsic_dim.py
src/analysis/topology.py
```

Important rule:

```text
Do not let topology analysis block the first model and MPC pipeline.
```

---

## 16. Language-Derived Mechanism Priors

Language-derived mechanism priors are a long-term direction, not a main variable in Paper 1.

Safe use in Paper 1:

```text
mechanism label taxonomy
representation probes
failure analysis labels
```

Do not add LLM input to Paper 1 models.

Future direction:

```text
Paper 2:
visual/video world models learn object-relation-mechanism representations.

Paper 2.5:
language-derived mechanism graphs / priors help compositional OOD.

Paper 3:
action-relevant representations are consumed by executors.
```

---

## 17. Minimal Runnable Milestone

The first runnable milestone is:

```text
dummy structured state
→ FlatEncoder
→ DynamicsHead + SubgoalHead
→ loss
→ backward
```

Required shapes:

```text
input:        [B, 6, 6, 16]
z:            [B, 256]
pred_delta:   [B, 3]
pred_subgoal: [B, 3]
loss:         scalar
```

The current debug script is:

```text
scripts/debug_dummy_forward.py
```

Expected output:

```text
input: torch.Size([4, 6, 6, 16])
z: torch.Size([4, 256])
pred_delta: torch.Size([4, 3])
pred_subgoal: torch.Size([4, 3])
loss: ...
grad_norm: ...
backward ok
```

After this works, implement:

```text
StateNormalizer
ObjectCentricEncoder
CausalityAwareEncoder
RIGWorldModel wrapper
```

---

## 18. Deferred Modules

The following are important but not part of the first runnable model path:

```text
RGB input
C-JEPA / visual world model
Diffusion / Flow executor
LLM-conditioned world model
full topology analysis
real robot adaptation
```

These should not block Paper 1 v0.1.

---

## 19. Current Implementation Priority

Current priority order:

```text
1. dummy flat model forward/backward
2. state normalizer
3. object-centric encoder
4. causality-aware encoder
5. unified RIGWorldModel wrapper
6. metadata schema
7. reset template generation
8. oracle-MPC capacity check
9. sim data collection
10. learned model + MPC evaluation
11. OOD gap
12. representation diagnostics
13. real-ID adapted OOD
```

The immediate engineering rule is:

> First make the smallest system run. Then expand.
