"""
Action-Inject World Model — action injected at input level (EE token).

Design (Option B):
    - State tokens [B, H, N, 16] + action [B, 2]
    - Action appended to EE token (idx 0) as features 16, 17
    - Other tokens padded with zeros at features 16, 17
    - Augmented tokens [B, H, N, 18] fed to encoder
    - Dynamics head takes z only (no action concatenation)

This is cleaner than latent-space concatenation because:
    - Encoder sees action from the first layer
    - Action has clear physical meaning (EE velocity command)
    - Same transformer attention can learn action-state interactions

Old code (RIGWorldModel with latent concat) is NOT modified.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.models.encoders import (
    FlatEncoder,
    ObjectCentricEncoder,
    CausalityAwareEncoder,
)


# ── Feature indices in canonical_state16 ──
IDX_EE = 0


class NoActionDynamicsHead(nn.Module):
    """Dynamics head that predicts delta from z only (no action input).

    Action is already embedded in the state via input injection,
    so this head just maps z → delta.
    """

    def __init__(self, z_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, z_dim] → pred_delta: [B, 3]"""
        return self.net(z)


class ActionInjectModel(nn.Module):
    """World model with action injection at the input level.

    Instead of concatenating action with the latent z (like RIGWorldModel),
    this model appends action to the EE token's raw features before encoding.

    Architecture:
        state [B, H, N, 16] + action [B, 2]
            → augment EE token: [B, H, N, 18]
            → Encoder (raw_token_dim=18)
            → z [B, z_dim]
            → NoActionDynamicsHead
            → pred_delta [B, 3]

    The encoder's transformer layers can now attend to action information
    from the first layer, enabling richer action-state interaction.
    """

    def __init__(
        self,
        encoder_type: Literal["flat", "object_centric", "causality_aware"] = "flat",
        history_len: int = 6,
        num_tokens: int = 6,
        raw_token_dim: int = 16,
        action_dim: int = 2,
        frame_embed_dim: int = 128,
        d_model: int = 128,
        transformer_layers: int = 2,
        nhead: int = 4,
        ffn_dim: int = 256,
        gru_hidden: int = 256,
        head_hidden_dim: int = 256,
        slot_dim: int = 32,
        valid_flag_index: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.history_len = history_len
        self.num_tokens = num_tokens
        self.raw_token_dim = raw_token_dim
        self.action_dim = action_dim
        self.augmented_token_dim = raw_token_dim + action_dim  # 16 + 2 = 18
        self.z_dim = gru_hidden

        # ── Encoder (accepts augmented 18-dim tokens) ──
        if encoder_type == "flat":
            self.encoder = FlatEncoder(
                history_len=history_len,
                num_tokens=num_tokens,
                raw_token_dim=self.augmented_token_dim,  # 18
                frame_embed_dim=frame_embed_dim,
                gru_hidden=gru_hidden,
            )
        elif encoder_type == "object_centric":
            self.encoder = ObjectCentricEncoder(
                history_len=history_len,
                num_tokens=num_tokens,
                raw_token_dim=self.augmented_token_dim,  # 18
                d_model=d_model,
                transformer_layers=transformer_layers,
                nhead=nhead,
                ffn_dim=ffn_dim,
                gru_hidden=gru_hidden,
                valid_flag_index=valid_flag_index,
                dropout=dropout,
            )
        elif encoder_type == "causality_aware":
            self.encoder = CausalityAwareEncoder(
                history_len=history_len,
                num_tokens=num_tokens,
                raw_token_dim=self.augmented_token_dim,  # 18
                d_model=d_model,
                transformer_layers=transformer_layers,
                nhead=nhead,
                ffn_dim=ffn_dim,
                gru_hidden=gru_hidden,
                slot_dim=slot_dim,
                valid_flag_index=valid_flag_index,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # ── Dynamics head (z only, no action) ──
        self.dynamics_head = NoActionDynamicsHead(
            z_dim=gru_hidden,
            hidden_dim=head_hidden_dim,
        )

        # ── Subgoal head (unchanged) ──
        from src.models.heads import SubgoalHead
        self.subgoal_head = SubgoalHead(
            z_dim=gru_hidden,
            hidden_dim=head_hidden_dim,
        )

    def _augment_state(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Append action to EE token, zero-pad other tokens.

        Args:
            x: [B, H, N, D_raw=16] state tokens
            action: [B, action_dim=2]

        Returns:
            augmented: [B, H, N, D_raw+action_dim=18]
        """
        B, H, N, D = x.shape
        device = x.device

        # Zero-pad all tokens to augmented dimension
        padding = torch.zeros(B, H, N, self.action_dim, device=device, dtype=x.dtype)
        augmented = torch.cat([x, padding], dim=-1)  # [B, H, N, 18]

        # Inject action into EE token (index 0) at features 16, 17
        action_expanded = action[:, None, None, :].expand(B, H, 1, self.action_dim)
        augmented[:, :, IDX_EE, D:] = action_expanded.squeeze(2)  # [B, H, 2]

        return augmented

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_slots: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B, H, N, D_raw=16] structured state
            action: [B, action_dim=2]
            mask: optional [B, H, N] valid-token mask
            return_slots: if True and causality_aware, return diagnostic slots

        Returns:
            dict with:
                z: [B, z_dim]
                pred_delta: [B, 3]
                pred_subgoal: [B, 3]
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x [B,H,N,D], got {x.shape}")
        if x.shape[1] != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {x.shape[1]}")
        if x.shape[2] != self.num_tokens:
            raise ValueError(f"Expected num_tokens={self.num_tokens}, got {x.shape[2]}")
        if x.shape[3] != self.raw_token_dim:
            raise ValueError(f"Expected raw_token_dim={self.raw_token_dim}, got {x.shape[3]}")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action.shape[-1]}")

        # ── Step 1: Augment state with action ──
        x_aug = self._augment_state(x, action)  # [B, H, N, 18]

        # ── Step 2: Encode ──
        if self.encoder_type == "causality_aware" and return_slots:
            enc_out = self.encoder(x_aug, mask=mask, return_slots=True)
            z = enc_out["z"]
        else:
            z = self.encoder(x_aug, mask=mask)
            enc_out = None

        # ── Step 3: Predict (no action in head — it's already in z) ──
        pred_delta = self.dynamics_head(z)
        pred_subgoal = self.subgoal_head(z)

        out = {
            "z": z,
            "pred_delta": pred_delta,
            "pred_subgoal": pred_subgoal,
        }

        if enc_out is not None:
            out.update({
                "z_stable": enc_out.get("z_stable"),
                "z_dynamics": enc_out.get("z_dynamics"),
                "z_affordance": enc_out.get("z_affordance"),
                "z_nuisance": enc_out.get("z_nuisance"),
            })

        return out


# ── Rollout wrapper for action-inject model ──

class ActionInjectRolloutModel:
    """Rollout model for ActionInjectModel.

    Same interface as LearnedRolloutModel, but action is injected
    at the input level instead of the dynamics head.
    """

    IDX_EE, IDX_OBJ, IDX_GOAL = 0, 1, 2
    FEAT_X, FEAT_Y, FEAT_SIN_THETA, FEAT_COS_THETA = 0, 1, 2, 3
    FEAT_VX, FEAT_VY, FEAT_OMEGA = 4, 5, 6

    def __init__(
        self,
        model: ActionInjectModel,
        device: torch.device | str = "cpu",
        normalizer: object | None = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.normalizer = normalizer
        self.model.to(self.device)
        self.model.eval()

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.normalizer is not None:
            return self.normalizer.transform(state)
        return state

    @torch.no_grad()
    def forward_step(
        self,
        state_tokens: np.ndarray,
        action: np.ndarray,
    ):
        """Single-step prediction. Same interface as LearnedRolloutModel."""
        from src.planners.rollout_model import LearnedRolloutStepResult

        squeeze = False
        if state_tokens.ndim == 3:
            state_tokens = state_tokens[np.newaxis]
            action = action[np.newaxis]
            squeeze = True

        state_norm = self._normalize_state(state_tokens)
        state_t = torch.from_numpy(state_norm).float().to(self.device)
        action_t = torch.from_numpy(action).float().to(self.device)

        out = self.model(state_t, action_t)
        delta_pose = out["pred_delta"].cpu().numpy()

        # Current object pose
        current_obj = state_tokens[:, -1, self.IDX_OBJ, :]
        current_xy = current_obj[:, [self.FEAT_X, self.FEAT_Y]]
        current_theta = np.arctan2(
            current_obj[:, self.FEAT_SIN_THETA],
            current_obj[:, self.FEAT_COS_THETA],
        )

        pred_xy = current_xy + delta_pose[:, :2]
        pred_theta = np.arctan2(
            np.sin(current_theta + delta_pose[:, 2]),
            np.cos(current_theta + delta_pose[:, 2]),
        )
        pred_object_pose = np.stack([pred_xy[:, 0], pred_xy[:, 1], pred_theta], axis=-1)

        if squeeze:
            delta_pose = delta_pose[0]
            pred_object_pose = pred_object_pose[0]

        return LearnedRolloutStepResult(
            delta_pose=delta_pose,
            pred_object_pose=pred_object_pose,
        )

    @torch.no_grad()
    def rollout_sequence(
        self,
        initial_state: np.ndarray,
        action_seq: np.ndarray,
    ):
        """Multi-step rollout. Same interface as LearnedRolloutModel."""
        from src.planners.rollout_model import LearnedRolloutResult

        squeeze = False
        if initial_state.ndim == 3:
            initial_state = initial_state[np.newaxis]
            action_seq = action_seq[np.newaxis]
            squeeze = True

        B = initial_state.shape[0]
        horizon = action_seq.shape[1]

        init_obj = initial_state[:, -1, self.IDX_OBJ, :]
        init_xy = init_obj[:, [self.FEAT_X, self.FEAT_Y]]
        init_theta = np.arctan2(
            init_obj[:, self.FEAT_SIN_THETA],
            init_obj[:, self.FEAT_COS_THETA],
        )
        init_ee = initial_state[:, -1, self.IDX_EE, :]
        init_ee_xy = init_ee[:, [self.FEAT_X, self.FEAT_Y]]

        object_traj = np.zeros((B, horizon + 1, 3), dtype=np.float64)
        delta_traj = np.zeros((B, horizon, 3), dtype=np.float64)
        ee_traj = np.zeros((B, horizon + 1, 2), dtype=np.float64)

        object_traj[:, 0] = np.stack([init_xy[:, 0], init_xy[:, 1], init_theta], axis=-1)
        ee_traj[:, 0] = init_ee_xy

        current_state = initial_state.copy()

        for t in range(horizon):
            action_t = action_seq[:, t]
            step_result = self.forward_step(current_state, action_t)

            delta_traj[:, t] = step_result.delta_pose
            object_traj[:, t + 1] = step_result.pred_object_pose

            # EE update: displacement = action_phys * control_dt
            # (action_seq is physical velocity, control_dt=0.1)
            ee_traj[:, t + 1] = ee_traj[:, t] + action_t * 0.1

            # Update state for next step
            current_state = self._update_state(
                current_state,
                step_result.pred_object_pose,
                ee_traj[:, t + 1],
            )

        if squeeze:
            object_traj = object_traj[0]
            delta_traj = delta_traj[0]
            ee_traj = ee_traj[0]

        return LearnedRolloutResult(
            object_traj=object_traj,
            delta_traj=delta_traj,
            ee_traj=ee_traj,
        )

    def _update_state(
        self,
        current_state: np.ndarray,
        new_object_pose: np.ndarray,
        new_ee_pos: np.ndarray,
    ) -> np.ndarray:
        """Update state tokens after a rollout step."""
        B, H, N, D = current_state.shape
        updated = current_state.copy()

        # Shift history
        updated[:, :-1] = updated[:, 1:]
        newest = updated[:, -1]

        # Update EE
        newest[:, self.IDX_EE, self.FEAT_X] = new_ee_pos[:, 0]
        newest[:, self.IDX_EE, self.FEAT_Y] = new_ee_pos[:, 1]

        # Update object
        newest[:, self.IDX_OBJ, self.FEAT_X] = new_object_pose[:, 0]
        newest[:, self.IDX_OBJ, self.FEAT_Y] = new_object_pose[:, 1]
        newest[:, self.IDX_OBJ, self.FEAT_SIN_THETA] = np.sin(new_object_pose[:, 2])
        newest[:, self.IDX_OBJ, self.FEAT_COS_THETA] = np.cos(new_object_pose[:, 2])

        # Clear velocities
        newest[:, self.IDX_OBJ, self.FEAT_VX] = 0.0
        newest[:, self.IDX_OBJ, self.FEAT_VY] = 0.0
        newest[:, self.IDX_OBJ, self.FEAT_OMEGA] = 0.0

        updated[:, -1] = newest
        return updated


# ── Loading helper ──

def load_action_inject_model(
    checkpoint_path: str,
    encoder_type: str = "flat",
    device: str = "cpu",
    normalizer_path: str | None = None,
) -> ActionInjectRolloutModel:
    """Load a trained ActionInjectModel from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        encoder_type: One of "flat", "object_centric", "causality_aware".
        device: Torch device.
        normalizer_path: Optional path to normalizer JSON.

    Returns:
        ActionInjectRolloutModel ready for rollout.
    """
    from src.data.state_normalizer import StateNormalizer

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect action_inject mode from checkpoint
    state_dict = checkpoint["model_state_dict"]
    is_action_inject = any("augmented_token_dim" in k or "NoAction" in str(type(k))
                          for k in state_dict.keys())
    # Better detection: check if dynamics_head has no action_embed
    has_action_embed = any("action_embed" in k for k in state_dict.keys())

    # Infer augmented_token_dim from encoder weight shapes
    # token_mlp.0.weight shape is [d_model, raw_token_dim] for object_centric
    # frame_mlp.0.weight shape is [frame_embed_dim, N*raw_token_dim] for flat
    if encoder_type == "flat":
        # frame_mlp.0.weight: [frame_embed_dim, N * D]
        w_shape = state_dict["encoder.frame_mlp.0.weight"].shape
        n_tokens = 6
        augmented_dim = w_shape[1] // n_tokens
    elif encoder_type in ("object_centric", "causality_aware"):
        # token_mlp.0.weight: [d_model, raw_token_dim]
        w_shape = state_dict["encoder.token_mlp.0.weight"].shape
        augmented_dim = w_shape[1]
    else:
        augmented_dim = 18  # default

    raw_token_dim = augmented_dim - 2  # 18 - 2 = 16

    model = ActionInjectModel(
        encoder_type=encoder_type,
        raw_token_dim=raw_token_dim,
        gru_hidden=256,
        d_model=128,
        head_hidden_dim=256,
    )

    # Patch lastframe encoder if needed
    if "encoder._lf_proj.weight" in state_dict:
        from src.models.encoders import FlatEncoder, ObjectCentricEncoder, CausalityAwareEncoder

        def _patch_lastframe(enc):
            if isinstance(enc, CausalityAwareEncoder):
                if hasattr(enc, 'object_backbone'):
                    _patch_lastframe(enc.object_backbone)
                return
            if isinstance(enc, FlatEncoder):
                frame_dim = enc.gru.input_size
                gru_h = enc.gru.hidden_size
            else:
                frame_dim = enc.d_model
                gru_h = enc.gru_hidden
            proj = nn.Linear(frame_dim, gru_h)
            enc.add_module('_lf_proj', proj)

            def _lf_fwd(x, mask=None, **kw):
                b, h, n, d = x.shape
                x_last = x[:, -1, :, :]
                if isinstance(enc, FlatEncoder):
                    fe = enc.frame_mlp(x_last.reshape(b, n * d))
                    return enc._lf_proj(fe)
                if mask is not None:
                    vf = mask[:, -1, :].bool()
                else:
                    vf = x_last[:, :, enc.valid_flag_index] > 0.5
                te = enc.token_mlp(x_last)
                nt = min(n, len(enc.token_type_ids))
                tid = enc.token_type_ids[:nt].unsqueeze(0).expand(b, nt)
                te[:, :nt] = te[:, :nt] + enc.type_embedding(tid)
                to = enc.transformer(te, src_key_padding_mask=~vf)
                vf_f = vf.float().unsqueeze(-1)
                pooled = (to * vf_f).sum(dim=1)
                denom = vf_f.sum(dim=1).clamp(min=1.0)
                return enc._lf_proj(pooled / denom)

            enc.forward = _lf_fwd

        _patch_lastframe(model.encoder)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    normalizer = None
    if normalizer_path and Path(normalizer_path).exists():
        normalizer = StateNormalizer.load(normalizer_path)

    return ActionInjectRolloutModel(model=model, device=device, normalizer=normalizer)
