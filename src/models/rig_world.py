from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.models.encoders import (
    FlatEncoder,
    ObjectCentricEncoder,
    CausalityAwareEncoder,
)
from src.models.heads import DynamicsHead, AdaLNDynamicsHead, SubgoalHead


ModelType = Literal["flat", "object_centric", "causality_aware"]


class RIGWorldModel(nn.Module):
    """
    Unified high-level world model wrapper for Paper 1.

    RIG = Representation Interface for Generalization.

    This wrapper provides a single interface for:

        FlatEncoder
        ObjectCentricEncoder
        CausalityAwareEncoder

    Shared output:

        {
            "z": [B, 256],
            "pred_delta": [B, 3],
            "pred_subgoal": [B, 3],
        }

    For causality_aware model, it can also return diagnostic slots:

        {
            "z_stable": [B, 32],
            "z_dynamics": [B, 32],
            "z_affordance": [B, 32],
            "z_nuisance": [B, 32],
        }

    Important:
        This wrapper is not itself the research comparison.
        The comparison is between the encoder variants under the same heads,
        same loss, same planner, and same data protocol.
    """

    def __init__(
        self,
        model_type: ModelType = "flat",
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
        use_action_embed: bool = True,
        dynamics_head_type: str = "concat",  # "concat" or "adaln"
    ):
        super().__init__()

        self.model_type = model_type
        self.history_len = history_len
        self.num_tokens = num_tokens
        self.raw_token_dim = raw_token_dim
        self.action_dim = action_dim
        self.z_dim = gru_hidden
        self.slot_dim = slot_dim

        if model_type == "flat":
            self.encoder = FlatEncoder(
                history_len=history_len,
                num_tokens=num_tokens,
                raw_token_dim=raw_token_dim,
                frame_embed_dim=frame_embed_dim,
                gru_hidden=gru_hidden,
            )

        elif model_type == "object_centric":
            self.encoder = ObjectCentricEncoder(
                history_len=history_len,
                num_tokens=num_tokens,
                raw_token_dim=raw_token_dim,
                d_model=d_model,
                transformer_layers=transformer_layers,
                nhead=nhead,
                ffn_dim=ffn_dim,
                gru_hidden=gru_hidden,
                valid_flag_index=valid_flag_index,
                dropout=dropout,
            )

        elif model_type == "causality_aware":
            self.encoder = CausalityAwareEncoder(
                history_len=history_len,
                num_tokens=num_tokens,
                raw_token_dim=raw_token_dim,
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
            raise ValueError(
                f"Unknown model_type={model_type}. "
                "Expected one of: flat, object_centric, causality_aware."
            )

        self.dynamics_head_type = dynamics_head_type

        if dynamics_head_type == "adaln":
            self.dynamics_head = AdaLNDynamicsHead(
                z_dim=gru_hidden,
                action_dim=action_dim,
                hidden_dim=head_hidden_dim,
                use_action_embed=use_action_embed,
            )
        else:
            self.dynamics_head = DynamicsHead(
                z_dim=gru_hidden,
                action_dim=action_dim,
                hidden_dim=head_hidden_dim,
                use_action_embed=use_action_embed,
            )

        self.subgoal_head = SubgoalHead(
            z_dim=gru_hidden,
            hidden_dim=head_hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_slots: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x:
                Structured state tensor with shape [B, H, N, D_raw].

            action:
                Action tensor with shape [B, action_dim].

            mask:
                Optional valid-token mask with shape [B, H, N].
                1 / True = valid token.
                0 / False = invalid padding token.

            return_slots:
                If True and model_type == "causality_aware",
                also return diagnostic causal slots.

        Returns:
            Dictionary with at least:
                z:            [B, 256]
                pred_delta:   [B, 3]
                pred_subgoal: [B, 3]

            If model_type == "causality_aware" and return_slots=True,
            also returns:
                z_stable:     [B, slot_dim]
                z_dynamics:   [B, slot_dim]
                z_affordance: [B, slot_dim]
                z_nuisance:   [B, slot_dim]
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected x with shape [B, H, N, D_raw], got {tuple(x.shape)}"
            )

        if x.shape[1] != self.history_len:
            raise ValueError(
                f"Expected history_len={self.history_len}, got {x.shape[1]}"
            )

        if x.shape[2] != self.num_tokens:
            raise ValueError(
                f"Expected num_tokens={self.num_tokens}, got {x.shape[2]}"
            )

        if x.shape[3] != self.raw_token_dim:
            raise ValueError(
                f"Expected raw_token_dim={self.raw_token_dim}, got {x.shape[3]}"
            )

        if action.dim() != 2:
            raise ValueError(
                f"Expected action with shape [B, action_dim], got {tuple(action.shape)}"
            )

        if x.shape[0] != action.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x batch={x.shape[0]}, "
                f"action batch={action.shape[0]}"
            )

        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {action.shape[-1]}"
            )

        if mask is not None and mask.shape != x.shape[:3]:
            raise ValueError(
                f"Expected mask shape {tuple(x.shape[:3])}, got {tuple(mask.shape)}"
            )

        if self.model_type == "causality_aware" and return_slots:
            enc_out = self.encoder(x, mask=mask, return_slots=True)
            z = enc_out["z"]
        else:
            z = self.encoder(x, mask=mask)
            enc_out = None

        pred_delta = self.dynamics_head(z, action)
        pred_subgoal = self.subgoal_head(z)

        out = {
            "z": z,
            "pred_delta": pred_delta,
            "pred_subgoal": pred_subgoal,
        }

        if enc_out is not None:
            out.update(
                {
                    "z_stable": enc_out["z_stable"],
                    "z_dynamics": enc_out["z_dynamics"],
                    "z_affordance": enc_out["z_affordance"],
                    "z_nuisance": enc_out["z_nuisance"],
                }
            )

        return out