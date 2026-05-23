# src/models/v2_encoder.py
"""Generic encoder for visual_structured_state_v2 with variable feature dimension."""

import torch
import torch.nn as nn


class V2FlatEncoder(nn.Module):
    """Flat MLP+GRU encoder for v2 features [B, H, N_features]."""

    def __init__(self, feature_dim: int, history_len: int = 6,
                 frame_embed_dim: int = 128, gru_hidden: int = 256):
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(feature_dim, frame_embed_dim),
            nn.ReLU(),
            nn.Linear(frame_embed_dim, frame_embed_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=frame_embed_dim, hidden_size=gru_hidden, batch_first=True)

    def forward(self, x, mask=None):
        """x: [B, H, N_features] → z: [B, gru_hidden]"""
        emb = self.frame_mlp(x)
        _, h_last = self.gru(emb)
        return h_last[-1]


class V2WorldModel(nn.Module):
    """World model for v2 features. Compatible with RIGWorldModel output interface."""

    def __init__(self, feature_dim: int, history_len: int = 6,
                 gru_hidden: int = 256, head_hidden_dim: int = 256):
        super().__init__()
        self.encoder = V2FlatEncoder(feature_dim, history_len, gru_hidden=gru_hidden)
        self.dynamics_head = nn.Sequential(
            nn.Linear(gru_hidden + 2, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 3),
        )
        self.subgoal_head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 3),
        )
        self.z_dim = gru_hidden
        self.action_dim = 2
        self.history_len = history_len
        self.feature_dim = feature_dim

    def forward(self, x, action):
        """x: [B, H, N_features], action: [B, 2] → dict with z, pred_delta, pred_subgoal"""
        z = self.encoder(x)
        pred_delta = self.dynamics_head(torch.cat([z, action], dim=-1))
        pred_subgoal = self.subgoal_head(z)
        return {'z': z, 'pred_delta': pred_delta, 'pred_subgoal': pred_subgoal}
