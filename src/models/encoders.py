from __future__ import annotations

import torch
import torch.nn as nn



class FlatEncoder(nn.Module):
    def __init__(
        self,
        history_len: int = 6,
        num_tokens: int = 6,
        raw_token_dim: int = 16,
        frame_embed_dim: int = 128,
        gru_hidden: int = 256,
    ):
        super().__init__()

        self.history_len = history_len
        self.num_tokens = num_tokens
        self.raw_token_dim = raw_token_dim
        self.frame_input_dim = num_tokens * raw_token_dim

        self.frame_mlp = nn.Sequential(
            nn.Linear(self.frame_input_dim, frame_embed_dim),
            nn.ReLU(),
            nn.Linear(frame_embed_dim, frame_embed_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=frame_embed_dim,
            hidden_size=gru_hidden,
            batch_first=True,
        )

    def forward(self, x, mask=None):
        """
        x: [B, H, N, D_raw]
        return: [B, gru_hidden]
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B,H,N,D], got {x.shape}")

        b, h, n, d = x.shape
        x = x.reshape(b, h, n * d)

        frame_emb = self.frame_mlp(x)
        _, h_last = self.gru(frame_emb)

        z = h_last[-1]
        return z


class ObjectCentricEncoder(nn.Module):
    """
    Object-centric encoder for Paper 1.

    Input:
        x: [B, H, N, D_raw]

    Token order:
        0: end-effector
        1: manipulated object
        2: goal
        3: obstacle_1
        4: obstacle_2
        5: obstacle_3

    Token type ids:
        0: ee
        1: object
        2: goal
        3: obstacle

    Output:
        z: [B, gru_hidden]

    v0.1 design:
        per-token MLP
        + token-type embedding
        + Transformer over object tokens per frame
        + masked mean pooling
        + GRU over history
    """

    def __init__(
        self,
        history_len: int = 6,
        num_tokens: int = 6,
        raw_token_dim: int = 16,
        d_model: int = 128,
        transformer_layers: int = 2,
        nhead: int = 4,
        ffn_dim: int = 256,
        gru_hidden: int = 256,
        valid_flag_index: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_tokens != 6:
            raise ValueError(
                "ObjectCentricEncoder v0.1 assumes num_tokens=6 "
                "with token order [ee, object, goal, obstacle1, obstacle2, obstacle3]."
            )

        self.history_len = history_len
        self.num_tokens = num_tokens
        self.raw_token_dim = raw_token_dim
        self.d_model = d_model
        self.gru_hidden = gru_hidden
        self.valid_flag_index = valid_flag_index

        self.token_mlp = nn.Sequential(
            nn.Linear(raw_token_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Token types:
        # 0 = ee
        # 1 = manipulated object
        # 2 = goal
        # 3 = obstacle
        self.type_embedding = nn.Embedding(4, d_model)

        token_type_ids = torch.tensor([0, 1, 2, 3, 3, 3], dtype=torch.long)
        self.register_buffer("token_type_ids", token_type_ids, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_layers,
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Structured state tensor with shape [B, H, N, D_raw].

            mask:
                Optional valid-token mask with shape [B, H, N].
                1 / True = valid token.
                0 / False = invalid padding token.

                If mask is None, valid_mask is read from x[..., valid_flag_index].

        Returns:
            z:
                Object-centric history representation with shape [B, gru_hidden].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B, H, N, D], got {tuple(x.shape)}")

        b, h, n, d = x.shape

        if h != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {h}")

        if n != self.num_tokens:
            raise ValueError(f"Expected num_tokens={self.num_tokens}, got {n}")

        if d != self.raw_token_dim:
            raise ValueError(f"Expected raw_token_dim={self.raw_token_dim}, got {d}")

        if mask is None:
            valid_mask = x[..., self.valid_flag_index] > 0.5
        else:
            if mask.shape != (b, h, n):
                raise ValueError(
                    f"Expected mask shape {(b, h, n)}, got {tuple(mask.shape)}"
                )
            valid_mask = mask > 0.5

        # [B, H, N, D] -> [B*H, N, D]
        x_flat = x.reshape(b * h, n, d)
        valid_flat = valid_mask.reshape(b * h, n).bool()

        # Each frame must contain at least one valid token.
        # In the intended schema, ee/object/goal should always be valid.
        if not valid_flat.any(dim=1).all():
            raise ValueError("Each frame must contain at least one valid token.")

        # Per-token embedding: [B*H, N, D_raw] -> [B*H, N, d_model]
        token_emb = self.token_mlp(x_flat)

        # Add token-type embeddings.
        type_ids = self.token_type_ids[:n].unsqueeze(0).expand(b * h, n)
        token_emb = token_emb + self.type_embedding(type_ids)

        # PyTorch Transformer uses True for positions that should be masked.
        key_padding_mask = ~valid_flat

        # Object-token interaction within each frame.
        token_out = self.transformer(
            token_emb,
            src_key_padding_mask=key_padding_mask,
        )

        # Masked mean pooling over tokens.
        valid_float = valid_flat.float().unsqueeze(-1)  # [B*H, N, 1]
        pooled = (token_out * valid_float).sum(dim=1)   # [B*H, d_model]
        denom = valid_float.sum(dim=1).clamp(min=1.0)   # [B*H, 1]
        frame_emb = pooled / denom                      # [B*H, d_model]

        # [B*H, d_model] -> [B, H, d_model]
        frame_emb = frame_emb.reshape(b, h, self.d_model)

        # Temporal aggregation over history.
        _, h_last = self.gru(frame_emb)

        # h_last: [num_layers, B, gru_hidden]
        z = h_last[-1]  # [B, gru_hidden]

        return z