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