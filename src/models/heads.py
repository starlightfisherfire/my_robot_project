import torch
import torch.nn as nn


class AdaLNDynamicsHead(nn.Module):
    """
    AdaLN-based dynamics head — drop-in replacement for DynamicsHead.

    Action conditioning via Adaptive Layer Normalization (inspired by LeWM):
        action → MLP → (gamma, beta)
        z' = (1 + gamma) * LayerNorm(z) + beta
        pred_delta = MLP(z')

    Same interface: forward(z, action) → [B, 3]
    Same parameter count: ~197K (vs concat's ~197K)
    """

    def __init__(self, z_dim: int = 256, action_dim: int = 2, hidden_dim: int = 256,
                 action_embed_dim: int = 64, use_action_embed: bool = True):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim

        # AdaLN modulator: action → (gamma, beta) for z
        self.adaln = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),  # gamma and beta
        )
        # Zero-init so modulation starts as identity
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

        self.norm = nn.LayerNorm(z_dim)

        # Prediction MLP (takes z_dim, not z_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, z, action):
        # AdaLN modulation
        params = self.adaln(action)        # [B, 2*z_dim]
        gamma, beta = params.chunk(2, dim=-1)  # each [B, z_dim]
        z_normed = self.norm(z)
        z_modulated = (1 + gamma) * z_normed + beta
        return self.net(z_modulated)


class DynamicsHead(nn.Module):
    def __init__(self, z_dim: int = 256, action_dim: int = 2, hidden_dim: int = 256,
                 action_embed_dim: int = 64, use_action_embed: bool = True):
        super().__init__()
        self.use_action_embed = use_action_embed
        if use_action_embed:
            self.action_embed = nn.Linear(action_dim, action_embed_dim)
            in_dim = z_dim + action_embed_dim
        else:
            self.action_embed = None
            in_dim = z_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, z, action):
        if self.use_action_embed and self.action_embed is not None:
            a_emb = self.action_embed(action)
            x = torch.cat([z, a_emb], dim=-1)
        else:
            x = torch.cat([z, action], dim=-1)
        return self.net(x)


class SubgoalHead(nn.Module):
    def __init__(self, z_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, z):
        return self.net(z)