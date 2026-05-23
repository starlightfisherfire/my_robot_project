"""
Tool Interface MVP — 模型定义

核心组件:
  1. StateEncoder (WM 简化版): 观测历史 → State_t
  2. ToolA_MLP: 快速 MLP 控制器
  3. ToolB_Diffusion: Diffusion Policy
  4. ToolSelector: 自适应选择器
  5. ToolInterface: 完整框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


# ============================================================
# 1. State Encoder (简化 WM)
# ============================================================

class StateEncoder(nn.Module):
    """
    从观测历史提取世界状态。
    
    简化版: GRU 替代完整 WM + Causal Transformer。
    后续可升级为 Frame Encoder + Prefix-LM Transformer。
    
    输入: obs_seq [B, T, obs_dim]
    输出: state [B, state_dim]
    """
    def __init__(self, obs_dim: int = 4, state_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=state_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.state_dim = state_dim
    
    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_seq: [B, T, obs_dim]  观测历史序列
        
        Returns:
            state: [B, state_dim]  全局状态
        """
        _, h_n = self.gru(obs_seq)
        return h_n[-1]  # 最后一层的 hidden state


# ============================================================
# 2. Tool A: MLP Controller (快速但单模)
# ============================================================

class ToolA_MLP(nn.Module):
    """
    快速 MLP 控制器。
    适合: 简单操作场景，大范围快速移动。
    特点: 单步推理, < 1ms, 但只能输出单模 (不适合对称/精确场景)。
    """
    def __init__(self, state_dim: int = 128, act_dim: int = 2,
                 hidden_dims: list = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
        Returns:
            action: [B, act_dim], 范围 [-1, 1]
        """
        return torch.tanh(self.net(state))


# ============================================================
# 3. Tool B: Diffusion Policy (精确但慢)
# ============================================================

class SinusoidalPosEmb(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, 1]
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb  # [B, dim]


class DiffusionDenoiser(nn.Module):
    """
    去噪网络 (1D, 无 UNet 架构的简化版 FiLM-conditioned MLP)
    
    后续可升级为 1D UNet 或 Transformer。
    """
    def __init__(self, act_dim: int = 2, state_dim: int = 128,
                 cond_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(32)
        
        # 时间嵌入投影
        self.time_proj = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        
        # FiLM conditioning: 从 (state, time_emb) → scale, shift
        self.cond_net = nn.Sequential(
            nn.Linear(state_dim + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # [scale, shift]
        )
        
        # 去噪 MLP
        self.denoise_net = nn.Sequential(
            nn.Linear(act_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, act_dim]  带噪声的动作
            t: [B, 1]        时间步
            state: [B, state_dim]
        
        Returns:
            predicted_noise: [B, act_dim]
        """
        B = x.shape[0]
        
        # 时间嵌入
        t_emb = self.time_emb(t)       # [B, 32]
        t_emb = self.time_proj(t_emb)  # [B, 128]
        
        # FiLM conditioning
        cond = torch.cat([state, t_emb], dim=-1)  # [B, state_dim+128]
        film_out = self.cond_net(cond)             # [B, hidden_dim*2]
        scale, shift = film_out.chunk(2, dim=-1)
        
        # FiLM: 对 denoise_net 的第一层输入做 modulation
        # 简化做法: 将 conditioning 拼接到输入
        h = torch.cat([x, scale * 0.1 + shift * 0.1], dim=-1)
        
        noise_pred = self.denoise_net(h)
        return noise_pred


class DiffusionScheduler:
    """DDPM 噪声调度器"""
    
    def __init__(self, n_steps: int = 100, beta_start: float = 0.0001,
                 beta_end: float = 0.02, scheduler: str = "cosine"):
        self.n_steps = n_steps
        
        if scheduler == "cosine":
            # Cosine schedule
            s = 0.008
            steps = torch.arange(n_steps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(
                (steps / n_steps + s) / (1 + s) * np.pi / 2
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            betas = torch.linspace(beta_start, beta_end, n_steps)
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向加噪"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        device = x0.device
        t_idx = t.long().cpu()
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t_idx].to(device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx].to(device)
        
        # 正确的维度广播
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        
        xt = sqrt_alpha * x0 + sqrt_one_minus * noise
        return xt, noise


class ToolB_Diffusion(nn.Module):
    """
    Diffusion Policy 动作生成器。
    适合: 精确操作场景，多模态动作分布。
    特点: K 步去噪推理，~5ms，能处理对称/精确场景。
    """
    def __init__(self, state_dim: int = 128, act_dim: int = 2,
                 cond_dim: int = 128, hidden_dim: int = 256,
                 n_steps: int = 100, n_infer_steps: int = 10):
        super().__init__()
        self.act_dim = act_dim
        self.n_steps = n_steps
        self.n_infer_steps = n_infer_steps
        
        self.denoiser = DiffusionDenoiser(act_dim, state_dim, cond_dim, hidden_dim)
        self.scheduler = DiffusionScheduler(n_steps)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练时: 随机时间步去噪
        """
        B = state.shape[0]
        device = state.device
        
        # 随机时间步 [1, n_steps], 减 1 转为 0-indexed
        t = torch.randint(0, self.n_steps, (B,), device=device)
        
        # 加噪 (使用 0-indexed t)
        noise = torch.randn(B, self.act_dim, device=device)
        xt, eps = self.scheduler.add_noise(action, t, noise)
        
        # 预测噪声 (使用 normalized timestep)
        t_norm = (t.float() + 1) / self.n_steps  # [B]
        eps_pred = self.denoiser(xt, t_norm.unsqueeze(-1), state)
        
        return eps_pred, eps
    
    @torch.no_grad()
    def sample(self, state: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        """
        推理时: DDIM 采样
        """
        if n_steps is None:
            n_steps = self.n_infer_steps
        
        B = state.shape[0]
        device = state.device
        
        # 从噪声开始
        x = torch.randn(B, self.act_dim, device=device)
        
        # DDIM 采样
        step_ratio = self.n_steps // n_steps
        for i in reversed(range(0, self.n_steps, step_ratio)):
            t = torch.full((B, 1), (i + 1) / self.n_steps, device=device)
            eps_pred = self.denoiser(x, t, state)
            
            # DDIM step
            alpha = self.scheduler.alphas_cumprod[i]
            if i - step_ratio >= 0:
                alpha_prev = self.scheduler.alphas_cumprod[i - step_ratio]
            else:
                alpha_prev = torch.tensor(1.0, device=device)
            
            # x0 prediction
            sqrt_alpha = alpha.sqrt()
            sqrt_one_minus = (1 - alpha).sqrt()
            x0_pred = (x - sqrt_one_minus * eps_pred) / sqrt_alpha
            
            # direction pointing to x
            dir_xt = (1 - alpha_prev).sqrt() * eps_pred
            x = alpha_prev.sqrt() * x0_pred + dir_xt
        
        return torch.tanh(x)  # 约束到 [-1, 1]


# ============================================================
# 4. Tool Selector
# ============================================================

class ToolSelector(nn.Module):
    """
    自适应 Tool 选择器。
    输入 State_t, 输出对每个 tool 的偏好。
    """
    def __init__(self, state_dim: int = 128, hidden_dim: int = 64,
                 n_tools: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_tools),
        )
        self.n_tools = n_tools
    
    def forward(self, state: torch.Tensor, temperature: float = 1.0,
                hard: bool = False) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
            temperature: Gumbel-Softmax 温度
            hard: 是否 hard sampling
        
        Returns:
            tool_probs: [B, n_tools]  软选择权重
        """
        logits = self.net(state)
        
        if self.training and not hard:
            return F.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            return F.softmax(logits / temperature, dim=-1)


# ============================================================
# 5. Tool Interface (完整框架)
# ============================================================

class ToolInterface(nn.Module):
    """
    Agentic WM 的 Tool Interface MVP。
    
    完整流程:
      1. Encoder(obs_seq) → State_t
      2. Selector(State_t) → tool_probs
      3. 选择 Tool A 或 Tool B 生成动作
    """
    def __init__(self, obs_dim: int = 4, state_dim: int = 128,
                 act_dim: int = 2, n_tools: int = 2):
        super().__init__()
        
        # WM
        self.encoder = StateEncoder(obs_dim, state_dim)
        
        # Tools
        self.tool_a = ToolA_MLP(state_dim, act_dim)
        self.tool_b = ToolB_Diffusion(state_dim, act_dim, state_dim)
        
        # Selector
        self.selector = ToolSelector(state_dim, n_tools=n_tools)
        
        self.state_dim = state_dim
        self.act_dim = act_dim
    
    def freeze_tools(self):
        """冻结工具，只训练 selector"""
        for p in self.tool_a.parameters():
            p.requires_grad = False
        for p in self.tool_b.parameters():
            p.requires_grad = False
    
    def unfreeze_tools(self):
        """解冻工具"""
        for p in self.tool_a.parameters():
            p.requires_grad = True
        for p in self.tool_b.parameters():
            p.requires_grad = True
    
    def forward(self, obs_seq: torch.Tensor, mode: str = 'train',
                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_seq: [B, T, obs_dim]
            mode: 'train' | 'eval'
            temperature: selector temperature
        
        Returns:
            action: [B, act_dim]
            tool_probs: [B, n_tools]
        """
        state = self.encoder(obs_seq)
        tool_probs = self.selector(state, temperature, hard=(mode != 'train'))
        
        if mode == 'train':
            # 软选择: 两个 tool 的动作加权混合 (可微分)
            action_a = self.tool_a(state)
            action_b = self.tool_b.sample(state)
            
            action = (tool_probs[:, 0:1] * action_a + 
                     tool_probs[:, 1:2] * action_b)
        else:
            # 硬选择: 选概率最大的 tool
            tool_idx = torch.argmax(tool_probs, dim=-1)
            actions_a = self.tool_a(state)
            actions_b = self.tool_b.sample(state)
            
            # 按 idx 选择
            mask_a = (tool_idx == 0).float().unsqueeze(-1)
            mask_b = (tool_idx == 1).float().unsqueeze(-1)
            action = mask_a * actions_a + mask_b * actions_b
        
        return action, tool_probs
    
    def get_selection(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """获取硬选择结果 (推理时用)"""
        state = self.encoder(obs_seq)
        logits = self.selector.net(state)
        return torch.argmax(logits, dim=-1)
