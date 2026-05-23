"""
训练 Tool Interface (Phase 2: 联合训练 Selector + Encoder, Tools 冻结)

先需要 Phase 1: 用 train_baselines.py 分别训练 MLP 和 Diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models import StateEncoder, ToolA_MLP, ToolB_Diffusion, ToolSelector


def load_pretrained_tools(config, checkpoint_dir):
    """加载预训练的 encoder + tools"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Encoder (共享 — 从 MLP baseline 加载)
    encoder = StateEncoder(
        obs_dim=config["wm"]["obs_dim"],
        state_dim=config["wm"]["state_dim"],
        num_layers=config["wm"]["num_layers"],
        dropout=config["wm"]["dropout"],
    ).to(device)
    
    # Tool A (from MLP checkpoint)
    tool_a = ToolA_MLP(
        state_dim=config["tool_mlp"]["state_dim"],
        act_dim=config["tool_mlp"]["act_dim"],
        hidden_dims=config["tool_mlp"]["hidden_dims"],
        dropout=config["tool_mlp"]["dropout"],
    ).to(device)
    
    # Tool B (from Diffusion checkpoint)
    tool_b = ToolB_Diffusion(
        state_dim=config["tool_diffusion"]["state_dim"],
        act_dim=config["tool_diffusion"]["act_dim"],
        cond_dim=config["tool_diffusion"]["cond_dim"],
        hidden_dim=config["tool_diffusion"]["hidden_dim"],
        n_steps=config["tool_diffusion"]["n_diffusion_steps"],
        n_infer_steps=config["tool_diffusion"]["n_inference_steps"],
    ).to(device)
    
    # 加载权重
    mlp_dir = Path(checkpoint_dir) / "baseline_mlp"
    diff_dir = Path(checkpoint_dir) / "baseline_diffusion"
    
    encoder.load_state_dict(torch.load(mlp_dir / "encoder.pt", map_location=device))
    tool_a.load_state_dict(torch.load(mlp_dir / "mlp.pt", map_location=device))
    tool_b.load_state_dict(torch.load(diff_dir / "diffusion.pt", map_location=device))
    
    print(f"  Loaded encoder from {mlp_dir}")
    print(f"  Loaded Tool A (MLP) from {mlp_dir}")
    print(f"  Loaded Tool B (Diffusion) from {diff_dir}")
    
    # 冻结 tools
    for p in tool_a.parameters():
        p.requires_grad = False
    for p in tool_b.parameters():
        p.requires_grad = False
    
    return encoder, tool_a, tool_b


def train_tool_interface(config: dict, data_path: str, 
                         checkpoint_dir: str, save_dir: str):
    """训练 Tool Interface"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Tool Interface on {device}")
    
    # 加载预训练组件
    encoder, tool_a, tool_b = load_pretrained_tools(config, checkpoint_dir)
    
    # Selector (从头训练)
    selector = ToolSelector(
        state_dim=config["selector"]["state_dim"],
        hidden_dim=config["selector"]["hidden_dim"],
        n_tools=2,
    ).to(device)
    
    # 只训练 selector + encoder (可选)
    trainable_params = list(selector.parameters())
    # 可选: 微调 encoder
    # trainable_params += list(encoder.parameters())
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["training"]["lr_selector"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Data
    from data import PushTDataset
    dataset = PushTDataset(data_path)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                        shuffle=True, drop_last=True)
    
    # 配置
    n_epochs = config["training"]["phase2_epochs"]
    lambda_task = config["training"]["lambda_task"]
    lambda_select = config["training"]["lambda_select"]
    
    print(f"  Training selector + optionally encoder for {n_epochs} epochs")
    print(f"  loss = {lambda_task} * task_loss + {lambda_select} * select_loss")
    
    tool_a.eval()
    tool_b.eval()
    
    for epoch in range(n_epochs):
        total_loss = 0
        total_task = 0
        total_select = 0
        correct_select = 0
        n_samples = 0
        
        for batch in loader:
            obs_seq = batch["obs_seq"].to(device)
            action_gt = batch["action"].to(device)
            difficulty = batch["difficulty"].to(device)  # 0=simple, 1=complex
            
            B = obs_seq.shape[0]
            
            # 编码状态
            state = encoder(obs_seq)
            
            # Tool selection (Gumbel-Softmax)
            logits = selector.net(state)
            tool_probs = F.gumbel_softmax(logits, tau=1.0, hard=False)
            
            # 生成动作 (两个 tool)
            with torch.no_grad():
                action_a = tool_a(state)
            
            action_b = tool_b.sample(state, n_steps=config["tool_diffusion"]["n_inference_steps"])
            
            # 加权混合
            action_pred = (tool_probs[:, 0:1] * action_a + 
                          tool_probs[:, 1:2] * action_b)
            
            # 任务 loss: 动作准确性
            loss_task = F.mse_loss(action_pred, action_gt)
            
            # 选择 loss: 监督 selector 选对
            # difficulty=0 → 应选 Tool A (MLP), difficulty=1 → 应选 Tool B (Diffusion)
            loss_select = F.cross_entropy(logits, difficulty)
            
            # 总 loss
            loss = lambda_task * loss_task + lambda_select * loss_select
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config["training"]["grad_clip"])
            optimizer.step()
            
            total_loss += loss.item()
            total_task += loss_task.item()
            total_select += loss_select.item()
            
            # 统计选择准确率
            pred_tool = torch.argmax(logits, dim=-1)
            correct_select += (pred_tool == difficulty).sum().item()
            n_samples += B
        
        acc = correct_select / n_samples if n_samples > 0 else 0
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | "
                  f"Total: {total_loss/len(loader):.4f} | "
                  f"Task: {total_task/len(loader):.4f} | "
                  f"Select: {total_select/len(loader):.4f} | "
                  f"Acc: {acc:.2%}")
    
    # 保存
    save_path = Path(save_dir) / "tool_interface"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), save_path / "encoder.pt")
    torch.save(tool_a.state_dict(), save_path / "tool_a.pt")
    torch.save(tool_b.state_dict(), save_path / "tool_b.pt")
    torch.save(selector.state_dict(), save_path / "selector.pt")
    print(f"  Saved to {save_path}")
    
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/push_t_mixed_500ep.pkl")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    if args.epochs is not None:
        config["training"]["phase2_epochs"] = args.epochs
    
    acc = train_tool_interface(config, args.data, 
                               args.checkpoint_dir, args.save_dir)
    print(f"\nFinal selection accuracy: {acc:.2%}")
