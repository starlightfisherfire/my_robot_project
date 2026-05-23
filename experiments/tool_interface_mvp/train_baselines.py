"""
训练 MLP 和 Diffusion baseline
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models import StateEncoder, ToolA_MLP, ToolB_Diffusion
from data import PushTDataset


def train_mlp(config: dict, data_path: str, save_dir: str):
    """训练 MLP Controller"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MLP on {device}")
    
    # Model
    encoder = StateEncoder(
        obs_dim=config["wm"]["obs_dim"],
        state_dim=config["wm"]["state_dim"],
        num_layers=config["wm"]["num_layers"],
        dropout=config["wm"]["dropout"],
    ).to(device)
    
    mlp = ToolA_MLP(
        state_dim=config["tool_mlp"]["state_dim"],
        act_dim=config["tool_mlp"]["act_dim"],
        hidden_dims=config["tool_mlp"]["hidden_dims"],
        dropout=config["tool_mlp"]["dropout"],
    ).to(device)
    
    # Data
    dataset = PushTDataset(data_path)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], 
                        shuffle=True, drop_last=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(mlp.parameters()),
        lr=config["training"]["lr_mlp"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Training
    n_epochs = config["training"]["phase1_epochs_mlp"]
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in loader:
            obs_seq = batch["obs_seq"].to(device)    # [B, T, obs_dim]
            action_gt = batch["action"].to(device)    # [B, act_dim]
            
            state = encoder(obs_seq)
            action_pred = mlp(state)
            
            loss = nn.functional.mse_loss(action_pred, action_gt)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(mlp.parameters()),
                config["training"]["grad_clip"]
            )
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  MLP Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.6f}")
    
    # Save
    save_path = Path(save_dir) / "baseline_mlp"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), save_path / "encoder.pt")
    torch.save(mlp.state_dict(), save_path / "mlp.pt")
    print(f"  Saved to {save_path}")


def train_diffusion(config: dict, data_path: str, save_dir: str):
    """训练 Diffusion Policy"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Diffusion on {device}")
    
    # Model
    encoder = StateEncoder(
        obs_dim=config["wm"]["obs_dim"],
        state_dim=config["wm"]["state_dim"],
        num_layers=config["wm"]["num_layers"],
        dropout=config["wm"]["dropout"],
    ).to(device)
    
    diffusion = ToolB_Diffusion(
        state_dim=config["tool_diffusion"]["state_dim"],
        act_dim=config["tool_diffusion"]["act_dim"],
        cond_dim=config["tool_diffusion"]["cond_dim"],
        hidden_dim=config["tool_diffusion"]["hidden_dim"],
        n_steps=config["tool_diffusion"]["n_diffusion_steps"],
        n_infer_steps=config["tool_diffusion"]["n_inference_steps"],
    ).to(device)
    
    # Data
    dataset = PushTDataset(data_path)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                        shuffle=True, drop_last=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(diffusion.parameters()),
        lr=config["training"]["lr_diffusion"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Training
    n_epochs = config["training"]["phase1_epochs_diffusion"]
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in loader:
            obs_seq = batch["obs_seq"].to(device)
            action_gt = batch["action"].to(device)
            
            state = encoder(obs_seq)
            eps_pred, eps = diffusion(state, action_gt)
            
            loss = nn.functional.mse_loss(eps_pred, eps)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(diffusion.parameters()),
                config["training"]["grad_clip"]
            )
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Diffusion Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.6f}")
    
    # Save
    save_path = Path(save_dir) / "baseline_diffusion"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), save_path / "encoder.pt")
    torch.save(diffusion.state_dict(), save_path / "diffusion.pt")
    print(f"  Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["mlp", "diffusion"], required=True)
    parser.add_argument("--data", type=str, default="./data/push_t_mixed_500ep.pkl")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    if args.epochs is not None:
        if args.tool == "mlp":
            config["training"]["phase1_epochs_mlp"] = args.epochs
        else:
            config["training"]["phase1_epochs_diffusion"] = args.epochs
    
    if args.tool == "mlp":
        train_mlp(config, args.data, args.save_dir)
    else:
        train_diffusion(config, args.data, args.save_dir)
