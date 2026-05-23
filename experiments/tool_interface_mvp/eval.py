"""
评估与对比实验

对比三个方案:
  1. MLP only
  2. Diffusion only
  3. Tool Interface (自适应)
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import yaml
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from models import (StateEncoder, ToolA_MLP, ToolB_Diffusion, 
                    ToolSelector, ToolInterface)
from envs import PushTEnv


def load_models(config, checkpoint_dir, device):
    """加载所有模型"""
    
    # Tool Interface
    ckpt_dir = Path(checkpoint_dir) / "tool_interface"
    has_tool_interface = ckpt_dir.exists()
    
    # MLP baseline
    encoder_mlp = StateEncoder(
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
    
    mlp_dir = Path(checkpoint_dir) / "baseline_mlp"
    encoder_mlp.load_state_dict(torch.load(mlp_dir / "encoder.pt", map_location=device))
    mlp.load_state_dict(torch.load(mlp_dir / "mlp.pt", map_location=device))
    encoder_mlp.eval()
    mlp.eval()
    
    # Diffusion baseline
    encoder_diff = StateEncoder(
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
    
    diff_dir = Path(checkpoint_dir) / "baseline_diffusion"
    encoder_diff.load_state_dict(torch.load(diff_dir / "encoder.pt", map_location=device))
    diffusion.load_state_dict(torch.load(diff_dir / "diffusion.pt", map_location=device))
    encoder_diff.eval()
    diffusion.eval()
    
    # Tool Interface
    selector = None
    encoder_ti = None
    if has_tool_interface:
        encoder_ti = StateEncoder(
            obs_dim=config["wm"]["obs_dim"],
            state_dim=config["wm"]["state_dim"],
            num_layers=config["wm"]["num_layers"],
            dropout=config["wm"]["dropout"],
        ).to(device)
        selector = ToolSelector(
            state_dim=config["selector"]["state_dim"],
            hidden_dim=config["selector"]["hidden_dim"],
            n_tools=2,
        ).to(device)
        
        encoder_ti.load_state_dict(torch.load(ckpt_dir / "encoder.pt", map_location=device))
        selector.load_state_dict(torch.load(ckpt_dir / "selector.pt", map_location=device))
        encoder_ti.eval()
        selector.eval()
    
    return {
        "encoder_mlp": encoder_mlp,
        "mlp": mlp,
        "encoder_diff": encoder_diff,
        "diffusion": diffusion,
        "encoder_ti": encoder_ti,
        "selector": selector,
    }


def run_episode(env: PushTEnv, models: dict, method: str, 
                device: torch.device, history_len: int = 5,
                render: bool = False) -> dict:
    """
    跑一个 episode
    
    返回:
      dict with: success, steps, total_reward, tool_selections, distances
    """
    obs = env.reset()
    obs_history = [obs] * history_len
    
    results = {
        "success": False,
        "steps": 0,
        "total_reward": 0.0,
        "tool_selections": [],  # 仅 Tool Interface
        "distances": [],
        "difficulty": env.difficulty,
    }
    
    for step in range(env.cfg.max_steps):
        obs_seq = torch.tensor(np.stack(obs_history), dtype=torch.float32
                              ).unsqueeze(0).to(device)  # [1, T, obs_dim]
        
        with torch.no_grad():
            if method == "mlp":
                state = models["encoder_mlp"](obs_seq)
                action = models["mlp"](state)
                tool_choice = 0
            elif method == "diffusion":
                state = models["encoder_diff"](obs_seq)
                action = models["diffusion"].sample(state)
                tool_choice = 1
            elif method == "tool_interface":
                state = models["encoder_ti"](obs_seq)
                logits = models["selector"].net(state)
                tool_idx = torch.argmax(logits, dim=-1).item()
                
                if tool_idx == 0:
                    action = models["mlp"](state)
                else:
                    action = models["diffusion"].sample(state)
                
                tool_choice = tool_idx
            else:
                raise ValueError(f"Unknown method: {method}")
        
        action = action.squeeze(0).cpu().numpy()
        obs, reward, done, info = env.step(action)
        
        results["steps"] += 1
        results["total_reward"] += reward
        results["distances"].append(info["distance"])
        results["tool_selections"].append(tool_choice)
        
        # 更新历史
        obs_history.pop(0)
        obs_history.append(obs)
        
        if done:
            results["success"] = info["success"]
            break
    
    return results


def evaluate(config: dict, checkpoint_dir: str, method: str = "all",
             n_episodes: int = 100, render: bool = False):
    """评估所有方法"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = load_models(config, checkpoint_dir, device)
    
    methods = ["mlp", "diffusion", "tool_interface"]
    if method != "all":
        methods = [method]
    
    print(f"\n{'='*60}")
    print(f"Evaluating on {n_episodes} episodes per method")
    print(f"{'='*60}\n")
    
    for method_name in methods:
        if method_name == "tool_interface" and models["selector"] is None:
            print(f"  [SKIP] Tool Interface: no checkpoint found")
            continue
        
        env = PushTEnv()
        all_results = []
        
        for ep in range(n_episodes):
            # 交替简单/复杂场景
            difficulty = "simple" if ep % 2 == 0 else "complex"
            env.difficulty = difficulty
            
            result = run_episode(env, models, method_name, device,
                                 history_len=config["data"]["history_len"],
                                 render=render)
            all_results.append(result)
        
        # 统计
        total = len(all_results)
        successes = sum(r["success"] for r in all_results)
        avg_steps = np.mean([r["steps"] for r in all_results])
        avg_dist = np.mean([r["distances"][-1] for r in all_results])
        
        # 按难度分
        simple_results = [r for r in all_results if r["difficulty"] == "simple"]
        complex_results = [r for r in all_results if r["difficulty"] == "complex"]
        
        simple_success = sum(r["success"] for r in simple_results) / max(len(simple_results), 1)
        complex_success = sum(r["success"] for r in complex_results) / max(len(complex_results), 1)
        
        # Tool Interface 的选择统计
        if method_name == "tool_interface":
            all_selections = []
            for r in all_results:
                all_selections.extend(r["tool_selections"])
            
            n_mlp = sum(1 for s in all_selections if s == 0)
            n_diff = sum(1 for s in all_selections if s == 1)
            total_sel = len(all_selections)
            
            # 在简单/复杂场景下的选择分布
            simple_sels = []
            complex_sels = []
            for r in simple_results:
                simple_sels.extend(r["tool_selections"])
            for r in complex_results:
                complex_sels.extend(r["tool_selections"])
            
            simple_mlp_pct = sum(1 for s in simple_sels if s == 0) / max(len(simple_sels), 1)
            complex_diff_pct = sum(1 for s in complex_sels if s == 1) / max(len(complex_sels), 1)
        
        # 输出
        print(f"  [{method_name.upper():^18}]")
        print(f"    Overall Success:  {successes}/{total} ({successes/total:.1%})")
        print(f"    Simple Success:   {simple_success:.1%}")
        print(f"    Complex Success:  {complex_success:.1%}")
        print(f"    Avg Steps:        {avg_steps:.1f}")
        print(f"    Avg Final Dist:   {avg_dist:.4f}")
        
        if method_name == "tool_interface":
            print(f"    Tool Selections:  MLP={n_mlp}({n_mlp/total_sel:.0%}) "
                  f"Diffusion={n_diff}({n_diff/total_sel:.0%})")
            print(f"    Simple→MLP:      {simple_mlp_pct:.0%}  "
                  f"Complex→Diffusion: {complex_diff_pct:.0%}")
        
        print()
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--method", type=str, default="all",
                       choices=["all", "mlp", "diffusion", "tool_interface"])
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    evaluate(config, args.checkpoint_dir, args.method, 
             args.n_episodes, args.render)
