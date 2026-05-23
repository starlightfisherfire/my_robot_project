"""
数据生成与加载

生成三类数据:
  - simple: 简单操作场景 (适合 MLP)
  - complex: 精确操作场景 (需要 Diffusion)
  - mixed: 混合场景 (训练 Tool Interface)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from typing import Optional

from envs import PushTEnv


def generate_data(
    n_episodes: int = 500,
    difficulty: str = "simple",
    save_dir: str = "./data",
    history_len: int = 5,
) -> str:
    """
    生成训练数据
    
    每条 trajectory:
      - obs_seq: [T, history_len, obs_dim]  (滑动窗口历史)
      - action: [T, act_dim]  (当前步专家动作)
      - difficulty: int (0=simple, 1=complex)
    """
    env = PushTEnv()
    samples = []
    
    for ep in range(n_episodes):
        if difficulty == "mixed":
            diff = "simple" if np.random.random() < 0.5 else "complex"
        else:
            diff = difficulty
        
        obs = env.reset(difficulty=diff)
        obs_history = [obs] * history_len  # padding 用初始帧
        
        traj = {
            "obs_seqs": [],
            "actions": [],
            "difficulties": [],
            "success": False,
        }
        
        for _ in range(env.cfg.max_steps):
            # 记录当前历史
            traj["obs_seqs"].append(np.stack(obs_history, axis=0))
            
            # 专家动作
            action = env.expert_action(obs, difficulty=diff)
            traj["actions"].append(action)
            diff_label = 1 if diff == "complex" else 0
            traj["difficulties"].append(diff_label)
            
            # 执行
            obs, _, done, info = env.step(action)
            
            # 更新历史
            obs_history.pop(0)
            obs_history.append(obs)
            
            if done:
                traj["success"] = info.get("success", False)
                break
        
        if len(traj["obs_seqs"]) > 0:
            samples.append(traj)
        
        if (ep + 1) % 200 == 0:
            print(f"  [{difficulty}] Generated {ep + 1}/{n_episodes} episodes")
    
    # 保存
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filepath = save_path / f"push_t_{difficulty}_{n_episodes}ep.pkl"
    
    with open(filepath, "wb") as f:
        pickle.dump(samples, f)
    
    n_traj = len(samples)
    n_steps = sum(len(t["obs_seqs"]) for t in samples)
    success_rate = sum(t["success"] for t in samples) / n_traj if n_traj > 0 else 0
    
    print(f"  Saved: {filepath}")
    print(f"    Trajectories: {n_traj}, Steps: {n_steps}, Success: {success_rate:.2%}")
    
    return str(filepath)


class PushTDataset(Dataset):
    """Push-T 数据集"""
    
    def __init__(self, data_path: str):
        with open(data_path, "rb") as f:
            self.trajectories = pickle.load(f)
        
        self.obs_seqs = []
        self.actions = []
        self.difficulties = []
        
        for traj in self.trajectories:
            self.obs_seqs.extend(traj["obs_seqs"])
            self.actions.extend(traj["actions"])
            self.difficulties.extend(traj["difficulties"])
        
        self.obs_seqs = torch.tensor(np.array(self.obs_seqs), dtype=torch.float32)
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        self.difficulties = torch.tensor(self.difficulties, dtype=torch.long)
        
        print(f"Dataset loaded: {len(self)} steps, "
              f"simple={sum(self.difficulties==0)}, "
              f"complex={sum(self.difficulties==1)}")
    
    def __len__(self):
        return len(self.obs_seqs)
    
    def __getitem__(self, idx):
        return {
            "obs_seq": self.obs_seqs[idx],       # [history_len, obs_dim]
            "action": self.actions[idx],          # [act_dim]
            "difficulty": self.difficulties[idx], # 0 or 1
        }


def get_dataloader(
    data_path: str,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    dataset = PushTDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate"], default="generate")
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--history_len", type=int, default=5)
    args = parser.parse_args()
    
    print("Generating simple scenarios...")
    generate_data(args.n_episodes, "simple", args.save_dir, args.history_len)
    
    print("\nGenerating complex scenarios...")
    generate_data(args.n_episodes, "complex", args.save_dir, args.history_len)
    
    print("\nGenerating mixed scenarios...")
    generate_data(args.n_episodes // 2, "mixed", args.save_dir, args.history_len)
    
    print("\nDone!")
