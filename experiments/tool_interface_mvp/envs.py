"""
Push-T 环境 (简化 MuJoCo 替代)

在 MuJoCo 不可用时，用纯 Python 实现的简化 2D Push-T 环境。
如有 MuJoCo，替换为 gym_mujoco 版本。
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PushTConfig:
    """Push-T 环境配置"""
    # 物体参数
    obj_size: float = 0.05       # T 形物体包围盒
    obj_mass: float = 0.1        # 质量 (kg)
    friction: float = 0.3        # 摩擦系数
    
    # 工作空间
    workspace_x: Tuple[float, float] = (0.0, 0.5)
    workspace_y: Tuple[float, float] = (0.0, 0.5)
    
    # 动作参数
    max_step: float = 0.03       # 单步最大位移
    
    # Episode
    max_steps: int = 100
    success_thresh: float = 0.05  # 物体离目标 < 此距离 = 成功
    
    # Difficulty
    difficulty: str = "mixed"     # "simple", "complex", "mixed"


class PushTEnv:
    """
    简化 Push-T 环境 (2D 平面推物体)
    
    状态: [obj_x, obj_y, target_x, target_y]  (4 维)
    动作: [dx, dy]  (末端执行器位移, 2 维)
    
    难度:
      - simple: 目标随机生成在远距离，需要快速大范围移动
      - complex: 目标随机生成在近距离，需要精确对准
    """
    
    def __init__(self, config: PushTConfig = None):
        self.cfg = config or PushTConfig()
        self.obs_dim = 4
        self.act_dim = 2
        self.reset()
    
    def reset(self, difficulty: Optional[str] = None) -> np.ndarray:
        """重置环境"""
        self.step_count = 0
        self.difficulty = difficulty or self.cfg.difficulty
        
        # 物体随机初始位置
        self.obj_pos = np.array([
            np.random.uniform(0.1, 0.4),
            np.random.uniform(0.1, 0.4)
        ])
        
        # 根据难度生成目标
        if self.difficulty == "simple":
            # 简单：目标距离 > 0.3，但用定向生成（更快）
            # 沿相反方向放置目标
            center = np.array([0.25, 0.25])
            obj_to_center = center - self.obj_pos
            if np.linalg.norm(obj_to_center) > 0.05:
                direction = obj_to_center / np.linalg.norm(obj_to_center)
            else:
                direction = np.array([1.0, 0.0])
            # 在反方向生成目标: obj + direction * (0.3 to 0.4)
            dist = np.random.uniform(0.3, 0.4)
            target = self.obj_pos + direction * dist
            target = np.clip(target, [0.05, 0.05], [0.45, 0.45])
            self.target = target
        elif self.difficulty == "complex":
            # 复杂：目标距离 < 0.15 (需要精确控制)
            # 在物体附近随机方向生成
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.05, 0.15)
            target = self.obj_pos + np.array([np.cos(angle), np.sin(angle)]) * dist
            target = np.clip(target, [0.05, 0.05], [0.45, 0.45])
            self.target = target
        else:  # mixed
            self.target = np.array([
                np.random.uniform(0.05, 0.45),
                np.random.uniform(0.05, 0.45)
            ])
            dist = np.linalg.norm(self.target - self.obj_pos)
            self.difficulty = "simple" if dist > 0.2 else "complex"
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """获取观测"""
        return np.concatenate([self.obj_pos, self.target]).astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作
        
        Args:
            action: [dx, dy] 位移，范围 [-1, 1]
        
        Returns:
            obs, reward_neg_dist, done, info
        """
        self.step_count += 1
        
        # 归一化动作 → 实际位移
        action = np.clip(action, -1, 1)
        displacement = action * self.cfg.max_step
        
        # 更新末端执行器位置 (简化：直接移动物体)
        # 实际 MuJoCo 中应该是推杆接触物体后的动力学
        new_obj = self.obj_pos + displacement
        
        # 边界约束
        new_obj[0] = np.clip(new_obj[0], *self.cfg.workspace_x)
        new_obj[1] = np.clip(new_obj[1], *self.cfg.workspace_y)
        
        self.obj_pos = new_obj
        
        # 计算到目标的距离
        dist = np.linalg.norm(self.obj_pos - self.target)
        
        # 判断结果
        success = dist < self.cfg.success_thresh
        timeout = self.step_count >= self.cfg.max_steps
        done = success or timeout
        
        # Reward: 负距离 (越小越好)
        reward = -dist
        
        info = {
            "success": success,
            "timeout": timeout,
            "distance": dist,
            "difficulty": self.difficulty,
        }
        
        return self._get_obs(), reward, done, info
    
    def expert_action(self, obs: np.ndarray, difficulty: str = None) -> np.ndarray:
        """
        生成专家动作 (指向目标的简单比例控制)
        用于生成训练数据的 ground truth
        
        简单场景: 大步移动，带噪声
        复杂场景: 小步精确调准
        """
        obj_pos = obs[:2]
        target = obs[2:4]
        
        vec = target - obj_pos
        dist = np.linalg.norm(vec)
        
        if dist < 1e-6:
            return np.zeros(2, dtype=np.float32)
        
        direction = vec / dist
        
        if difficulty == "simple" or (difficulty is None and dist > 0.2):
            # 大步: 速度快，带一些噪声 (模拟次优控制)
            step_size = min(1.0, dist / self.cfg.max_step)
            action = direction * step_size
            action += np.random.normal(0, 0.05, 2)  # 噪声
        else:
            # 小步: 精确控制
            step_size = min(0.3, dist / self.cfg.max_step)
            action = direction * step_size
            action += np.random.normal(0, 0.02, 2)  # 小噪声
        
        return np.clip(action, -1, 1).astype(np.float32)


def make_env(difficulty: str = "mixed") -> PushTEnv:
    """创建环境"""
    cfg = PushTConfig(difficulty=difficulty)
    return PushTEnv(cfg)
