from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    # 基础训练控制
    episodes: int = 2000              # 增加总轮数，4090 跑得很快
    max_steps_per_episode: int = 450  # 核心改动：从 220 增加到 450，给 Agent 容错和绕路空间
    
    # 学习效率
    learning_rate: float = 8e-5       # 稍微调高，1e-5 太保守，4090 适合更大步长
    batch_size: int = 256             # 核心改动：从 64 增加到 256，充分利用显存，稳定 Loss
    gamma: float = 0.99
    target_update_interval: int = 1500 # 增加更新间隔，配合大 Batch 提升稳定性
    
    # 经验回放
    replay_capacity: int = 150_000    # 扩容，保存更多探索样本
    min_replay_size: int = 5_000      # 积累更多样本后再开始学习，防止早期过拟合失败经验
    
    # 探索策略 (Exploration)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05         # 降低底限，后期更相信模型
    epsilon_decay_steps: int = 200_000 # 核心改动：从 8w 增加到 20w，拉长探索期，防止过早变猥琐
    
    # 启发式引导
    heuristic_start: float = 0.6
    heuristic_end: float = 0.1
    heuristic_decay_steps: int = 150_000
    
    # 奖励权重优化 (Reward Shaping)
    goal_reward: float = 500.0        # 增加目标奖励，提高终点诱惑力
    step_penalty: float = -0.3        # 减小步数惩罚，鼓励多尝试
    timeout_penalty: float = -200.0   # 减小超时惩罚，防止 Agent 产生由于怕被扣分而不敢走的恐惧
    blocked_move_penalty: float = -2.0 # 核心改动：从 -8 减小到 -2，撞墙不可怕，不敢走才可怕
    
    # 风险与引导
    risk_lambda: float = 1.5          # 增加对风险的关注，但在奖励函数里要平衡
    guide_omega: float = 2.0          # 强化距离引导，让 Agent 像闻到味一样往终点走
    
    # 环境与评估
    enemy_jitter: int = 1
    start_jitter: int = 1
    observation_size: int = 64
    eval_interval_episodes: int = 50
    eval_episodes: int = 20
    seed: int = 42


def load_scenario(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
