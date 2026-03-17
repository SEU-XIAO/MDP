from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    # --- 基础训练控制 ---
    episodes: int = 3000               # 4090 性能强劲，2000 轮足以完成复杂场景收敛
    max_steps_per_episode: int = 500   # 配合 Dijkstra 引导，400 步足以支持复杂的避障绕路
    
    # --- 学习效率 (针对大 Batch 优化) ---
    learning_rate: float = 3e-5        # 配合 256 Batch，适中的学习率可防止梯度震荡
    batch_size: int = 256              # 充分利用显存，提供更稳定的梯度估计
    gamma: float = 0.99                # 保持对远期奖励（终点）的高度关注
    target_update_interval: int = 1500  # 增加延迟更新，提升大 Batch 下的训练稳定性
    
    # --- 经验回放 ---
    replay_capacity: int = 500_000     # 扩容以存储更多样化的探索样本
    min_replay_size: int = 10_000      # 积累足够样本再开始训练，确保初始批次具有代表性
    
    # --- 探索策略 (Exploration) ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02          # 训练后期给予模型更高的决策信任度
    epsilon_decay_steps: int = 250_000 # 在训练中期左右完成探索，随后进入精细化路径优化
    
    # --- 启发式引导 (Heuristic/Dijkstra) ---
    # 这部分参数将控制 Dijkstra 引导在训练过程中的衰减（如果你的逻辑中有用到）
    heuristic_start: float = 0.6
    heuristic_end: float = 0.1
    heuristic_decay_steps: int = 300_000
    
    # --- 奖励权重优化 (Reward Shaping) ---
    goal_reward: float = 300.0         # 目标奖励，与步数惩罚保持合理的量级比例
    step_penalty: float = -0.2         # 适度的时间压力，促使 Agent 寻找更短的避障路径
    timeout_penalty: float = -50.0    # 惩罚超时，但避免过高导致 Agent 产生“原地卡死”的恐惧
    blocked_move_penalty: float = -2.0 # 引入 Dijkstra 后，撞墙概率降低，惩罚可调轻
    
    # --- 风险与引导 (核心算法适配) ---
    risk_lambda: float = 1.2           # 配合环境中的 risk^2，保持对高风险区域的绝对警惕
    guide_omega: float = 2.5           # 强化引导权重：Dijkstra 提供的路径是全局最优，值得高度追随
    
    # --- 环境与评估 ---
    enemy_jitter: int = 1
    start_jitter: int = 1
    observation_size: int = 64
    eval_interval_episodes: int = 50   # 每 50 轮评估一次，观察收敛趋势
    eval_episodes: int = 20
    seed: int = 42
    eval_enemy_jitter: int = 1  
    eval_start_jitter: int = 1

    high_risk_block_threshold: float = 0.8


def load_scenario(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)