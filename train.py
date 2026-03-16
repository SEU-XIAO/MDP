from __future__ import annotations

import argparse
import random
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

# 请确保你的项目结构如下，或根据实际情况修改 import
from src.agent.d3qn_agent import AgentConfig, D3QNAgent
from src.config import TrainConfig, load_scenario
from src.environment.risk_grid_env import RewardWeights, RiskAwareGridEnv
from src.replay.per_buffer import PrioritizedReplayBuffer

# --- 1. 动态课程学习生成器 (保持原逻辑并增强) ---
class RandomScenarioGenerator:
    def __init__(self, grid_size=64, start_pos=(1, 1), goal_pos=(62, 62)):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos

    def generate(self, episode_idx: int):
        # 难度随轮次增加
        max_enemies = min(12, 6 + (episode_idx // 300))
        num_enemies = random.randint(4, max_enemies)
        
        enemies = []
        for _ in range(num_enemies):
            ex = random.randint(8, self.grid_size - 9)
            ey = random.randint(8, self.grid_size - 9)
            enemies.append({
                "pos": [ex, ey], 
                "detection_zones": [{"r": random.uniform(3.5, 8.0), "p": random.uniform(0.7, 0.95)}]
            })

        # U型胡同概率随进度提升
        u_prob = min(0.4, 0.15 + (episode_idx / 3000))
        if random.random() < u_prob:
            enemies.extend(self._create_u_shape(episode_idx))

        return {
            "map": {"grid_size": self.grid_size, "start_pos": self.start_pos, "goal_pos": self.goal_pos},
            "enemies": enemies,
        }

    def _create_u_shape(self, episode_idx):
        u_points = []
        bx, by = random.randint(15, 40), random.randint(15, 40)
        length = min(16, 10 + (episode_idx // 400))
        for i in range(length):
            u_points.append({"pos": [bx + i, by + 10], "detection_zones": [{"r": 2.5, "p": 1.0}]})
        for i in range(length - 2):
            u_points.append({"pos": [bx, by + i], "detection_zones": [{"r": 2.5, "p": 1.0}]})
            u_points.append({"pos": [bx + length - 1, by + i], "detection_zones": [{"r": 2.5, "p": 1.0}]})
        return u_points

# --- 2. 辅助工具 ---
def linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0: return end
    return start + (end - start) * min(step / total_steps, 1.0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7: Final Curriculum on 4090")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v7_4090_fix")
    parser.add_argument("--batch-size", type=int, default=256) # 4090 标配
    parser.add_argument("--lr", type=float, default=8e-5)      # 配合大 Batch 稍微调高
    parser.add_argument("--episodes", type=int, default=3000)
    return parser.parse_args()

def evaluate_greedy(agent, env, episodes, seed):
    success = 0
    rewards = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        ep_reward = 0.0
        for _ in range(env.max_steps):
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if term: success += 1
            if term or trunc: break
        rewards.append(ep_reward)
    return success / episodes, np.mean(rewards)

# --- 3. 主流程 ---
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=f"runs/v7_curriculum_4090_fix")
    
    # 初始化环境示例（为了获取维度）
    generator = RandomScenarioGenerator(grid_size=64)
    temp_env = RiskAwareGridEnv(generator.generate(0), observation_size=64)
    
    # 核心：自动对齐维度，彻底解决 index out of bounds
    obs_shape = temp_env.observation_space.shape # (3, 64, 64)
    n_actions = temp_env.action_space.n          # 应该是 8
    
    cfg = TrainConfig(episodes=args.episodes)
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.min_replay_size = 15000  # 4090 跑得快，多存点样本再开始

    replay = PrioritizedReplayBuffer(capacity=200000) # 扩容
    agent = D3QNAgent(
        state_shape=obs_shape,
        num_actions=n_actions, 
        replay_buffer=replay,
        config=AgentConfig(gamma=0.99, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size),
        device=str(device)
    )

    # 评估基准（静态地图）
    eval_scenario = generator.generate(1000) # 生成一个中等难度的作为基准
    eval_env = RiskAwareGridEnv(eval_scenario, observation_size=64, seed=999)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step, best_eval_success = 0, 0.0
    print(f"🔥 V7 Fixed Training Start | Device: {device} | Actions: {n_actions}")

    try:
        pbar = trange(cfg.episodes, desc="V7 Training")
        env = None
        for ep in pbar:
            # 每 10 轮换一张新随机地图，增加泛化能力
            if ep % 10 == 0 or env is None:
                new_scenario = generator.generate(ep)
                # 动态 guide_omega 衰减
                current_omega = max(0.6, 2.0 * (0.97 ** (ep // 100)))
                env = RiskAwareGridEnv(
                    scenario=new_scenario,
                    observation_size=64,
                    reward_weights=RewardWeights(goal_reward=500, risk_lambda=3.0, guide_omega=current_omega),
                    seed=42 + ep
                )

            state, _ = env.reset()
            ep_reward, ep_losses = 0.0, []

            for _ in range(env.max_steps):
                epsilon = linear_schedule(1.0, 0.05, global_step, 400000)
                beta = linear_schedule(0.4, 1.0, global_step, 400000)
                h_prob = linear_schedule(0.5, 0.02, global_step, 500000)

                # 使用 Dijkstra 引导
                action = agent.select_action(state, epsilon=epsilon, heuristic_prob=h_prob, heuristic_fn=env.heuristic_action)
                next_state, reward, term, trunc, _ = env.step(action)
                
                agent.remember(state, action, reward, next_state, term or trunc)
                
                if len(replay) >= cfg.min_replay_size:
                    m = agent.learn(beta=beta)
                    ep_losses.append(m["loss"])

                state = next_state
                ep_reward += reward
                global_step += 1
                if term or trunc: break

            # 日志记录
            avg_loss = np.mean(ep_losses) if ep_losses else 0
            writer.add_scalar("Train/Reward", ep_reward, ep)
            writer.add_scalar("Train/Loss", avg_loss, ep)

            if (ep + 1) % 50 == 0:
                success, _ = evaluate_greedy(agent, eval_env, 20, 999)
                writer.add_scalar("Eval/SuccessRate", success, ep)
                if success >= best_eval_success:
                    best_eval_success = success
                    agent.save(checkpoint_dir / "best_model.pt")
                pbar.set_postfix({"Best_Succ": f"{best_eval_success:.2%}", "Loss": f"{avg_loss:.4f}"})

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    agent.save(checkpoint_dir / "final_model.pt")
    writer.close()

if __name__ == "__main__":
    main()