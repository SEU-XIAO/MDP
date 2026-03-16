from __future__ import annotations

import argparse
import random
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from src.agent.d3qn_agent import AgentConfig, D3QNAgent
from src.config import TrainConfig, load_scenario
from src.environment.risk_grid_env import RewardWeights, RiskAwareGridEnv
from src.replay.per_buffer import PrioritizedReplayBuffer

# --- 1. 动态课程学习生成器 ---
class RandomScenarioGenerator:
    def __init__(self, grid_size=64, start_pos=(1, 1), goal_pos=(62, 62)):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos

    def generate(self, episode_idx: int):
        """
        根据训练进度动态调整难度：课程学习 (Curriculum Learning)
        """
        # 风险区数量随轮次增加：0轮时3-5个，2000轮时6-10个
        max_enemies = min(10, 5 + (episode_idx // 400))
        num_enemies = random.randint(3, max_enemies)
        
        enemies = []
        for _ in range(num_enemies):
            ex = random.randint(10, self.grid_size - 11)
            ey = random.randint(10, self.grid_size - 11)
            # 难度越高，风险半径越随机
            enemies.append({
                "pos": [ex, ey], 
                "detection_zones": [{"r": random.uniform(3.0, 7.5), "p": random.uniform(0.7, 1.0)}]
            })

        # U型胡同出现概率随进度提升：最高 35%
        u_prob = min(0.35, 0.1 + (episode_idx / 4000))
        if random.random() < u_prob:
            enemies.extend(self._create_u_shape(episode_idx))

        return {
            "map": {"grid_size": self.grid_size, "start_pos": self.start_pos, "goal_pos": self.goal_pos},
            "enemies": enemies,
        }

    def _create_u_shape(self, episode_idx):
        u_points = []
        # 随机位置，但确保开口可能阻挡路径
        bx = random.randint(15, 35)
        by = random.randint(15, 35)
        # 随着轮次增加，U型墙可能变得更长更深
        length = min(15, 10 + (episode_idx // 500))
        
        for i in range(length): # 底部
            u_points.append({"pos": [bx + i, by + 10], "detection_zones": [{"r": 2.2, "p": 1.0}]})
        for i in range(length - 2): # 两翼
            u_points.append({"pos": [bx, by + i], "detection_zones": [{"r": 2.2, "p": 1.0}]})
            u_points.append({"pos": [bx + length - 1, by + i], "detection_zones": [{"r": 2.2, "p": 1.0}]})
        return u_points

# --- 2. 辅助函数 ---
def linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0: return end
    return start + (end - start) * min(step / total_steps, 1.0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7: Curriculum Learning on 4090")
    parser.add_argument("--scenario", type=str, default="configs/scenario.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v7_curriculum")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3000) # 建议增加到 3000
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-5)
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

# --- 3. 主训练流程 ---
def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # 初始化日志
    writer = SummaryWriter(log_dir=f"runs/v7_curriculum_4090")
    
    # 加载基础配置并根据建议调优
    cfg = TrainConfig(episodes=args.episodes)
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.epsilon_decay_steps = 500000  # 延长探索期
    cfg.min_replay_size = 20000       # 预热更久，见多识广
    cfg.goal_reward = 500.0           # 强化终点诱惑
    cfg.step_penalty = -0.4           # 给绕路留出容错

    # 初始化生成器
    generator = RandomScenarioGenerator(grid_size=cfg.observation_size)
    env = None 
    
    # 评估环境（静态 Benchmark）
    eval_scenario = load_scenario(args.scenario)
    eval_env = RiskAwareGridEnv(
        scenario=eval_scenario,
        observation_size=cfg.observation_size,
        max_steps=cfg.max_steps_per_episode,
        reward_weights=RewardWeights(goal_reward=500, step_penalty=-0.4, risk_lambda=2.5, guide_omega=1.5),
        seed=999
    )

    replay = PrioritizedReplayBuffer(capacity=cfg.replay_capacity)
    agent = D3QNAgent(
        state_shape=(3, 64, 64), # 确保与环境一致
        num_actions=5,
        replay_buffer=replay,
        config=AgentConfig(gamma=cfg.gamma, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size),
        device=str(device)
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_eval_success = 0.0

    print(f"🔥 V7 Curriculum Training Start | Device: {device}")

    try:
        pbar = trange(cfg.episodes, desc="V7 Training")
        for ep in pbar:
            # --- 课程学习核心：动态更新地图与权重 ---
            if ep % 10 == 0 or env is None:
                new_scenario = generator.generate(ep)
                # 动态权重：引导奖励 guide_omega 随轮次缓慢下降 (2.0 -> 0.8)
                current_omega = max(0.8, 2.0 * (0.95 ** (ep // 100)))
                
                rw = RewardWeights(
                    goal_reward=cfg.goal_reward,
                    step_penalty=cfg.step_penalty,
                    risk_lambda=cfg.risk_lambda,
                    guide_omega=current_omega,
                    timeout_penalty=cfg.timeout_penalty,
                    blocked_move_penalty=cfg.blocked_move_penalty
                )
                
                env = RiskAwareGridEnv(
                    scenario=new_scenario,
                    observation_size=cfg.observation_size,
                    max_steps=cfg.max_steps_per_episode,
                    enemy_jitter=5, # 直接拉满泛化压力
                    start_jitter=5,
                    reward_weights=rw,
                    seed=cfg.seed + ep
                )

            state, _ = env.reset()
            ep_reward, ep_losses = 0.0, []

            for _ in range(cfg.max_steps_per_episode):
                epsilon = linear_schedule(cfg.epsilon_start, cfg.epsilon_end, global_step, cfg.epsilon_decay_steps)
                beta = linear_schedule(0.4, 1.0, global_step, cfg.epsilon_decay_steps)
                
                # 启发式选择概率也随全局步数衰减
                h_prob = linear_schedule(0.6, 0.05, global_step, 600000)

                action = agent.select_action(state, epsilon=epsilon, heuristic_prob=h_prob, heuristic_fn=env.heuristic_action)
                next_state, reward, term, trunc, info = env.step(action)
                
                agent.remember(state, action, reward, next_state, term or trunc)
                
                if len(replay) >= cfg.min_replay_size:
                    m = agent.learn(beta=beta)
                    ep_losses.append(m["loss"])

                state = next_state
                ep_reward += reward
                global_step += 1
                if term or trunc: break

            # Tensorboard 记录
            avg_loss = np.mean(ep_losses) if ep_losses else 0
            writer.add_scalar("Train/Reward", ep_reward, ep)
            writer.add_scalar("Train/Loss", avg_loss, ep)
            writer.add_scalar("Params/Guide_Omega", current_omega, ep)

            # 评估与保存
            if (ep + 1) % cfg.eval_interval_episodes == 0:
                success, rew = evaluate_greedy(agent, eval_env, cfg.eval_episodes, cfg.seed + ep)
                writer.add_scalar("Eval/SuccessRate", success, ep)
                if success >= best_eval_success:
                    best_eval_success = success
                    agent.save(checkpoint_dir / "best_model.pt")
                pbar.set_postfix({"Best_Succ": f"{best_eval_success:.2%}", "Curr_Loss": f"{avg_loss:.4f}"})

    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")
        agent.save(checkpoint_dir / "interrupted.pt")
    
    agent.save(checkpoint_dir / "final_model.pt")
    writer.close()

if __name__ == "__main__":
    main()