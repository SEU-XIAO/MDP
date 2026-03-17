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
            while True:
                ex = random.randint(0, self.grid_size - 1)
                ey = random.randint(0, self.grid_size - 1)
                # 禁止出现在四角
                if (ex, ey) not in [
                    (0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)
                ]:
                    break
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
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
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

    # 加载评估测试集
    import json
    eval_dataset_path = "src/environment/eval_dataset_v2.json"
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    def evaluate_on_dataset(agent, cfg):
        results = {}
        for level, scenarios in eval_dataset.items():
            success = 0
            for scenario in scenarios:
                env = RiskAwareGridEnv(
                    scenario=scenario,
                    observation_size=cfg.observation_size,
                    max_steps=cfg.max_steps_per_episode,
                    enemy_jitter=cfg.enemy_jitter,
                    start_jitter=cfg.start_jitter,
                    reward_weights=RewardWeights(
                        goal_reward=cfg.goal_reward,
                        step_penalty=cfg.step_penalty,
                        risk_lambda=cfg.risk_lambda,
                        guide_omega=cfg.guide_omega,
                        timeout_penalty=cfg.timeout_penalty,
                        blocked_move_penalty=cfg.blocked_move_penalty
                    ),
                    blocked_risk_threshold=cfg.high_risk_block_threshold,
                    seed=cfg.seed
                )
                state, _ = env.reset()
                for _ in range(env.max_steps):
                    action = agent.select_action(state, epsilon=0.0)
                    next_state, reward, term, trunc, _ = env.step(action)
                    state = next_state
                    if term or trunc:
                        if term:
                            success += 1
                        break
            results[level] = success / max(len(scenarios), 1)
        return results

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=f"runs/v1")

    cfg = TrainConfig(episodes=args.episodes)
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.min_replay_size = 15000  # 4090 跑得快，多存点样本再开始

    generator = RandomScenarioGenerator(grid_size=cfg.observation_size, start_pos=(1, 1), goal_pos=(cfg.observation_size-2, cfg.observation_size-2))
    temp_env = RiskAwareGridEnv(generator.generate(0), observation_size=cfg.observation_size, max_steps=cfg.max_steps_per_episode, enemy_jitter=cfg.enemy_jitter, start_jitter=cfg.start_jitter, reward_weights=RewardWeights(
        goal_reward=cfg.goal_reward,
        step_penalty=cfg.step_penalty,
        risk_lambda=cfg.risk_lambda,
        guide_omega=cfg.guide_omega,
        timeout_penalty=cfg.timeout_penalty,
        blocked_move_penalty=cfg.blocked_move_penalty
    ), blocked_risk_threshold=cfg.high_risk_block_threshold, seed=cfg.seed)

    obs_shape = temp_env.observation_space.shape
    n_actions = temp_env.action_space.n

    replay = PrioritizedReplayBuffer(capacity=cfg.replay_capacity)
    agent = D3QNAgent(
        state_shape=obs_shape,
        num_actions=n_actions,
        replay_buffer=replay,
        config=AgentConfig(gamma=cfg.gamma, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size),
        device=str(device)
    )

    eval_scenario = generator.generate(1000)
    eval_env = RiskAwareGridEnv(eval_scenario, observation_size=cfg.observation_size, max_steps=cfg.max_steps_per_episode, enemy_jitter=cfg.enemy_jitter, start_jitter=cfg.start_jitter, reward_weights=RewardWeights(
        goal_reward=cfg.goal_reward,
        step_penalty=cfg.step_penalty,
        risk_lambda=cfg.risk_lambda,
        guide_omega=cfg.guide_omega,
        timeout_penalty=cfg.timeout_penalty,
        blocked_move_penalty=cfg.blocked_move_penalty
    ), blocked_risk_threshold=cfg.high_risk_block_threshold, seed=cfg.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step, best_eval_success = 0, 0.0
    print(f"🔥 V7 Fixed Training Start | Device: {device} | Actions: {n_actions}")

    try:
        pbar = trange(cfg.episodes, desc="V7 Training")
        # 优化：只初始化一次env，后续直接reset新场景
        env = RiskAwareGridEnv(
            scenario=generator.generate(0),
            observation_size=cfg.observation_size,
            max_steps=cfg.max_steps_per_episode,
            enemy_jitter=cfg.enemy_jitter,
            start_jitter=cfg.start_jitter,
            reward_weights=RewardWeights(
                goal_reward=cfg.goal_reward,
                step_penalty=cfg.step_penalty,
                risk_lambda=cfg.risk_lambda,
                guide_omega=cfg.guide_omega,
                timeout_penalty=cfg.timeout_penalty,
                blocked_move_penalty=cfg.blocked_move_penalty
            ),
            blocked_risk_threshold=cfg.high_risk_block_threshold,
            seed=cfg.seed
        )
        for ep in pbar:
            # 只reset新场景，不重建env
            new_scenario = generator.generate(ep)
            env.reset(scenario=new_scenario)

            ep_reward, ep_losses = 0.0, []
            state, _ = env.reset(scenario=new_scenario)

            for step_idx in range(env.max_steps):
                epsilon = linear_schedule(cfg.epsilon_start, cfg.epsilon_end, global_step, cfg.epsilon_decay_steps)
                beta = linear_schedule(0.4, 1.0, global_step, 400000)
                h_prob = linear_schedule(cfg.heuristic_start, cfg.heuristic_end, global_step, cfg.heuristic_decay_steps)

                action = agent.select_action(state, epsilon=epsilon, heuristic_prob=h_prob, heuristic_fn=env.heuristic_action)
                next_state, reward, term, trunc, _ = env.step(action)

                agent.remember(state, action, reward, next_state, term or trunc)

                # 优化：每4步才learn一次
                if len(replay) >= cfg.min_replay_size and (step_idx % 4 == 0):
                    m = agent.learn(beta=beta)
                    ep_losses.append(m["loss"])

                state = next_state
                ep_reward += reward
                global_step += 1
                if term or trunc: break

            avg_loss = np.mean(ep_losses) if ep_losses else 0
            writer.add_scalar("Train/Reward", ep_reward, ep)
            writer.add_scalar("Train/Loss", avg_loss, ep)

            # 前500轮不评估，之后每100轮评估一次
            if ep + 1 > 500 and (ep + 1 - 500) % 100 == 0:
                eval_results = evaluate_on_dataset(agent, cfg)
                for level, rate in eval_results.items():
                    writer.add_scalar(f"Eval/{level}_SuccessRate", rate, ep)
                print(f"[Eval] 难度分级成功率: " + " | ".join([f"{level}: {rate:.2%}" for level, rate in eval_results.items()]))
                if eval_results.get("normal", 0) >= best_eval_success:
                    best_eval_success = eval_results.get("normal", 0)
                    agent.save(checkpoint_dir / "best_model.pt")
                pbar.set_postfix({"Best_Succ": f"{best_eval_success:.2%}", "Loss": f"{avg_loss:.4f}"})

    except KeyboardInterrupt:
        print("\nInterrupted. Saving interrupt model...")
        agent.save(checkpoint_dir / "interrupt_model.pt")
    agent.save(checkpoint_dir / "final_model.pt")
    writer.close()

if __name__ == "__main__":
    main()