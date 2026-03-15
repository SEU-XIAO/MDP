from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from src.agent.d3qn_agent import AgentConfig, D3QNAgent
from src.config import TrainConfig, load_scenario
from src.environment.risk_grid_env import RewardWeights, RiskAwareGridEnv
from src.replay.per_buffer import PrioritizedReplayBuffer


def linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    ratio = min(step / total_steps, 1.0)
    return start + (end - start) * ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train D3QN on risk-aware grid world (Optimized for 4090)")
    # 基础路径与设备
    parser.add_argument("--scenario", type=str, default="configs/scenario.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--resume", type=str, default=None)
    
    # 训练循环控制
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-global-step", type=int, default=None)

    # 4090 核心参数优化
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--learning-rate", "--lr", type=float, default=8e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=450, help="Max steps per episode")
    parser.add_argument("--epsilon-decay-steps", type=int, default=200000, help="Steps for epsilon decay")
    
    # 环境抖动控制
    parser.add_argument("--enemy-jitter", type=int, default=None)
    parser.add_argument("--start-jitter", type=int, default=None)
    parser.add_argument("--eval-enemy-jitter", type=int, default=None)
    parser.add_argument("--eval-start-jitter", type=int, default=None)
    
    return parser.parse_args()


def _resolve_model_path(path_text: str) -> Path:
    p = Path(path_text)
    if p.exists():
        return p
    fallback = Path("checkpoints") / p.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Model file not found: {p}")


def evaluate_greedy(agent: D3QNAgent, env: RiskAwareGridEnv, episodes: int, seed: int) -> tuple[float, float]:
    success = 0
    rewards: list[float] = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        ep_reward = 0.0
        for _ in range(env.max_steps):
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if terminated:
                success += 1
            if terminated or truncated:
                break
        rewards.append(ep_reward)
    return success / max(episodes, 1), float(np.mean(rewards))


def main() -> None:
    args = parse_args()

    # --- 设备检测 ---
    if args.device:
        device_type = args.device
    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(device_type)
    print("="*60)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀  Training started on GPU: {gpu_name}")
    else:
        print("💻  Training started on CPU")
    print(f"📍  Device object: {device}")
    print("="*60)

    # --- 配置加载 ---
    cfg = TrainConfig(episodes=args.episodes, seed=args.seed)
    
    # 覆盖命令行输入的参数
    cfg.learning_rate = args.learning_rate
    cfg.batch_size = args.batch_size
    cfg.max_steps_per_episode = args.max_steps
    cfg.epsilon_decay_steps = args.epsilon_decay_steps
    
    if args.enemy_jitter is not None:
        cfg.enemy_jitter = args.enemy_jitter
    if args.start_jitter is not None:
        cfg.start_jitter = args.start_jitter
    if args.eval_enemy_jitter is not None:
        cfg.eval_enemy_jitter = args.eval_enemy_jitter
    if args.eval_start_jitter is not None:
        cfg.eval_start_jitter = args.eval_start_jitter

    scenario = load_scenario(args.scenario)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 奖励权重逻辑
    reward_weights = RewardWeights(
        goal_reward=cfg.goal_reward,
        step_penalty=cfg.step_penalty,
        risk_lambda=cfg.risk_lambda,
        guide_omega=cfg.guide_omega,
        timeout_penalty=cfg.timeout_penalty,
        blocked_move_penalty=cfg.blocked_move_penalty,
    )
    
    # --- 实例化训练环境 ---
    # 修复：直接使用 0.95 作为默认风险阈值，避免引用 cfg 中不存在的属性
    env = RiskAwareGridEnv(
        scenario=scenario,
        observation_size=cfg.observation_size,
        max_steps=cfg.max_steps_per_episode,
        enemy_jitter=cfg.enemy_jitter,
        start_jitter=cfg.start_jitter,
        reward_weights=reward_weights,
        blocked_risk_threshold=0.95,
        seed=cfg.seed,
    )

    # --- 实例化评估环境 ---
    eval_env = RiskAwareGridEnv(
        scenario=scenario,
        observation_size=cfg.observation_size,
        max_steps=cfg.max_steps_per_episode,
        enemy_jitter=cfg.eval_enemy_jitter,
        start_jitter=cfg.eval_start_jitter,
        reward_weights=reward_weights,
        blocked_risk_threshold=0.95,
        seed=cfg.seed + 10000,
    )

    # --- 实例化 Agent ---
    replay = PrioritizedReplayBuffer(capacity=cfg.replay_capacity, alpha=0.6) # 默认 alpha
    agent = D3QNAgent(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        replay_buffer=replay,
        config=AgentConfig(
            gamma=cfg.gamma,
            learning_rate=cfg.learning_rate,
            target_update_interval=cfg.target_update_interval,
            batch_size=cfg.batch_size,
        ),
        device=device_type,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_path = _resolve_model_path(args.resume)
        agent.load(resume_path)
        print(f"Resumed model from: {resume_path.resolve()}")

    # 训练步数控制
    global_step = 0
    if args.initial_global_step is not None:
        global_step = max(0, int(args.initial_global_step))
    elif args.resume:
        global_step = cfg.epsilon_decay_steps
    
    best_eval_success = float("-inf")
    best_eval_reward = float("-inf")

    interrupted = False
    try:
        # 使用 trange 显示进度条
        for ep in trange(cfg.episodes, desc="Training", ncols=100):
            state, _ = env.reset()
            ep_reward = 0.0
            ep_risk = 0.0
            losses: list[float] = []
            reached_goal = False

            for _ in range(cfg.max_steps_per_episode):
                # 计算各种衰减系数
                epsilon = linear_schedule(
                    cfg.epsilon_start, cfg.epsilon_end,
                    global_step, cfg.epsilon_decay_steps,
                )
                heuristic_prob = linear_schedule(
                    cfg.heuristic_start, cfg.heuristic_end,
                    global_step, cfg.heuristic_decay_steps,
                )
                # PER Beta 衰减，默认为 0.4 -> 1.0
                beta = linear_schedule(0.4, 1.0, global_step, cfg.epsilon_decay_steps)

                action = agent.select_action(
                    state,
                    epsilon=epsilon,
                    heuristic_prob=heuristic_prob,
                    heuristic_fn=env.heuristic_action,
                )

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.remember(state, action, reward, next_state, done)

                if len(replay) >= cfg.min_replay_size:
                    metrics = agent.learn(beta=beta)
                    losses.append(metrics["loss"])

                state = next_state
                ep_reward += reward
                ep_risk += info.get("risk", 0.0)
                global_step += 1

                if terminated:
                    reached_goal = True
                if done:
                    break

            # 定期评估
            if (ep + 1) % cfg.eval_interval_episodes == 0:
                eval_success, eval_reward = evaluate_greedy(
                    agent=agent, env=eval_env, episodes=cfg.eval_episodes, seed=cfg.seed + (ep + 1) * 100
                )

                # 更新最佳模型
                if eval_success >= best_eval_success:
                    if eval_success > best_eval_success or eval_reward > best_eval_reward:
                        best_eval_success = eval_success
                        best_eval_reward = eval_reward
                        agent.save(checkpoint_dir / "best_model.pt")

                print(
                    f"\n[Eval] ep={ep + 1:4d} | success={eval_success:.2%} | "
                    f"avg_reward={eval_reward:8.2f} | best_success={best_eval_success:.2%}"
                )

            # 定期打印日志
            if (ep + 1) % 20 == 0:
                avg_loss = float(np.mean(losses)) if losses else 0.0
                avg_risk = ep_risk / max(env.step_count, 1)
                print(
                    f"Episode {ep + 1:4d} | reward={ep_reward:8.2f} | "
                    f"avg_risk={avg_risk:6.3f} | loss={avg_loss:7.4f} | "
                    f"goal={int(reached_goal)} | epsilon={epsilon:.3f}"
                )
                
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user. Saving model...")
        agent.save(checkpoint_dir / "interrupted_model.pt")

    # 保存最终模型
    agent.save(checkpoint_dir / "final_model.pt")
    print(f"Training Complete. Best eval success: {best_eval_success:.2%}")


if __name__ == "__main__":
    main()