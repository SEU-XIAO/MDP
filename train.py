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
    parser = argparse.ArgumentParser(description="Train D3QN on risk-aware grid world")
    parser.add_argument("--scenario", type=str, default="configs/scenario.json")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


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

    cfg = TrainConfig(episodes=args.episodes, seed=args.seed)
    scenario = load_scenario(args.scenario)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    reward_weights = RewardWeights(
        goal_reward=cfg.goal_reward,
        step_penalty=cfg.step_penalty,
        risk_lambda=cfg.risk_lambda,
        guide_omega=cfg.guide_omega,
        timeout_penalty=cfg.timeout_penalty,
    )
    env = RiskAwareGridEnv(
        scenario=scenario,
        observation_size=cfg.observation_size,
        max_steps=cfg.max_steps_per_episode,
        enemy_jitter=cfg.enemy_jitter,
        start_jitter=cfg.start_jitter,
        reward_weights=reward_weights,
        seed=cfg.seed,
    )

    eval_env = RiskAwareGridEnv(
        scenario=scenario,
        observation_size=cfg.observation_size,
        max_steps=cfg.max_steps_per_episode,
        enemy_jitter=cfg.eval_enemy_jitter,
        start_jitter=cfg.eval_start_jitter,
        reward_weights=reward_weights,
        seed=cfg.seed + 10_000,
    )

    replay = PrioritizedReplayBuffer(capacity=cfg.replay_capacity, alpha=cfg.per_alpha)
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
        device=args.device,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_eval_success = float("-inf")
    best_eval_reward = float("-inf")

    for ep in trange(cfg.episodes, desc="Training", ncols=100):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_risk = 0.0
        losses: list[float] = []
        reached_goal = False

        for _ in range(cfg.max_steps_per_episode):
            epsilon = linear_schedule(
                cfg.epsilon_start,
                cfg.epsilon_end,
                global_step,
                cfg.epsilon_decay_steps,
            )
            heuristic_prob = linear_schedule(
                cfg.heuristic_start,
                cfg.heuristic_end,
                global_step,
                cfg.heuristic_decay_steps,
            )
            beta = linear_schedule(
                cfg.per_beta_start,
                cfg.per_beta_end,
                global_step,
                cfg.per_beta_steps,
            )

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
            ep_risk += info["risk"]
            global_step += 1

            if terminated:
                reached_goal = True
            if done:
                break

        if (ep + 1) % cfg.eval_interval_episodes == 0:
            eval_success, eval_reward = evaluate_greedy(
                agent=agent,
                env=eval_env,
                episodes=cfg.eval_episodes,
                seed=cfg.seed + (ep + 1) * 100,
            )

            should_save = False
            if eval_success > best_eval_success:
                should_save = True
            elif eval_success == best_eval_success and eval_reward > best_eval_reward:
                should_save = True

            if should_save:
                best_eval_success = eval_success
                best_eval_reward = eval_reward
                agent.save(checkpoint_dir / "best_model.pt")

            print(
                f"[Eval] ep={ep + 1:4d} | success={eval_success:.2%} | "
                f"avg_reward={eval_reward:8.2f} | best_success={best_eval_success:.2%}"
            )

        if (ep + 1) % 20 == 0:
            avg_loss = float(np.mean(losses)) if losses else 0.0
            avg_risk = ep_risk / max(env.step_count, 1)
            print(
                f"Episode {ep + 1:4d} | reward={ep_reward:8.2f} | "
                f"avg_risk={avg_risk:6.3f} | loss={avg_loss:7.4f} | "
                f"goal={int(reached_goal)} | epsilon={epsilon:.3f}"
            )

    if best_eval_success < 0.0:
        agent.save(checkpoint_dir / "best_model.pt")
        best_eval_success = 0.0
        best_eval_reward = 0.0

    agent.save(checkpoint_dir / "final_model.pt")
    print("Training finished.")
    print(f"Best eval success: {best_eval_success:.2%}")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"Saved checkpoints to: {checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()
