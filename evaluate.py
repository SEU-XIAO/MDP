from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.agent.d3qn_agent import D3QNAgent
from src.config import TrainConfig, load_scenario
from src.environment.risk_grid_env import RewardWeights, RiskAwareGridEnv
from src.replay.per_buffer import PrioritizedReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained D3QN model")
    parser.add_argument("--scenario", type=str, default="configs/scenario.json")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--enemy-jitter", type=int, default=None)
    parser.add_argument("--start-jitter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _resolve_model_path(path_text: str) -> Path:
    p = Path(path_text)
    if p.exists():
        return p
    fallback = Path("checkpoints") / p.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Model file not found: {p}")


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()
    scenario = load_scenario(args.scenario)
    enemy_jitter = cfg.eval_enemy_jitter if args.enemy_jitter is None else args.enemy_jitter
    start_jitter = cfg.eval_start_jitter if args.start_jitter is None else args.start_jitter

    reward_weights = RewardWeights(
        goal_reward=cfg.goal_reward,
        step_penalty=cfg.step_penalty,
        risk_lambda=cfg.risk_lambda,
        guide_omega=cfg.guide_omega,
        timeout_penalty=cfg.timeout_penalty,
        blocked_move_penalty=cfg.blocked_move_penalty,
    )
    env = RiskAwareGridEnv(
        scenario=scenario,
        observation_size=cfg.observation_size,
        max_steps=cfg.max_steps_per_episode,
        enemy_jitter=enemy_jitter,
        start_jitter=start_jitter,
        reward_weights=reward_weights,
        blocked_risk_threshold=cfg.high_risk_block_threshold,
        seed=args.seed,
    )
    replay_stub = PrioritizedReplayBuffer(capacity=1)
    agent = D3QNAgent(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        replay_buffer=replay_stub,
        device=args.device,
    )

    model_path = _resolve_model_path(args.model)
    agent.load(model_path)

    rewards = []
    success = 0
    mean_risks = []

    for _ in range(args.episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_risk = 0.0

        for _ in range(env.max_steps):
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_risk += info["risk"]
            state = next_state
            if terminated:
                success += 1
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        mean_risks.append(ep_risk / max(env.step_count, 1))

    print(f"Model: {model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Jitter: enemy={enemy_jitter}, start={start_jitter}")
    print(f"Success rate: {success / args.episodes:.2%}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average step risk: {np.mean(mean_risks):.4f}")


if __name__ == "__main__":
    main()
