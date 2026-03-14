from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.agent.d3qn_agent import D3QNAgent
from src.config import load_scenario
from src.environment.risk_grid_env import RiskAwareGridEnv
from src.replay.per_buffer import PrioritizedReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained D3QN model")
    parser.add_argument("--scenario", type=str, default="configs/scenario.json")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = load_scenario(args.scenario)

    env = RiskAwareGridEnv(scenario=scenario, observation_size=64, max_steps=220)
    replay_stub = PrioritizedReplayBuffer(capacity=1)
    agent = D3QNAgent(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        replay_buffer=replay_stub,
        device=args.device,
    )

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
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
    print(f"Success rate: {success / args.episodes:.2%}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average step risk: {np.mean(mean_risks):.4f}")


if __name__ == "__main__":
    main()
