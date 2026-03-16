from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from src.agent.d3qn_agent import D3QNAgent
from src.config import TrainConfig, load_scenario
from src.environment.risk_grid_env import RewardWeights, RiskAwareGridEnv
from src.replay.per_buffer import PrioritizedReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize policy trajectory in real time")
    parser.add_argument("--scenario", type=str, default="configs/scenario.json")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--pause", type=float, default=0.08)
    parser.add_argument("--observation-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--enemy-jitter", type=int, default=0)
    parser.add_argument("--start-jitter", type=int, default=0)
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


def visualize_episode(
    env: RiskAwareGridEnv,
    agent: D3QNAgent,
    epsilon: float,
    pause_seconds: float,
    episode_index: int,
    seed: int,
) -> tuple[bool, float, float, int]:
    state, _ = env.reset(seed=seed)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    fig.canvas.manager.set_window_title(f"Policy Visualization - Episode {episode_index}")

    risk_image = ax.imshow(state[0], origin="lower", cmap="magma", vmin=0.0, vmax=1.0, interpolation="bicubic")
    plt.colorbar(risk_image, ax=ax, fraction=0.046, pad=0.04, label="Risk Probability")

    scale = (env.observation_size - 1) / max(env.world_size - 1, 1)
    zone_colors = ["#ef4444", "#f97316", "#facc15", "#22c55e"]
    for enemy in env.current_enemies:
        ex, ey = env._world_to_obs(tuple(enemy["pos"]))
        zones = sorted(enemy["detection_zones"], key=lambda z: z["r"], reverse=True)
        for zi, zone in enumerate(zones):
            radius_obs = float(zone["r"]) * scale
            color = zone_colors[zi % len(zone_colors)]
            circle_fill = Circle((ex, ey), radius_obs, facecolor=color, edgecolor="none", alpha=0.08)
            circle_edge = Circle((ex, ey), radius_obs, fill=False, edgecolor=color, linewidth=1.2, alpha=0.8)
            ax.add_patch(circle_fill)
            ax.add_patch(circle_edge)

    sx, sy = env._world_to_obs(tuple(env.start_pos))
    gx, gy = env._world_to_obs(tuple(env.goal_pos))
    ax.scatter([sx], [sy], c="lime", s=80, marker="s", label="Start")
    ax.scatter([gx], [gy], c="red", s=80, marker="s", label="Goal")

    path_points_x: list[int] = []
    path_points_y: list[int] = []
    path_line, = ax.plot([], [], color="cyan", linewidth=2.0, alpha=0.85, label="Path")
    agent_scatter = ax.scatter([], [], c="white", s=60, edgecolors="black", linewidths=0.8, label="Agent")

    info_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 6, "edgecolor": "none"},
        color="white",
    )

    ax.set_title("Risk Map + Real-Time Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, env.observation_size - 0.5)
    ax.set_ylim(-0.5, env.observation_size - 0.5)
    ax.grid(color="white", linestyle="--", linewidth=0.3, alpha=0.25)
    ax.legend(loc="upper right")

    cumulative_reward = 0.0
    cumulative_risk = 0.0
    success = False

    for t in range(env.max_steps):
        action = agent.select_action(state, epsilon=epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)

        cumulative_reward += reward
        cumulative_risk += info["risk"]

        ax_pos_x, ax_pos_y = env._world_to_obs(env.agent_pos)
        path_points_x.append(ax_pos_x)
        path_points_y.append(ax_pos_y)

        risk_image.set_data(next_state[0])
        path_line.set_data(path_points_x, path_points_y)
        agent_scatter.set_offsets(np.array([[ax_pos_x, ax_pos_y]], dtype=np.float32))

        avg_risk = cumulative_risk / (t + 1)
        info_text.set_text(
            f"step={t + 1}\n"
            f"reward={cumulative_reward:.2f}\n"
            f"avg_risk={avg_risk:.3f}\n"
            f"cell_risk={info.get('risk', 0.0):.3f}\n"
            f"dijkstra_cost={info.get('dijkstra_cost', 0.0):.2f}"
        )

        plt.pause(pause_seconds)
        state = next_state

        if terminated or truncated:
            success = terminated
            break

    final_steps = env.step_count
    final_avg_risk = cumulative_risk / max(final_steps, 1)
    status = "SUCCESS" if success else "FAILED"
    ax.set_title(f"Episode {episode_index} - {status}")
    plt.pause(max(0.5, pause_seconds))
    plt.show(block=True)
    plt.close(fig)

    return success, cumulative_reward, final_avg_risk, final_steps


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()

    scenario = load_scenario(args.scenario)
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
        observation_size=args.observation_size,
        max_steps=args.max_steps,
        enemy_jitter=args.enemy_jitter,
        start_jitter=args.start_jitter,
        reward_weights=reward_weights,
        blocked_risk_threshold=getattr(cfg, 'high_risk_block_threshold', 0.8),
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

    print(f"Model: {model_path.resolve()}")
    print(f"Scenario: {Path(args.scenario).resolve()}")
    print(f"Episodes: {args.episodes}")
    print(f"Jitter: enemy={args.enemy_jitter}, start={args.start_jitter}")

    all_success = 0
    all_rewards: list[float] = []
    all_risks: list[float] = []

    for ep in range(1, args.episodes + 1):
        success, reward, avg_risk, steps = visualize_episode(
            env=env,
            agent=agent,
            epsilon=args.epsilon,
            pause_seconds=args.pause,
            episode_index=ep,
            seed=args.seed + ep,
        )
        all_success += int(success)
        all_rewards.append(reward)
        all_risks.append(avg_risk)
        print(
            f"Episode {ep:3d} | success={int(success)} | "
            f"reward={reward:8.2f} | avg_risk={avg_risk:6.3f} | steps={steps:3d}"
        )

    print("=" * 72)
    print(f"Success rate: {all_success / max(args.episodes, 1):.2%}")
    print(f"Average reward: {np.mean(all_rewards):.2f}")
    print(f"Average step risk: {np.mean(all_risks):.4f}")


if __name__ == "__main__":
    main()
