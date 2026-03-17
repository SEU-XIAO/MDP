
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.patches import Circle
import torch
from src.agent.d3qn_agent import D3QNAgent
from src.environment.risk_grid_env import RiskAwareGridEnv, RewardWeights
from src.replay.per_buffer import PrioritizedReplayBuffer

# 加载场景
with open("eval_dataset_v2.json", "r", encoding="utf-8") as f:
    eval_dataset = json.load(f)
scenario = eval_dataset["normal"][0]  # 可选任意场景

grid_size = scenario["map"]["grid_size"]
start_pos = scenario["map"]["start_pos"]
goal_pos = scenario["map"]["goal_pos"]
enemies = scenario["enemies"]

# 构建环境
env = RiskAwareGridEnv(
    scenario=scenario,
    observation_size=grid_size,
    max_steps=400,
    enemy_jitter=0,
    start_jitter=0,
    reward_weights=RewardWeights(),
    blocked_risk_threshold=0.95,
    seed=42
)

# 加载模型（可自行替换路径）
model_path = "checkpoints/best_model.pt"
replay_stub = PrioritizedReplayBuffer(capacity=1)
agent = D3QNAgent(
    state_shape=env.observation_space.shape,
    num_actions=env.action_space.n,
    replay_buffer=replay_stub,
    device="cpu"
)
agent.load(model_path)

# 推理轨迹
state, _ = env.reset(seed=42)
path_points_x, path_points_y = [], []
for t in range(env.max_steps):
    action = agent.select_action(state, epsilon=0.0)
    next_state, reward, terminated, truncated, info = env.step(action)
    ax_pos_x, ax_pos_y = env.agent_pos
    path_points_x.append(ax_pos_x)
    path_points_y.append(ax_pos_y)
    state = next_state
    if terminated or truncated:
        break

# 生成空栅格
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_xticks(np.arange(0, grid_size, 1))
ax.set_yticks(np.arange(0, grid_size, 1))
ax.grid(True, color="gray", linewidth=0.5, alpha=0.5)
ax.set_aspect("equal")

# 绘制敌人探测区（圆形）
for enemy in enemies:
    ex, ey = enemy["pos"]
    for zone in enemy["detection_zones"]:
        r = zone["r"]
        circle = Circle((ex, ey), r, edgecolor="red", facecolor="none", linewidth=2, alpha=0.7)
        ax.add_patch(circle)
    ax.scatter([ex], [ey], c="red", s=60, marker="o", label="Enemy" if enemy==enemies[0] else None)

# 绘制起点终点
ax.scatter([start_pos[0]], [start_pos[1]], c="lime", s=100, marker="s", label="Start")
ax.scatter([goal_pos[0]], [goal_pos[1]], c="blue", s=100, marker="*", label="Goal")

# 绘制真实模型轨迹
ax.plot(path_points_x, path_points_y, color="cyan", linewidth=3, label="Real Path")

ax.legend(loc="upper right")
plt.title("Path Visualization (Grid, Enemy Circles, Real Trajectory)")
plt.show()
