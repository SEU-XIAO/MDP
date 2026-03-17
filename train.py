from __future__ import annotations

import argparse
import random
import os
import json
import copy
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

# --- 1. 辅助工具 ---
def linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0: return end
    return start + (end - start) * min(step / total_steps, 1.0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7: Final Curriculum on 4090")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v3")
    parser.add_argument("--batch-size", type=int, default=256) 
    parser.add_argument("--lr", type=float, default=3e-5)      # 建议微调阶段用 TrainConfig 里的 3e-5
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--model", type=str, default=None, help="Path to pretrained v2 weights")
    return parser.parse_args()

# --- 2. 场景变换工具 ---
def rotate_scenario(scen, k):
    grid = scen["map"]["grid_size"]
    def rot_xy(x, y):
        for _ in range(k):
            x, y = y, grid-1-x
        return x, y
    new = copy.deepcopy(scen)
    new["map"]["start_pos"] = list(rot_xy(*scen["map"]["start_pos"]))
    new["map"]["goal_pos"] = list(rot_xy(*scen["map"]["goal_pos"]))
    for e in new["enemies"]:
        e["pos"] = list(rot_xy(*e["pos"]))
    return new

def mirror_scenario(scen):
    grid = scen["map"]["grid_size"]
    def mirror_xy(x, y):
        return grid-1-x, y
    new = copy.deepcopy(scen)
    new["map"]["start_pos"] = list(mirror_xy(*scen["map"]["start_pos"]))
    new["map"]["goal_pos"] = list(mirror_xy(*scen["map"]["goal_pos"]))
    for e in new["enemies"]:
        e["pos"] = list(mirror_xy(*e["pos"]))
    return new

def perturb_enemies(scen, offset=3):
    grid = scen["map"]["grid_size"]
    new = copy.deepcopy(scen)
    for e in new["enemies"]:
        ex, ey = e["pos"]
        ex += random.randint(-offset, offset)
        ey += random.randint(-offset, offset)
        ex = np.clip(ex, 0, grid-1)
        ey = np.clip(ey, 0, grid-1)
        e["pos"] = [int(ex), int(ey)]
    return new

def evaluate_on_scenarios(agent, cfg, scenarios):
    results = {}
    for idx, scenario in enumerate(scenarios):
        success = 0
        # 这里为了评估稳定，固定环境
        env_eval = RiskAwareGridEnv(
            scenario=scenario,
            observation_size=cfg.observation_size,
            max_steps=cfg.max_steps_per_episode,
            enemy_jitter=0,
            start_jitter=0,
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
        for _ in range(cfg.eval_episodes):
            state, _ = env_eval.reset()
            for _ in range(env_eval.max_steps):
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, term, trunc, _ = env_eval.step(action)
                state = next_state
                if term or trunc:
                    if term: success += 1
                    break
        results[f"eval_{idx+1}"] = success / cfg.eval_episodes
    return results

# --- 3. 主流程 ---
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=f"runs/v3_finetune")

    # 配置加载
    cfg = TrainConfig(episodes=args.episodes)
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    
    # --- 集成预加载场景与dijkstra_map ---
    from glob import glob
    scenario_files = sorted(glob("precompute/data/scenario_*.json"))
    dijkstra_files = sorted(glob("precompute/data/dijkstra_*.npy"))
    assert len(scenario_files) == len(dijkstra_files), "场景与dijkstra_map数量不一致"
    precomputed_data = []
    for scen_path, dij_path in zip(scenario_files, dijkstra_files):
        with open(scen_path, "r") as f:
            scenario = json.load(f)
        dijkstra_map = np.load(dij_path)
        precomputed_data.append({"scenario": scenario, "dijkstra_map": dijkstra_map})

    # 评估集：场景1、场景2、场景4 (用于验证泛化性)
    eval_scenarios = [
        load_scenario("src/environment/scenario_1.json"),
        load_scenario("src/environment/scenario_2.json"),
        load_scenario("src/environment/scenario_4.json"),
    ]

    # 初始化环境以获取空间信息（用预生成场景）
    env = RiskAwareGridEnv(
        scenario=precomputed_data[0]["scenario"],
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
    
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    replay = PrioritizedReplayBuffer(capacity=cfg.replay_capacity)
    agent = D3QNAgent(
        state_shape=obs_shape,
        num_actions=n_actions,
        replay_buffer=replay,
        config=AgentConfig(gamma=cfg.gamma, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size),
        device=str(device)
    )

    if args.model:
        agent.load(args.model)
        print(f"✅ Loaded pretrained model weights from: {args.model}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step, best_eval_success = 0, 0.0
    print(f"🔥 V3 Fine-tuning Start | Device: {device} | Batch: {cfg.batch_size}")

    try:
        pbar = trange(cfg.episodes, desc="Train")
        for ep in pbar:
            # --- 训练时直接从预生成数据中选取场景和dijkstra_map ---
            data_idx = random.randint(0, len(precomputed_data)-1)
            dynamic_scen = precomputed_data[data_idx]["scenario"]
            dijkstra_map = precomputed_data[data_idx]["dijkstra_map"]
            state, _ = env.reset(scenario=dynamic_scen, dijkstra_map=dijkstra_map)
            ep_reward, ep_losses = 0.0, []

            for step_idx in range(env.max_steps):
                epsilon = linear_schedule(cfg.epsilon_start, cfg.epsilon_end, global_step, cfg.epsilon_decay_steps)
                beta = linear_schedule(0.4, 1.0, global_step, 400000)
                h_prob = linear_schedule(cfg.heuristic_start, cfg.heuristic_end, global_step, cfg.heuristic_decay_steps)

                action = agent.select_action(state, epsilon=epsilon, heuristic_prob=h_prob, heuristic_fn=env.heuristic_action)
                next_state, reward, term, trunc, _ = env.step(action)

                agent.remember(state, action, reward, next_state, term or trunc)

                if len(replay) >= cfg.min_replay_size and (global_step % 4 == 0):
                    m = agent.learn(beta=beta)
                    ep_losses.append(m["loss"])

                state = next_state
                ep_reward += reward
                global_step += 1
                if term or trunc: break

            # Logging
            avg_loss = np.mean(ep_losses) if ep_losses else 0
            writer.add_scalar("Train/Reward", ep_reward, ep)
            writer.add_scalar("Train/Loss", avg_loss, ep)
            writer.add_scalar("Train/Epsilon", epsilon, ep)

            # 评估逻辑
            if (ep + 1) >= 500 and (ep + 1) % 100 == 0:
                eval_results = evaluate_on_scenarios(agent, cfg, eval_scenarios)
                for scen, rate in eval_results.items():
                    writer.add_scalar(f"Eval/{scen}_SuccessRate", rate, ep)
                
                print(f"\n[Eval Ep {ep+1}] " + " | ".join([f"{k}: {v:.2%}" for k, v in eval_results.items()]))
                
                # 以场景4（泛化场景）作为最佳模型保存依据
                current_gen_score = eval_results.get("eval_3", 0)
                if current_gen_score >= best_eval_success:
                    best_eval_success = current_gen_score
                    agent.save(checkpoint_dir / "best_model.pt")
                
                pbar.set_postfix({"Best_Gen_Succ": f"{best_eval_success:.2%}", "Loss": f"{avg_loss:.4f}"})

    except KeyboardInterrupt:
        print("\nSaving interrupt model...")
        agent.save(checkpoint_dir / "interrupt_model.pt")
    
    agent.save(checkpoint_dir / "final_model.pt")
    writer.close()

if __name__ == "__main__":
    main()