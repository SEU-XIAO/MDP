from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import numpy as np

# 导入你项目中的模块
from src.agent.d3qn_agent import D3QNAgent, AgentConfig
from src.config import TrainConfig, load_scenario
from src.environment.risk_grid_env import RewardWeights, RiskAwareGridEnv
from src.replay.per_buffer import PrioritizedReplayBuffer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained D3QN model")
    parser.add_argument("--scenario", type=str, default="configs/eval_dataset.json", help="Path to scenario file")
    parser.add_argument("--model", type=str, default="checkpoints/v1/best_model.pth", help="Path to model file")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per scenario")
    parser.add_argument("--enemy-jitter", type=int, default=None)
    parser.add_argument("--start-jitter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    return parser.parse_args()

def _resolve_model_path(path_text: str) -> Path:
    """解析模型路径，支持绝对路径或相对于checkpoints的路径"""
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
    
    # 1. 奖励权重与基础配置
    reward_weights = RewardWeights(
        goal_reward=cfg.goal_reward,
        step_penalty=cfg.step_penalty,
        risk_lambda=cfg.risk_lambda,
        guide_omega=cfg.guide_omega,
        timeout_penalty=cfg.timeout_penalty,
        blocked_move_penalty=cfg.blocked_move_penalty
    )

    # 2. 加载场景数据
    scenario_path = args.scenario
    if scenario_path.endswith('.json'):
        with open(scenario_path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
    else:
        # 如果不是json，尝试作为单个场景路径加载
        scenario_data = load_scenario(scenario_path)

    # 3. 初始化临时环境以获取维度信息
    # 取数据中的第一个场景作为样板
    if isinstance(scenario_data, dict) and 'map' not in scenario_data:
        # 多难度字典格式
        sample_level = list(scenario_data.keys())[0]
        sample_scenario = scenario_data[sample_level][0]
    else:
        # 单场景格式
        sample_scenario = scenario_data

    temp_env = RiskAwareGridEnv(
        scenario=sample_scenario,
        observation_size=cfg.observation_size,
        max_steps=cfg.max_steps_per_episode,
        enemy_jitter=cfg.enemy_jitter,
        start_jitter=cfg.start_jitter,
        reward_weights=reward_weights,
        blocked_risk_threshold=cfg.high_risk_block_threshold,
        seed=cfg.seed
    )
    
    obs_shape = temp_env.observation_space.shape
    n_actions = temp_env.action_space.n

    # 4. 初始化 Agent
    # 评估不需要真正的Buffer，但Agent结构需要传入
    replay = PrioritizedReplayBuffer(capacity=100) 
    
    # 确定设备
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = D3QNAgent(
        state_shape=obs_shape,
        num_actions=n_actions,
        replay_buffer=replay,
        config=AgentConfig(
            gamma=cfg.gamma, 
            learning_rate=cfg.learning_rate, 
            batch_size=cfg.batch_size
        ),
        device=device
    )
    
    # 5. 加载权重
    model_path = _resolve_model_path(args.model)
    agent.load(str(model_path))
    print(f"✅ Successfully loaded model from: {model_path} on {device}")

    # 6. 开始评估逻辑
    if isinstance(scenario_data, dict) and 'map' not in scenario_data:
        print(f"🚀 Starting Batch Evaluation on {len(scenario_data)} levels...")
        results = {}
        risk_stats = {}
        step_stats = {}
        for level, scenarios in scenario_data.items():
            level_success = 0
            level_risk = []
            level_step_ratio = []
            for sc in scenarios:
                env = RiskAwareGridEnv(
                    scenario=sc,
                    observation_size=cfg.observation_size,
                    max_steps=cfg.max_steps_per_episode,
                    enemy_jitter=args.enemy_jitter if args.enemy_jitter is not None else cfg.eval_enemy_jitter,
                    start_jitter=args.start_jitter if args.start_jitter is not None else cfg.eval_start_jitter,
                    reward_weights=reward_weights,
                    blocked_risk_threshold=cfg.high_risk_block_threshold,
                    seed=cfg.seed
                )
                state, _ = env.reset()
                risk_sum = 0.0
                step_count = 0
                path = []
                for _ in range(env.max_steps):
                    path.append(env.agent_pos)
                    risk_sum += env._risk_at(env.agent_pos)
                    action = agent.select_action(state, epsilon=0.0)
                    next_state, reward, term, trunc, _ = env.step(action)
                    state = next_state
                    step_count += 1
                    if term or trunc:
                        if term:
                            level_success += 1
                        break
                avg_risk = risk_sum / max(step_count, 1)
                # Dijkstra最短路径步数（终点距离场+1）
                gx, gy = env.goal_pos
                startx, starty = env.start_pos
                dijkstra_steps = int(env.dijkstra_map[startx, starty]) + 1
                step_ratio = step_count / max(dijkstra_steps, 1)
                level_risk.append(avg_risk)
                level_step_ratio.append(step_ratio)
            rate = level_success / max(len(scenarios), 1)
            results[level] = rate
            risk_stats[level] = np.mean(level_risk)
            step_stats[level] = np.mean(level_step_ratio)
            print(f"   - {level}: {rate:.2%} | Avg Risk Exposure: {risk_stats[level]:.4f} | Step Ratio: {step_stats[level]:.4f}")
        print("\nFinal Results Summary:")
        for level in results:
            print(f" {level:10}: {results[level]:.2%} | Avg Risk Exposure: {risk_stats[level]:.4f} | Step Ratio: {step_stats[level]:.4f}")
            
    else:
        # 单场景多轮评估逻辑
        print(f"🚀 Starting Single Scenario Evaluation for {args.episodes} episodes...")
        success_count = 0
        risk_list = []
        step_ratio_list = []
        for i in range(args.episodes):
            env = RiskAwareGridEnv(
                scenario=scenario_data,
                observation_size=cfg.observation_size,
                max_steps=cfg.max_steps_per_episode,
                enemy_jitter=args.enemy_jitter if args.enemy_jitter is not None else cfg.eval_enemy_jitter,
                start_jitter=args.start_jitter if args.start_jitter is not None else cfg.eval_start_jitter,
                reward_weights=reward_weights,
                blocked_risk_threshold=cfg.high_risk_block_threshold,
                seed=cfg.seed + i # 每轮使用不同随机种子
            )
            state, _ = env.reset()
            risk_sum = 0.0
            step_count = 0
            for _ in range(env.max_steps):
                risk_sum += env._risk_at(env.agent_pos)
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, term, trunc, _ = env.step(action)
                state = next_state
                step_count += 1
                if term or trunc:
                    if term:
                        success_count += 1
                    break
            avg_risk = risk_sum / max(step_count, 1)
            gx, gy = env.goal_pos
            startx, starty = env.start_pos
            dijkstra_steps = int(env.dijkstra_map[startx, starty]) + 1
            step_ratio = step_count / max(dijkstra_steps, 1)
            risk_list.append(avg_risk)
            step_ratio_list.append(step_ratio)
            if (i+1) % 5 == 0:
                print(f"   Progress: {i+1}/{args.episodes}...")

        print(f"\nSingle Scenario Success Rate: {success_count}/{args.episodes} ({success_count/args.episodes:.2%})")
        print(f"Avg Risk Exposure: {np.mean(risk_list):.4f}")
        print(f"Step Ratio: {np.mean(step_ratio_list):.4f}")

if __name__ == "__main__":
    main()