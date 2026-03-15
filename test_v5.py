import argparse
import random
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from src.agent.d3qn_agent import D3QNAgent
from src.config import TrainConfig, load_scenario
from src.environment.risk_grid_env import RewardWeights, RiskAwareGridEnv

def test_model():
    parser = argparse.ArgumentParser(description="Test D3QN Performance (V5 4090)")
    parser.add_argument("--model-path", type=str, default="checkpoints/retrain_v5_4090/best_model.pt")
    parser.add_argument("--scenario", type=str, default="configs/scenario.json")
    parser.add_argument("--num-tests", type=int, default=100) # 跑100轮统计
    parser.add_argument("--enemy-jitter", type=int, default=1) # 默认抖动
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. 加载配置与环境
    cfg = TrainConfig()
    scenario = load_scenario(args.scenario)
    
    # 模拟高压环境进行测试
    env = RiskAwareGridEnv(
        scenario=scenario,
        observation_size=cfg.observation_size,
        max_steps=450,
        enemy_jitter=args.enemy_jitter,
        start_jitter=1,
        reward_weights=RewardWeights(goal_reward=500, step_penalty=-0.3, risk_lambda=1.5),
        seed=12345 # 使用固定随机种子进行测试
    )

    # 2. 加载模型
    # 注意：这里的 state_shape 和 num_actions 必须与训练时一致
    agent = D3QNAgent(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        replay_buffer=None, # 测试不需要 buffer
        device=args.device
    )
    agent.load(args.model_path)
    print(f"✅ Loaded model from: {args.model_path}")

    # 3. 开始压力测试
    success_count = 0
    total_rewards = []
    total_steps = []
    total_risks = []
    collision_count = 0

    print(f"🚀 Starting Stress Test: {args.num_tests} episodes, Enemy Jitter = {args.enemy_jitter}")
    
    for i in range(args.num_tests):
        state, _ = env.reset(seed=12345 + i)
        ep_reward = 0
        ep_steps = 0
        ep_risk = 0
        
        for _ in range(env.max_steps):
            # 严格使用贪婪策略 (epsilon=0)
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            ep_steps += 1
            ep_risk += info.get("risk", 0)
            
            # 统计无效动作（撞墙/越界）
            if reward <= -2.0: # 对应你设置的 blocked_move_penalty
                collision_count += 1
                
            state = next_state
            if terminated:
                success_count += 1
                break
            if truncated:
                break
        
        total_rewards.append(ep_reward)
        total_steps.append(ep_steps)
        total_risks.append(ep_risk / ep_steps if ep_steps > 0 else 0)

    # 4. 输出战报
    print("\n" + "="*40)
    print(f"📊 TEST RESULTS FOR V5 MODEL")
    print("="*40)
    print(f"🏆 Success Rate: {success_count}/{args.num_tests} ({success_count/args.num_tests:.2%})")
    print(f"💰 Avg Reward:  {np.mean(total_rewards):.2f}")
    print(f"⏱️  Avg Steps:   {np.mean(total_steps):.1f} steps")
    print(f"⚠️  Avg Risk/Step: {np.mean(total_risks):.4f}")
    print(f"💥 Total Collisions: {collision_count} (across all episodes)")
    print("="*40)

if __name__ == "__main__":
    test_model()