import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import json
import copy
from pathlib import Path
from src.config import load_scenario
from src.environment.risk_grid_env import RiskAwareGridEnv, RewardWeights

# 基础场景路径
base_paths = [
    "src/environment/scenario_1.json",
    "src/environment/scenario_2.json",
    "src/environment/scenario_3.json",
]

# 旋转与镜像

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

# 敌人扰动

def perturb_enemies(scen, offset=3):
    grid = scen["map"]["grid_size"]
    new = copy.deepcopy(scen)
    for e in new["enemies"]:
        ex, ey = e["pos"]
        ex += np.random.randint(-offset, offset+1)
        ey += np.random.randint(-offset, offset+1)
        ex = np.clip(ex, 0, grid-1)
        ey = np.clip(ey, 0, grid-1)
        e["pos"] = [int(ex), int(ey)]
    return new

# 生成3000个场景变体

def generate_variants(num=3000, offset_range=(3,5)):
    base_scenarios = [load_scenario(p) for p in base_paths]
    layouts = []
    for scen in base_scenarios:
        for k in range(4):
            layouts.append(rotate_scenario(scen, k))
            layouts.append(mirror_scenario(rotate_scenario(scen, k)))
    variants = []
    for i in range(num):
        base = layouts[np.random.randint(len(layouts))]
        offset = np.random.randint(offset_range[0], offset_range[1]+1)
        variant = perturb_enemies(base, offset=offset)
        variants.append(variant)
    return variants

# 预计算dijkstra_map

def precompute_dijkstra(variants, save_dir):
    Path(save_dir).mkdir(exist_ok=True)
    for idx, scen in enumerate(variants):
        env = RiskAwareGridEnv(
            scenario=scen,
            observation_size=scen["map"]["grid_size"],
            max_steps=400,
            enemy_jitter=0,
            start_jitter=0,
            reward_weights=RewardWeights(),
            blocked_risk_threshold=0.4,
            seed=42
        )
        env.reset()
        np.save(f"{save_dir}/dijkstra_{idx}.npy", env.dijkstra_map)
        with open(f"{save_dir}/scenario_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(scen, f, ensure_ascii=False, indent=2)
        print(f"Saved scenario_{idx}.json and dijkstra_{idx}.npy")

if __name__ == "__main__":
    variants = generate_variants(num=3000, offset_range=(3,5))
    precompute_dijkstra(variants, save_dir="/tempdisk2/xfh/MDP/precompute/data")
