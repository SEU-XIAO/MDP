from __future__ import annotations

import math
import random
import heapq
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RewardWeights:
    """经过 Dijkstra 逻辑调优后的权重"""
    goal_reward: float = 200.0         # 参数实际由config.py统一管理
    step_penalty: float = -0.5         # 参数实际由config.py统一管理
    risk_lambda: float = 2.5           # 参数实际由config.py统一管理
    guide_omega: float = 1.5           # 参数实际由config.py统一管理
    timeout_penalty: float = -150.0    # 参数实际由config.py统一管理
    blocked_move_penalty: float = -1.0 # 参数实际由config.py统一管理

class RiskAwareGridEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    # 8 连通动作
    ACTIONS = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1),
    ]

    def __init__(
        self,
        scenario: dict[str, Any],
        observation_size: int = 64,
        max_steps: int = 400,
        enemy_jitter: int = 1,
        start_jitter: int = 1,
        reward_weights: RewardWeights | None = None,
        goal_sigma: float = 10.0,
        blocked_risk_threshold: float | None = None,
        seed: int | None = None,
        pregenerated_data: dict | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.world_size = int(scenario["map"]["grid_size"])
        self.start_pos = tuple(scenario["map"]["start_pos"])
        self.goal_pos = tuple(scenario["map"]["goal_pos"])
        self.base_enemies = scenario["enemies"]

        self.observation_size = observation_size
        self.max_steps = max_steps
        self.enemy_jitter = enemy_jitter
        self.start_jitter = start_jitter
        self.goal_sigma = goal_sigma
        self.blocked_risk_threshold = blocked_risk_threshold
        self.reward_weights = reward_weights or RewardWeights()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, observation_size, observation_size),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # DataManager集成
        from src.environment.data_manager import DataManager
        self.data_manager = DataManager(pregenerated_data) if pregenerated_data else None

        self.agent_pos = self.start_pos
        self.current_enemies = self.base_enemies
        self.step_count = 0
        self.dijkstra_map = np.zeros((self.world_size, self.world_size))
        self.obs_buffer = np.zeros((3, self.observation_size, self.observation_size), dtype=np.float32)

    def _compute_dijkstra_map(self) -> np.ndarray:
        """核心改进：向量化计算全图风险，Dijkstra循环查表，极大提速"""
        h, w = self.world_size, self.world_size
        # --- 提速核心：先一次性向量化计算全图风险 ---
        xs = np.arange(h)
        ys = np.arange(w)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        survival = np.ones((h, w), dtype=np.float32)
        for enemy in self.base_enemies:
            ex, ey = enemy["pos"]
            dist_sq = (xx - ex)**2 + (yy - ey)**2
            pk = np.zeros_like(survival)
            for zone in sorted(enemy["detection_zones"], key=lambda z: z["r"]):
                mask = (dist_sq <= zone["r"]**2) & (pk == 0)
                pk[mask] = zone["p"]
            survival *= (1.0 - pk)
        full_risk_grid = 1.0 - survival
        # ------------------------------------------

        dist_map = np.full((h, w), float('inf'), dtype=np.float32)
        gx, gy = self.goal_pos
        dist_map[gx, gy] = 0.0
        pq = [(0.0, gx, gy)]
        risk_weight = 50.0 

        while pq:
            d, x, y = heapq.heappop(pq)
            if d > dist_map[x, y]: continue
            for dx, dy in self.ACTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    move_cost = math.sqrt(dx**2 + dy**2)
                    # 提速点：这里直接查表，不再调用 _risk_at
                    step_risk = full_risk_grid[nx, ny]
                    total_step_cost = move_cost + (step_risk * risk_weight)
                    new_dist = d + total_step_cost
                    if new_dist < dist_map[nx, ny]:
                        dist_map[nx, ny] = new_dist
                        heapq.heappush(pq, (new_dist, nx, ny))
        return dist_map

    def heuristic_action(self, state=None) -> int:
        """
        Dijkstra 引导接口：供 train.py 调用
        寻找邻域内让代价场下降最快的动作
        """
        x, y = self.agent_pos
        best_action = 0
        min_cost = float('inf')

        for i, (dx, dy) in enumerate(self.ACTIONS):
            nx = int(np.clip(x + dx, 0, self.world_size - 1))
            ny = int(np.clip(y + dy, 0, self.world_size - 1))
            
            cost = self.dijkstra_map[nx, ny]
            # 加入微小扰动打破对称性
            cost += self.rng.uniform(0, 0.01)
            
            if cost < min_cost:
                min_cost = cost
                best_action = i
        return best_action

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, scenario: dict[str, Any] | None = None, dijkstra_map: np.ndarray | None = None, pregenerated_data: dict | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        # 只在场景变化时重新计算Dijkstra地图，否则复用
        scenario_changed = False
        if scenario is not None:
            self.scenario = scenario
            self.world_size = int(scenario["map"]["grid_size"])
            self.start_pos = tuple(scenario["map"]["start_pos"])
            self.goal_pos = tuple(scenario["map"]["goal_pos"])
            self.base_enemies = scenario["enemies"]
            scenario_changed = True
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.step_count = 0
        if pregenerated_data is not None:
            # 直接复用预生成数据
            self.current_enemies = pregenerated_data.get('enemies', self.base_enemies)
            self.agent_pos = tuple(pregenerated_data.get('agent_pos', self.start_pos))
        else:
            self.current_enemies = self._randomize_enemies(self.base_enemies)
            self.agent_pos = self._randomize_start(self.start_pos)

        # 优先使用DataManager的查表与切片
        if pregenerated_data is not None:
            from src.environment.data_manager import DataManager
            self.data_manager = DataManager(pregenerated_data)
            self.dijkstra_map = self.data_manager.get_dijkstra_map()
            self.full_risk_grid = self.data_manager.get_full_risk_grid()
            self.coord_map = self.data_manager.coord_map
            self.obs_buffer[0:2] = self.data_manager.get_obs_layers()
        elif dijkstra_map is not None:
            self.dijkstra_map = dijkstra_map
        elif scenario_changed:
            self.dijkstra_map = self._compute_dijkstra_map()

        # 初始化agent_map
        self.obs_buffer[2] = 0
        ax, ay = self._world_to_obs(self.agent_pos)
        self.obs_buffer[2, ay, ax] = 1.0

        obs = self.obs_buffer.copy()
        info = {"risk": self._risk_at(self.agent_pos), "cost": self.dijkstra_map[self.agent_pos]}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        prev_pos = self.agent_pos
        prev_cost = self.dijkstra_map[prev_pos[0], prev_pos[1]]

        dx, dy = self.ACTIONS[action]
        nx = int(np.clip(prev_pos[0] + dx, 0, self.world_size - 1))
        ny = int(np.clip(prev_pos[1] + dy, 0, self.world_size - 1))

        blocked = self._is_blocked_by_risk((nx, ny))
        if blocked:
            self.agent_pos = prev_pos
        else:
            self.agent_pos = (nx, ny)

        curr_cost = self.dijkstra_map[self.agent_pos[0], self.agent_pos[1]]
        risk_prob = self._risk_at(self.agent_pos)

        # 奖励计算逻辑
        r_goal = self.reward_weights.goal_reward if self.agent_pos == self.goal_pos else 0.0
        r_step = self.reward_weights.step_penalty
        r_risk = -self.reward_weights.risk_lambda * (risk_prob**2)
        r_guide = self.reward_weights.guide_omega * (prev_cost - curr_cost)
        r_blocked = self.reward_weights.blocked_move_penalty if blocked else 0.0
        r_timeout = self.reward_weights.timeout_penalty if (self.step_count >= self.max_steps and self.agent_pos != self.goal_pos) else 0.0

        reward = r_goal + r_step + r_risk + r_guide + r_timeout + r_blocked

        terminated = self.agent_pos == self.goal_pos
        truncated = self.step_count >= self.max_steps

        # 只更新agent_map
        self.obs_buffer[2] = 0
        ax, ay = self._world_to_obs(self.agent_pos)
        self.obs_buffer[2, ay, ax] = 1.0

        obs = self.obs_buffer.copy()
        info = {
            "risk": risk_prob,
            "dijkstra_cost": curr_cost,
            "reward_breakdown": {
                "goal": r_goal, "step": r_step, "risk": r_risk, "guide": r_guide, "blocked": r_blocked
            },
        }
        return obs, float(reward), terminated, truncated, info

    def _risk_at(self, pos: tuple[int, int]) -> float:
        # 查表，彻底消除遍历
        if hasattr(self, 'full_risk_grid'):
            return float(self.full_risk_grid[pos[0], pos[1]])
        elif hasattr(self, 'data_manager') and self.data_manager is not None:
            return self.data_manager.get_risk(pos)
        else:
            # 兼容旧逻辑
            px, py = pos
            survival = 1.0
            for enemy in self.current_enemies:
                ex, ey = enemy["pos"]
                dist = math.dist((px, py), (ex, ey))
                pk = 0.0
                for zone in sorted(enemy["detection_zones"], key=lambda z: z["r"]):
                    if dist <= zone["r"]:
                        pk = float(zone["p"])
                        break
                survival *= 1.0 - pk
            return float(1.0 - survival)

    def _is_blocked_by_risk(self, pos: tuple[int, int]) -> bool:
        if self.blocked_risk_threshold is None: return False
        return self._risk_at(pos) >= float(self.blocked_risk_threshold)

    def _randomize_enemies(self, enemies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"pos": [int(np.clip(e["pos"][0] + self.rng.randint(-self.enemy_jitter, self.enemy_jitter), 0, self.world_size-1)),
                         int(np.clip(e["pos"][1] + self.rng.randint(-self.enemy_jitter, self.enemy_jitter), 0, self.world_size-1))],
                 "detection_zones": e["detection_zones"]} for e in enemies]

    def _randomize_start(self, start_pos: tuple[int, int]) -> tuple[int, int]:
        nx = int(np.clip(start_pos[0] + self.rng.randint(-self.start_jitter, self.start_jitter), 0, self.world_size-1))
        ny = int(np.clip(start_pos[1] + self.rng.randint(-self.start_jitter, self.start_jitter), 0, self.world_size-1))
        return nx, ny

    def _world_to_obs(self, pos: tuple[int, int]) -> tuple[int, int]:
        # 查表，彻底消除浮点运算
        if hasattr(self, 'coord_map'):
            ox = self.coord_map[pos[0]]
            oy = self.coord_map[pos[1]]
            return ox, oy
        elif hasattr(self, 'data_manager') and self.data_manager is not None:
            return self.data_manager.get_coord(pos)
        else:
            ox = int(round(pos[0] * (self.observation_size - 1) / (self.world_size - 1)))
            oy = int(round(pos[1] * (self.observation_size - 1) / (self.world_size - 1)))
            return ox, oy

    def _build_observation(self, agent_pos: tuple[int, int]) -> np.ndarray:
        # 兼容旧逻辑，实际已用obs_buffer
        return self.obs_buffer.copy()