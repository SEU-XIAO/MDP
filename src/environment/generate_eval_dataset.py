import json
import random
import os

class EvalDatasetGenerator:
    def __init__(self, grid_size=64, start_pos=[5, 5], goal_pos=[58, 58]):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos

    def generate_scenario(self, episode_idx, seed):
        """
        生成验证场景。允许敌人位置贴边以增加泛化性测试。
        """
        random.seed(seed)
        
        # 1. 敌人数量随轮次(难度)增长
        num_enemies = random.randint(4, min(12, 6 + (episode_idx // 300)))
        
        # 2. U型障碍出现的概率
        u_prob = min(0.4, 0.15 + episode_idx / 3000)
        
        enemies = []
        for i in range(num_enemies):
            # --- 修改点：位置范围设为 [0, grid_size-1]，允许贴边 ---
            # 为了防止敌人直接刷在起点或终点脸上，这里加一个简单的距离判断
            while True:
                pos = [
                    random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1)
                ]
                # 距离起点和终点至少保持一定距离（如 5 个单位），避免开局即结束
                dist_to_start = ((pos[0]-self.start_pos[0])**2 + (pos[1]-self.start_pos[1])**2)**0.5
                dist_to_goal = ((pos[0]-self.goal_pos[0])**2 + (pos[1]-self.goal_pos[1])**2)**0.5
                if dist_to_start > 8 and dist_to_goal > 8:
                    break
            
            is_u_obstacle = random.random() < u_prob
            
            if is_u_obstacle:
                # U型障碍组件：半径小，100%覆盖
                r = 2.5
                p = 1.0
            else:
                # 随机探测区：半径与概率均动态生成
                r = round(random.uniform(3.5, 8.0), 2)
                p = round(random.uniform(0.7, 0.95), 2)

            enemies.append({
                "id": i + 1,
                "pos": pos,
                "detection_shape": "circle",
                "detection_zones": [{"r": r, "p": p}]
            })

        return {
            "map": {
                "grid_size": self.grid_size,
                "start_pos": self.start_pos,
                "goal_pos": self.goal_pos
            },
            "enemies": enemies
        }

    def create_full_dataset(self, filename="eval_dataset_v2.json"):
        # 分层设计：涵盖从新手期到极限难度的四种配置
        configs = {
            "easy": 0,       # 4-6 敌，少U障
            "normal": 900,   # 6-9 敌
            "hard": 1800,    # 6-12 敌，中等U障
            "extreme": 3000  # 12 敌上限，高概率U障
        }
        
        full_data = {}
        for level, ep_idx in configs.items():
            level_scenarios = []
            for i in range(25): # 每个等级生成 25 个固定场景
                # 使用复合种子确保每个场景唯一且固定
                seed = hash(f"eval_{level}_{i}") % (10**6)
                level_scenarios.append(self.generate_scenario(ep_idx, seed))
            full_data[level] = level_scenarios

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        
        print(f"验证集生成成功！文件：{filename}")
        print(f"结构：4个难度分级，每个分级25个场景，共100个测试用例。")

if __name__ == "__main__":
    generator = EvalDatasetGenerator()
    generator.create_full_dataset()