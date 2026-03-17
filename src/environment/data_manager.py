import numpy as np

class DataManager:
    """
    高效管理预生成场景、dijkstra_map、risk_map、goal_map、obs_layers等，支持快速切片与查表。
    """
    def __init__(self, pregenerated_data: dict):
        self.dijkstra_map = pregenerated_data['dijkstra_map']
        self.risk_map = pregenerated_data['risk_map']
        self.goal_map = pregenerated_data['goal_map']
        self.obs_layers = pregenerated_data['obs_layers']
        self.full_risk_grid = pregenerated_data['full_risk_grid']
        self.coord_map = pregenerated_data['coord_map']

    def get_obs_layers(self):
        return self.obs_layers

    def get_full_risk_grid(self):
        return self.full_risk_grid

    def get_coord(self, world_pos):
        return self.coord_map[world_pos[0]], self.coord_map[world_pos[1]]

    def get_dijkstra_map(self):
        return self.dijkstra_map

    def get_risk(self, pos):
        return float(self.full_risk_grid[pos[0], pos[1]])

    def get_goal_map(self):
        return self.goal_map

    def get_risk_map(self):
        return self.risk_map
