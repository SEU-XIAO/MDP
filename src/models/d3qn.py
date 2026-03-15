from __future__ import annotations

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        
        # 1. 特征提取器 (Feature Extractor)
        # 针对 64x64 输入进行设计，旨在平衡局部细节与宏观感受野
        self.feature_extractor = nn.Sequential(
            # 第一层：保持 64x64 原尺寸提取局部细节
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), # 增加 BN 层提升 4090 大 Batch 训练下的稳定性
            nn.ReLU(inplace=True),
            
            # 第二层：下采样到 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第三层：下采样到 16x16
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第四层：保持 16x16，加深语义理解（识别障碍物群拓扑）
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 自动计算展平后的维度 (128 * 16 * 16 = 32768)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            out = self.feature_extractor(dummy)
            self.feat_dim = out.view(1, -1).size(1)

        # 2. 状态价值流 (State Value Stream)
        # 用于判断当前位置的整体安全性和价值
        self.value_stream = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),  # 轻微 Dropout 缓解 4090 高速拟合时的过拟合
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        
        # 3. 优势动作流 (Action Advantage Stream)
        # 用于判断在当前位置采取各个动作的相对优劣
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 归一化检查：如果输入是 0-255，这里需要手动归一化
        if x.max() > 1.0:
            x = x / 255.0

        # 1. 提取卷积特征
        feat = self.feature_extractor(x)
        feat = feat.flatten(start_dim=1)

        # 2. 计算 V 和 A
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        
        # 3. 结合为 Q 值 (中心化处理)
        # Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        # 这确保了优势函数的均值为 0，增强了训练的辨识度
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q