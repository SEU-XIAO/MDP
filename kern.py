import torch
import matplotlib.pyplot as plt
import os
from src.models.d3qn import DuelingDQN

# 1. 解决服务器中文字体报错
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 统统改用英文显示
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
model = DuelingDQN(in_channels=3, num_actions=8)
MODEL_PATH = 'checkpoints/v2/best_model.pt'

if os.path.exists(MODEL_PATH):
    # 根据你的 D3QNAgent 加载逻辑，如果是从整个 Agent 加载，注意 key 的对应
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    # 如果保存的是整个 state_dict，可能需要提取其中的 model 部分
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Successfully loaded model from {MODEL_PATH}")

# 获取第一层卷积核权重 (out_channels, in_channels, h, w)
# 假设是 (32, 3, 3, 3)
weights = model.feature_extractor[0].weight.data.cpu().numpy()

# 我们选取前 8 个 Filter，展示它们分别对 3 个通道的敏感度
num_filters = 8
fig, axes = plt.subplots(num_filters, 4, figsize=(12, 16))

for i in range(num_filters):
    filt = weights[i] # shape (3, 3, 3)
    
    for j in range(3):
        ax = axes[i, j]
        # 单通道展示
        im = ax.imshow(filt[j], cmap='RdBu', interpolation='nearest')
        ax.axis('off')
        if i == 0:
            channel_names = ["Ch1: Pos", "Ch2: Risk", "Ch3: Aux"]
            ax.set_title(channel_names[j])
        if j == 0:
            ax.set_ylabel(f'Filter {i}', rotation=0, labelpad=40, size='large')

    # 第 4 列展示该 Filter 的综合热力图（绝对值累加）
    ax_sum = axes[i, 3]
    combined = sum([abs(filt[k]) for k in range(3)])
    ax_sum.imshow(combined, cmap='viridis')
    ax_sum.axis('off')
    if i == 0:
        ax_sum.set_title("Combined Activity")

plt.suptitle('Conv1 Weights Analysis (Per Channel)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 建议保存为文件，因为 K8s 环境弹出 GUI 窗口可能不稳定
plt.savefig('conv_weights_analysis.png')
print("Analysis image saved as 'conv_weights_analysis.png'")
plt.show()