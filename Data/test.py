import torch

# 假设 original_tensor 是你的原始张量，形状为 [20, 2, 1]
original_tensor = torch.randn(20, 2, 1,1)

# 使用 repeat 方法正确地重复张量
expanded_tensor = original_tensor.repeat(1, 1, 300, 1)

# 检查新张量的形状
print(expanded_tensor.shape)  # 应该输出 torch.Size([20, 2, 300, 1])
