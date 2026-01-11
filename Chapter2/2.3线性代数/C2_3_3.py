# 2.3.2 矩阵
import torch

x = torch.arange(20).reshape(4, 5)
print(x, "\n")

# 转置
print(x.T)

# 判断是否相等
y = torch.ones(4,4)
print(y == y.T)
