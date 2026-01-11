# 2.3.6 降维
import torch

x = torch.arange(4, dtype=torch.float32)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)

print(x)
print(x.sum(),"\n")

print(A)
# 按axis=1求和
print(A.sum(axis=0))
# 按axis=0求和
print(A.sum(axis=1),"\n")

# 平均值
print(A.mean())
print(A.sum() / A.numel(),"\n")

# 按axis=1求和
print(A.sum(axis=0,keepdim=True))
# 按axis=0求和
print(A.sum(axis=1,keepdim=True))
# 广播机制
print(A / A.sum(axis=1,keepdim=True))

print(A.cumsum(axis=0))