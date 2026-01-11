# 2.3.8 矩阵-向量积
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)


print(A)
print(A.shape)

print(x)
print(x.shape)

print(torch.mv(A, x))
