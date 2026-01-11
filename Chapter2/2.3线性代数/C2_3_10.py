# 2.3.10 范数
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=torch.float32)
x = torch.tensor([-3,4],dtype=torch.float32)

print(torch.norm(x))
print(torch.abs(x).sum())

print(torch.norm(A))
