# 2.3.9 矩阵-矩阵乘法
import torch

A = torch.arange(20).reshape(5,4)
B = torch.arange(20).reshape(4,5)

print(A)
print(B)

print(torch.mm(A,B))
print(torch.mm(A,B).shape)