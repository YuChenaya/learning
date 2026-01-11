# 2.3.7 向量点积
import torch

x = torch.tensor([1,2,3],dtype=torch.float32)
y = torch.rand(3,dtype=torch.float32)

print(x,"  ",y)

print(x*y)
print(sum(x*y))

print(torch.dot(x,y))