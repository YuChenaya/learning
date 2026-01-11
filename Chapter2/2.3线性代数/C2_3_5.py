# 2.3.5 张量算法的基本性质
import torch

x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(x)

y = x.clone()
print(x+y)
print(x*y)