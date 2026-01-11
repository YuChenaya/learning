# 2.5.3 分离计算
import torch

x = torch.arange(4.0, requires_grad=True)
print(x)

y = x * x
u = y.detach()

z = u * x
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)