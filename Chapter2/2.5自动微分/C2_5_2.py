# 2.5.2 非标量变量的反向传播
import torch

x = torch.arange(4.0, requires_grad=True)
print(x)

y = x * x

# y.sum().backward()
# y.backward(torch.ones(len(x)))
y.backward(torch.ones_like(y))
print(x.grad)
