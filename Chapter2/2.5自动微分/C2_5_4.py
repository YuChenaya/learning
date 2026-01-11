# 2.5.4 Python控制流的梯度计算
import torch


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b
            return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)
