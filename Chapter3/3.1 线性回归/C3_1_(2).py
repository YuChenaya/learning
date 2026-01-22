# =====================
# 3.1 线性回归
# 3.1.2 线性回归的表示方法
# 3.1.2.2 矢量计算表达式
# =====================

import torch
from time import time

num = 1000000
a = torch.ones(num)
b = torch.ones(num)

start1 = time()
c = torch.zeros(num)
for i in range(num):
    c[i] = a[i] + b[i]
runtime1 = time() - start1
print(f"{runtime1:.8f}")

d = torch.zeros(num)
start2 = time()
d = a + b
runtime2 = time() - start2
print(f"{runtime2:.8f}")

print(f"两种方法时间差为：{runtime2 - runtime1:.8f}")
