import torch

# =====================
# 2.3 线性代数
# 2.3.1 标量
# =====================
a = torch.tensor(3.0)
b = torch.tensor(2.0)

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** b)

# =====================
# 2.3.2 向量
# =====================
x = torch.arange(4)
print(x)

# 访问特定元素
print(x[2])

# 获取向量长度
print(len(x))

# 向量维度
print(x.shape)

# =====================
# 2.3.3 矩阵
# =====================
x = torch.arange(20).reshape(4, 5)
print(x, "\n")

# 转置
print(x.T)

# 判断是否相等
y = torch.ones(4, 4)
print(y == y.T)

# =====================
# 2.3.4 张量
# =====================
x = torch.arange(30).reshape(2, 3, 5)
print(x)

# =====================
# 2.3.5 张量算法的基本性质
# =====================

x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(x)

y = x.clone()
print(x + y)
print(x * y)

# =====================
# 2.3.6 降维
# =====================

x = torch.arange(4, dtype=torch.float32)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)

print(x)
print(x.sum(), "\n")

print(A)
# 按axis=1求和
print(A.sum(axis=0))
# 按axis=0求和
print(A.sum(axis=1), "\n")

# 平均值
print(A.mean())
print(A.sum() / A.numel(), "\n")

# 按axis=1求和
print(A.sum(axis=0, keepdim=True))
# 按axis=0求和
print(A.sum(axis=1, keepdim=True))
# 广播机制
print(A / A.sum(axis=1, keepdim=True))

print(A.cumsum(axis=0))

# =====================
# 2.3.7 向量点积
# =====================

x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.rand(3, dtype=torch.float32)

print(x, "  ", y)

print(x * y)
print(sum(x * y))

print(torch.dot(x, y))

# =====================
# 2.3.8 矩阵-向量积
# =====================

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

print(A)
print(A.shape)

print(x)
print(x.shape)

print(torch.mv(A, x))

# =====================
# 2.3.9 矩阵-矩阵乘法
# =====================

A = torch.arange(20).reshape(5, 4)
B = torch.arange(20).reshape(4, 5)

print(A)
print(B)

print(torch.mm(A, B))
print(torch.mm(A, B).shape)

# =====================
# 2.3.10 范数
# =====================

A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
x = torch.tensor([-3, 4], dtype=torch.float32)

print(torch.norm(x))
print(torch.abs(x).sum())

print(torch.norm(A))
