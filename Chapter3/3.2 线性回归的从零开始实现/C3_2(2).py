import torch
import numpy as np
import random

# =====================
# 3.2 线性回归的从零开始实现
# 3.2.1. 生成数据集
# =====================
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = torch.matmul(features, torch.tensor(true_w, dtype=torch.float32)) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)


# print(features[0], "\n", labels[0])

# def use_svg_display():
#     # 用矢量图显示
#     set_matplotlib_formats('svg')
#
# def set_figsize(figsize=(3.5, 2.5)):
#     use_svg_display()
#     # 设置图的尺寸
#     plt.rcParams['figure.figsize'] = figsize

# # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# import sys
# sys.path.append("..")
# from d2lzh_pytorch import *

# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()

# =====================
# 3.2.2. 读取数据
# =====================
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        # yield features.index_select(0, j), labels.index_select(0, j)
        yield features[j], labels[j]


batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, "\n\n" ,y)
#     break

# =====================
# 3.2.3. 初始化模型参数
# =====================
# 使用正态分布随机初始化权重，requires_grad=True表示需要计算梯度
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)),
                 dtype=torch.float32,
                 requires_grad=True)
# 偏置为0，requires_grad=True表示需要计算梯度
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)


# =====================
# 3.2.4. 定义模型
# =====================
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# =====================
# 3.2.5. 定义损失函数
# =====================
def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以2
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# =====================
# 3.2.6. 定义优化函数
# =====================
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


# =====================
# 3.2.7. 训练模型
# =====================
lr = 0.03
num_epochs = 30
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。
    # X和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    if (epoch + 1) % 5 == 0:
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(f"w真实值：{true_w}, \n预测值：{w}" )
print(f"b真实值：{true_b}, \n预测值：{b}" )
