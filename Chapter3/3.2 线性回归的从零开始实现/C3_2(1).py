import random
import torch


def synthetic_data(w, b, num_examples):
    """
    生成线性回归的合成数据集
    Params:
        w: 权重向量
        b: 偏置项
        num_examples: 样本数量
    Returns:
        X: 特征矩阵，形状为(num_examples, len(w))
        y: 标签向量，形状为(num_examples, 1)
    """
    # 生成均值为0，标准差为1的正态分布随机特征矩阵
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算线性关系 y = Xw + b
    y = torch.matmul(X, w) + b
    # 添加均值为0，标准差为0.01的噪声
    y += torch.normal(0, 0.01, y.shape)
    # 将y重塑为列向量
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """
    小批量随机采样数据迭代器
    Params:
        batch_size: 每个批量的样本数
        features: 特征矩阵
        labels: 标签向量
    Returns:
        每次迭代返回一个批量的特征和标签
    """
    num_examples = len(features)
    # 创建索引列表
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    # 按批量大小迭代数据
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # 返回当前批量的特征和标签
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """
    线性回归模型
    Params:
        X: 特征矩阵
        w: 权重向量
        b: 偏置项
    Returns:
        线性回归预测值
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    均方损失函数
    Params:
        y_hat: 预测值
        y: 真实值
    Returns:
        均方损失
    """
    # 将y重塑为与y_hat相同的形状，然后计算均方误差
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降
    Params:
        params: 需要更新的参数列表
        lr: 学习率
        batch_size: 批量大小
    """
    with torch.no_grad():  # 在不计算梯度的情况下更新参数
        for param in params:
            # 参数更新公式: param = param - lr * gradient / batch_size
            param -= lr * param.grad / batch_size
            # 将梯度清零，为下一次迭代做准备
            param.grad.zero_()


# 设置真实的权重和偏置
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成合成数据集
features, labels = synthetic_data(true_w, true_b, 1000)

# 设置批量大小
batch_size = 10
# 打印第一个批量的数据，查看数据格式
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# 初始化模型参数
# 使用正态分布随机初始化权重，requires_grad=True表示需要计算梯度
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
# 偏置初始化为0
b = torch.zeros(1, requires_grad=True)

# 设置超参数
lr = 0.03  # 学习率
num_epochs = 3  # 训练轮数
net = linreg  # 线性模型
loss = squared_loss  # 损失函数

# 训练模型
for epoch in range(num_epochs):
    # 在每个epoch中，对整个数据集进行一次迭代
    for X, y in data_iter(batch_size, features, labels):
        # 计算当前批量的损失
        l = loss(net(X, w, b), y)
        # 计算梯度
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        # 使用参数的梯度更新参数
        sgd([w, b], lr, batch_size)
    # 每个epoch结束后，计算在整个数据集上的损失并打印
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 打印估计的参数与真实参数之间的误差
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
