import torch
from torch.utils import data
from torch import nn


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


def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个PyTorch数据迭代器
    Params:
        data_arrays: 包含特征和标签的元组
        batch_size: 批量大小
        is_train: 是否为训练模式，如果是，则打乱数据顺序
    Returns:
        DataLoader对象，可用于迭代数据
    """
    # 将数据包装成TensorDataset对象
    dataset = data.TensorDataset(*data_arrays)
    # 创建DataLoader对象，用于批量加载数据
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 设置真实的权重和偏置
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成合成数据集
features, labels = synthetic_data(true_w, true_b, 1000)

# 设置批量大小
batch_size = 10
# 创建数据迭代器，用于批量读取数据
data_iter = load_array((features, labels), batch_size)

# 使用nn.Sequential构建神经网络模型
# nn.Linear(2, 1)表示一个线性层，输入特征数为2，输出特征数为1
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
# 使用正态分布初始化权重，均值为0，标准差为0.01
net[0].weight.data.normal_(0, 0.01)
# 将偏置初始化为0
net[0].bias.data.fill_(0)

# 定义损失函数
# nn.MSELoss()表示均方误差损失函数
loss = nn.MSELoss()

# 定义优化器
# 使用随机梯度下降(SGD)优化器，学习率为0.03
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 设置训练轮数
num_epochs = 300
# 开始训练模型
for epoch in range(num_epochs):
    # 每个epoch遍历整个数据集一次
    for X, y in data_iter:
        # 计算当前批量的损失
        l = loss(net(X), y)
        # 清空梯度
        trainer.zero_grad()
        # 反向传播计算梯度
        l.backward()
        # 更新模型参数
        trainer.step()
    # 每个epoch结束后，计算在整个数据集上的损失
    l = loss(net(features), labels)
    # 打印当前epoch的损失
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)