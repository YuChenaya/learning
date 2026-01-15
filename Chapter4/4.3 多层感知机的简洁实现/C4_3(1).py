import torch
from torch import nn
import d2l

# =====================
# 定义模型
# =====================
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))


def init_weights(m):
    """
    初始化模型权重的函数
    参数:
        m: 模型的层，这里特指nn.Linear层
    功能:
        使用正态分布(均值为0，标准差为0.01)来初始化线性层的权重
    """
    if type(m) == nn.Linear:
        # 判断当前层是否为全连接层(nn.Linear)
        # 如果是，则使用正态分布初始化其权重
        nn.init.normal_(m.weight, std=0.01)


# 将这个初始化函数应用到模型的所有层上
net.apply(init_weights)

# =====================
# 定义超参数
# =====================
batch_size, lr, num_epochs = 256, 0.1, 5

# =====================
# 定义损失函数
# =====================
loss = nn.CrossEntropyLoss(reduction='none')

# =====================
# 定义优化器
# =====================
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# =====================
# 训练模型
# =====================
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
