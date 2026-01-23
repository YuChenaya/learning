import torch
from torch import nn
from torch.nn import init
import d22l as d2l
from collections import OrderedDict

# =====================
# 3.7.1. 读取数据
# =====================
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# =====================
# 3.7.2. 定义和初始化模型
# =====================
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y


net = LinearNet(num_inputs, num_outputs)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# =====================
# 3.7.3. softmax和交叉熵损失函数
# =====================
loss = nn.CrossEntropyLoss()

# =====================
# 3.7.4. 定义优化算法
# =====================
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# =====================
# 3.7.5. 训练模型
# =====================
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
