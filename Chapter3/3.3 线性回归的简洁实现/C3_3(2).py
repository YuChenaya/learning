import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
from torch.nn import init

# =====================
# 3.3.1. 生成数据集
# =====================
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = torch.matmul(features, torch.tensor(true_w, dtype=torch.float32)) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

# =====================
# 3.3.2. 读取数据
# =====================
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# for X, y in data_iter:
#     print(X, "\n\n", y)
#     break

# =====================
# 3.3.3. 定义模型
# =====================
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
)

# 查看模型
# print(net,"\n")
# print(net[0])

# 查看模型所有的可学习参数
# for param in net.parameters():
#     print(param)

# =====================
# 3.3.4. 初始化模型参数
# =====================
# 原地操作(更简洁)
net[0].weight.data.normal_(0, 0.01)
# init提供接口(需导入)
# init.normal_(net[0].weight, mean=0, std=0.01)
net[0].bias.data.fill_(0)
# init.constant_(net[0].bias, val=0)

# =====================
# 3.3.5. 定义损失函数
# =====================
loss = nn.MSELoss()

# =====================
# 3.3.6. 定义优化算法
# =====================
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
# print(optimizer)

# 调整学习率(动态学习率，可能会丢失动量等状态信息，损失函数收敛震荡)
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍


# =====================
# 3.3.7. 训练模型
# =====================
num_epochs = 3
for epoch in range(0, num_epochs):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch + 1, l.item()))

print(true_w, net[0].weight)
print(true_b, net[0].bias)
