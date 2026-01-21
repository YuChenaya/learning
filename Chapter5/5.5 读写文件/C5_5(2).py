import torch
from torch import nn

# =====================
# 5.5 读取和存储
# 5.5.1 读写Tensor
# =====================
x = torch.ones(3)
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
# print(x2)

y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
# print(xy_list)

torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
# print(xy)


# =====================
# 5.5.2 读写模型
# =====================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
print(net.state_dict())

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())
