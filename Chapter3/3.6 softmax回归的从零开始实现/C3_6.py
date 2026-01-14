import torch
from IPython import display
from d2l import torch as d2l

# =====================
# 1. 数据加载
# =====================
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# =====================
# 2. 模型参数
# =====================
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, (num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# =====================
# 3. 模型定义
# =====================
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(X.reshape((-1, num_inputs)) @ W + b)

# =====================
# 4. 损失函数
# =====================
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

# =====================
# 5. 准确率计算
# =====================
def accuracy(y_hat, y):
    if y_hat.ndim > 1:
        y_hat = y_hat.argmax(dim=1)
    return float((y_hat == y).sum())

# =====================
# 6. 累加器
# =====================
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]

# =====================
# 7. 评估函数
# =====================
def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# =====================
# 8. 训练一个 epoch
# =====================
def train_epoch_ch3(net, train_iter, loss, updater):
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        l.sum().backward()
        updater(X.shape[0])
        metric.add(l.sum(), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

# =====================
# 9. 动画器
# =====================
class Animator:
    def __init__(self, xlabel, legend, xlim, ylim):
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots()
        self.xlabel = xlabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
        self.X, self.Y = None, None

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        if not hasattr(x, "__len__"):
            x = [x] * len(y)

        if self.X is None:
            self.X = [[] for _ in y]
            self.Y = [[] for _ in y]

        for i, (a, b) in enumerate(zip(x, y)):
            self.X[i].append(a)
            self.Y[i].append(b)

        self.axes.cla()
        for x, y in zip(self.X, self.Y):
            self.axes.plot(x, y)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_xlim(self.xlim)
        self.axes.set_ylim(self.ylim)
        self.axes.legend(self.legend)
        display.display(self.fig)
        display.clear_output(wait=True)

# =====================
# 10. 训练主函数
# =====================
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(
        xlabel='epoch',
        legend=['train loss', 'train acc', 'test acc'],
        xlim=[1, num_epochs],
        ylim=[0.3, 1.0]
    )

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))

# =====================
# 11. 参数更新函数
# =====================
lr = 0.1
def updater(batch_size):
    d2l.sgd([W, b], lr, batch_size)

# =====================
# 12. 开始训练
# =====================
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# =====================
# 13. 预测可视化
# =====================
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1))
    titles = [t + '\n' + p for t, p in zip(trues, preds)]
    d2l.show_images(X[:n].reshape((n, 28, 28)), 1, n, titles=titles[:n])

predict_ch3(net, test_iter)
