import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------
# 1. 定义 CNN 网络
# -------------------------

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for MNIST classification.

    Architecture:
    - 2 Convolutional layers with ReLU and Batch Normalization.
    - 2 Max Pooling layers.
    - 2 Fully Connected (FC) layers.
    - Regularization:
        - 2 Dropout.
        - 2 Batch Normalization.
    - Note: No explicit Softmax (applies by nn.CrossEntropyLoss).
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # 卷积层1: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)

        # 卷积层2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)

        # 池化
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout
        self.dropout1 = nn.Dropout(0.45)
        self.dropout2 = nn.Dropout(0.35)

    def forward(self, x):
        # Conv1 -> BN -> ReLU -> Pool
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))

        # Conv2 -> BN -> ReLU -> Pool
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        # 展平
        x = x.view(x.size(0), -1)

        # FC1 -> ReLU -> Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)

        # FC2
        x = self.fc2(x)

        return x


# -------------------------
# 2. 训练函数
# -------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------
# 3. 验证 / 测试函数
# -------------------------

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


# -------------------------
# 4. 主函数
# -------------------------

def main():

    # 超参数
    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST 数据集
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 初始化模型
    model = SimpleCNN().to(device)

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -------------------------
    # 训练循环
    # -------------------------

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc:.2f}%"
        )

    print("Training Finished!")


# -------------------------
# 5. 入口
# -------------------------

if __name__ == "__main__":
    main()
