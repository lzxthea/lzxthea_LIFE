"""
PyTorch 基础训练循环示例 - 图像分类 toy dataset
本demo展示一个完整的训练循环：前向传播、损失计算、反向传播、参数更新
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# ============ 超参数配置 ============
BATCH_SIZE = 32          # 批量大小
LEARNING_RATE = 0.001    # 学习率
EPOCHS = 10              # 训练轮数
NUM_CLASSES = 10         # 类别数量（0-9）
INPUT_CHANNELS = 1       # 输入通道数（灰度图）
IMAGE_SIZE = 8           # 图像尺寸 8x8
HIDDEN_DIM = 64          # 隐藏层维度
NUM_SAMPLES = 1000       # 样本总数
VAL_SPLIT = 0.2          # 验证集比例
RANDOM_SEED = 42         # 随机种子，保证可复现

# 设置随机种子
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=== PyTorch 基础训练循环 ===")
print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print(f"超参数: batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, epochs={EPOCHS}")

# ============ 1. 构造伪图像数据 ============
print("\n=== 步骤1: 构造数据集 ===")
# X: (1000, 1, 8, 8) 的伪图像数据，y: 10类分类标签 (0-9)
X = torch.randn(NUM_SAMPLES, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))

print(f"X shape: {X.shape}, dtype: {X.dtype}")
print(f"y shape: {y.shape}, dtype: {y.dtype}")
print(f"标签分布: {torch.unique(y, return_counts=True)}")

# 划分训练集和验证集
split_idx = int(NUM_SAMPLES * (1 - VAL_SPLIT))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# 封装成 Dataset 和 DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

# ============ 2. 构建模型 ============
print("\n=== 步骤2: 构建 CNN 模型 ===")
# 使用 nn.Sequential 构建简单 CNN
# Conv2d → ReLU → MaxPool2d → Flatten → Linear → ReLU → Linear → Softmax
model = nn.Sequential(
    nn.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, padding=1),  # (B,1,8,8) -> (B,16,8,8)
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),                      # (B,16,8,8) -> (B,16,4,4)
    nn.Flatten(),                                                # (B,16,4,4) -> (B,256)
    nn.Linear(16 * 4 * 4, HIDDEN_DIM),                          # (B,256) -> (B,64)
    nn.ReLU(),
    nn.Linear(HIDDEN_DIM, NUM_CLASSES),                          # (B,64) -> (B,10)
    nn.Softmax(dim=1)                                           # 输出概率分布
)

print("模型结构:")
print(model)

# ============ 3. 损失函数和优化器 ============
print("\n=== 步骤3: 损失函数和优化器 ===")
criterion = nn.CrossEntropyLoss()  # 适用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch减半学习率

print(f"损失函数: CrossEntropyLoss")
print(f"优化器: Adam, 初始学习率: {LEARNING_RATE}")
print(f"学习率调度器: StepLR, step_size=5, gamma=0.5")

# ============ 4. 训练循环 ============
print("\n=== 步骤4: 开始训练 ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

global_step = 0

for epoch in range(EPOCHS):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 将数据移到设备上
        inputs, targets = inputs.to(device), targets.to(device)

        # ===== 训练步骤 =====
        # 1. 清零梯度（防止梯度累积）
        optimizer.zero_grad()

        # 2. 前向传播
        outputs = model(inputs)

        # 3. 计算损失
        loss = criterion(outputs, targets)

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        # ===== 统计信息 =====
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        global_step += 1

        # 每 20 个 step 打印一次
        if global_step % 20 == 0:
            avg_loss = running_loss / 20
            accuracy = 100. * correct / total
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{global_step}], "
                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            running_loss = 0.0
            correct = 0
            total = 0

    # 每个 epoch 结束后更新学习率
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f">>> Epoch {epoch+1} 完成, 当前学习率: {current_lr:.6f}")

# ============ 5. 验证集评估 ============
print("\n=== 步骤5: 验证集评估 ===")
model.eval()  # 设置为评估模式
val_loss = 0.0
val_correct = 0
val_total = 0

with torch.no_grad():  # 评估时不计算梯度
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        val_total += targets.size(0)
        val_correct += (predicted == targets).sum().item()

avg_val_loss = val_loss / len(val_loader)
val_accuracy = 100. * val_correct / val_total
print(f"验证集 Loss: {avg_val_loss:.4f}")
print(f"验证集 Accuracy: {val_accuracy:.2f}%")

print("\n=== 训练完成 ===")
