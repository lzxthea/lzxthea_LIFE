"""
PyTorch AMP 混合精度训练示例
本demo展示如何使用自动混合精度（Automatic Mixed Precision）来减少显存占用和加速训练
AMP尤其适用于GPU显存不够时使用，通过float16计算减少显存消耗
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

print("=== PyTorch AMP 混合精度训练 ===")

# ============ 超参数配置 ============
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
INPUT_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 10
NUM_SAMPLES = 1000

# ============ 1. 检查设备并配置 ============
print("\n=== 步骤1: 检查设备 ===")
# 检查 CUDA 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()

print(f"CUDA 可用: {use_cuda}")
if use_cuda:
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("将使用 CPU 运行（AMP自动回退到FP32）")

print(f"设备类型: {device}")

# ============ 2. 构造 Toy 数据集 ============
print("\n=== 步骤2: 构造数据集 ===")
X = torch.randn(NUM_SAMPLES, INPUT_DIM)
y = torch.randint(0, OUTPUT_DIM, (NUM_SAMPLES,))

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"数据集大小: {NUM_SAMPLES}, batch_size: {BATCH_SIZE}")
print(f"每个epoch的batch数: {len(dataloader)}")

# ============ 3. 定义简单模型 ============
print("\n=== 步骤3: 定义模型 ===")

class SimpleModel(nn.Module):
    """简单的 MLP 模型用于演示"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
print(f"模型结构:\n{model}")

# ============ 4. 损失函数和优化器 ============
print("\n=== 步骤4: 损失函数和优化器 ===")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"损失函数: CrossEntropyLoss")
print(f"优化器: Adam, lr={LEARNING_RATE}")

# ============ 5. AMP 训练核心代码 ============
print("\n=== 步骤5: AMP 训练配置 ===")

# 创建 GradScaler，用于缩放loss和梯度，防止float16下溢
# 注意：torch.amp.GradScaler 是 torch 内置的，不需要额外安装
scaler = torch.amp.GradScaler('cuda') if use_cuda else None

print(f"是否使用 AMP GradScaler: {scaler is not None}")
if use_cuda:
    print("AMP 模式: 使用 float16 计算，梯度自动缩放防止下溢")
else:
    print("CPU 模式: AMP 自动回退到 float32（FP32）")

# ============ 6. 训练循环（带 AMP） ============
print("\n=== 步骤6: 开始训练 ===")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # ===== AMP 核心代码段 =====
        # 1. autocast：自动将计算切换到 float16（如果可用）
        # 2. loss 计算在 autocast 上下文中
        # 3. scaler.scale：对 loss 进行缩放，防止 float16 下溢
        # 4. scaler.step：先unscale梯度再更新参数
        # 5. scaler.update：更新 scaler 的 scale factor

        if use_cuda:
            # GPU 模式：使用真正的 AMP
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # scaled loss backward
            scaler.scale(loss).backward()

            # unscale gradients + clip + parameter update
            scaler.step(optimizer)

            # update scale factor for next iteration
            scaler.update()
        else:
            # CPU 模式：自动回退到普通 FP32
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        # ===== AMP 核心代码段结束 =====

        # 统计信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        if (batch_idx + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss: {running_loss/5:.4f}, Acc: {100*correct/total:.2f}%")
            running_loss = 0.0
            correct = 0
            total = 0

# ============ 7. 验证 AMP 效果 ============
print("\n=== 步骤7: 验证训练结果 ===")
model.eval()
with torch.no_grad():
    # 取一个batch验证
    val_inputs, val_targets = next(iter(dataloader))
    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

    if use_cuda:
        with torch.amp.autocast('cuda'):
            val_outputs = model(val_inputs)
    else:
        val_outputs = model(val_inputs)

    _, val_predicted = torch.max(val_outputs.data, 1)
    val_acc = (val_predicted == val_targets).sum().item() / val_targets.size(0)

print(f"验证集 Accuracy: {val_acc*100:.2f}%")

# ============ 8. 对比普通模式和 AMP 模式的显存使用 ============
print("\n=== 步骤8: 显存使用对比 ===")
if use_cuda:
    print(f"当前 GPU 显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU 显存峰值占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print("\nAMP 优势:")
    print("  - 减少显存占用约 30-50%")
    print("  - 加速训练 1.5-3x（视具体模型和GPU而定）")
    print("  - 适用于显存不够的大 batch 训练场景")
else:
    print("当前运行在 CPU 模式，无法展示显存对比")
    print("如果有 GPU 可用，AMP 可以显著减少显存占用")

# ============ 9. AMP 关键点总结 ============
print("\n=== AMP 关键点总结 ===")
print("""
1. torch.amp.autocast('cuda'):
   - 自动将前向传播中的计算切换到 float16
   - loss 计算在 autocast 上下文中完成

2. torch.amp.GradScaler('cuda'):
   - scale loss 后 backward，防止 float16 下溢（梯度太小）
   - unscale 梯度后再 step 更新参数
   - 自动调整 scale factor

3. 适用场景：
   - GPU 显存不够时（推荐系统大模型、LLM等）
   - 想要加速训练时
   - 混合精度训练已成为深度学习标配

4. 注意事项：
   - 不是所有操作都支持 float16，某些操作会自动回退到 FP32
   - loss scaling 对收敛影响不大，但需要正确使用 scaler
""")

print("\n=== Demo 完成 ===")
