"""
PyTorch DDP 分布式训练示例 - 单机模式对照
本demo展示普通训练和DDP（DistributedDataParallel）训练的对比
DDP是工业级大规模训练的基础，需要理解其核心原理
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np

print("=== PyTorch DDP 分布式训练 ===")

# ============ 超参数配置 ============
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 3
INPUT_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 10
NUM_SAMPLES = 200

# ============ 1. 普通训练版本（单机单卡基准） ============
print("\n=== 普通训练版本（单机单卡基准）===")

class SimpleModel(nn.Module):
    """简单MLP模型"""
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

def train_normal():
    """普通单进程训练（Baseline）"""
    print("\n--- 普通训练模式 ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 构造数据
    X = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, OUTPUT_DIM, (NUM_SAMPLES,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型
    model = SimpleModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    print("普通训练完成")
    return model

# ============ 2. DDP 训练版本 ============
print("\n=== DDP 分布式训练版本 ===")

def setup_ddp(rank, world_size):
    """
    初始化分布式环境
    DDP的关键：每个进程独立运行，通过 all-reduce 同步梯度
    """
    # 设置每个进程能看到的主节点地址和端口
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    # backend='gloo' 适用于CPU训练，'nccl' 适用于GPU训练
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def cleanup_ddp():
    """清理分布式环境"""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    """
    DDP 训练函数
    每个进程有独立的 model 副本，独立进行 forward/backward，
    然后通过 all-reduce 在所有进程间同步梯度
    """
    print(f"\n--- DDP 训练模式, Rank={rank}/{world_size} ---")

    # 初始化当前进程的分布式环境
    setup_ddp(rank, world_size)

    # 将模型移动到当前进程的设备上
    # 注意：local_rank 表示当前进程在本地节点中的编号
    local_rank = rank  # 单机模式下 rank 就是 local_rank
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

    # ===== DDP 核心代码 =====
    # 1. 用 DDP 包装模型
    # 2. DDP 会自动：
    #    - 在 forward 时保持模型同步（可选）
    #    - 在 backward 后自动执行 all-reduce 同步梯度
    #    - 所有进程的模型参数保持一致
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    # ===== DDP 核心代码结束 =====

    # 构造数据
    # 注意：实际应用中每个进程应该加载不同的数据子集（数据分片）
    # 这里为了演示简单，所有进程用相同的数据
    X = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, OUTPUT_DIM, (NUM_SAMPLES,))
    dataset = TensorDataset(X, y)

    # 使用 DistributedSampler 确保每个进程加载不同的数据子集
    # sampler 会将数据分片，每个进程只看到自己的数据
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,  # 使用 DistributedSampler，不使用 shuffle
        num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    model.train()
    for epoch in range(EPOCHS):
        # 每个 epoch 重置 sampler，以保证跨 epoch 的数据分片正确
        sampler.set_epoch(epoch)

        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            # DDP 会在 backward 后自动执行 all-reduce
            # 所有进程的梯度会被同步平均
            loss.backward()

            # 参数更新
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # 只在 rank 0 打印，避免重复输出
        if rank == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # 清理分布式环境
    cleanup_ddp()

    print(f"Rank {rank}: DDP 训练完成")

def run_ddp_multiprocess():
    """
    启动多进程 DDP 训练
    使用 spawn 方式创建进程，更安全
    """
    print("\n=== 启动 DDP 多进程训练 ===")

    world_size = 2  # 使用2个进程（模拟2卡）

    # 使用 spawn 创建进程
    # 每个进程会执行 train_ddp 函数，并传入 (rank, world_size)
    mp.spawn(
        train_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

# ============ 3. DDP 核心原理解释 ============
print("\n=== DDP 核心原理解释 ===")
print("""
DDP（Distributed Data Parallel）的关键概念：

1. 进程与模型副本：
   - 每个 GPU/进程有独立的模型副本
   - 每个进程独立进行前向传播和反向传播

2. 梯度同步（All-Reduce）：
   - 每个进程计算出自己的梯度
   - 通过 All-Reduce 操作，所有进程的梯度被同步平均
   - 同步后的梯度用于更新参数

3. 为什么 DDP 比 DataParallel 快：
   - DataParallel 是中心化的，有主节点瓶颈
   - DDP 是去中心化的，通信与计算重叠
   - 梯度同步是高效的 ring-all-reduce

4. 关键代码流程：
   a) torch.distributed.init_process_group() 初始化进程组
   b) model = DDP(model) 包装模型
   c) 独立 forward → 独立 backward → 自动 all-reduce 梯度同步
   d) torch.distributed.destroy_process_group() 清理

5. 与普通训练的区别：
   - 需要初始化进程组
   - 需要使用 DistributedSampler 或自定义分片
   - 需要在结束时清理进程组
""")

# ============ 4. 启动脚本说明 ============
print("\n=== DDP 启动方式说明 ===")
print("""
启动 DDP 训练有两种推荐方式：

方式1: torchrun（推荐，新API）
  torchrun --nproc_per_node=2 train_script.py

方式2: torch.distributed.launch（旧API）
  python -m torch.distributed.launch --nproc_per_node=2 train_script.py

方式3: 手动多进程（适用于 Jupyter/测试）
  mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)

关键参数：
  --nproc_per_node: 每个节点的进程数，通常等于 GPU 数量
  --master_addr: 主节点地址
  --master_port: 主节点端口
""")

# ============ 5. 运行示例 ============
print("\n=== 运行普通训练 ===")
normal_model = train_normal()

print("\n=== 运行 DDP 训练（2进程）===")
print("注意：DDP 需要多进程环境，这里使用 mp.spawn 启动")
run_ddp_multiprocess()

print("\n=== Demo 完成 ===")
