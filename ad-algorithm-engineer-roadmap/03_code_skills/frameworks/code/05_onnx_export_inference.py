"""
双塔召回模型导出 ONNX + ONNXRuntime 推理
本demo展示推荐系统中经典的双塔模型（UserTower + ItemTower）的训练、导出和推理
双塔模型是工业推荐系统的核心组件，用于高效的向量召回
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=== 双塔召回模型 ONNX 导出与推理 ===")

# ============ 超参数配置 ============
USER_NUM = 100         # 用户数量
ITEM_NUM = 500         # 商品数量
EMBEDDING_DIM = 32     # Embedding 维度
MLP_DIM = 64           # MLP 隐藏层维度
OUTPUT_DIM = 32        # Tower 输出 embedding 维度
BATCH_SIZE = 8
LEARNING_RATE = 0.001
TRAIN_STEPS = 50

# ============ 1. 构造双塔模型结构 ============
print("\n=== 步骤1: 定义双塔模型结构 ===")

class Tower(nn.Module):
    """
    共享结构的 Tower（UserTower 和 ItemTower 共用）
    Embedding + MLP → 输出固定维度的 embedding
    """
    def __init__(self, num_ids, embedding_dim, mlp_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_ids, embedding_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim)
        )

    def forward(self, ids):
        """
        Args:
            ids: (batch_size,) 用户ID或商品ID
        Returns:
            emb: (batch_size, output_dim) 塔的输出embedding
        """
        # Embedding 查找
        emb = self.embedding(ids)  # (batch_size, embedding_dim)
        # 通过 MLP
        output = self.mlp(emb)      # (batch_size, output_dim)
        # L2 归一化（用于余弦相似度计算）
        output = nn.functional.normalize(output, p=2, dim=1)
        return output

class TwoTowerModel(nn.Module):
    """
    双塔召回模型
    - UserTower: 处理用户特征，输出 user_embedding
    - ItemTower: 处理商品特征，输出 item_embedding
    - score = cosine(user_emb, item_emb)
    """
    def __init__(self, user_num, item_num, embedding_dim, mlp_dim, output_dim):
        super().__init__()
        self.user_tower = Tower(user_num, embedding_dim, mlp_dim, output_dim)
        self.item_tower = Tower(item_num, embedding_dim, mlp_dim, output_dim)

    def forward(self, user_ids, item_ids):
        """
        Args:
            user_ids: (batch_size,) 用户ID
            item_ids: (batch_size,) 商品ID
        Returns:
            user_emb: (batch_size, output_dim)
            item_emb: (batch_size, output_dim)
            scores: (batch_size,) 余弦相似度
        """
        user_emb = self.user_tower(user_ids)
        item_emb = self.item_tower(item_ids)
        # 余弦相似度
        scores = torch.sum(user_emb * item_emb, dim=1)
        return user_emb, item_emb, scores

print("双塔模型结构定义完成:")
print("  - UserTower: Embedding + MLP + L2 normalize")
print("  - ItemTower: Embedding + MLP + L2 normalize")
print("  - Score: cosine(user_emb, item_emb)")

# ============ 2. 构造训练数据 ============
print("\n=== 步骤2: 构造训练数据 ===")
# 模拟用户-商品交互数据
np.random.seed(42)
torch.manual_seed(42)

user_ids = torch.randint(1, USER_NUM, (100,))    # 用户ID (1~99，避免0)
item_ids = torch.randint(1, ITEM_NUM, (100,))    # 商品ID
labels = torch.tensor(np.random.randint(0, 2, 100), dtype=torch.float32)  # 点击标签

print(f"user_ids shape: {user_ids.shape}, 范围: [{user_ids.min()}, {user_ids.max()}]")
print(f"item_ids shape: {item_ids.shape}, 范围: [{item_ids.min()}, {item_ids.max()}]")
print(f"labels shape: {labels.shape}")

# ============ 3. 初始化模型和训练 ============
print("\n=== 步骤3: 初始化模型和训练 ===")

# 初始化双塔模型
model = TwoTowerModel(
    user_num=USER_NUM,
    item_num=ITEM_NUM,
    embedding_dim=EMBEDDING_DIM,
    mlp_dim=MLP_DIM,
    output_dim=OUTPUT_DIM
)

print(f"\n模型结构:")
print(model)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n损失函数: BCEWithLogitsLoss")
print(f"优化器: Adam, lr={LEARNING_RATE}")

# 简单训练几个 step
print(f"\n训练 {TRAIN_STEPS} 个 step:")
model.train()
for step in range(TRAIN_STEPS):
    optimizer.zero_grad()

    # 随机采样一个 batch
    indices = torch.randint(0, len(user_ids), (BATCH_SIZE,))
    batch_user_ids = user_ids[indices]
    batch_item_ids = item_ids[indices]
    batch_labels = labels[indices]

    # 前向传播
    user_emb, item_emb, scores = model(batch_user_ids, batch_item_ids)

    # 计算损失
    loss = criterion(scores, batch_labels)

    # 反向传播
    loss.backward()
    optimizer.step()

    if (step + 1) % 10 == 0:
        print(f"  Step {step+1}/{TRAIN_STEPS}, Loss: {loss.item():.4f}")

print("训练完成!")

# ============ 4. 导出 ONNX ============
print("\n=== 步骤4: 导出 ONNX 模型 ===")

# 创建独立的 UserTower 和 ItemTower 用于导出
# 注意：导出时需要先创建独立的 tower 实例并加载权重
user_tower = Tower(USER_NUM, EMBEDDING_DIM, MLP_DIM, OUTPUT_DIM)
item_tower = Tower(ITEM_NUM, EMBEDDING_DIM, MLP_DIM, OUTPUT_DIM)

# 从主模型中复制权重
user_tower.load_state_dict(model.user_tower.state_dict())
item_tower.load_state_dict(model.item_tower.state_dict())

# 设置为 eval 模式
user_tower.eval()
item_tower.eval()

# 准备示例输入
example_user_ids = torch.randint(1, USER_NUM, (BATCH_SIZE,), dtype=torch.long)
example_item_ids = torch.randint(1, ITEM_NUM, (BATCH_SIZE,), dtype=torch.long)

print(f"示例输入 shape: user_ids={example_user_ids.shape}, item_ids={example_item_ids.shape}")

# 导出 UserTower
print("\n导出 UserTower...")
torch.onnx.export(
    user_tower,
    (example_user_ids,),
    'user_tower.onnx',
    input_names=['user_id'],
    output_names=['user_emb'],
    dynamic_axes={
        'user_id': {0: 'batch_size'},
        'user_emb': {0: 'batch_size'}
    },
    opset_version=14,
    do_constant_folding=True
)
print("UserTower 导出完成: user_tower.onnx")

# 导出 ItemTower
print("\n导出 ItemTower...")
torch.onnx.export(
    item_tower,
    (example_item_ids,),
    'item_tower.onnx',
    input_names=['item_id'],
    output_names=['item_emb'],
    dynamic_axes={
        'item_id': {0: 'batch_size'},
        'item_emb': {0: 'batch_size'}
    },
    opset_version=14,
    do_constant_folding=True
)
print("ItemTower 导出完成: item_tower.onnx")

# ============ 5. 使用 ONNXRuntime 推理 ============
print("\n=== 步骤5: ONNXRuntime 推理 ===")

try:
    import onnxruntime as ort
    print("ONNXRuntime 可用")

    # 加载 ONNX 模型
    print("\n加载 user_tower.onnx...")
    user_session = ort.InferenceSession('user_tower.onnx')
    print(f"UserTower 输入: {[inp.name for inp in user_session.get_inputs()]}")
    print(f"UserTower 输出: {[out.name for out in user_session.get_outputs()]}")

    print("\n加载 item_tower.onnx...")
    item_session = ort.InferenceSession('item_tower.onnx')
    print(f"ItemTower 输入: {[inp.name for inp in item_session.get_inputs()]}")
    print(f"ItemTower 输出: {[out.name for out in item_session.get_outputs()]}")

    # 准备测试数据
    test_user_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    test_item_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    print(f"\n测试数据:")
    print(f"  user_ids: {test_user_ids}")
    print(f"  item_ids: {test_item_ids}")

    # User Tower 推理
    user_emb_np = user_session.run(
        None,
        {'user_id': test_user_ids.reshape(-1, 1)}  # ONNX 需要 batch 维度
    )[0]
    print(f"\nUser Embedding shape: {user_emb_np.shape}")

    # Item Tower 推理
    item_emb_np = item_session.run(
        None,
        {'item_id': test_item_ids.reshape(-1, 1)}
    )[0]
    print(f"Item Embedding shape: {item_emb_np.shape}")

    # 计算余弦相似度
    # cosine(u, v) = sum(u * v) / (||u|| * ||v||)
    # 由于我们已经做了 L2 normalize，所以直接点积就是 cosine similarity
    user_emb_norm = user_emb_np / np.linalg.norm(user_emb_np, axis=1, keepdims=True)
    item_emb_norm = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    cosine_scores = np.sum(user_emb_norm * item_emb_norm, axis=1)

    print(f"\n余弦相似度分数:")
    for i, score in enumerate(cosine_scores):
        print(f"  用户{test_user_ids[i]} vs 商品{test_item_ids[i]}: {score:.4f}")

    # 验证 PyTorch 和 ONNX 输出一致性
    print("\n验证 PyTorch 和 ONNX 输出一致性:")
    with torch.no_grad():
        pt_user_emb = user_tower(torch.tensor(test_user_ids, dtype=torch.long))
        pt_item_emb = item_tower(torch.tensor(test_item_ids, dtype=torch.long))

        pt_cosine = torch.sum(pt_user_emb * pt_item_emb, dim=1).numpy()

        diff = np.abs(cosine_scores - pt_cosine)
        print(f"最大差异: {diff.max():.6f}")
        print(f"一致性验证: {'通过 ✓' if diff.max() < 1e-5 else '失败 ✗'}")

    print("\n=== ONNX 导出和推理验证成功 ===")

except ImportError:
    print("\n⚠️ onnxruntime 未安装，无法进行推理验证")
    print("请运行: pip install onnxruntime-gpu")
    print("或: pip install onnxruntime")
    print("\nONNX 模型文件已导出到当前目录:")
    print("  - user_tower.onnx")
    print("  - item_tower.onnx")
    print("安装 onnxruntime 后可进行推理验证")

# ============ 6. 双塔模型在推荐系统中的意义 ============
print("\n=== 双塔模型在推荐系统中的意义 ===")
print("""
1. 问题背景：
   - 推荐系统中需要从海量商品中快速找到用户可能喜欢的商品
   - 暴力计算 user-item 相似度代价太大 O(N)

2. 双塔模型的优势：
   - User Tower: 把用户特征编码成固定维度的向量
   - Item Tower: 把商品特征编码成固定维度的向量
   - 离线计算好所有 item 的 embedding（Item Tower）
   - 在线时只计算 user embedding，然后做向量检索（如 Faiss）

3. 向量检索优化：
   - 余弦相似度或内积作为 score
   - 使用近似最近邻（ANN）算法加速检索
   - 典型方案：Faiss, Milvus, Elasticsearch HNSW

4. 工程实践：
   - 双塔模型是召回（recall）阶段的核心模型
   - 后面还会接排序（rank）阶段的精排模型
   - Embedding 质量直接影响召回效果

5. ONNX 导出：
   - 训练用 PyTorch，推理可以用 ONNXRuntime 部署
   - 支持跨平台部署（服务器、移动端、边缘设备）
""")

print("\n=== Demo 完成 ===")
