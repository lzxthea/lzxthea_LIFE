"""
自定义 Dataset + DataLoader 示例 - 推荐系统变长序列场景
本demo展示如何在推荐系统中处理用户行为序列（长度不一致）的padding问题
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("=== 自定义 Dataset + DataLoader ===")
print("场景：推荐系统中用户行为序列长度不一致，需要 padding 对齐")

# ============ 超参数配置 ============
MAX_LEN = 8           # 固定序列长度（padding后的长度）
PADDING_VALUE = 0     # padding 填充值
BATCH_SIZE = 4        # 批量大小
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============ 1. 构造 Toy 推荐系统数据集 ============
print("\n=== 步骤1: 构造 Toy 推荐数据集 ===")
# 模拟用户行为数据：
# - user_id: 用户ID
# - item_id: 商品ID序列（每个用户历史点击的多个商品，长度不一）
# - category_id: 商品类别序列
# - clicked: 是否点击（标签）

# 模拟5个用户，每个用户有不同长度的行为序列
# 用户0: 3个行为, 用户1: 5个行为, 用户2: 8个行为(最大), 用户3: 2个行为, 用户4: 6个行为
user_behaviors = [
    {'user_id': 0, 'item_ids': [101, 205, 330], 'categories': [10, 20, 30], 'clicked': 1},
    {'user_id': 1, 'item_ids': [102, 203, 304, 405, 506], 'categories': [11, 21, 31, 41, 51], 'clicked': 0},
    {'user_id': 2, 'item_ids': [103, 201, 302, 403, 504, 605, 706, 807], 'categories': [12, 22, 32, 42, 52, 62, 72, 82], 'clicked': 1},
    {'user_id': 3, 'item_ids': [104, 209], 'categories': [13, 23], 'clicked': 0},
    {'user_id': 4, 'item_ids': [105, 202, 303, 404, 505, 606], 'categories': [14, 24, 34, 44, 54, 64], 'clicked': 1},
]

print(f"用户行为序列原始长度:")
for i, b in enumerate(user_behaviors):
    print(f"  用户{b['user_id']}: {len(b['item_ids'])} 个行为")

# ============ 2. 自定义 Dataset ============
print("\n=== 步骤2: 自定义 MyDataset 类 ===")

class MyDataset(Dataset):
    """
    推荐系统用户行为序列 Dataset
    在 __getitem__ 中做固定长度 padding（padding到MAX_LEN）
    """
    def __init__(self, user_behaviors, max_len=8, padding_value=0):
        """
        Args:
            user_behaviors: 用户行为列表
            max_len: 序列最大长度（不足则padding）
            padding_value: padding填充值
        """
        self.user_behaviors = user_behaviors
        self.max_len = max_len
        self.padding_value = padding_value

    def __len__(self):
        """返回数据集大小"""
        return len(self.user_behaviors)

    def __getitem__(self, idx):
        """
        获取单个样本，并对序列进行padding
        注意：这里演示固定长度padding，实际中常用 collate_fn 做动态padding
        """
        behavior = self.user_behaviors[idx]

        user_id = behavior['user_id']
        item_ids = behavior['item_ids']
        categories = behavior['categories']
        clicked = behavior['clicked']

        # 将序列padding到固定长度
        # 注意：这里简化处理，实际推荐系统中padding逻辑可能更复杂
        if len(item_ids) < self.max_len:
            # 不足max_len的部分用padding_value填充
            item_ids = item_ids + [self.padding_value] * (self.max_len - len(item_ids))
            categories = categories + [self.padding_value] * (self.max_len - len(categories))
        else:
            # 超过max_len的截断
            item_ids = item_ids[:self.max_len]
            categories = categories[:self.max_len]

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(item_ids, dtype=torch.long),
            torch.tensor(categories, dtype=torch.long),
            torch.tensor(clicked, dtype=torch.long)
        )

print("MyDataset 类定义完成，实现了 __len__ 和 __getitem__")

# ============ 3. 自定义 collate_fn（动态padding） ============
print("\n=== 步骤3: 自定义 collate_fn（动态padding） ===")

def dynamic_collate_fn(batch):
    """
    动态padding函数：找出batch中最长序列，然后pad到该长度
    这是推荐系统中更常用的做法，比在Dataset里固定padding更灵活

    实际意义：不同用户的行为序列长度差异很大，
    如果固定padding到MAX_LEN会造成大量无效计算；
    动态padding可以根据每个batch的实际最大长度来padding，节省资源
    """
    user_ids, item_ids, categories, clicked = zip(*batch)

    # 找出当前batch中最长序列
    max_len = max(len(ids) for ids in item_ids)
    print(f"  [collate_fn] batch内最大序列长度: {max_len}")

    # 动态padding到当前batch的最大长度
    padded_item_ids = []
    padded_categories = []
    for ids in item_ids:
        if len(ids) < max_len:
            padded = ids + [0] * (max_len - len(ids))
        else:
            padded = ids
        padded_item_ids.append(padded)

    for cats in categories:
        if len(cats) < max_len:
            padded = cats + [0] * (max_len - len(cats))
        else:
            padded = cats
        padded_categories.append(padded)

    return (
        torch.stack(user_ids),
        torch.tensor(padded_item_ids, dtype=torch.long),
        torch.tensor(padded_categories, dtype=torch.long),
        torch.stack(clicked)
    )

print("collate_fn 定义完成：动态padding到batch内最大长度")

# ============ 4. 创建 Dataset 和 DataLoader ============
print("\n=== 步骤4: 创建 DataLoader ===")

# 使用固定padding的Dataset（演示两种方式）
dataset_fixed = MyDataset(user_behaviors, max_len=MAX_LEN, padding_value=PADDING_VALUE)

# 使用动态padding的DataLoader
dataloader_dynamic = DataLoader(
    dataset_fixed,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=dynamic_collate_fn  # 使用自定义collate_fn
)

print(f"DataLoader 配置: batch_size={BATCH_SIZE}, shuffle=True, collate_fn=dynamic_collate_fn")

# ============ 5. 遍历 DataLoader 验证效果 ============
print("\n=== 步骤5: 遍历 DataLoader 验证 padding 效果 ===")

print("\n--- 方式1: 固定padding（Dataset内部处理）---")
dataset_fixed_loader = DataLoader(dataset_fixed, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
for batch_idx, (user_ids, item_ids, categories, clicked) in enumerate(dataset_fixed_loader):
    print(f"Batch {batch_idx}:")
    print(f"  user_ids shape: {user_ids.shape} - {user_ids.tolist()}")
    print(f"  item_ids shape: {item_ids.shape}")
    print(f"  categories shape: {categories.shape}")
    print(f"  clicked: {clicked.tolist()}")
    print(f"  item_ids[0] (原始3个+padding 5个): {item_ids[0].tolist()}")

print("\n--- 方式2: 动态padding（collate_fn处理）---")
print("说明：collate_fn会根据每个batch的实际最大长度动态调整padding长度")
for batch_idx, (user_ids, item_ids, categories, clicked) in enumerate(dataloader_dynamic):
    print(f"\nBatch {batch_idx}:")
    print(f"  user_ids: {user_ids.tolist()}")
    print(f"  item_ids shape: {item_ids.shape}")
    print(f"  item_ids (每行的padding长度可能不同):")
    for i, row in enumerate(item_ids.tolist()):
        # 找出非padding值的位置
        non_padding = [x for x in row if x != 0]
        print(f"    用户{user_ids[i].item()}: 有效值={non_padding}, pad后长度={len(row)}")

# ============ 6. 说明推荐系统中 padding 的实际意义 ============
print("\n=== 步骤6: 总结 - 推荐系统中 padding 的意义 ===")
print("""
在推荐系统中处理用户行为序列时，padding是非常重要的技术：

1. 问题背景：
   - 不同用户的点击/购买行为数量差异巨大
   - 有的用户可能有成百上千次行为，有的可能只有几次
   - 深度学习模型需要固定形状的输入

2. Padding 的作用：
   - 将不同长度的序列填充到相同长度，便于批量处理
   - 填充值通常选择0或特殊标记（如-1），模型需要学会忽略这些无意义的值

3. 动态 padding vs 固定 padding：
   - 固定 padding：所有样本都pad到统一最大长度，简单但可能浪费计算资源
   - 动态 padding：每个batch pad到该batch内的最大长度，更灵活高效
   - 实际工业应用中，动态padding配合pack_padded_sequence使用效果更好

4. 注意事项：
   - 需要记录原始序列长度，以便在模型中正确处理
   - 可以使用 mask 机制让模型"看到"哪些是真实的padding位置
""")

print("\n=== Demo 完成 ===")
