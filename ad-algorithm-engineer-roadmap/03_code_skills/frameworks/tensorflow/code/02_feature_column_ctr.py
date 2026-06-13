"""
tf.feature_column + DenseFeatures 做 CTR 预估（推荐系统经典写法）

主题：使用 tf.feature_column 处理 user_id/item_id/category/hour 等特征，
      通过 DenseFeatures + MLP 构建 CTR 预估模型。

运行：python 02_feature_column_ctr.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import AUC


# ==================== 1. 生成 Toy 数据（模拟 CTR 数据）====================

def generate_ctr_data(num_samples=50000, seed=42):
    """
    生成模拟 CTR 数据：
    - user_id: 1~5000 用户 ID
    - item_id: 1~10000 商品 ID
    - category_id: 1~20 商品类目
    - hour: 0~23 行为发生时间（小时）
    - is_click: 0/1 点击标签
    """
    np.random.seed(seed)
    
    # 生成基础特征
    user_ids = np.random.randint(1, 5001, size=num_samples)          # 5000 个用户
    item_ids = np.random.randint(1, 10001, size=num_samples)          # 10000 个商品
    category_ids = np.random.randint(1, 21, size=num_samples)          # 20 个类目
    hours = np.random.randint(0, 24, size=num_samples)                 # 24 小时
    
    # 构造点击标签（模拟真实 CTR：user_id 和 item_id 有偏好）
    # 用户对某些 item_id 有偏好，category_id 也影响点击率
    click_probs = (
        0.1 * (user_ids % 10) / 10 +                  # 用户活跃度
        0.1 * (item_ids % 10) / 10 +                  # 商品热度
        0.1 * (category_ids % 5) / 5 +                 # 类目偏好
        0.05 * (hours >= 19).astype(float)             # 晚间高峰
    )
    click_probs = np.clip(click_probs, 0.01, 0.5)
    is_click = (np.random.random(num_samples) < click_probs).astype(np.float32)
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'category_id': category_ids,
        'hour': hours,
        'is_click': is_click
    })
    
    return df


# ==================== 2. 定义 Feature Columns ====================

def build_feature_columns(user_vocab_size=5000, item_vocab_size=10000, 
                           category_vocab_size=20, hour_boundaries=None):
    """
    构建 tf.feature_column 特征列：
    
    1. hour -> numeric_column -> bucketized_column（分桶，模拟时间特征）
    2. user_id -> categorical_column_with_identity -> embedding_column
    3. item_id -> categorical_column_with_identity -> embedding_column
    4. category_id -> categorical_column_with_vocabulary_list -> indicator_column
    5. user_id × category_id -> crossed_column -> embedding_column
    """
    
    if hour_boundaries is None:
        hour_boundaries = [6, 12, 18, 21]  # 凌晨/上午/下午/晚间
    
    # 2.1 hour: 连续值 -> 分桶
    hour_col = tf.feature_column.numeric_column('hour')
    hour_bucket = tf.feature_column.bucketized_column(hour_col, boundaries=hour_boundaries)
    
    # 2.2 user_id: 类别 ID -> embedding
    user_id_col = tf.feature_column.categorical_column_with_identity(
        'user_id', num_buckets=user_vocab_size)
    user_emb_col = tf.feature_column.embedding_column(
        user_id_col, dimension=16, combiner='mean')
    
    # 2.3 item_id: 类别 ID -> embedding
    item_id_col = tf.feature_column.categorical_column_with_identity(
        'item_id', num_buckets=item_vocab_size)
    item_emb_col = tf.feature_column.embedding_column(
        item_id_col, dimension=16, combiner='mean')
    
    # 2.4 category_id: 词表类别 -> indicator_column (one-hot)
    category_col = tf.feature_column.categorical_column_with_vocabulary_list(
        'category_id', vocabulary_list=list(range(1, category_vocab_size + 1)))
    category_ind_col = tf.feature_column.indicator_column(category_col)
    
    # 2.5 特征交叉: user_id × category_id -> embedding（解决冷启动泛化）
    crossed_col = tf.feature_column.crossed_column(
        [user_id_col, category_col], hash_bucket_size=1000)
    crossed_emb_col = tf.feature_column.embedding_column(crossed_col, dimension=8)
    
    # 所有特征列
    feature_columns = {
        'hour_bucket': hour_bucket,
        'user_emb': user_emb_col,
        'item_emb': item_emb_col,
        'category_ind': category_ind_col,
        'crossed_emb': crossed_emb_col,
    }
    
    return feature_columns


# ==================== 3. 构建 DenseFeatures + MLP 模型 ====================

def build_ctr_model(feature_columns, dnn_units=[256, 128, 64]):
    """
    使用 DenseFeatures 层接入 feature_columns，构建 MLP CTR 模型
    
    Args:
        feature_columns: dict of feature columns
        dnn_units: DNN 每层单元数
    """
    # 输入层（字典形式，key 与 feature_column 名称对应）
    inputs = {
        'hour': Input(shape=(1,), dtype=tf.int32, name='hour'),
        'user_id': Input(shape=(1,), dtype=tf.int32, name='user_id'),
        'item_id': Input(shape=(1,), dtype=tf.int32, name='item_id'),
        'category_id': Input(shape=(1,), dtype=tf.int32, name='category_id'),
    }
    
    # DenseFeatures 层（将 sparse 特征转为 dense）
    dense_features = tf.keras.layers.DenseFeatures(
        list(feature_columns.values()), name='dense_features'
    )(inputs)
    
    # MLP
    x = Dense(dnn_units[0], activation='relu')(dense_features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(dnn_units[1], activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = Dense(dnn_units[2], activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # 输出层：sigmoid 做 CTR 预估
    output = Dense(1, activation='sigmoid', name='pctr')(x)
    
    model = Model(inputs=inputs, outputs=output, name='ctr_model')
    return model


# ==================== 4. DataFrame -> tf.data.Dataset ====================

def df_to_dataset(df, feature_columns, batch_size=256, shuffle=True):
    """
    将 pandas DataFrame 转为 tf.data.Dataset
    
    Args:
        df: DataFrame，包含特征和标签
        feature_columns: dict of feature columns
        batch_size: batch 大小
        shuffle: 是否打乱
    """
    # 标签
    labels = df.pop('is_click')
    
    # 创建 Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels.values))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ==================== 5. 自定义 Callback：打印中间指标 ====================

class MetricsLogger(Callback):
    """每隔若干 epoch 打印训练指标"""
    
    def __init__(self, print_freq=2):
        super().__init__()
        self.print_freq = print_freq
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq == 0:
            logs = logs or {}
            print(f"    Epoch {epoch+1:03d} - "
                  f"loss: {logs.get('loss', 0):.4f}, "
                  f"auc: {logs.get('auc', 0):.4f}, "
                  f"val_loss: {logs.get('val_loss', 0):.4f}, "
                  f"val_auc: {logs.get('val_auc', 0):.4f}")


# ==================== 6. 主流程 ====================

if __name__ == '__main__':
    
    print("=" * 60)
    print("tf.feature_column + DenseFeatures CTR 预估 Demo")
    print("=" * 60)
    
    # 超参数
    BATCH_SIZE = 256
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    
    # 1. 生成 Toy 数据
    print("\n[1] 生成 Toy CTR 数据（50000条）...")
    df = generate_ctr_data(num_samples=50000)
    print(f"    数据形状: {df.shape}")
    print(f"    点击率: {df['is_click'].mean():.4f}")
    print(f"    样例:\n{df.head()}")
    
    # 2. 划分训练集/验证集
    print("\n[2] 划分训练集/验证集...")
    val_size = int(len(df) * 0.2)
    df_val = df.iloc[:val_size]
    df_train = df.iloc[val_size:]
    print(f"    训练集: {len(df_train)} 条")
    print(f"    验证集: {len(df_val)} 条")
    
    # 3. 构建 Feature Columns
    print("\n[3] 构建 Feature Columns...")
    feature_columns = build_feature_columns()
    print(f"    特征列: {list(feature_columns.keys())}")
    
    # 4. 构建模型
    print("\n[4] 构建 DenseFeatures + MLP CTR 模型...")
    model = build_ctr_model(feature_columns, dnn_units=[256, 128, 64])
    model.summary()
    
    # 5. 编译模型
    print("\n[5] 编译模型...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[AUC(name='auc')]
    )
    
    # 6. 准备 tf.data.Dataset
    print("\n[6] 准备 tf.data.Pipeline...")
    train_dataset = df_to_dataset(df_train.copy(), feature_columns, 
                                   batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = df_to_dataset(df_val.copy(), feature_columns, 
                                  batch_size=BATCH_SIZE, shuffle=False)
    
    # 7. 训练
    print("\n[7] 开始训练...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[MetricsLogger(print_freq=2)],
        verbose=1
    )
    
    # 8. 评估
    print("\n[8] 评估模型...")
    results = model.evaluate(val_dataset, verbose=1)
    print(f"    验证集 Loss: {results[0]:.4f}")
    print(f"    验证集 AUC: {results[1]:.4f}")
    
    # 9. 导出 SavedModel（用于 TF Serving）
    print("\n[9] 导出 SavedModel...")
    export_dir = './saved_model_dir/02_feature_column_ctr'
    os.makedirs(export_dir, exist_ok=True)
    model.save(export_dir, save_format='tf')
    print(f"    SavedModel 已保存至: {export_dir}")
    
    # 10. 加载并推理一条样本
    print("\n[10] 加载 SavedModel 并推理...")
    loaded = tf.saved_model.load(export_dir)
    infer = loaded.signatures['serving_default']
    
    # 构造单条样本输入
    sample_input = {
        'hour': tf.constant([[19]], dtype=tf.int32),          # 晚间
        'user_id': tf.constant([[1001]], dtype=tf.int32),      # 用户1001
        'item_id': tf.constant([[5001]], dtype=tf.int32),      # 商品5001
        'category_id': tf.constant([[5]], dtype=tf.int32),    # 类目5
    }
    result = infer(**sample_input)
    pctr = result['pctr'].numpy()[0][0]
    
    print(f"    样本: hour=19, user_id=1001, item_id=5001, category=5")
    print(f"    预测 CTR: {pctr:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("=" * 60)
    print("\n启动 TensorBoard 查看训练曲线：")
    print("  tensorboard --logdir ./logs --port 6006")
