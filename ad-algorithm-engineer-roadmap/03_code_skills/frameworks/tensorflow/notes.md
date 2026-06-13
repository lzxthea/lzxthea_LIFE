# TensorFlow / Keras · 知识点笔记

前言

TensorFlow 是谷歌开源的深度学习框架，在工业级推荐系统（Recommendation System）、广告系统（Advertising System）中占据核心地位。与 PyTorch 相比，TF 的静态图执行模式（TF1.x）和 XLA 编译（TF2.x）在大规模分布式训练与线上推理场景下具备更成熟的生态——尤其是 TF Serving、TF Lite、TFX 等组件为搜广推系统的模型部署提供了端到端闭环。因此，掌握 TensorFlow 是广告/推荐算法工程师的必备技能。

搜广推场景对 TensorFlow 的使用主要体现在三个层面：特征工程（tf.feature_column）、离线训练（tf.estimator / tf.keras）与线上推理（SavedModel + TF Serving）。本文档围绕工业实践组织知识点，所有章节均标注"在搜广推哪里见过"。

---

## 1. TF 版本与 Keras 三种建模方式

### 1.1 TF1.x vs TF2.x 对比表

| 维度 | TF1.x | TF2.x |
|------|-------|-------|
| 执行模式 | Graph（静态图） | Eager（动态图）+ @tf.function 编译 |
| 高层 API | Estimator / Slim / Keras（tf.keras） | tf.keras（主力）+ Estimator（兼容） |
| 会话管理 | tf.Session() | 无需 Session，Eager 即时执行 |
| 变量管理 | tf.global_variable() | tf.Variable / tf.keras.layers |
| 分布式 | tf.train.MonitoredTrainingSession | tf.distribute.Strategy |
| API 风格 | imperative（命令式） | 混合：命令式+Eager，函数式+tf.function |

**在搜广推哪里见过**：TF1.x 的 Estimator + feature_column 组合在早期BAT广告系统中极为常见，很多内部项目至今仍有 TF1.x 代码在运行。TF2.x 的 Keras 是新项目首选。

### 1.2 Keras 三种建模方式

#### Sequential API（线性堆叠）

```python
import tensorflow as tf

# 适用于单输入单输出、无分支的简单模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

#### Functional API（函数式，多输入多输出）

```python
# 适用于多输入/多输出、有共享层、有残差连接等复杂结构
user_input = tf.keras.Input(shape=(64,), name='user_feat')
item_input = tf.keras.Input(shape=(64,), name='item_feat')

# 共享编码层
shared = tf.keras.layers.Dense(128, activation='relu')
user_encoded = shared(user_input)
item_encoded = shared(item_input)

# 内积交互
dot = tf.reduce_sum(user_encoded * item_encoded, axis=-1, keepdims=True)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dot)

model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
```

#### Subclassing（面向对象自定义）

```python
# 适用于需要自定义训练循环、复杂层逻辑、GAN、强化学习等
class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)

model = CustomModel()
```

**在搜广推哪里见过**：Functional API 是搜广推中最常用的建模方式，因为推荐系统模型（如Wide&Deep）天然具有多输入（user特征、item特征、交叉特征）和多输出（点击率、转化率）结构。Sequential 用于快速搭建 DNN 基线。Subclassing 在多任务学习（MMoE、PLE）和自定义Attention时使用较多。

---

## 2. 推荐系统/广告中的 TF 典型用法

### 2.1 tf.feature_column

tf.feature_column 是 TF 提供的特征处理工具，在 CTR 预估场景中负责将原始特征（类别 ID、连续值、交叉特征）转换为 Dense Tensor。

```python
import tensorflow as tf

# 1. 连续值特征 → 直接输入 DNN
age = tf.feature_column.numeric_column('age')

# 2. 连续值 → 分桶离散化（模拟树模型分裂效果）
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 35, 45, 60])

# 3. 类别 ID（整数编码）→ embedding（推荐系统核心操作）
user_id = tf.feature_column.categorical_column_with_identity('user_id', num_buckets=10000)
user_id_emb = tf.feature_column.embedding_column(user_id, dimension=16)

# 4. 类别特征（已知词表）→ one-hot（高基数用 embedding）
category = tf.feature_column.categorical_column_with_vocabulary_list(
    'category',
    vocabulary_list=['electronics', 'clothing', 'food', 'beauty']
)
category_ind = tf.feature_column.indicator_column(category)  # 转为 one-hot

# 5. 特征交叉：user_id × category → embedding（解决冷启动泛化）
crossed = tf.feature_column.crossed_column([user_id, category], hash_bucket_size=1000)
crossed_emb = tf.feature_column.embedding_column(crossed, dimension=8)
```

**在搜广推哪里见过**：tf.feature_column 是 TF1.x/2.x 时代做 CTR 模型特征工程的标准写法。阿里M人群中台、字节DNN模型均大量使用。CrossedColumn 用于建模 user × item 的二阶交叉。

### 2.2 tf.data 标准 Pipeline

```python
import tensorflow as tf

def parse_fn(example):
    """解析 TFRecord 示例"""
    description = {
        'user_id': tf.io.FixedLenFeature([], tf.int64),
        'item_id': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example, description)
    return parsed['user_id'], parsed['label']

# 标准 Pipeline
dataset = tf.data.TFRecordDataset('./data/train.tfrecord')
dataset = dataset.map(parse_fn)           # 解析
dataset = dataset.shuffle(buffer_size=10000)  # 打乱
dataset = dataset.batch(256)              # batch
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 预取（性能关键）
```

**在搜广推哪里见过**：搜广推系统日均数十亿样本，离线训练依赖 tf.data 流水线。prefetch 和 map 并行化是防止 GPU 空闲的核心手段。

### 2.3 tf.estimator 自定义 model_fn

```python
def model_fn(features, labels, mode, params):
    """自定义 Estimator 模型函数"""
    
    # ==================== 1. Forward Pass ====================
    user_emb = tf.keras.layers.Embedding(
        params['user_vocab'], params['emb_dim'])(features['user_id'])
    item_emb = tf.keras.layers.Embedding(
        params['item_vocab'], params['emb_dim'])(features['item_id'])
    
    emb_sum = tf.reduce_sum(user_emb * item_emb, axis=-1, keepdims=True)
    x = tf.keras.layers.Dense(128, activation='relu')(emb_sum)
    logits = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # ==================== 2. Train 分支 ====================
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(labels, logits))
        train_op = tf.optimizers.Adam(0.001).minimize(
            loss, global_step=tf.compat.v1.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
    # ==================== 3. Eval 分支 ====================
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(labels, logits))
        auc = tf.metrics.auc(labels, logits)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops={'auc': auc})
    
    # ==================== 4. Predict 分支 ====================
    else:
        predictions = {'prob': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
```

**在搜广推哪里见过**：TF1.x 时代的经典写法，很多线上 Serving 系统基于 Estimator 构建。虽然 TF2.x 建议迁移到 Keras，但老项目（尤其是 TF1.x 存量）仍大量存在。

### 2.4 tf.distribute 分布式策略

```python
import tensorflow as tf

# 策略1：MirroredStrategy（单服务器多卡，数据并行）
strategy1 = tf.distribute.MirroredStrategy()

# 策略2：ParameterServerStrategy（多服务器参数服务器，推荐系统标配）
# 注意：PS 策略在 TF2.x 中需配合 tf.distribute.experimental.ParameterServerStrategy
strategy2 = tf.distribute.experimental.ParameterServerStrategy()

# 使用方式：创建模型时套上策略作用域
with strategy1.scope():
    model = build_model()  # 所有变量自动镜像到各卡
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
```

**在搜广推哪里见过**：搜广推系统训练数据量极大（十亿级别），必须使用多卡甚至多机分布式训练。PS 策略（Parameter Server）是召回、粗排、精排多阶段训练的标准架构。

### 2.5 SavedModel + TF Serving

```python
import tensorflow as tf

# 导出 SavedModel
model.save('saved_model_dir', save_format='tf')

# TF Serving 所需 input signature
def serving_input_receiver_fn():
    """定义线上推理时 TF Serving 接收的输入格式"""
    input_spec = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec={
            'user_id': tf.io.FixedLenFeature([], tf.int64),
            'item_id': tf.io.FixedLenFeature([], tf.int64),
        }
    )()
    return input_spec

# 导出供 TF Serving 使用
estimator.export_saved_model('export_dir', serving_input_receiver_fn)

# 推理时加载
loaded = tf.saved_model.load('saved_model_dir')
infer = loaded.signatures['serving_default']
result = infer(user_id=tf.constant([[1001]]), item_id=tf.constant([[2002]]))
```

**在搜广推哪里见过**：线上精排模型全量依赖 TF Serving。TF Serving 支持热更新、A/B 流量切换、模型版本管理，是搜广推系统的标准推理框架。

---

## 3. 推荐系统典型模型的 TF 实现要点

### 3.1 Wide&Deep

Wide 部分处理记忆（memorization），Deep 部分处理泛化（generalization）。

```python
# Wide: Linear（直接记忆原始特征交互）
wide = tf.keras.layers.Dense(1)(wide_features)  # 或用 tf.keras.experimental.LinearModel

# Deep: DNN（泛化学习）
deep = tf.keras.layers.Dense(256, activation='relu')(deep_features)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
deep = tf.keras.layers.Dense(1)(deep)

# 合并
output = tf.keras.layers.Add()([wide, deep])
output = tf.keras.activations.sigmoid(output)
```

### 3.2 DeepFM（二阶交互技巧）

DeepFM 的二阶交互通过 **sum(emb_sum^2 - emb_sum_squared^2)** 实现，无需显式两两配对：

```python
# 经典二阶交互实现（无需 O(n^2) 遍历）
# emb_sum = sum(emb_i)
# emb_squared_sum = sum(emb_i^2)
# interaction = sum(emb_sum^2 - emb_squared_sum) / 2
emb_sum = tf.reduce_sum(emb, axis=1)  # [batch, emb_dim]
emb_squared_sum = tf.reduce_sum(tf.square(emb), axis=1)  # [batch, emb_dim]
interaction = 0.5 * tf.reduce_sum(
    tf.square(emb_sum) - emb_squared_sum, axis=-1, keepdims=True)  # [batch, 1]
```

**在搜广推哪里见过**：DeepFM 是 2017 年左右字节跳动、腾讯广告系统的标配模型，FM 二阶交互公式几乎是面试必备。

### 3.3 DIN（Deep Interest Network）

DIN 通过 Attention 机制对用户行为序列建模：

```python
class DINAttention(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(dim, activation='prelu')
    
    def call(self, queries, keys, keys_length):
        # queries: [batch, dim] 当前候选 item
        # keys: [batch, seq_len, dim] 用户历史行为序列
        # keys_length: [batch] 有效序列长度（mask 用）
        
        queries_expanded = tf.expand_dims(queries, axis=1)  # [batch, 1, dim]
        attention_scores = tf.reduce_sum(
            self.dense(queries_expanded * keys), axis=-1)  # [batch, seq_len]
        
        # Mask 填充部分
        mask = tf.sequence_mask(keys_length, maxlen=tf.shape(keys)[1])
        attention_scores = tf.where(mask, attention_scores, -1e9)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        output = tf.reduce_sum(
            tf.expand_dims(attention_weights, -1) * keys, axis=1)  # [batch, dim]
        return output
```

**在搜广推哪里见过**：DIN 是阿里妈妈广告团队2018年的工作，在淘宝猜你喜欢、头条等信息流广告中有大量落地。用户行为序列的 Attention 建模是搜广推核心考点。

### 3.4 MMoE（Multi-Task Modeling）

MMoE 通过多个专家网络和门控机制实现多任务学习：

```python
class MMoE(tf.keras.layers.Layer):
    def __init__(self, num_experts, expert_dim, num_tasks):
        super().__init__()
        self.experts = [
            tf.keras.layers.Dense(expert_dim, activation='relu')
            for _ in range(num_experts)
        ]
        self.gates = [
            tf.keras.layers.Dense(num_experts, activation='softmax')
            for _ in range(num_tasks)
        ]
    
    def call(self, inputs):
        # inputs: [batch, dim]
        expert_outputs = [expert(inputs) for expert in self.experts]  # num_experts × [batch, expert_dim]
        expert_outputs = tf.stack(expert_outputs, axis=1)  # [batch, num_experts, expert_dim]
        
        outputs = []
        for gate in self.gates:
            gate_weights = gate(inputs)  # [batch, num_experts]
            gate_weights = tf.expand_dims(gate_weights, -1)  # [batch, num_experts, 1]
            task_output = tf.reduce_sum(gate_weights * expert_outputs, axis=1)  # [batch, expert_dim]
            outputs.append(task_output)
        return outputs  # list of num_tasks × [batch, expert_dim]
```

**在搜广推哪里见过**：MMoE 是快手、抖音等多任务推荐系统的标准方案。美团、字节的多目标优化（点击+时长+关注）均基于 MMoE 架构。

### 3.5 双塔召回（DSSM）

双塔模型将 user 和 item 分别编码，线上只需计算内积或 cosine：

```python
class DSSM(tf.keras.Model):
    def __init__(self, user_vocab, item_vocab, emb_dim=64):
        super().__init__()
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(user_vocab, emb_dim),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64),
        ])
        self.item_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(item_vocab, emb_dim),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64),
        ])
    
    def call(self, inputs, training=False):
        user_emb = self.user_tower(inputs['user_id'])
        item_emb = self.item_tower(inputs['item_id'])
        # 线上只需算 user_emb @ item_emb.T 做召回
        return tf.reduce_sum(user_emb * item_emb, axis=-1, keepdims=True)
    
    def export_user_tower(self):
        """导出 User 塔（用于线上 ANN 索引）"""
        user_input = tf.keras.Input(shape=(1,), dtype=tf.int64, name='user_id')
        user_emb = self.user_tower(user_input)
        return tf.keras.Model(user_input, user_emb)
    
    def export_item_tower(self):
        """导出 Item 塔（用于线上实时召回）"""
        item_input = tf.keras.Input(shape=(1,), dtype=tf.int64, name='item_id')
        item_emb = self.item_tower(item_input)
        return tf.keras.Model(item_input, item_emb)
```

**在搜广推哪里见过**：双塔召回是所有推荐系统（抖音、淘宝、微信看一看）的标配架构。Faiss/Milvus ANN 索引 + 双塔 embedding 是工业级召回方案。

---

## 4. TF 工程化常见坑

### 4.1 TF1.x → TF2.x 迁移

| 问题 | TF1.x | TF2.x |
|------|-------|-------|
| Session 管理 | `sess.run(op)` | Eager 即时执行，`@tf.function` 编译 |
| 全局变量 | `tf.global_variable()` | `tf.Variable`，自动管理 |
| Dropout | `keep_prob` placeholder | `rate` 参数，`training=True/False` 控制 |
| BatchNorm | `is_training` placeholder | `training=True/False`（Keras 统一接口） |
| 特征列 | `tf.feature_column.input_layer` | `tf.keras.layers.DenseFeatures` |

### 4.2 Variable 创建时机

TF2.x 中，所有 `tf.Variable` 应在 **策略作用域（strategy.scope()）内** 创建：

```python
with strategy.scope():
    model = build_model()  # 模型变量在此创建
```

在 scope 外创建的变量不会被分布式策略管理，导致多卡训练失败。

### 4.3 BatchNorm 的 training=True/False

推理时必须设置 `training=False`，否则 BatchNorm 会使用 batch 统计量而非全局移动平均：

```python
# 正确做法：Keras 自动处理
outputs = model(inputs, training=False)  # 推理

# 自定义层需显式控制
class CustomBatchNorm(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        if training:
            return tf.nn.batch_normalization(inputs, ...)
        return tf.nn.batch_normalization(inputs, ...)  # 使用 moving_mean/moving_var
```

### 4.4 Serving Input Receiver 格式

导出 TF Serving 模型时，输入必须是原始 tensor（非字典）：

```python
def serving_input_receiver_fn():
    # 错误：返回字典
    # return {'user_id': tf.placeholder(tf.int64, [None])}
    
    # 正确：返回 ServingInputReceiver
    return tf.estimator.export.ServingInputReceiver(
        features={'user_id': tf.keras.Input(shape=(1,), dtype=tf.int64, name='user_id')},
        receiver_tensors={'user_id': tf.keras.Input(shape=(1,), dtype=tf.int64, name='user_id')},
        # 或用 raw tensors：
        # features=tf.keras.Input(shape=(1,), dtype=tf.int64),
        # receiver_tensors=...
    )
```

**在搜广推哪里见过**：线上模型导出版本不对（input name 不匹配）是 TF Serving 上线时的头号问题，几乎每个算法工程师都踩过。

---

## 5. TensorBoard 使用指南

### 5.1 安装与启动

```bash
# 安装（独立包，无需装全部 tf）
pip install tensorboard

# 启动（指定 logdir 和端口）
tensorboard --logdir ./logs --port 6006

# 多实验对比：不同实验写到同一 --logdir 的不同子目录
# runs/exp1, runs/exp2, runs/exp3...
tensorboard --logdir ./runs --port 6006
```

### 5.2 PyTorch 用法

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/exp1')

# 标量曲线
writer.add_scalar('train/loss', loss_value, global_step)
writer.add_scalar('eval/auc', auc_value, global_step)

# 直方图（权重/梯度分布）
writer.add_histogram('layer1/weights', layer1.weight, global_step)

# 图像
writer.add_image('input/image', img_tensor, global_step)

# Embedding 可视化
writer.add_embedding(features, metadata=labels, global_step=global_step)

writer.flush()
```

### 5.3 TF/Keras 用法

```python
# 方式1：Keras Callback（推荐）
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,      # 每 1 个 epoch 记录一次权重直方图
    write_graph=True,     # 记录计算图
    update_freq='epoch',  # 或 'batch'
)
model.fit(x_train, y_train, epochs=10, callbacks=[tb_callback])

# 方式2：TF Summary API
writer = tf.summary.create_file_writer('./logs')
with writer.as_default():
    tf.summary.scalar('train/loss', loss_value, step=global_step)
    tf.summary.histogram('layer1/weights', weights, step=global_step)
```

### 5.4 8个核心面板

| 面板 | 用途 | 常见指标 |
|------|------|---------|
| **Scalars** | 标量曲线（loss、AUC、LR） | train/loss, eval/auc, learning_rate |
| **Images** | 输入图片、attention 可视化 | input_image, attention_map |
| **Histograms** | 权重/梯度分布（判断梯度消失/爆炸） | layer1/weights, layer1/gradients |
| **Distributions** | Histogram 的另一种视图 | 同上 |
| **Graphs** | 计算图结构（调试用） | model.graph |
| **Embeddings** | embedding 降维可视化 | user_emb, item_emb |
| **HPARAMS** | 超参数组合对比 | lr, batch_size, embedding_dim |
| **PR Curves** | Precision-Recall 曲线（不平衡数据） | pctr/precision_recall |

### 5.5 常见坑

1. **路径错误**：`--logdir` 必须是绝对路径或相对于启动目录
2. **flush 不及时**：训练过程中想实时看曲线，加 `writer.flush()` 或设置 `update_freq='batch'`
3. **histogram 频率控制**：`histogram_freq=0` 关闭直方图（省磁盘）；频繁记录影响训练速度
4. **多实验覆盖**：同一个 `runs/` 目录下写多个实验会自动对比，非常方便

**在搜广推哪里见过**：面试时会问"你线上模型效果怎么观察的"，TensorBoard 是标准答案之一。HPARAMS 面板在调参阶段（搜广推场景学习率、embedding维度对AUC的影响）非常有用。

---

## 6. 下一步

- **Demo 实战** → `code/01_keras_basic_training.py`（Keras 训练循环）
- **特征工程** → `code/02_feature_column_ctr.py`（feature_column CTR 预估）
- **工程化** → `code/03_estimator_and_saved_model.py`（Estimator + TF Serving）
- **扩展资源** → `resources.md`
