"""
自定义 tf.estimator + SavedModel 导出（老项目维护必备）

主题：手写 model_fn 实现 TRAIN/EVAL/PREDICT 三分支，
      构建 Embedding + 内积 + MLP 的 CTR 模型，
      使用 tf.estimator.train_and_evaluate 训练，
      并通过 serving_input_receiver_fn 导出 TF Serving 模型。

运行：python 03_estimator_and_saved_model.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf


# ==================== 1. 生成 Toy 数据 ====================

def generate_toy_data(num_samples=50000, user_vocab=5000, item_vocab=10000, seed=42):
    """
    生成模拟 CTR 数据
    """
    np.random.seed(seed)
    
    user_ids = np.random.randint(0, user_vocab, size=num_samples)
    item_ids = np.random.randint(0, item_vocab, size=num_samples)
    
    # 简单点击率模型：user_emb @ item_emb + bias
    click_probs = 0.1 + 0.05 * (user_ids % 10) / 10 + 0.05 * (item_ids % 10) / 10
    click_probs = np.clip(click_probs, 0.01, 0.4)
    labels = (np.random.random(num_samples) < click_probs).astype(np.float32)
    
    df = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids, 'label': labels})
    return df


# ==================== 2. 自定义 model_fn ====================

def model_fn(features, labels, mode, params):
    """
    自定义 Estimator 模型函数
    
    Args:
        features: dict of tensors，输入特征
        labels: tensor，标签
        mode: tf.estimator.ModeKeys（TRAIN/EVAL/PREDICT）
        params: dict，超参数
    
    Returns:
        EstimatorSpec（根据 mode 返回不同分支）
    """
    
    # ==================== 前向传播 ====================
    user_id = features['user_id']           # [batch_size]
    item_id = features['item_id']           # [batch_size]
    
    # Embedding 查表
    user_emb = tf.keras.layers.Embedding(
        params['user_vocab'], params['emb_dim'],
        embeddings_initializer=tf.random_uniform_initializer(-0.01, 0.01)
    )(user_id)                               # [batch_size, emb_dim]
    
    item_emb = tf.keras.layers.Embedding(
        params['item_vocab'], params['emb_dim'],
        embeddings_initializer=tf.random_uniform_initializer(-0.01, 0.01)
    )(item_id)                               # [batch_size, emb_dim]
    
    # Embedding 内积（一阶交互）
    interaction = tf.reduce_sum(user_emb * item_emb, axis=-1, keepdims=True)  # [batch_size, 1]
    
    # MLP 二阶交互 + 分类
    concat = tf.concat([user_emb, item_emb], axis=-1)  # [batch_size, 2*emb_dim]
    x = tf.keras.layers.Dense(128, activation='relu')(concat)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(1)(x)  # logits
    
    # 最终预测 = sigmoid(logits + 内积)
    logits = x + interaction
    pctr = tf.nn.sigmoid(logits, name='pctr')  # [batch_size, 1]
    
    # ==================== TRAIN 分支 ====================
    if mode == tf.estimator.ModeKeys.TRAIN:
        # 计算损失
        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(labels, pctr))
        
        # 优化器
        optimizer = tf.optimizers.Adam(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
        
        # EstimatorSpec（TRAIN 模式需要 loss 和 train_op）
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )
    
    # ==================== EVAL 分支 ====================
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(labels, pctr))
        
        # 自定义评估指标：AUC
        auc_metric = tf.metrics.auc(labels, pctr)
        
        # eval_metric_ops 要求 value 和 update_op 两个 op
        eval_metric_ops = {
            'auc': (auc_metric[0], auc_metric[1])
        }
        
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )
    
    # ==================== PREDICT 分支 ====================
    else:
        # PREDICT 模式返回 predictions
        predictions = {
            'user_id': user_id,
            'item_id': item_id,
            'pctr': pctr,
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )


# ==================== 3. Serving Input Receiver ====================

def serving_input_receiver_fn():
    """
    定义 TF Serving 接收的输入格式
    
    线上推理时，TF Serving 会将 JSON/Proto 输入映射到这里的 placeholder
    
    Returns:
        ServingInputReceiver
    """
    # 定义输入特征的 feature_spec
    # 线上传的是原始 user_id / item_id（int64）
    features = {
        'user_id': tf.compat.v1.placeholder(tf.int64, shape=[None, 1], name='user_id'),
        'item_id': tf.compat.v1.placeholder(tf.int64, shape=[None, 1], name='item_id'),
    }
    
    receiver_tensors = features.copy()
    
    return tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors
    )


# ==================== 4. 主流程 ====================

if __name__ == '__main__':
    
    print("=" * 60)
    print("自定义 tf.estimator + SavedModel 导出 Demo")
    print("=" * 60)
    
    # ==================== 超参数 ====================
    USER_VOCAB = 5000
    ITEM_VOCAB = 10000
    EMB_DIM = 16
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 256
    TRAIN_STEPS = 1000
    EVAL_STEPS = 200
    
    # 模型参数（传入 model_fn）
    params = {
        'user_vocab': USER_VOCAB,
        'item_vocab': ITEM_VOCAB,
        'emb_dim': EMB_DIM,
        'learning_rate': LEARNING_RATE
    }
    
    # ==================== 1. 生成数据 ====================
    print("\n[1] 生成 Toy CTR 数据...")
    df = generate_toy_data(num_samples=50000, user_vocab=USER_VOCAB, item_vocab=ITEM_VOCAB)
    print(f"    数据形状: {df.shape}")
    print(f"    点击率: {df['label'].mean():.4f}")
    
    # 分割训练/验证
    val_size = int(len(df) * 0.2)
    df_train = df.iloc[val_size:]
    df_val = df.iloc[:val_size]
    print(f"    训练集: {len(df_train)} 条")
    print(f"    验证集: {len(df_val)} 条")
    
    # ==================== 2. 创建 Estimator ====================
    print("\n[2] 创建 Estimator...")
    model_dir = './estimator_model_dir'
    os.makedirs(model_dir, exist_ok=True)
    
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        model_dir=model_dir,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=100,
            log_step_count_steps=50
        )
    )
    print(f"    Estimator model_dir: {model_dir}")
    
    # ==================== 3. 准备输入函数 ====================
    def train_input_fn():
        """训练输入函数"""
        dataset = tf.data.Dataset.from_tensor_slices((
            {'user_id': df_train['user_id'].values, 'item_id': df_train['item_id'].values},
            df_train['label'].values
        ))
        dataset = dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def eval_input_fn():
        """评估输入函数"""
        dataset = tf.data.Dataset.from_tensor_slices((
            {'user_id': df_val['user_id'].values, 'item_id': df_val['item_id'].values},
            df_val['label'].values
        ))
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    # ==================== 4. TrainSpec / EvalSpec ====================
    print("\n[3] 配置 TrainSpec / EvalSpec...")
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=TRAIN_STEPS
    )
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=EVAL_STEPS,
        throttle_secs=30  # 至少 30 秒做一次评估
    )
    
    # ==================== 5. train_and_evaluate ====================
    print("\n[4] 执行 tf.estimator.train_and_evaluate...")
    print(f"    训练 {TRAIN_STEPS} 步，评估 {EVAL_STEPS} 步")
    
    # 注意：TF2.x 中 Estimator 需要关闭 v2 behavior 才能正常工作
    tf.compat.v1.disable_v2_behavior()
    
    results = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    print(f"\n    评估结果:")
    print(f"    - Loss: {results['loss']:.4f}")
    print(f"    - AUC: {results['auc']:.4f}")
    
    # ==================== 6. 导出 SavedModel（供 TF Serving 使用）====================
    print("\n[5] 导出 SavedModel（TF Serving）...")
    export_dir = './saved_model_dir/03_estimator_ctr'
    
    # 先启用 v2 behavior（否则 export 会失败）
    tf.compat.v1.enable_v2_behavior()
    
    # 导出
    estimator.export_saved_model(
        export_dir,
        serving_input_receiver_fn=serving_input_receiver_fn
    )
    print(f"    SavedModel 已导出至: {export_dir}")
    
    # ==================== 7. 加载 SavedModel 并推理 ====================
    print("\n[6] 加载 SavedModel 并推理...")
    
    # 列出导出的版本目录
    versions = [d for d in os.listdir(export_dir) if d.isdigit()]
    latest_version = max(versions)
    model_path = os.path.join(export_dir, latest_version)
    
    print(f"    加载模型版本: {latest_version}")
    
    loaded = tf.saved_model.load(model_path)
    infer = loaded.signatures['serving_default']
    
    # 打印签名
    print(f"    输入 key: {list(infer.structured_input_signature[1].keys())}")
    print(f"    输出 key: {list(infer.structured_outputs.keys())}")
    
    # 构造样本并推理
    sample_user_id = np.array([[1001]], dtype=np.int64)
    sample_item_id = np.array([[5001]], dtype=np.int64)
    
    result = infer(
        user_id=tf.constant(sample_user_id),
        item_id=tf.constant(sample_item_id)
    )
    
    pctr = result['pctr'].numpy()[0][0]
    print(f"\n    样本: user_id=1001, item_id=5001")
    print(f"    预测 CTR: {pctr:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("=" * 60)
