"""
Keras Functional API 训练循环 + Callbacks + SavedModel 导出/推理

主题：使用 Keras Functional API 构建图像分类模型，
      演示 callbacks（早停/学习率调度/模型保存/TensorBoard），
      以及 SavedModel 导出与加载推理。

运行：python 01_keras_basic_training.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.metrics import AUC, Accuracy

# ==================== 1. 构建模型（Functional API）====================

def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    使用 Functional API 构建卷积图像分类模型
    """
    # 输入层
    inputs = Input(shape=input_shape, name='image_input')
    
    # 卷积块1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    
    # 全连接层
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # 输出层
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='image_classifier')
    return model


# ==================== 2. 生成模拟数据 ====================

def generate_toy_data(num_samples=10000, input_shape=(28, 28, 1), num_classes=10):
    """
    生成模拟图像分类数据（用于演示，无需真实数据集）
    """
    x_train = np.random.randn(num_samples, *input_shape).astype(np.float32)
    x_train = (x_train - x_train.mean()) / (x_train.std() + 1e-7)  # 标准化
    
    y_train = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    
    x_val = np.random.randn(int(num_samples * 0.2), *input_shape).astype(np.float32)
    x_val = (x_val - x_val.mean()) / (x_val.std() + 1e-7)
    y_val = np.random.randint(0, num_classes, size=(int(num_samples * 0.2),)).astype(np.int32)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    
    return x_train, y_train, x_val, y_val


# ==================== 3. 训练配置 ====================

def get_callbacks(log_dir='./logs', save_dir='./checkpoints'):
    """
    获取训练 callbacks 列表
    
    - EarlyStopping: 监控 val_loss，patience=5 轮不降则停止
    - ReduceLROnPlateau: 监控 val_loss，patience=3 轮不降则降学习率
    - ModelCheckpoint: 保存 val_auc 最高的模型
    - TensorBoard: 记录训练指标和权重直方图
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    callbacks = [
        # 早停：val_loss 5 轮不降停止
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率调度：val_loss 3 轮不降则降低学习率
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # 模型保存：保存 val_auc 最高的模型
        ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.keras'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # TensorBoard：记录 scalar 和 histogram
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    return callbacks


# ==================== 4. SavedModel 导出与推理 ====================

def export_saved_model(model, export_dir='./saved_model_dir'):
    """
    将训练好的模型导出为 SavedModel 格式（TF Serving / tf.saved_model.load 直接使用）
    """
    os.makedirs(export_dir, exist_ok=True)
    
    # SavedModel 格式导出
    model.save(export_dir, save_format='tf')
    print(f"[导出] SavedModel 已保存至: {export_dir}")
    return export_dir


def load_and_infer_saved_model(export_dir, sample_input):
    """
    使用 tf.saved_model.load 加载 SavedModel 并推理
    
    Args:
        export_dir: SavedModel 路径
        sample_input: 形状为 (1, 28, 28, 1) 的 numpy 数组
    """
    # 加载模型
    loaded = tf.saved_model.load(export_dir)
    
    # 获取 serving default 签名
    infer = loaded.signatures['serving_default']
    
    # 打印输入输出 key
    print(f"[推理] 输入 key: {list(infer.structured_input_signature[1].keys())}")
    print(f"[推理] 输出 key: {list(infer.structured_outputs.keys())}")
    
    # 构造输入（注意 key 应与导出时的 Input layer name 一致）
    # 这里使用的是 'image_input'（build_model 中定义）
    result = infer(image_input=tf.constant(sample_input))
    
    # 解析结果
    probs = result['predictions'].numpy()
    pred_class = np.argmax(probs, axis=-1)
    
    print(f"[推理] 输入形状: {sample_input.shape}")
    print(f"[推理] 输出概率: {probs}")
    print(f"[推理] 预测类别: {pred_class}")
    
    return probs, pred_class


# ==================== 5. 主训练流程 ====================

if __name__ == '__main__':
    
    print("=" * 60)
    print("Keras Functional API 训练循环 Demo")
    print("=" * 60)
    
    # 超参数
    INPUT_SHAPE = (28, 28, 1)
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    
    # 1. 生成模拟数据
    print("\n[1] 生成模拟数据...")
    x_train, y_train, x_val, y_val = generate_toy_data(
        num_samples=5000, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES
    )
    print(f"    训练集: {x_train.shape}, 标签: {y_train.shape}")
    print(f"    验证集: {x_val.shape}, 标签: {y_val.shape}")
    
    # 2. 构建模型
    print("\n[2] 构建模型（Functional API）...")
    model = build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.summary()
    
    # 3. 编译模型
    print("\n[3] 编译模型...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[AUC(name='auc'), Accuracy(name='acc')]
    )
    
    # 4. 获取 callbacks
    callbacks = get_callbacks(
        log_dir='./logs/01_keras_basic',
        save_dir='./checkpoints/01_keras_basic'
    )
    
    # 5. 训练
    print("\n[4] 开始训练...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. 导出 SavedModel
    print("\n[5] 导出 SavedModel...")
    export_dir = export_saved_model(
        model,
        export_dir='./saved_model_dir/01_keras_basic'
    )
    
    # 7. 加载并推理
    print("\n[6] 加载 SavedModel 并推理...")
    # 构造一条样本
    sample = np.random.randn(1, *INPUT_SHAPE).astype(np.float32)
    sample = (sample - sample.mean()) / (sample.std() + 1e-7)
    
    probs, pred_class = load_and_infer_saved_model(export_dir, sample)
    
    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("=" * 60)
    print("\n启动 TensorBoard 查看训练曲线：")
    print("  tensorboard --logdir ./logs --port 6006")
