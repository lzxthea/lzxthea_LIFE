# TensorFlow 知识点笔记 · 搜广推工程师指南

本模块涵盖 TensorFlow 在推荐系统/广告系统中工程落地的核心知识点与 Demo。

---

## 目录结构

```
tensorflow/
├── README.md              # 本文件
├── notes.md               # TensorFlow 知识点笔记
├── resources.md           # 扩展学习资源
└── code/
    ├── 01_keras_basic_training.py       # Keras 训练 + SavedModel 导出
    ├── 02_feature_column_ctr.py          # feature_column CTR 预估
    └── 03_estimator_and_saved_model.py  # Estimator + TF Serving
```

---

## 场景 → 方案映射

| 场景 | 推荐方案 | 对应文件 |
|------|---------|---------|
| 新项目快速建模 | tf.keras + Sequential/Functional API | `01_keras_basic_training.py` |
| CTR/CVR 预估特征处理 | tf.feature_column + DenseFeatures | `02_feature_column_ctr.py` |
| 老项目维护/线上 Serving | tf.estimator + export_saved_model | `03_estimator_and_saved_model.py` |
| 分布式训练（十亿级样本） | tf.distribute.MirroredStrategy / PSStrategy | `notes.md` §2.4 |
| 线上推理服务 | SavedModel + TF Serving | `01_keras_basic_training.py` / `03_estimator_and_saved_model.py` |
| 训练过程可视化 | TensorBoard | `notes.md` §5 |

---

## 运行方式

### 环境准备

```bash
pip install tensorflow
pip install pandas numpy  # 数据处理
pip install tensorboard    # 可视化（独立包）
```

### 运行 Demo

```bash
# 1. Keras 训练 + SavedModel 导出
python code/01_keras_basic_training.py

# 2. feature_column CTR 预估
python code/02_feature_column_ctr.py

# 3. Estimator + TF Serving
python code/03_estimator_and_saved_model.py
```

### 查看 TensorBoard

```bash
# 训练时自动写入 logs/
tensorboard --logdir ./logs --port 6006
```

---

## 知识点速查

| 知识点 | 位于 |
|--------|------|
| TF1.x vs TF2.x 对比 | `notes.md` §1.1 |
| Keras 三种建模方式 | `notes.md` §1.2 |
| feature_column 6 种列类型 | `notes.md` §2.1 |
| tf.data 标准 Pipeline | `notes.md` §2.2 |
| estimator model_fn 三分支 | `notes.md` §2.3 |
| 分布式策略（Mirrored/PS） | `notes.md` §2.4 |
| SavedModel + TF Serving | `notes.md` §2.5 |
| DeepFM 二阶交互公式 | `notes.md` §3.2 |
| DIN Attention 实现 | `notes.md` §3.3 |
| MMoE 多任务架构 | `notes.md` §3.4 |
| 双塔召回 DSSM | `notes.md` §3.5 |
| TF1→TF2 迁移要点 | `notes.md` §4 |
| BatchNorm training=True/False | `notes.md` §4.3 |
| TensorBoard 8 面板 | `notes.md` §5.4 |

---

## 下一步

- **扩展资源** → `resources.md`
- **PyTorch 对比** → 同级目录 `pytorch/`
- **特征工程专题** → `../features/`
