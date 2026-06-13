# 框架模块：PyTorch 与 TensorFlow 核心技能

本模块聚焦广告/推荐算法工程师必须掌握的深度学习框架核心技能，以 PyTorch 为主，TensorFlow 为辅。

## 目录结构

```
frameworks/
├── README.md
├── resources.md
├── notes.md                        # 框架核心概念笔记
├── code/                           # PyTorch Demo（5个）
│   ├── 01_tensor_basics.py         # 张量操作与自动求导
│   ├── 02_nn_module.py             # 神经网络模块与层
│   ├── 03_training_loop.py         # 训练循环与优化
│   ├── 04_pretrained_models.py     # 预训练模型与Fine-tuning
│   └── 05_recommender_demo.py      # 推荐系统简易实现
└── tensorflow/                     # TensorFlow Demo（3个）
    ├── 01_tf_basics.py             # TF2 张量与自动微分
    ├── 02_keras_api.py             # Keras High-Level API
    └── 03_wide_deep.py             # Wide&Deep 模型实现
```

## PyTorch 5个 Demo

| 序号 | 文件名 | 主题 |
|------|--------|------|
| 1 | `01_tensor_basics.py` | 张量操作与自动求导 |
| 2 | `02_nn_module.py` | 神经网络模块与层 |
| 3 | `03_training_loop.py` | 训练循环与优化 |
| 4 | `04_pretrained_models.py` | 预训练模型与Fine-tuning |
| 5 | `05_recommender_demo.py` | 推荐系统简易实现 |

## TensorFlow 3个 Demo

| 序号 | 文件名 | 主题 |
|------|--------|------|
| 1 | `01_tf_basics.py` | TF2 张量与自动微分 |
| 2 | `02_keras_api.py` | Keras High-Level API |
| 3 | `03_wide_deep.py` | Wide&Deep 模型实现 |

## 框架对比速览

| 维度 | PyTorch | TensorFlow |
|------|---------|------------|
| 编程风格 | 动态图，Pythonic | 静态图（TF1）/动态图（TF2） |
| API 设计 | 简洁直观 | Keras 封装后较简洁 |
| 部署生态 | TorchScript, ONNX | TF Serving, TF Lite, SavedModel |
| 推荐系统支持 | DeepCTR-Torch, DeepMatch | TF Recommenders, DeepCTR-TF |
| 工业部署成熟度 | 中（正在追赶） | 高（TF Serving 成熟） |
| 学习曲线 | 较平缓 | 较陡（TF1历史包袱） |
| 社区活跃度 | 非常高 | 高 |
| GPU 加速 | 原生 CUDA | 原生 CUDA |

## 广告/推荐工程师的框架选型建议

1. **主攻方向：PyTorch**
   - 推荐算法实验、快速原型开发
   - 论文复现、研究创新
   - 当下业界主流选择

2. **补充方向：TensorFlow**
   - TensorFlow Serving 模型部署
   - TF Recommenders 库学习
   - TFX 完整 pipeline 构建

3. **选型原则**
   - 小团队/实验阶段：优先 PyTorch
   - 大规模工业部署：了解 TF Serving
   - 兼顾两者：先精通一个，再了解另一个

## 学习路径建议

1. **先读 `notes.md`**：理解框架核心概念（计算图、张量、自动求导、模块化设计）
2. **再看 `code/` 目录**：通过 PyTorch Demo 动手实践
3. **最后 `tensorflow/` 子目录**：对比学习 TensorFlow，特别是 Wide&Deep 实现

## 下一步

- 完成本模块所有 Demo 代码
- 阅读 `resources.md` 中的推荐资料
- 开始 `model_architectures/` 模块学习（经典推荐模型）
