# 01_foundation / math · 数学基础

> 数学是广告/推荐算法工程师的「语言」——
> 理解 Embedding 需要线性代数，理解 CTR 预估的 loss 需要概率论，
> 理解 AdamW 需要微积分与凸优化。
> 本模块用**知识点笔记 + NumPy 代码 Demo**把抽象公式跑起来。

---

## 本模块目录

```
math/
├── README.md                 # 本文件：导航 + 学习路径
├── notes.md                  # 四大板块完整知识点（含「在搜广推哪里见过」）
├── resources.md              # 书籍 / 课程 / 可视化工具推荐
└── code/                     # 5 个独立可运行的 NumPy Demo
    ├── 01_linear_algebra_basics.py      # 向量/矩阵/秩/SVD/PCA
    ├── 02_matrix_calculus_chain_rule.py # 矩阵求导 + LR 梯度 + 数值验证
    ├── 03_probability_and_kl_divergence.py # 分布采样/MLE/熵/交叉熵/KL
    ├── 04_gradient_descent_variants.py   # GD vs SGD vs Momentum vs RMSProp vs Adam
    └── 05_adamw_weight_decay_intuition.py # AdamW：解耦权重衰减的直觉
```

---

## 内容大纲

### 一、线性代数（最高优先级）

| 知识点 | 与广告/推荐的直接关联 |
|--------|---------------------|
| 向量 / 矩阵 / 秩 | Embedding 就是向量；评分矩阵做 SVD 做降维 |
| **SVD / EVD** | 矩阵分解 (MF)、推荐系统中的低秩近似；LSA |
| 内积 / 余弦相似度 | 双塔召回的 score；Faiss / Milvus 向量检索 |
| **PCA** | Embedding 可视化（t-SNE 之前的预处理 baseline） |
| 矩阵求导 | 手推 LR / Softmax / FM 的梯度，理解反向传播 |

### 二、概率论 & 统计（高优先级）

| 知识点 | 与广告/推荐的直接关联 |
|--------|---------------------|
| 贝叶斯公式 / 全概率公式 | 朴素贝叶斯；A/B 测试；贝叶斯优化 |
| 期望 / 方差 / 协方差 | 特征相关分析；高方差 → 过拟合 |
| **伯努利 / Beta / Dirichlet** | CTR 先验是 Beta；Thompson Sampling；LDA 主题模型 |
| 中心极限定理 CLT | A/B 测试的置信区间计算 |
| **熵 / 交叉熵 / KL 散度** | 分类任务用的交叉熵 loss；推荐多样性的度量；GAN |
| **MLE / MAP** | 所有概率模型的参数估计方法 |

### 三、微积分（中优先级，聚焦梯度）

| 知识点 | 与广告/推荐的直接关联 |
|--------|---------------------|
| 梯度 / Hessian | 优化器核心；牛顿法；Hessian-free 优化 |
| **链式法则** | 反向传播 Backpropagation 的数学基础 |
| 泰勒展开 | Adam / RMSProp 等优化器的理论背景 |
| 拉格朗日乘数法 | SVM 对偶问题；约束优化 |

### 四、优化理论（高优先级，直接跟训练挂钩）

| 知识点 | 与广告/推荐的直接关联 |
|--------|---------------------|
| **GD / SGD / Mini-batch SGD** | 所有 NN 训练的起点 |
| **Momentum / RMSProp / Adam / AdamW** | Transformer / DeepFM 的标配；AdamW 是现代首选 |
| 坐标下降 / 近端梯度 | Lasso；稀疏模型 FTRL-Proximal |
| KKT 条件 | SVM 对偶；理解最优解的一阶条件 |

---

## 推荐的学习路径

### 第一遍（快速扫，建立大局观）
1. 通读 [notes.md](notes.md) 的四大板块，把每个知识点的「在搜广推哪里见过」读一遍 → 建立「数学 → 业务」的映射
2. 跑通 [code/](code/) 的 5 个 Demo，每个 2–5 分钟 → 把抽象公式「跑」成数字

### 第二遍（深入理解，手推关键公式）
1. 手推 LR 的梯度：$\nabla_w L = X^T (p - y) / N$
2. 手推 SVD 的 $U, \Sigma, V^T$ 分解与低秩近似
3. 手推 AdamW 的更新公式，理解「解耦权重衰减」为什么重要
4. 挑一篇你工作相关的论文（比如 DeepFM 或 DIN），把公式部分读一遍

### 第三遍（结合业务问题，持续应用）
- 做 A/B test 时，想清楚中心极限定理如何提供置信区间
- 调 AdamW 的 lr / weight decay 时，回忆「weight decay 是在每步把 w 乘 (1 - lr * λ)」
- 遇到模型不收敛，先想「是优化器问题、学习率问题，还是数据问题」

---

## 学习优先级建议（按业务方向）

| 你的主要工作 | 重点板块 |
|------------|---------|
| **CTR/CVR 精排模型**（DeepFM / DIN / MMoE） | 一 + 二 + 四 |
| **召回 / 向量检索**（双塔 / SVD / GraphSAGE） | 一（重点）+ 二 |
| **A/B 测试 & 实验平台** | 二（重点）+ 四 |
| **优化器 / 训练加速**（大模型 / 分布式） | 四（重点）+ 三 |

---

## 下一步

- 看完 [notes.md](notes.md) 后可以开始写自己的学习笔记（附在 notes.md 末尾）
- 看完 5 个 code Demo 后，建议用你工作中的真实数据重写一遍（例如拿一份真实的用户-物品评分矩阵做 SVD）
- 下一模块：**01_foundation/statistics**（贝叶斯推断 / 假设检验 / A/B 测试）
