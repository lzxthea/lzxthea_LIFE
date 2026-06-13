# 数学基础 · 知识点笔记

**前言**

广告与推荐算法工程师的日常，本质上是在"建模用户行为 → 优化决策 → 评估效果"这三个环节里反复打转。无论是 CTR/CVR 预估中的逻辑回归、推荐召回中的矩阵分解、还是广告出价中的 Bandit，背后都有数学作为骨架。很多工程师写公式时对符号不敏感（梯度方向、正则化项、KL 散度的不对称性），结果就是看论文慢、调参靠直觉、出问题查半天。这份笔记的目标是：**把你在论文和代码里反复遇到的数学符号，做一次集中式的"落地"整理**。不追求教科书级别的证明，但保证你看到 EVD、SVD、AdamW、KL divergence 这些词时，能立刻想起"这东西在搜广推里干过什么"。

---

## 一、线性代数

### 1.1 向量 & 矩阵运算 & 矩阵的秩

向量 $\boldsymbol{x} \in \mathbb{R}^d$ 可以理解为"一个对象的 d 维数字描述"——在推荐里它可能是 user embedding、item embedding 或 某个特征的 one-hot。矩阵 $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ 是线性变换：把 $\mathbb{R}^n$ 里的向量映射到 $\mathbb{R}^m$ 里。

矩阵加法、数乘、矩阵乘法是基础，关键是理解**矩阵乘法的三种视角**：

1. 按列：$\boldsymbol{A}\boldsymbol{x}$ 是 $\boldsymbol{A}$ 的列向量的线性组合；
2. 按行：$\boldsymbol{A}\boldsymbol{x}$ 每行是 $\boldsymbol{A}$ 对应行向量与 $\boldsymbol{x}$ 的内积；
3. 外积展开：$\boldsymbol{A}\boldsymbol{B} = \sum_k \boldsymbol{A}_{:,k} \boldsymbol{B}_{k,:}^T$。

矩阵的秩 rank($\boldsymbol{A}$) 是矩阵列空间/行空间的维度。**rank 越低 = 信息越冗余 = 越容易压缩/低秩近似**。

在搜广推哪里见过：

- **FM (Factorization Machine)**：交互项 $w_{ij} x_i x_j$ 写成 $\boldsymbol{v}_i^T \boldsymbol{v}_j x_i x_j$，把一个满秩交互矩阵强行压成低秩，参数量从 $O(n^2)$ 降到 $O(kn)$；
- **Deep & Wide / Wide & Deep**：Wide 部分的线性权重向量 $\boldsymbol{w}$ 本质上就是一个特征权重向量，Deep 部分把离散特征 embedding 成向量后做矩阵乘法；
- **Embedding 召回 (如 DSSM)**：query embedding $\boldsymbol{u}$ 与 doc embedding $\boldsymbol{v}$ 做内积或余弦相似度，就是典型的向量点积打分；
- **矩阵低秩近似**：在新闻推荐或社交网络中，用户-物品点击矩阵 $\boldsymbol{R}$ 的低秩近似 $\boldsymbol{R} \approx \boldsymbol{U}\boldsymbol{V}^T$，直接用 rank 来解释需要保留多少个隐因子。

---

### 1.2 特征值分解 (EVD) & 奇异值分解 (SVD)

对**实对称方阵** $\boldsymbol{A} \in \mathbb{R}^{n \times n}$，特征值分解 (Eigenvalue Decomposition, EVD) 为：

$$\boldsymbol{A} = \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^T$$

其中 $\boldsymbol{Q}$ 的列是正交的单位特征向量，$\boldsymbol{\Lambda} = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$ 是特征值对角阵。

对**任意矩阵** $\boldsymbol{A} \in \mathbb{R}^{m \times n}$，奇异值分解 (Singular Value Decomposition, SVD) 为：

$$\boldsymbol{A} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^T$$

其中 $\boldsymbol{U} \in \mathbb{R}^{m \times r}, \boldsymbol{V} \in \mathbb{R}^{n \times r}$ 列正交，$\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1, \dots, \sigma_r)$ 是非负奇异值。$\sigma_i$ 是 $\boldsymbol{A}\boldsymbol{A}^T$ 特征值的平方根。

**直观理解**：EVD/SVD 把一个变换"拧回"到正交基上，让每个方向的"拉伸强度"一目了然。取前 k 个最大奇异值，就是最优 rank-k 近似（在 Frobenius 范数意义下）。

在搜广推哪里见过：

- **MF (Matrix Factorization) / SVD 推荐**：经典的 NetFlix 问题里，用户-物品评分矩阵 $\boldsymbol{R} \approx \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T$，SVD 是最早一批协同滤波方法；
- **LSA (潜在语义分析)**：在文本推荐/搜索引擎中，词-文档共现矩阵 $\boldsymbol{A}$ 做 SVD，前 k 个奇异向量就是"话题向量"；
- **特征预处理（相关性去重）**：特征矩阵做 SVD 或 EVD 后丢弃小奇异值方向，等价于处理高度共线特征，减轻 LR/FM 训练的数值问题；
- **Graph Embedding 预训练**：对邻接矩阵或转移矩阵做谱分解（本质是拉普拉斯矩阵的 EVD），对应早期的 Laplacian Eigenmaps。

---

### 1.3 向量内积 / 外积 / 范数 / 余弦相似度

向量 $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d$：

- 内积（点积）：$\boldsymbol{x}^T \boldsymbol{y} = \sum_i x_i y_i$，衡量"方向相似性 + 幅度"；
- 外积：$\boldsymbol{x}\boldsymbol{y}^T \in \mathbb{R}^{d \times d}$，得到一个 rank ≤ 1 的矩阵；
- $L_p$ 范数：$\|\boldsymbol{x}\|_p = (\sum_i |x_i|^p)^{1/p}$，常用 $p=1, 2, \infty$；
- Frobenius 范数：$\|\boldsymbol{A}\|_F = \sqrt{\sum_{ij} A_{ij}^2}$；
- 余弦相似度：

$$\cos(\boldsymbol{x}, \boldsymbol{y}) = \frac{\boldsymbol{x}^T \boldsymbol{y}}{\|\boldsymbol{x}\|_2 \|\boldsymbol{y}\|_2}$$

它只衡量方向相似度，跟向量长短无关。

在搜广推哪里见过：

- **两塔召回模型（DSSM /双塔）**：user 塔输出 $\boldsymbol{u}$，item 塔输出 $\boldsymbol{v}$，线上用内积或 cosine 做 ANN 近似最近邻搜索（Faiss、HNSW 都靠这个）；
- **Word2Vec / Item2Vec**：skip-gram 的目标是让中心词向量与上下文词向量的内积变大（通过 softmax 归一化后）；
- **L1 / L2 正则化**：LR / FM / Deep 模型里 $\|\boldsymbol{w}\|_1$（稀疏化）或 $\|\boldsymbol{w}\|_2^2$（权重衰减），就是向量范数直接当约束；
- **Attention（Transformer 族）**：Query 与 Key 的相似度就是 scaled dot-product，本质是内积除以 $\sqrt{d_k}$。

---

### 1.4 PCA 主成分分析

给定零均值数据矩阵 $\boldsymbol{X} \in \mathbb{R}^{n \times d}$（每行一个样本），协方差矩阵 $\boldsymbol{\Sigma} = \frac{1}{n-1}\boldsymbol{X}^T\boldsymbol{X}$。PCA 找一组正交方向，使数据在这些方向上的方差最大。等价于对 $\boldsymbol{\Sigma}$ 做 EVD：

$$\boldsymbol{\Sigma} = \boldsymbol{W} \boldsymbol{\Lambda} \boldsymbol{W}^T$$

取前 k 个最大特征值对应的特征向量构成投影矩阵 $\boldsymbol{W}_k$，降维后的数据为 $\boldsymbol{Z} = \boldsymbol{X}\boldsymbol{W}_k$。

也可以通过对 $\boldsymbol{X}$ 做 SVD 得到同样结果（数值上更稳）。

在搜广推哪里见过：

- **特征降维**：广告系统里 raw features 有时几万维，先 PCA 到几百维再喂入 GBDT / FM，减少计算量和噪声；
- **Embedding 可视化**：把几百维 user embedding 用 PCA 压到 2D/3D，看用户群体是否可分；
- **异常检测**：在线反作弊中，用 PCA 重构误差（样本在主成分外的残差能量）做离群点识别；
- **多路召回后的特征融合**：不同召回通路输出的 embedding 拼在一起后，PCA 用来去冗余再送给精排。

---

### 1.5 矩阵求导（链式法则 + 常用公式表）

机器学习里经常要求 $\frac{\partial L}{\partial \boldsymbol{W}}$ 或 $\frac{\partial L}{\partial \boldsymbol{x}}$，只要记住"标量对向量求导得到同形状向量，标量对矩阵求导得到同形状矩阵"即可（采用**分子布局**还是**分母布局**要统一，这里约定用分母布局：$\nabla_{\boldsymbol{x}} f \in \mathbb{R}^d$）。

**常用公式**（$\boldsymbol{A}$ 为常矩阵，$\boldsymbol{x}, \boldsymbol{y}$ 为向量，$f, g$ 为标量函数）：

| 函数 | 导数 |
|---|---|
| $f = \boldsymbol{a}^T \boldsymbol{x}$ | $\nabla_{\boldsymbol{x}} f = \boldsymbol{a}$ |
| $f = \boldsymbol{x}^T \boldsymbol{x}$ | $\nabla_{\boldsymbol{x}} f = 2\boldsymbol{x}$ |
| $f = \boldsymbol{x}^T \boldsymbol{A} \boldsymbol{x}$ | $\nabla_{\boldsymbol{x}} f = (\boldsymbol{A} + \boldsymbol{A}^T)\boldsymbol{x}$ |
| $f = \|\boldsymbol{x}\|_2^2$ | $\nabla_{\boldsymbol{x}} f = 2\boldsymbol{x}$ |
| $L = f(\boldsymbol{y}),\ \boldsymbol{y} = \boldsymbol{A}\boldsymbol{x}$ | $\nabla_{\boldsymbol{x}} L = \boldsymbol{A}^T \nabla_{\boldsymbol{y}} f$（链式法则） |
| $L = f(\boldsymbol{Y}),\ \boldsymbol{Y} = \boldsymbol{X}\boldsymbol{W}$ | $\nabla_{\boldsymbol{W}} L = \boldsymbol{X}^T \nabla_{\boldsymbol{Y}} f$ |

**链式法则的直觉**：上游梯度一路"反传"，每层乘上该层的雅可比（或它的转置，取决于布局）。

在搜广推哪里见过：

- **LR 反向传播**：$\sigma'(z) = \sigma(z)(1-\sigma(z))$ 配合链式法则求 $\nabla_{\boldsymbol{w}} L$，就是 SGD 更新公式；
- **DLRM / DIN / SIM 等推荐模型**：embedding 层、MLP、Attention 的梯度全靠链式法则一层层 backprop；
- **FM 中对隐向量 $\boldsymbol{v}_i$ 的梯度**：交叉项对 $\boldsymbol{v}_i$ 求导后形式紧凑，这是 FM 能高效训练的关键；
- **Adam 等优化器内部**：损失对参数矩阵 $\boldsymbol{W}$ 的梯度 $\nabla_{\boldsymbol{W}} L$ 是第一公民，之后再做动量/自适应学习率处理。

---

## 二、概率论与统计学

### 2.1 概率公理 / 条件概率 / 贝叶斯公式 / 全概率公式

三条 Kolmogorov 公理：$P(A) \geq 0$，$P(\Omega) = 1$，可数可加。基础但不能跳过——所有概率建模都在这个地基上。

条件概率：

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)},\quad P(B) > 0$$

全概率公式（样本空间 $B_1, \dots, B_n$ 是一个划分）：

$$P(A) = \sum_i P(A \mid B_i) P(B_i)$$

贝叶斯公式（Bayes' theorem）：

$$P(B_j \mid A) = \frac{P(A \mid B_j) P(B_j)}{\sum_i P(A \mid B_i) P(B_i)}$$

**一句话**：先验 $P(B_j)$ + 似然 $P(A \mid B_j)$ → 后验 $P(B_j \mid A)$。

在搜广推哪里见过：

- **贝叶斯平滑 / 点击率修正**：小广告位曝光很少，直接用 clicks/impressions 做 CTR 会很抖。引入 Beta 先验 $Beta(\alpha, \beta)$，后验均值为 $\frac{C+\alpha}{C+V+\alpha+\beta}$，用于平滑 CTR 估计；
- **Naive Bayes 分类器（早期 CTR 预估）**：$P(click \mid features) \propto P(click) \prod P(feature_i \mid click)$，做邮件广告/文本广告分类；
- **贝叶斯 Bandit（Thompson Sampling）**：对每个 arm 的奖励后验分布（通常 Beta 或 Gaussian）做采样，选择最大样本的 arm，广告投放 A/B test 中常用；
- **EM 算法的 E 步**：隐变量 $Z$ 的后验 $P(Z \mid X, \theta_{old})$ 就是靠 Bayes 公式求出来的（见 GMM / pLSA）。

---

### 2.2 随机变量 / 期望 / 方差 / 协方差矩阵

随机变量 $X$ 把样本点映射到数。离散情况 $E[X] = \sum x_i p(x_i)$，连续情况 $E[X] = \int x f(x) dx$。

方差与标准差：

$$\mathrm{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2,\quad \sigma = \sqrt{\mathrm{Var}(X)}$$

协方差：$\mathrm{Cov}(X, Y) = E[(X-E[X])(Y-E[Y])]$。对随机向量 $\boldsymbol{X} \in \mathbb{R}^d$，协方差矩阵 $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$：

$$\boldsymbol{\Sigma}_{ij} = \mathrm{Cov}(X_i, X_j)$$

$\boldsymbol{\Sigma}$ 半正定对称，对角线是各分量方差。

在搜广推哪里见过：

- **CTR 点估计 & 置信区间**：每个广告位的 CTR 是个 Bernoulli 的 $p$，估计方差 $\hat{p}(1-\hat{p})/n$ 直接决定 A/B test 是否显著；
- **特征相关性分析**：协方差矩阵/相关系数矩阵用来判断两个特征是否高度线性相关，决定要不要做特征选择或 PCA；
- **Gaussian Embedding**：把每个 item 表示成一个正态分布 $N(\mu_i, \sigma_i^2)$，用 KL 散度做相似度（视频推荐/图像检索的不确定性建模）；
- **Bandit 的 UCB 指标**：上置信界 $\mu + c \cdot \sqrt{\frac{2\ln n}{t}}$ 的第二项本质上是均值估计的标准差。

---

### 2.3 常用分布：伯努利 / 二项 / 正态 / Beta / Dirichlet

**Bernoulli**：一次点击事件 $X \sim \mathrm{Ber}(p)$，$P(X=1)=p, P(X=0)=1-p$。

**Binomial**：$n$ 次独立伯努利中的成功次数，$X \sim B(n, p)$，$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$。

**Gaussian (Normal)**：$X \sim N(\mu, \sigma^2)$，密度：

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Beta**（定义在 [0,1] 上，共轭先验于 Bernoulli / Binomial）：

$$f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$$

**Dirichlet**（Beta 的多维推广，共轭先验于 Multinomial）：

$$f(\boldsymbol{x}; \boldsymbol{\alpha}) = \frac{\Gamma(\sum \alpha_i)}{\prod \Gamma(\alpha_i)} \prod_{k=1}^K x_k^{\alpha_k-1},\quad \sum x_k = 1$$

在搜广推哪里见过：

- **CTR 标签建模**：label 是 click / no-click → Bernoulli；一天内 $n$ 次曝光里的点击数 → Binomial；
- **贝叶斯平滑**：用 Beta($\alpha, \beta$) 做 CTR 的先验，加起来就得到 smoothed CTR；
- **假设检验**：Gaussian 假设是 A/B test 的主力（Z-test / t-test），也用 CLT 近似支撑；
- **话题模型 (LDA)**：文档-话题分布 $\theta_d \sim \mathrm{Dir}(\alpha)$，话题-词分布 $\phi_k \sim \mathrm{Dir}(\beta)$，是 Dirichlet-Multinomial 的典型用法。

---

### 2.4 大数定律 + 中心极限定理

**大数定律 (LLN)**：i.i.d. 样本 $X_1, \dots, X_n$ 期望为 $\mu$，则 $\bar{X}_n = \frac{1}{n}\sum X_i \xrightarrow{} \mu$（依概率或几乎必然）。**"样本平均 → 真值"**。

**中心极限定理 (CLT)**：若 $\mathrm{Var}(X_i) = \sigma^2 < \infty$，则：

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

**"不管分布是什么，样本平均的分布在大样本下近似正态"**。

在搜广推哪里见过：

- **A/B test 显著性**：CLT 保证点击率差异近似正态，直接用 Z-test 就能判断新版是否显著提升；
- **曝光量阈值**：曝光 < 100 的广告不纳入 A/B 分析，因为 CLT 还没生效，样本均值不稳定；
- **offline metric 置信区间**：离线 AUC、gAUC 等指标的波动区间，用 bootstrap 或 CLT 近似给出；
- **SGD 的收敛性理解**：mini-batch 梯度是真实梯度的无偏估计，样本量够大时波动变小，这是 LLN/CLT 直觉的反复使用。

---

### 2.5 信息论基础：熵 / 交叉熵 / KL 散度

离散分布 $P$ 的**熵 (Entropy)** 衡量不确定性：

$$H(P) = -\sum_i p_i \log p_i$$

**交叉熵 (Cross Entropy)**：用分布 $Q$ 去编码来自 $P$ 的样本的平均编码长度：

$$H(P, Q) = -\sum_i p_i \log q_i$$

**Kullback-Leibler divergence (KL divergence)**：衡量两个分布"差多少"（不对称、非负）：

$$D_{\mathrm{KL}}(P \parallel Q) = H(P, Q) - H(P) = \sum_i p_i \log \frac{p_i}{q_i} \geq 0$$

当 $P=Q$ 时 $D_{\mathrm{KL}}=0$。注意 $D_{\mathrm{KL}}(P \parallel Q) \neq D_{\mathrm{KL}}(Q \parallel P)$。

在搜广推哪里见过：

- **二分类交叉熵损失**：LR / DNN 做 CTR 预估时，损失就是 $L = -\frac{1}{N}\sum [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$，本质是标签 Bernoulli $P$ 与模型预测 $Q$ 之间的 cross entropy；
- **多分类 softmax + cross entropy**：候选广告召回/精排多分类打分，$L = -\sum y_i \log \text{softmax}(f_i)$；
- **Word2Vec / 负采样**：skip-gram 的目标是最大化中心词与上下文词的"互信息"，可以写成减小真实分布 $P$ 与模型分布 $Q$ 之间的 KL；
- **Variational Autoencoder (VAE) 推荐**：损失 = 重构损失 + $D_{\mathrm{KL}}(q(z \mid x) \parallel p(z))$，约束后验不要离先验太远。

---

### 2.6 MLE 极大似然估计 & MAP 极大后验估计

给定样本 $\mathcal{D} = \{x_1, \dots, x_n\}$，**MLE** 选择使似然最大的参数：

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta}\ P(\mathcal{D} \mid \theta) = \arg\max_{\theta}\ \sum \log P(x_i \mid \theta)$$

**MAP** 引入先验 $P(\theta)$，最大化后验：

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta}\ P(\theta \mid \mathcal{D}) = \arg\max_{\theta}\ \big[\log P(\mathcal{D} \mid \theta) + \log P(\theta)\big]$$

当 $P(\theta) = N(0, \lambda^{-1}I)$（高斯先验），MAP 等价于 MLE + L2 正则；当先验是 Laplace 分布，MAP 等价于 MLE + L1 正则。

在搜广推哪里见过：

- **LR / DNN 训练**：最小化交叉熵 = 最大化 Bernoulli 似然，标准的 MLE；加 $L2\ \|\boldsymbol{w}\|_2^2$ 项就是 MAP（高斯先验）；
- **贝叶斯平滑的另一种理解**：对 CTR 的 Beta 先验 + Binomial 似然 → Beta 后验，后验均值就是对 $p$ 的 MAP 估计；
- **朴素贝叶斯分类器**：对类先验 $P(C)$ 和条件分布 $P(x_i \mid C)$ 做频率估计，本质也是 MLE；
- **GBDT 中"对数损失"的含义**：GBDT 拟合负梯度，而负梯度来自"最小化 logistic 交叉熵 = 最大化 Bernoulli 似然"这一目标。

---

## 三、微积分

### 3.1 偏导 / 梯度 / 海森矩阵 / 雅可比矩阵

多变量函数 $f(x_1, \dots, x_d)$ 对 $x_i$ 的偏导记为 $\frac{\partial f}{\partial x_i}$。

**梯度 (Gradient)**：偏导数拼成的向量，指向函数增长最快的方向：

$$\nabla f(\boldsymbol{x}) = \left(\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_d}\right)^T$$

**海森矩阵 (Hessian)**：二阶偏导构成的对称方阵：

$$\boldsymbol{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

$\boldsymbol{H}$ 的正定性告诉我们临界点是极小/极大/鞍点。

**雅可比矩阵 (Jacobian)**：对向量值函数 $\boldsymbol{f}: \mathbb{R}^n \to \mathbb{R}^m$，$J_{ij} = \frac{\partial f_i}{\partial x_j}$。

在搜广推哪里见过：

- **SGD / Adam 的参数更新**：$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$，梯度是优化的第一公民；
- **牛顿法 / 自然梯度**：在某些在线学习里用 $\boldsymbol{H}^{-1} \nabla L$ 更新，收敛更快但计算代价高；广告大规模模型通常不直接用，但理解它可以让你看懂 TRPO / Natural gradient 论文；
- **FM / NFM / DeepFM 里的二阶交互**：Hessian 的非对角线项直觉上对应特征之间的"相互影响强度"；
- **对抗样本 / 对抗训练**：沿着梯度方向加扰动，看 CTR 预估模型是否稳，用于作弊检测/鲁棒性分析。

---

### 3.2 链式法则（标量 + 矩阵形式）

标量版：若 $y = f(g(x))$，则 $\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$。

向量版：若 $L = f(\boldsymbol{y}), \boldsymbol{y} = \boldsymbol{g}(\boldsymbol{x})$，则：

$$\nabla_{\boldsymbol{x}} L = J_{\boldsymbol{g}}^T\ \nabla_{\boldsymbol{y}} f$$

矩阵版（常见于 DL）：若 $\boldsymbol{Y} = \boldsymbol{X}\boldsymbol{W}$，$L = f(\boldsymbol{Y})$，则：

$$\nabla_{\boldsymbol{W}} L = \boldsymbol{X}^T \nabla_{\boldsymbol{Y}} f,\quad \nabla_{\boldsymbol{X}} L = (\nabla_{\boldsymbol{Y}} f)\ \boldsymbol{W}^T$$

这几条是 backprop 的全部，记熟即可。

在搜广推哪里见过：

- **MLP backpropagation**：从 loss 一路反传到 embedding 层，每层都用链式法则；
- **Attention / Transformer 反向传播**：softmax、残差、LayerNorm 每一段的雅可比都在链式法则下组合；
- **GBDT 的负梯度拟合**：每棵树拟合的是 $-\frac{\partial L}{\partial \hat{y}}$，这是链式法则在集成方法中的体现；
- **BPR (Bayesian Personalized Ranking)**：pairwise loss $L = -\log \sigma(\hat{y}_{u,i} - \hat{y}_{u,j})$ 对 user/item 向量求导，全靠链式法则展开。

---

### 3.3 泰勒展开

对多元函数 $f$ 在 $\boldsymbol{x}_0$ 附近做二阶近似：

$$f(\boldsymbol{x}_0 + \boldsymbol{\Delta}) \approx f(\boldsymbol{x}_0) + \nabla f(\boldsymbol{x}_0)^T \boldsymbol{\Delta} + \frac{1}{2} \boldsymbol{\Delta}^T \boldsymbol{H}(\boldsymbol{x}_0) \boldsymbol{\Delta}$$

只取前两项 = 线性近似；加上二次项 = 二次近似。

直觉：**任何光滑函数局部都像一个二次函数**，所以分析局部极小/极大时看 Hessian 就够了。

在搜广推哪里见过：

- **牛顿法更新步**：对 $f(\theta)$ 做二阶泰勒展开，令导数为零，得到 $\Delta \theta = -\boldsymbol{H}^{-1} \nabla f$；
- **L-BFGS / Trust Region**：不直接求逆 Hessian，但核心仍是"用二次模型近似目标函数"；
- **Bias-Variance 分解**：对期望损失 $E[(y - \hat{f}(x))^2]$ 做平方展开，得到 $\mathrm{Bias}^2 + \mathrm{Var} + \mathrm{noise}$，这是泰勒/平方展开的典型用法；
- **某些在线优化的稳定性分析**：用二阶信息判断 step-size 太大是否会发散（推荐系统的线上学习器常见）。

---

### 3.4 拉格朗日乘数法

等式约束优化问题：

$$\min_{\boldsymbol{x}} f(\boldsymbol{x}),\quad \text{s.t.}\ g_i(\boldsymbol{x}) = 0$$

引入拉格朗日函数 $L(\boldsymbol{x}, \boldsymbol{\lambda}) = f(\boldsymbol{x}) + \sum \lambda_i g_i(\boldsymbol{x})$，必要条件是 $\nabla_{\boldsymbol{x}} L = 0$，$\nabla_{\boldsymbol{\lambda}} L = 0$。

**直观理解**：约束曲面的法线与目标函数的梯度在最优解处必须共线——"沿着可行方向再走一步也不会变好"。

在搜广推哪里见过：

- **SVM 硬间隔**：$\min \frac{1}{2}\|\boldsymbol{w}\|^2$ s.t. $y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) \geq 1$，写成拉格朗日对偶后用 SMO 解；虽然搜广推里直接用 SVM 少，但它背后的对偶/拉格朗日框架大量出现；
- **Budget 约束下的广告出价**：在总花费上限 $\sum \mathrm{cost}_i \leq B$ 下最大化点击数，不等式约束会过渡到 KKT（见 4.5）；
- **PCA 推导**：在 $\|\boldsymbol{w}\|=1$ 约束下最大化方差 $\boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w}$，用拉格朗日乘数法推出主方向 = 最大特征向量；
- **最大熵模型 / 指数族**：在矩匹配约束下最大化熵，拉格朗日解的形式就是指数族分布（softmax 是特殊形式）。

---

## 四、优化理论

### 4.1 凸集 / 凸函数 / 凸优化基本概念

集合 $\mathcal{C}$ 是**凸集**：任意 $\boldsymbol{x}, \boldsymbol{y} \in \mathcal{C}, \theta \in [0,1]$，有 $\theta \boldsymbol{x} + (1-\theta)\boldsymbol{y} \in \mathcal{C}$。

函数 $f: \mathcal{C} \to \mathbb{R}$ 是**凸函数**（convex）：

$$f(\theta \boldsymbol{x} + (1-\theta)\boldsymbol{y}) \leq \theta f(\boldsymbol{x}) + (1-\theta) f(\boldsymbol{y})$$

二阶可导时等价于 $\boldsymbol{H}(\boldsymbol{x}) \succeq 0$（Hessian 半正定）。

**凸优化**：在凸集上最小化凸函数 → "任何局部最优都是全局最优"，这是能找到稳定解的最重要原因。

在搜广推哪里见过：

- **线性回归 / Logistic Regression / SVM**：都是凸优化问题（LR 的交叉熵 + L2 是强凸），所以我们确信用 GD 找到的就是全局最小；
- **FM 中线性部分 + 正则化**：线性项关于 $\boldsymbol{w}$ 是凸的；加上正则化保持凸性；注意 FM 的二阶隐向量参数整体上不再凸，但工程上仍能用 SGD 优化；
- **Lasso (L1) / Ridge (L2)**：Ridge 保持强凸，Lasso 虽然不可微但仍是凸（用次梯度）；特征选择/稀疏化常用；
- **广告分配的 LP 松弛**：把 0/1 整数约束放松成 [0,1] 得到 LP（线性规划，凸优化的子类），再 rounding，大规模投放系统常见。

---

### 4.2 梯度下降 / SGD / Mini-batch SGD

**梯度下降 (Gradient Descent, GD)**：

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

每次用全部样本算梯度。样本量 N 很大时一次迭代巨慢。

**随机梯度下降 (Stochastic Gradient Descent, SGD)**：每次只抽 1 个样本算梯度：

$$\theta_{t+1} = \theta_t - \eta_t \nabla \ell_i(\theta_t)$$

梯度是无偏估计但噪声大，通常需要 $\eta_t$ 衰减（如 $\eta_t = \eta_0 / \sqrt{t}$）。

**Mini-batch SGD**：每次抽大小为 B 的 batch 做梯度，兼顾稳定性和并行能力。这是现代深度学习的 default。

在搜广推哪里见过：

- **LR / FM / GBDT（部分实现）**：广告工业里 LR 用 SGD 在线增量训练是老传统；
- **DIN / DLRM / DIEN / SIM / MMoE 等主流 DNN 推荐模型**：清一色的 Adam/AdamW + mini-batch；
- **FTRL (Follow-the-Regularized-Leader)**：Google 经典的在线 LR 算法，本质是带 L1 正则的 SGD 变种，用于大规模 CTR 预估；
- **负采样训练**：Word2Vec / Item2Vec 每次采一条 (center, context, negative) 三元组更新，是典型的 SGD 思路。

---

### 4.3 Momentum / RMSProp / Adam / AdamW（重点讲解 AdamW）

**Momentum**：用指数加权平均积累历史梯度，帮助冲过局部极小和鞍点：

$$\boldsymbol{v}_t = \beta_1 \boldsymbol{v}_{t-1} + (1-\beta_1) \boldsymbol{g}_t,\quad \theta_{t+1} = \theta_t - \eta \boldsymbol{v}_t$$

**RMSProp**：对梯度的**平方**做 EWMA，用它来归一化每个参数的步长（稀疏特征自动获得更大步长）：

$$\boldsymbol{s}_t = \beta_2 \boldsymbol{s}_{t-1} + (1-\beta_2) \boldsymbol{g}_t \odot \boldsymbol{g}_t,\quad \theta_{t+1} = \theta_t - \eta \frac{\boldsymbol{g}_t}{\sqrt{\boldsymbol{s}_t} + \epsilon}$$

**Adam (Adaptive Moment Estimation)**：把 Momentum + RMSProp 合起来，并加 bias correction（因为 $\boldsymbol{v}_0, \boldsymbol{s}_0$ 是 0，早期有偏）：

$$
\begin{aligned}
\boldsymbol{m}_t &= \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1) \boldsymbol{g}_t \\
\boldsymbol{v}_t &= \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2) \boldsymbol{g}_t \odot \boldsymbol{g}_t \\
\hat{\boldsymbol{m}}_t &= \frac{\boldsymbol{m}_t}{1-\beta_1^t},\quad \hat{\boldsymbol{v}}_t = \frac{\boldsymbol{v}_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon}
\end{aligned}
$$

默认 $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$。

**AdamW（Decoupled Weight Decay Regularization, 重点）**：Adam 里若直接把 L2 正则放进 $L$ 中再求导，权重衰减 $\theta_{t+1} = (1-\lambda) \theta_t - \eta \cdot (\text{Adam term})$ 会被自适应学习率 $\sqrt{\hat{v}_t}$ 缩放，导致正则化效果被"吃掉"。AdamW 把**权重衰减从梯度中解耦**出来，每步显式乘 $(1-\lambda)$：

$$
\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon}
$$

$\lambda$ 是 weight decay 系数（PyTorch 里 `weight_decay` 参数）。AdamW 是目前主流推荐/大模型的默认优化器。

在搜广推哪里见过：

- **Transformer 系推荐模型（BST / DIN 带自注意力 / MMoE）**：默认 AdamW + 余弦退火调度；
- **双塔召回 /多兴趣召回（MIND / ComiRec）**：item 塔和 user 塔同步用 AdamW 训练；
- **预训练语言模型做 CTR / 检索（BERT + Click）**：BERT 的 fine-tune 标配就是 AdamW（$\lambda$ 典型值 0.01）；
- **对比学习（SimCLR / CLIP 在推荐中的应用）**：InfoNCE loss + AdamW，weight decay 对表示质量影响很大。

---

### 4.4 坐标下降 / 近端梯度

**坐标下降 (Coordinate Descent, CD)**：每次固定其他维度，只对一个维度做一维优化。对某些问题（如 Lasso、SVM dual）一维子问题有闭式解，收敛非常快。

**近端梯度 (Proximal Gradient Descent, PGD)**：处理不可微但凸的目标 $f + g$（如交叉熵 + L1）。对光滑部分 $f$ 走一步梯度下降，对非光滑部分 $g$ 做一次"投影/soft-threshold"：

$$\theta_{t+1} = \mathrm{prox}_{\eta g}(\theta_t - \eta \nabla f(\theta_t))$$

L1 的 prox 就是软阈值算子 $\mathrm{soft}(x, \alpha) = \mathrm{sign}(x)\max(|x|-\alpha, 0)$。

在搜广推哪里见过：

- **Lasso / 广义线性模型的稀疏解**：广告/推荐里特征动辄几十万维，用 coordinate descent 或 PGD 训练带 L1 的 LR，可以把大部分特征权重压到 0，做自动特征选择；
- **MF 交替最小二乘 (ALS)**：固定 $\boldsymbol{U}$ 解 $\boldsymbol{V}$，固定 $\boldsymbol{V}$ 解 $\boldsymbol{U}$，每次子问题是 ridge regression 有闭式解——这是坐标下降在推荐里最经典的形式；
- **FTRL-Proximal**：Google 的 FTRL 可以看成是 proximal 思想 + 正则化，在线稀疏 CTR 预估的标杆；
- **核方法 / 早期 CTR 的 SGD + L1**：用 soft-threshold 每步把小权重置零，配合正则化保持模型小而稳。

---

### 4.5 KKT 条件（直观理解，不做推导）

考虑带不等式约束的凸优化：

$$\min f(\boldsymbol{x}),\quad \text{s.t.}\ g_i(\boldsymbol{x}) \leq 0,\ h_j(\boldsymbol{x}) = 0$$

在满足 Slater 条件（存在严格内点）下，最优解 $\boldsymbol{x}^*$ 必满足 KKT 条件：

1. 稳定性：$\nabla f(\boldsymbol{x}^*) + \sum_i \lambda_i \nabla g_i(\boldsymbol{x}^*) + \sum_j \mu_j \nabla h_j(\boldsymbol{x}^*) = 0$
2. 原始可行性：$g_i(\boldsymbol{x}^*) \leq 0,\ h_j(\boldsymbol{x}^*) = 0$
3. 对偶可行性：$\lambda_i \geq 0$
4. 互补松弛 (Complementary Slackness)：$\lambda_i g_i(\boldsymbol{x}^*) = 0$

**直觉**：互补松弛说的是——"如果约束没被触到（$g_i < 0$），它对最优解没影响（$\lambda_i=0$）；一旦它被触到变成 active constraint（$g_i=0$），它就会产生一个"拉"着最优解的力（$\lambda_i>0$）"。

在搜广推哪里见过：

- **带预算约束的投放优化**：目标是最大化收益 $\sum r_i x_i$，约束是成本上限 $\sum c_i x_i \leq B, x_i \in \{0,1\}$。连续松弛后，互补松弛告诉我们"要么花光预算，要么收益/性价比最高的都选了"；
- **SVM 对偶 / kernel 方法**：KKT 的互补松弛意味着"只有支持向量的 $\alpha_i > 0$，其它样本不影响解"；广告的早期核 SVM 排名模型就建立在这套语言上；
- **最大熵模型 / LR 的统一视角**：softmax 回归 / 最大熵分类器都可以从"满足特征期望约束下最大化熵"的凸优化问题推出，最优性由 KKT 刻画；
- **公平性约束的推荐**：约束 "某用户群体的曝光率 ≥ α" 变成不等式约束，求解时看哪些约束是 active 的，由对偶变量判断"这个公平约束花了多少性价比"。

---

## 学习建议 & 下一步

**怎么用这份笔记**

1. **不要死记证明**，上面出现的公式（尤其是 AdamW、KL、交叉熵、SVD、KKT 的互补松弛）必须做到"看符号就知道它在说什么、在哪个模块起什么作用"；
2. **每看一篇新论文（比如 DIN、DCNv2、SimCSE、MWUF、TWOWING、DCNv3 等），遇到符号先翻这份笔记定位它在哪个数学主题下**，如果发现缺漏，就补到自己的版本；
3. **每学一个新模型，顺手把它的 loss、优化器、训练范式对应到 "二、概率 + 四、优化"**。比如 DLRM = embedding lookup + MLP + dot-interaction + 交叉熵 + AdamW，全部是这套工具链的组合。

**下一步推荐的学习顺序**

- 如果你刚入行：**ML 基础（LR、FM、GBDT）→ 工程（Spark / TensorFlow / PyTorch / 参数服务器）→ 优化器与调参（本笔记 4.3）→ 召回的矩阵分解和双塔**；
- 如果你做精排为主：本笔记的**2.5 交叉熵 + 3.1 Hessian + 4.3 AdamW** 是日常高频；
- 如果你做召回为主：**1.2 SVD + 1.3 cosine + 2.5 KL / 互信息 + 4.2 SGD** 是重点；
- 如果你做广告出价 / 流量分配 / 探索-利用：**2.1 贝叶斯 + 2.3 Beta + 2.6 MAP + 4.5 KKT** 是核心。

**一句话收尾**：搜广推里的数学，永远是"为计算服务"——只要你能把论文里的符号翻译成"这段代码会怎么写"，你就算掌握它了。
