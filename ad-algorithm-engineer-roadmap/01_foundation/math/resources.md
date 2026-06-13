# math · 学习资源

> 精选的数学学习资料（书籍 / 课程 / 博客 / 可视化工具），面向广告/推荐算法工程师。
> 优先级：⭐⭐⭐ 必读 / ⭐⭐ 推荐 / ⭐ 可选

---

## 一、书籍

| 书名 / 作者 | 中译 | 优先级 | 说明 |
|-------------|------|--------|------|
| **《Linear Algebra Done Right》** (Axler) | 《线性代数应该这样学》 | ⭐⭐⭐ | 从向量空间角度讲线性代数，不强调行列式，对理解 Embedding/SVD 很有帮助 |
| **《Introduction to Linear Algebra》** (Strang) | 《线性代数导论》 | ⭐⭐⭐ | 巨著，配合 MIT 18.06 课程一起看；讲直觉大于推导 |
| **《Pattern Recognition and Machine Learning》** (Bishop) | 《模式识别与机器学习》 | ⭐⭐⭐ | PRML —— 贝叶斯视角的 ML 圣经；第 1-4 章和矩阵求导/高斯/EM 必读 |
| **《The Elements of Statistical Learning》** (Hastie/Tibshirani/Friedman) | 《统计学习基础》 | ⭐⭐⭐ | ESL —— 频率派经典；SVD/PCA/Boosting/SVM 讲得非常细 |
| **《统计学习方法》** (李航) | — | ⭐⭐⭐ | **中文首选**，第 2 版；逻辑清晰、推导完整；LR/SVM/EM/HMM 必读章节 |
| **《Convex Optimization》** (Boyd/Vandenberghe) | 《凸优化》 | ⭐⭐ | 优化理论权威参考书；前 5 章必看，其余按需翻 |
| **《Deep Learning》** (Goodfellow/Bengio/Courville) | — | ⭐⭐⭐ | 花书，第 2-5 章（线性代数/概率/数值计算/ML 基础）对算法工程师是基础必备 |
| **《All of Statistics》** (Wasserman) | 《统计学完全教程》 | ⭐⭐ | 一本覆盖概率论+统计推断；公式密集但紧凑 |
| **《Convex Optimization & Gradient Descent》** (Boyd 讲义) | — | ⭐⭐ | Boyd 教授的免费讲义，配合课程视频 |

---

## 二、课程 / 视频

| 课程 | 平台 | 语言 | 优先级 | 说明 |
|------|------|------|--------|------|
| **MIT 18.06 线性代数** (Strang) | MIT OCW / YouTube | 英文 | ⭐⭐⭐ | 经典中的经典；看前 15 讲足够 |
| **MIT 18.065 机器学习中的线性代数** (Strang) | MIT OCW | 英文 | ⭐⭐⭐ | 2018 新课；SVD/PCA/随机矩阵/深度学习；直接对接本模块主题 |
| **Stanford CS229 机器学习** (Ng) | Stanford / YouTube | 中英字幕 | ⭐⭐⭐ | 线性代数/概率/优化复习最佳起点 |
| **《动手学深度学习》D2L** (李沐团队) | https://zh.d2l.ai/ | 中文 | ⭐⭐⭐ | **强烈推荐** —— PyTorch 代码 + 中文讲解；第 2 章是数学预备 |
| **DeepLearning.AI 数学专项** (DeepLearning.AI) | Coursera | 英文 | ⭐⭐ | 三门课：线性代数/微积分/概率；面向 ML 学习者 |
| **3Blue1Brown · Essence of Linear Algebra** | YouTube / Bilibili | 中英字幕 | ⭐⭐⭐ | 可视化神作；**看课程前先看这个建立直觉** |
| **3Blue1Brown · Essence of Calculus** | YouTube / Bilibili | 中英字幕 | ⭐⭐⭐ | 理解链式法则的最佳可视化 |

---

## 三、博客 / 文章

| 作者 / 主题 | 链接 | 说明 |
|------------|------|------|
| **colah's blog · Neural Networks, Manifolds, and Topology** | https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/ | 理解 Embedding 流形的最佳文章 |
| **The Matrix Cookbook** | https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf | **矩阵求导公式速查表** |
| **Khan Academy · 线性代数 / 概率** | https://www.khanacademy.org | 中文讲解；适合概念不清时快速复习 |
| **Distill.pub · A Visual Article** | https://distill.pub/ | 可视化 ML 文章；直观理解 SVD、优化器等 |
| **Jay Alammar · The Illustrated Transformer / Word2Vec** | https://jalammar.github.io/ | 直觉+图解；尤其是 Attention/Softmax 部分 |
| **Machine Learning Mastery** | https://machinelearningmastery.com/ | Jason Brownlee；实用向，每篇讲一个主题 |
| **知乎专栏 · 「机器学习与数学」** | — | 中文资源丰富；搜索「SVD 推荐系统」「FM 推导」能找到好文章 |

---

## 四、可视化 / 交互工具

| 工具 | 链接 | 用途 |
|------|------|------|
| **Desmos** | https://www.desmos.com/calculator | 在线绘制函数（凸/非凸、loss 曲线） |
| **GeoGebra** | https://www.geogebra.org/ | 几何与代数可视化；理解线性变换/PCA |
| **TensorFlow Playground** | https://playground.tensorflow.org/ | 直观理解 NN 如何分类数据；不同优化器/学习率的效果 |
| **Seeing Theory** | https://seeing-theory.brown.edu/ | 概率与统计的交互可视化（贝叶斯、分布、p-value） |
| **Setosa.io · Visual Eigenvectors** | https://setosa.io/ev/eigenvectors-and-eigenvalues/ | 特征值/特征向量的交互演示 |
| **Visualizing Matrix SVD** (UC Davis) | — | Google "SVD visualization"，找能展示 U/Σ/V 的交互图 |

---

## 五、本模块对应的关键论文（拓展阅读）

| 论文 | 主题 | 与本模块关联 |
|------|------|-------------|
| **Matrix Factorization Techniques for Recommender Systems** (Koren/Bell/Volinsky, 2009) | 推荐系统 | SVD 直接应用；Netflix Prize |
| **Factorization Machines** (Rendle, 2010) | FM 模型 | 二阶交互 = 矩阵求导 + 梯度下降 |
| **Adam: A Method for Stochastic Optimization** (Kingma/Ba, 2014) | Adam 优化器 | 掌握 Adam 公式后推荐精读原文 |
| **Decoupled Weight Decay Regularization** (Loshchilov/Hutter, 2017) | AdamW | 「解耦权重衰减」的出处；DeepFM 等模型标配 |
| **Large-Scale Machine Learning with Stochastic Gradient Descent** (Bottou, 2010) | SGD 理论 | 理解 SGD 收敛性质 |

---

## 六、工具 / 参考速查

- **NumPy Linear Algebra** —— `np.linalg.*` (svd, eig, inv, norm, det, solve)
- **SciPy** —— `scipy.stats` (分布采样、p-value), `scipy.optimize`
- **Matrix Cookbook** (Petersen/Kaas) —— 矩阵求导 / 迹 / 行列式 / 逆，**随时查**
- **Wolfram MathWorld** —— 查定义级别的数学概念
