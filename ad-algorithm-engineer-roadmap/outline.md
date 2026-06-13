# 广告算法工程师 · 知识体系大纲

> 面向「搜索 / 广告 / 推荐」方向的算法工程师能力地图。
> 每个模块的详细知识点请前往对应子目录查阅 notes.md；参考资源见全局 resources/。

---

## 01 · 基础理论 (Foundation)

- 1.1 **数学** —— 线性代数 / 概率论 / 微积分 / 优化理论
- 1.2 **统计学** —— 描述性统计 / 假设检验 / MLE & MAP / 贝叶斯推断 / A/B 测试 / 因果推断基础
- 1.3 **计算机科学基础** —— 数据结构 / 算法与复杂度 / 操作系统 / 网络 / 数据库

📁 目录：[01_foundation/](./01_foundation/)

---

## 02 · 机器学习理论 (ML Theory)

- 2.1 **经典机器学习** —— 线性模型 / 树模型 / SVM / 聚类 / 降维 / 集成学习 / 特征选择 / 评估指标 / 正则化
- 2.2 **深度学习** —— 神经网络基础 / CNN / RNN & LSTM / Attention & Transformer / 预训练与微调
- 2.3 **进阶方向** —— 强化学习 RL / 图神经网络 GNN / 生成模型 / 元学习 / 联邦学习 / 因果推断

📁 目录：[02_ml_theory/](./02_ml_theory/)

---

## 03 · 代码与工程能力 (Code Skills)

- 3.1 **Python** —— 语言进阶 / 面向对象 / 常用库 (NumPy / Pandas / scikit-learn) / 性能优化 / Git / TensorBoard
- 3.2 **深度学习框架** —— PyTorch / TensorFlow / Keras / ONNX Runtime
- 3.3 **数据结构与算法** —— 数组 / 链表 / 栈 & 队列 / 哈希表 / 树 & 图 / 动态规划 / 贪心

📁 目录：[03_code_skills/](./03_code_skills/)

---

## 04 · 工程与 MLOps (Engineering)

- 4.1 **数据处理** —— 数据清洗 / 特征工程 / Spark / Flink / Kafka
- 4.2 **模型部署** —— TorchScript / ONNX / TensorRT / FastAPI / Docker / 推理优化
- 4.3 **MLOps** —— 实验管理 (MLflow / wandb) / 特征平台 / 模型注册 / CI-CD / A/B 平台

📁 目录：[04_engineering/](./04_engineering/)

---

## 05 · 实践项目 (Practice)

- 5.1 **竞赛** —— Kaggle / 天池 / KDD Cup / RecSys Challenge
- 5.2 **小项目** —— 文本分类 / CTR 预估 / 推荐召回 Demo
- 5.3 **论文复现** —— 经典模型复现 + ablation study

📁 目录：[05_practice/](./05_practice/)

---

## 06 · 软实力 (Soft Skills)

- 6.1 **论文阅读** —— 快速读懂 / 做笔记 / 写 Summary
- 6.2 **技术写作与沟通** —— 技术文档 / Blog / 专利

📁 目录：[06_soft_skills/](./06_soft_skills/)

---

## 07 · 搜索 / 广告 / 推荐 专项

- 7.1 **推荐系统 (RecSys)** —— 召回 → 粗排 → 精排 → 重排 → 策略层 / 冷启动 / 评估
- 7.2 **计算广告 (Computational Advertising)** —— CTR/CVR 预估 / 竞价策略 / 创意优化 / 归因 / 反作弊
- 7.3 **搜索与排序 (Search & LTR)** —— Query 理解 / 倒排 + 向量召回 / Learning to Rank / 搜索广告
- 7.4 **大规模系统** —— 参数服务器 / 分布式 ML / 实时计算 / 向量检索 / 在线推理架构

📁 目录：[07_search_ads_recommendation/](./07_search_ads_recommendation/)

---

## 全局资源

- 📚 推荐书籍 / 课程 / 顶会论文 / GitHub 工具：[resources/](./resources/)

---

## 学习路径建议（7 阶段）

1. **打基础** —— 数学 + 统计 + Python
2. **机器学习入门** —— 经典 ML（LR → GBDT → SVM → 聚类）
3. **深度学习** —— 神经网络 → CNN → RNN → Transformer
4. **推荐系统** —— MF → FM → Wide&Deep → DIN → MMoE
5. **计算广告** —— CTR/CVR 预估 → 竞价策略 → 归因 → 反作弊
6. **工程化** —— Spark / Flink / Docker / MLOps / 在线推理
7. **前沿方向** —— RL / 因果推断 / 大模型 —— 结合业务问题应用

## 当前进度

- ✅ 01.1 数学（线性代数 / 概率 / 微积分 / 优化）
- ⏳ 01.2 统计学（下一模块）
- ⏳ 其他模块后续补充
