# 广告算法工程师 · 学习路线

> 面向「搜索 / 广告 / 推荐」方向的算法工程师能力地图。
> 用「**笔记 + 代码 Demo + 资源链接**」三件套，系统地记录从基础理论到业务专项的全链路知识。

---

## 📁 目录结构

```
ad-algorithm-engineer-roadmap/
├── outline.md                       # 完整大纲（简洁目录形式，带链接）
├── README.md                        # 本文件
│
├── 01_foundation/                   # 基础理论
│   └── math/                        # ✅ 数学（线性代数 / 概率 / 微积分 / 优化）
│       ├── README.md                # 模块导航 + 学习路径
│       ├── notes.md                 # 四大板块完整知识点（含「在搜广推哪里见过」）
│       ├── resources.md             # 书籍 / 课程 / 博客 / 可视化工具
│       └── code/                    # 5 个独立可运行的 NumPy Demo
│
├── 02_ml_theory/                    # ⏳ 机器学习理论（经典 ML / 深度学习 / 进阶）
├── 03_code_skills/                  # ⏳ 代码与工程能力（Python / 框架 / 算法）
├── 04_engineering/                  # ⏳ 工程与 MLOps
├── 05_practice/                     # ⏳ 实践项目
├── 06_soft_skills/                  # ⏳ 软实力
├── 07_search_ads_recommendation/    # ⏳ 搜广推专项（核心业务方向）
└── resources/                       # ⏳ 全局资源（书籍 / 课程 / 顶会 / GitHub）
```

---

## 🎯 定位与特色

- **面向广告/推荐算法工程师** —— 每个知识点都标注「在搜广推哪里见过」，学完能直接映射到业务
- **两条线并行** —— 理论笔记（notes.md）+ 可运行代码 Demo（code/），把抽象公式跑出来
- **持续更新** —— 按模块推进，已完成模块打 ✅，未完成打 ⏳

---

## 🚀 如何使用

### 1. 看大纲
打开 [outline.md](./outline.md) 了解完整的 7 大模块目录结构和学习路径。

### 2. 学模块
每个子模块内部有：
- **README.md** —— 模块导航与学习路径建议
- **notes.md** —— 知识点笔记（含业务关联）
- **code/** —— 可运行的代码 Demo（`python xxx.py` 直接跑）
- **resources.md** —— 对应主题的书籍 / 课程 / 工具推荐

### 3. 跟着代码 Demo 动手
```bash
cd ad-algorithm-engineer-roadmap/01_foundation/math/code
pip install numpy
python 01_linear_algebra_basics.py
python 04_gradient_descent_variants.py
```

---

## 📊 当前进度

| 模块 | 状态 | 说明 |
|------|------|------|
| **01.1 数学** | ✅ | 线性代数 / 概率 / 微积分 / 优化 — 含 notes + 5 个 NumPy Demo |
| 01.2 统计学 | ⏳ | 下一模块：假设检验 / A/B 测试 / 因果推断 |
| 01.3 CS 基础 | ⏳ | 数据结构 / 算法与复杂度 |
| 02 ML 理论 | ⏳ | 经典 ML / 深度学习 / 进阶 |
| 03 代码与工程 | ⏳ | Python / 框架 / 算法题 |
| 04 工程与 MLOps | ⏳ | 数据处理 / 模型部署 / MLOps |
| 05 实践项目 | ⏳ | 竞赛 / 小项目 / 论文复现 |
| 06 软实力 | ⏳ | 论文阅读 / 技术写作 |
| 07 搜广推专项 | ⏳ | 推荐系统 / 计算广告 / 搜索 / 大规模系统 |

---

## 📌 Git 分支约定

- **`main`** —— 稳定分支，重要里程碑合入
- **`lzx_dev`** —— 开发分支（当前分支），每次改动立刻 push 到远端

```bash
git pull origin lzx_dev     # 同步最新内容
git checkout lzx_dev        # 切到开发分支
```

在 GitHub 上查看最新内容：https://github.com/lzxthea/lzxthea_LIFE/tree/lzx_dev
