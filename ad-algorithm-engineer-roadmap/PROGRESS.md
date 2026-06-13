# 项目进度追踪

> ⚠️ 每次完成一个模块后，必须立即更新本文件并 push 到远端。
> 本文件是项目状态的唯一真实来源，每次开始工作前必须先读一遍。

---

## 模块完成状态

| 模块 | 路径 | 状态 | commit hash | 备注 |
|------|------|------|------------|------|
| **数学** | `01_foundation/math/` | ✅ 完成 | `06f8bb4` | notes + 5 code + README + resources |
| **框架 (PyTorch)** | `03_code_skills/frameworks/code/` | ✅ 完成 | `cfb22ef` | 5个 Demo |
| **框架 (TensorFlow)** | `03_code_skills/frameworks/tensorflow/` | ✅ 完成 | `cfb22ef` | notes + 3 code + README |
| **框架 (resources)** | `03_code_skills/frameworks/resources.md` | ✅ 完成 | `cfb22ef` | 资源总表 |
| 统计学 | `01_foundation/statistics/` | ⏳ 未开始 | — | — |
| 计算机科学基础 | `01_foundation/cs_basics/` | ⏳ 未开始 | — | — |
| 经典机器学习 | `02_ml_theory/classical_ml/` | ⏳ 未开始 | — | — |
| 深度学习理论 | `02_ml_theory/deep_learning/` | ⏳ 未开始 | — | — |
| 进阶方向 | `02_ml_theory/advanced/` | ⏳ 未开始 | — | — |
| Python | `03_code_skills/python/` | ⏳ 未开始 | — | — |
| 数据结构与算法 | `03_code_skills/data_structures_algo/` | ⏳ 未开始 | — | — |
| 数据处理 | `04_engineering/data_processing/` | ⏳ 未开始 | — | — |
| 模型部署 | `04_engineering/model_deployment/` | ⏳ 未开始 | — | — |
| MLOps | `04_engineering/mlops/` | ⏳ 未开始 | — | — |
| 竞赛 | `05_practice/competitions/` | ⏳ 未开始 | — | — |
| 小项目 | `05_practice/mini_projects/` | ⏳ 未开始 | — | — |
| 论文复现 | `05_practice/paper_reproduction/` | ⏳ 未开始 | — | — |
| 论文阅读 | `06_soft_skills/paper_reading/` | ⏳ 未开始 | — | — |
| 技术写作 | `06_soft_skills/communication/` | ⏳ 未开始 | — | — |
| 推荐系统 | `07_search_ads_recommendation/recsys/` | ⏳ 未开始 | — | — |
| 计算广告 | `07_search_ads_recommendation/computational_advertising/` | ⏳ 未开始 | — | — |
| 搜索排序 | `07_search_ads_recommendation/search_ranking/` | ⏳ 未开始 | — | — |
| 大规模系统 | `07_search_ads_recommendation/large_scale_systems/` | ⏳ 未开始 | — | — |
| 全局资源 | `resources/` | ⏳ 未开始 | — | — |

---

## 每次开始工作的 SOP（必须遵守）

1. **先读本文件**：`cat PROGRESS.md`，确认「上次做到哪里」
2. **检查本地状态**：`git branch && git status --short && find . -name "*.py" | wc -l`
3. **检查远端最新**：`git fetch origin && git log origin/lzx_dev --oneline -5`
4. **如果本地落后远端**：`git pull origin lzx_dev --rebase`，不要直接 force push
5. **完成模块后**：
   - 更新本文件的对应行（状态→✅，commit hash，备注）
   - `git add PROGRESS.md && git commit -m "docs: 更新 PROGRESS.md" && git push origin lzx_dev`
6. **永远不要对 lzx_dev 做 force push**（除非明确知道远端有错误 commit 必须回退）

---

## Git 分支约定

- `main` —— 稳定分支，永远不直接 push 到 main
- `lzx_dev` —— 开发分支，每次 push 之前确认远端状态
- 重要原则：**push 之前必须 pull**，避免覆盖别人的提交

---

## 远端地址

```
https://github.com/lzxthea/lzxthea_LIFE/tree/lzx_dev
```

每次 push 后可以在 GitHub 上刷新确认文件是否到位。
