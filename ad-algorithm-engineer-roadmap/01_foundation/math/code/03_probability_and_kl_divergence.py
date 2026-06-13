"""
概率论 & 信息论 Demo
====================
覆盖内容：
  1. Beta 分布采样（Beta(2,5) 和 Beta(5,2) —— 打印样本统计量
  2. 伯努利分布的 MLE：p_MLE = mean(x)
  3. 离散分布 P、Q：熵 H(P)、交叉熵 H(P,Q)、KL(P||Q)

关键公式：
    H(P)    = - Σ P_i log(P_i)
    H(P, Q) = - Σ P_i log(Q_i)
    KL(P||Q) = H(P,Q) - H(P) = Σ P_i log(P_i / Q_i)

"为什么 NN 用交叉熵而不是 MSE（均方误差）？
  - 交叉熵对分类任务，softmax+sigmoid 的组合：
    —— loss 曲面更"平坦"，梯度消失问题比 MSE 小很多；
    —— 交叉熵 = -log(1 时，softmax 的梯度就是 p-y，简单稳定。
  - 而 MSE = (p-y)^2：
    —— 当 p 远离 y 且靠近 0/1 时，sigmoid 导数趋近 0，梯度消失；
    —— loss 曲面非凸，训练更容易卡在局部最优。
  - 一句话：**分类任务请用交叉熵，回归任务用 MSE。

"为什么 KL 不对称 —— KL(P||Q) ≠ KL(Q||P)"
  - KL 衡量"用 Q 去近似 P 时多出来的信息损失"，是方向量度；
  - 直觉：
      P = 真实分布（比如 [0.6, 0.3, 0.1]）
      Q = 我们的估计分布
      KL(P||Q)：从 P 的视角看 Q 的"不匹配程度"
      KL(Q||P)：从 Q 的视角看 P 的"不匹配程度"
  - 数学上：加权求和时权重用的是 P（或 Q），所以方向不同；
  - 因此在 VAE / GAN / 信息论模型里，方向选择很重要。
"""

import numpy as np


def section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# 1. Beta(2,5) 与 Beta(5,2)：采样并打印统计量
# ============================================================
def demo_beta_sampling(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)

    # Beta(α, β)：
    #   - 均值 = α/(α+β)
    #   - 方差 = αβ / ((α+β)^2 * (α+β+1))
    samples_A = rng.beta(a=2, b=5, size=n_samples)   # 均值小，偏向 0
    samples_B = rng.beta(a=5, b=2, size=n_samples)   # 均值大，偏向 1

    def stats(name, s, alpha, beta):
        mean_t = alpha / (alpha + beta)
        var_t  = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        print(f"\n[{name}] Beta({alpha},{beta})")
        print(f"  样本形状  = {s.shape}")
        print(f"  样本均值  = {s.mean():.4f}   理论均值  = {mean_t:.4f}")
        print(f"  样本方差  = {s.var(ddof=1):.4f}   理论方差  = {var_t:.4f}")
        print(f"  样本最小值 = {s.min():.4f},  最大值 = {s.max():.4f}")
        # 分位数
        q = np.percentile(s, [10, 25, 50, 75, 90])
        print(f"  分位数 [10/25/50/75/90] = {q.round(3)}")

    stats("A", samples_A, 2, 5)
    stats("B", samples_B, 5, 2)

    print()
    print("  → Beta 是 (0,1) 上的分布，常用于 CTR / 转化率建模。")
    print("  → Beta(2,5) 形状右偏（偏向小值），Beta(5,2) 形状左偏。")

    return samples_A, samples_B


# ============================================================
# 2. 伯努利分布的最大似然估计 (MLE)
# ============================================================
def demo_bernoulli_mle(seed=42):
    """
    伯努利分布：x ~ Bernoulli(p), P(X=1) = p, P(X=0) = 1-p
    似然函数：L(p) = ∏ p^{x_i} (1-p)^{1-x_i}
    对数似然：log L(p) = Σ [x_i log p + (1-x_i) log (1-p)]
    对 p 求导并令 = 0，解得：
        p_MLE = ( Σ x_i ) / N = mean(x)
    """
    rng = np.random.default_rng(seed)
    p_true = 0.3
    N = 1000
    x = rng.binomial(n=1, p=p_true, size=N).astype(np.float64)

    p_mle = x.mean()
    print(f"真实 p = {p_true}")
    print(f"样本数 N = {N},  Σ x = {int(x.sum())}")
    print(f"p_MLE = mean(x) = {p_mle:.4f}")
    print()
    print("  → 伯努利 MLE 的直观理解：成功频率 = 概率。")
    print("  → 在 CTR / 点击率建模中，MLE 是最自然的起点。")

    return p_mle


# ============================================================
# 3. 熵 / 交叉熵 / KL 散度：手动计算并与 numpy 实现
# ============================================================
def demo_entropy_and_kl():
    """
    构造两个离散分布 P 和 Q（比如三分类问题）：
      P = [0.6, 0.3, 0.1]   (真实分布，更"确定")
      Q = [0.3, 0.4, 0.3]   (模型预测分布，更"混乱")
    """
    P = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    Q = np.array([0.3, 0.4, 0.3], dtype=np.float64)

    assert abs(P.sum() - 1.0) < 1e-9, "P 必须是概率分布"
    assert abs(Q.sum() - 1.0) < 1e-9, "Q 必须是概率分布"

    print(f"P = {P}  (真实分布)")
    print(f"Q = {Q}  (模型预测分布)")
    print()

    # —— 熵 H(P) = - Σ P_i log P_i ——
    # 熵衡量 P 自身的"不确定性"；越大越混乱。
    H_P = float(-np.sum(P * np.log(P + 1e-12)))
    print(f"H(P)        = -Σ P log P     = {H_P:.6f}")

    # —— 交叉熵 H(P,Q) = - Σ P_i log Q_i ——
    # 用 Q 来编码 P 时的"平均编码长度"。
    H_PQ = float(-np.sum(P * np.log(Q + 1e-12)))
    print(f"H(P,Q)     = -Σ P log Q     = {H_PQ:.6f}")

    # —— KL 散度 KL(P||Q) = H(P,Q) - H(P) = Σ P_i log(P_i / Q_i) ——
    # 衡量 "用 Q 近似 P 时额外浪费的信息量"。
    KL_PQ_1 = H_PQ - H_P
    KL_PQ_2 = float(np.sum(P * np.log((P + 1e-12) / (Q + 1e-12))))
    print(f"KL(P||Q)    = H(P,Q) - H(P) = {KL_PQ_1:.6f}   "
          f"(= Σ P log(P/Q) = {KL_PQ_2:.6f})")

    # —— 反向 KL(Q||P) ——
    H_Q = float(-np.sum(Q * np.log(Q + 1e-12)))
    H_QP = float(-np.sum(Q * np.log(P + 1e-12)))
    KL_QP = H_QP - H_Q
    print(f"\n[反向]")
    print(f"H(Q)         = {H_Q:.6f}")
    print(f"H(Q,P)       = {H_QP:.6f}")
    print(f"KL(Q||P)     = {KL_QP:.6f}")
    print()
    print(f"  → KL(P||Q) = {KL_PQ_1:.4f}  ≠  KL(Q||P) = {KL_QP:.4f}，不对称 ✅")
    print("  → 这正是为什么 KL 是有向散度，而不是距离。")

    # —— 额外：当 Q = P 时，KL = 0 ——
    KL_PP = float(-np.sum(P * np.log(P + 1e-12))) - H_P
    print(f"\n[验证] KL(P||P) = {KL_PP:.6f}   (应为 0)")


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    print()
    print("*" * 60)
    print("*  概率论 & 信息论 Demo (numpy only)")
    print("*" * 60)

    section("1. Beta(2,5) & Beta(5,2) 采样，各 5000 个样本")
    demo_beta_sampling(n_samples=5000, seed=42)

    section("2. 伯努利分布 MLE：p_MLE = mean(x)")
    demo_bernoulli_mle(seed=42)

    section("3. 熵 H(P) / 交叉熵 H(P,Q) / KL(P||Q)")
    demo_entropy_and_kl()

    section("4. 小结")
    print(
        "要点：\n"
        "  - Beta 分布常用于 CTR/转化率的先验分布；\n"
        "  - 伯努利 MLE = 成功频率；\n"
        "  - 交叉熵 H(P,Q) 是 NN 分类任务的首选损失；\n"
        "  - KL 散度衡量两个分布差异，不对称；\n"
        "  - 三者关系：KL(P||Q) = H(P,Q) - H(P)。"
    )

    print()
    print("=" * 60)
    print("  Demo 结束")
    print("=" * 60)
