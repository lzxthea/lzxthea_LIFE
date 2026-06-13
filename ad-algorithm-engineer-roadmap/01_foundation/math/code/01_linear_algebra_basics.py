"""
线性代数基础 Demo
==================
覆盖内容：
  1. 向量 / 矩阵基本运算
  2. 矩阵的秩 (rank)
  3. 内积 / 外积 / 范数 / 余弦相似度
  4. SVD 奇异值分解 & 低秩近似
  5. PCA 主成分分析

依赖：仅 numpy（不使用 matplotlib 等图形库，方便服务器运行）
"""

import numpy as np


def section(title: str):
    """打印分隔栏，便于在 terminal 中阅读"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# 1. 构造 "用户-物品" 评分矩阵 X：6 个用户，4 个物品
# ============================================================
# 行 = 用户 (u1..u6)，列 = 物品 (i1..i4)，元素 = 评分 (1~5)
X = np.array([
    [5, 3, 0, 1],   # u1
    [4, 0, 0, 1],   # u2
    [1, 1, 0, 5],   # u3
    [1, 0, 0, 4],   # u4
    [0, 1, 5, 4],   # u5
    [0, 2, 5, 3],   # u6
], dtype=np.float64)


# ============================================================
# 2. 向量 / 矩阵运算  &  秩
# ============================================================
def demo_basic_ops_and_rank():
    section("1. 矩阵形状 / 基本运算 / 秩")

    print(f"X.shape = {X.shape}")              # (6, 4)
    print(f"X = \n{X}")

    # —— 基本矩阵运算 ——
    print("\n-- X 的逐元素平方：")
    print(X ** 2)

    # —— 矩阵的秩 rank ——
    # rank = 矩阵线性无关行（列）向量的最大数量
    # numpy 使用 SVD 计算数值秩：对小于 tol 的奇异值视为 0
    rk = np.linalg.matrix_rank(X)
    print(f"\nrank(X) = {rk}")                  # 应接近 4

    # —— X @ X.T：用户-用户 相似度矩阵（未归一化）——
    # (6,4) @ (4,6) -> (6,6)，第 (i,j) 项 = <row_i, row_j>
    XXt = X @ X.T
    print(f"\nX @ X.T 形状 = {XXt.shape}")
    print(XXt)


# ============================================================
# 3. 内积 / 外积 / 范数 / 余弦相似度
# ============================================================
def demo_inner_outer_norm_cosine():
    section("2. 内积 / 外积 / 范数 / 余弦相似度")

    a = X[0]   # u1 的评分向量
    b = X[2]   # u3 的评分向量
    print(f"a = u1 评分 = {a}")
    print(f"b = u3 评分 = {b}")

    # —— 内积（点积）——
    # <a, b> = Σ a_i * b_i
    inner = np.dot(a, b)
    print(f"\n内积 <a,b> = a·b = {inner}")

    # —— 外积 ——
    # a ⊗ b = a.reshape(n,1) @ b.reshape(1,m)  -> 矩阵 (n,m)
    outer = np.outer(a, b)
    print(f"\n外积 shape = {outer.shape}：")
    print(outer)

    # —— 范数 ——
    # L1 = Σ|x|,  L2 = √(Σx²),  L∞ = max(|x|)
    print(f"\n||a||_1 = {np.linalg.norm(a, 1):.4f}")
    print(f"||a||_2 = {np.linalg.norm(a, 2):.4f}")
    print(f"||a||_∞ = {np.linalg.norm(a, np.inf):.4f}")

    # —— 行向量两两余弦相似度 ——
    # cos(a, b) = <a, b> / (||a|| * ||b||)
    norms = np.linalg.norm(X, axis=1, keepdims=True)   # (6,1)
    X_norm = X / (norms + 1e-12)                        # 行归一化
    cosine = X_norm @ X_norm.T                          # (6,6)
    print("\n行向量（用户）两两 余弦相似度矩阵：")
    print(np.round(cosine, 3))
    # 对角元 = 1（自己与自己夹角=0，cos=1）；越接近 1 表示两个用户越相似


# ============================================================
# 4. SVD 奇异值分解 & 低秩近似
# ============================================================
def demo_svd_low_rank():
    section("3. SVD 奇异值分解 & 低秩近似 (rank-2)")

    # X = U @ diag(s) @ Vt
    #   U: (6,6) 列正交，左奇异向量（"用户空间"）
    #   s: (4,)   奇异值，降序
    #   Vt:(4,4) 行正交，右奇异向量（"物品空间"）
    U, s, Vt = np.linalg.svd(X, full_matrices=False)   # 紧凑模式：U(6,4), s(4), Vt(4,4)
    print(f"U.shape = {U.shape},  s.shape = {s.shape},  Vt.shape = {Vt.shape}")
    print(f"奇异值 s = {s.round(2)}")

    # 奇异值平方 ≈ 协方差矩阵的特征值（信息含量）
    print(f"奇异值占比 (解释方差) = {((s**2).cumsum() / (s**2).sum()).round(3)}")

    # —— 只保留前 2 个奇异值，做低秩近似 ——
    k = 2
    U_k = U[:, :k]                   # (6,2)
    s_k = s[:k]                      # (2,)
    Vt_k = Vt[:k, :]                 # (2,4)
    X_approx = U_k @ np.diag(s_k) @ Vt_k

    print(f"\n前 {k} 个奇异值：{s_k.round(2)}")
    print(f"X_approx.shape = {X_approx.shape}")
    print("X_approx ≈")
    print(X_approx.round(2))

    # —— 误差衡量：Frobenius 范数 ||X - X_approx||_F ——
    err = np.linalg.norm(X - X_approx, 'fro')
    print(f"\n低秩近似误差 ||X - X_k||_F = {err:.4f}")
    print(f"理论上 = 被丢弃奇异值平方和开根号 = {np.sqrt((s[k:]**2).sum()):.4f}")


# ============================================================
# 5. PCA 主成分分析
# ============================================================
def demo_pca():
    section("4. PCA 主成分分析（手工实现：中心化 -> 协方差 -> 特征值分解）")

    # —— 步骤 1：数据中心化 ——
    #   让每个特征（列）均值为 0，这样协方差矩阵 = Xc.T @ Xc / (N-1)
    mu = X.mean(axis=0)          # (4,)，每列（物品）的平均评分
    Xc = X - mu                  # 中心化，形状仍 (6,4)
    print(f"每列（物品）均值 mu = {mu.round(2)}")
    print(f"Xc 行和（接近 0）= {Xc.sum(axis=0).round(4)}")

    # —— 步骤 2：协方差矩阵 (4,4) ——
    N = X.shape[0]
    Cov = Xc.T @ Xc / (N - 1)    # 无偏估计除以 N-1
    print(f"\nCov.shape = {Cov.shape}")
    print(Cov.round(3))

    # —— 步骤 3：对 Cov 做特征值分解 ——
    #   Cov = W @ diag(λ) @ W.T,  W 列正交（主成分方向）
    eigvals, eigvecs = np.linalg.eigh(Cov)
    # eigh 返回的特征值是升序，需要翻转成降序
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    print(f"\n特征值 λ（降序） = {eigvals.round(3)}")
    print(f"解释方差比例   = {(eigvals / eigvals.sum()).round(3)}")
    print(f"累计解释方差   = {(eigvals.cumsum() / eigvals.sum()).round(3)}")

    # —— 步骤 4：取前 2 维做投影 ——
    k = 2
    W_k = eigvecs[:, :k]          # (4,2) 前 2 个主成分方向
    Z = Xc @ W_k                  # (6,2) = 每个用户在 2 维主空间的坐标
    print(f"\nW_k (前 {k} 主成分方向) shape = {W_k.shape}")
    print(W_k.round(3))
    print(f"\nZ = Xc @ W_k (用户在 2D PCA 空间的投影) shape = {Z.shape}")
    print(Z.round(3))

    # —— 额外验证：PCA 与 SVD 的联系 ——
    #   对 Xc 做 SVD：Xc = U s Vt
    #   则主成分得分 Z = U_k * s_k
    #   与上面 "Xc @ W_k" 得到的结果应一致（相差 ±1 的列符号，因为特征向量方向不定）
    U2, s2, Vt2 = np.linalg.svd(Xc, full_matrices=False)
    Z_svd = U2[:, :k] * s2[:k]
    print(f"\n[验证] 用 SVD 得到的 Z ≈")
    print(Z_svd.round(3))
    print("(与上方 Z 相比，每列可能整体差一个 ± 号，属于 PCA / SVD 的符号不定正常现象)")


if __name__ == '__main__':
    print()
    print("*" * 60)
    print("*  线性代数基础 Demo (numpy only)")
    print("*" * 60)

    demo_basic_ops_and_rank()
    demo_inner_outer_norm_cosine()
    demo_svd_low_rank()
    demo_pca()

    print()
    print("=" * 60)
    print("  Demo 结束")
    print("=" * 60)
