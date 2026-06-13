"""
矩阵求导 & 链式法则 Demo
========================
主题：用 NumPy 从零实现 Logistic Regression，并做"解析梯度 vs 数值梯度"的验证。

覆盖内容：
  1. 构造二分类 toy dataset（两个高斯团）
  2. 手推 LR 的前向 + 反向（梯度）
     - sigmoid(z) = 1 / (1 + exp(-z))
     - BCE loss  L = -(1/N) Σ [y*log(p) + (1-y)*log(1-p)]
     - 梯度    ∂L/∂w = (1/N) X.T @ (p - y)
               ∂L/∂b = (1/N) Σ (p - y)
  3. 用纯梯度下降训练 200 个 epoch，记录 loss 并打印
  4. 用"有限差分" f(x+h)-f(x-h))/(2h) 做数值梯度校验
     —— 证明解析梯度是对的

「梯度在反向传播中是什么」
  反向传播 = 对神经网络做"链式法则的求导工具"。
  每一层都需要：
    (a) 前向：计算 y = f(x, θ)，以及中间量（存下来）
    (b) 反向：接收从后一层传来的"上游梯度 ∂L/∂y"，
              用链式法则推出 ∂L/∂x 传给前一层，
              同时计算 ∂L/∂θ 作为参数更新梯度。
  对 Logistic Regression 来说：
    网络结构 = 线性层 z = X w + b  →  σ(z)  →  BCE loss
    反向传播时，先从 loss 出发得到 ∂L/∂p = -(y/p - (1-y)/(1-p))
    再推到 ∂L/∂z = p - y（这一步简化的关键！）
    再推到 ∂L/∂w = X.T @ ∂L/∂z / N，∂L/∂b = mean(∂L/∂z)
"""

import numpy as np


def section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# 1. 构造二分类 toy dataset（两个高斯团）
# ============================================================
def make_toy_dataset(n_per_class=200, seed=42):
    """在 2D 平面造两个高斯簇：正类在 (2,2)，负类在 (-2,-2)"""
    rng = np.random.default_rng(seed)

    X_pos = rng.normal(loc=(2.0, 2.0), scale=(0.8, 0.8), size=(n_per_class, 2))
    X_neg = rng.normal(loc=(-2.0, -2.0), scale=(0.8, 0.8), size=(n_per_class, 2))
    X = np.vstack([X_pos, X_neg])                          # (400, 2)
    y = np.concatenate([np.ones(n_per_class),
                        np.zeros(n_per_class)])            # (400,)
    y = y.reshape(-1, 1)                                   # (400,1) 统一成列向量

    # 手动打乱（保证正负样本混合）
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]


# ============================================================
# 2. 核心：LR 的前向 & 反向（解析梯度）
# ============================================================
def sigmoid(z):
    """sigmoid(z) = 1 / (1 + exp(-z))"""
    # clipping 防止溢出
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def forward(X, w, b):
    """
    前向传播：z = X @ w + b,  p = sigmoid(z)
    返回: p (N,1)
    """
    z = X @ w + b
    return sigmoid(z)


def bce_loss(y, p, eps=1e-12):
    """
    Binary Cross-Entropy：
        L = -(1/N) Σ [y*log(p) + (1-y)*log(1-p)]
    """
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def backward(X, y, p):
    """
    反向传播（解析梯度）：
        ∂L/∂z = p - y        (N, 1)
        ∂L/∂w = (1/N) X.T @ (p - y)   (D, 1)
        ∂L/∂b = (1/N) Σ (p - y)       (1,)
    """
    N = X.shape[0]
    dz = p - y                     # 误差项
    dw = X.T @ dz / N              # (D,1)
    db = dz.mean(axis=0, keepdims=True)   # (1,1)
    return dw, db


# ============================================================
# 3. 用解析梯度做梯度下降训练 200 epoch
# ============================================================
def train_by_gradient_descent(X, y, D, epochs=200, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.1, size=(D, 1))
    b = np.zeros((1, 1))

    losses = []
    for epoch in range(1, epochs + 1):
        p = forward(X, w, b)
        loss = bce_loss(y, p)
        dw, db = backward(X, y, p)

        w -= lr * dw
        b -= lr * db

        losses.append(loss)
        if epoch % 20 == 0 or epoch == 1:
            acc = ((p >= 0.5).astype(int) == y).mean()
            print(f"  epoch {epoch:3d}: loss = {loss:.6f},  acc = {acc:.4f}")
    return w, b, losses


# ============================================================
# 4. 数值梯度验证（有限差分）
# ============================================================
def numerical_gradient(theta_flat, X, y, h=1e-5):
    """
    把 (w, b) 压扁成一个向量 theta；
    对每个维度 i 做： (f(theta + h*e_i) - f(theta - h*e_i)) / (2h)
    返回与 theta 形状相同的数值梯度向量。
    """
    grad = np.zeros_like(theta_flat)
    for i in range(len(theta_flat)):
        old = theta_flat[i]
        theta_flat[i] = old + h
        lp = loss_from_flat(theta_flat, X, y)
        theta_flat[i] = old - h
        ln = loss_from_flat(theta_flat, X, y)
        theta_flat[i] = old
        grad[i] = (lp - ln) / (2 * h)
    return grad


def loss_from_flat(theta_flat, X, y):
    """把扁平参数恢复成 (w,b) 并计算 BCE loss"""
    D = X.shape[1]
    w = theta_flat[:D].reshape(D, 1)
    b = theta_flat[D:].reshape(1, 1)
    p = forward(X, w, b)
    return bce_loss(y, p)


def grad_from_flat(theta_flat, X, y):
    """把扁平参数恢复成 (w,b) 并计算解析梯度，再扁平返回"""
    D = X.shape[1]
    w = theta_flat[:D].reshape(D, 1)
    b = theta_flat[D:].reshape(1, 1)
    p = forward(X, w, b)
    dw, db = backward(X, y, p)
    return np.concatenate([dw.flatten(), db.flatten()])


def check_gradient(X, y, seed=0):
    """用一小部分数据（如 20 个样本）做数值梯度 vs 解析梯度的对比"""
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    w = rng.normal(scale=0.5, size=(D, 1))
    b = rng.normal(scale=0.5, size=(1, 1))
    theta = np.concatenate([w.flatten(), b.flatten()])

    g_analytic = grad_from_flat(theta.copy(), X, y)
    g_numeric  = numerical_gradient(theta.copy(), X, y)

    print("  参数 theta =", theta.round(4))
    print("  解析梯度   =", g_analytic.round(6))
    print("  数值梯度   =", g_numeric.round(6))

    # 相对误差 = ||g_a - g_n|| / (||g_a|| + ||g_n||)
    diff = np.linalg.norm(g_analytic - g_numeric)
    norm_sum = np.linalg.norm(g_analytic) + np.linalg.norm(g_numeric) + 1e-12
    rel_err = diff / norm_sum
    print(f"  L2 差值   = {diff:.2e}")
    print(f"  相对误差   = {rel_err:.2e}")
    print("  (若相对误差 < 1e-6，说明解析梯度推导正确 ✅)")


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    print()
    print("*" * 60)
    print("*  矩阵求导 & 链式法则：从 0 实现 LR + 梯度校验")
    print("*" * 60)

    # --- 数据 ---
    section("1. 构造二分类 toy dataset（两个高斯团）")
    X, y = make_toy_dataset(n_per_class=200, seed=42)
    D = X.shape[1]
    print(f"X.shape = {X.shape},  y.shape = {y.shape}")
    print(f"正样本数 = {int(y.sum())},  负样本数 = {int((1 - y).sum())}")
    print(f"前 5 行 X =")
    print(X[:5].round(3))

    # --- 前向 / 反向测试（小样本，直观理解） ---
    section("2. 前向 & 反向（解析梯度）演示 —— 用一个 5 样本小 batch")
    rng = np.random.default_rng(0)
    w_demo = rng.normal(scale=0.1, size=(D, 1))
    b_demo = np.zeros((1, 1))
    X_small = X[:5]
    y_small = y[:5]
    p_demo = forward(X_small, w_demo, b_demo)
    print(f"w = {w_demo.flatten().round(4)},  b = {b_demo.flatten().round(4)}")
    print(f"p = sigmoid(Xw+b) = {p_demo.flatten().round(4)}")
    print(f"BCE loss  = {bce_loss(y_small, p_demo):.6f}")
    dw_demo, db_demo = backward(X_small, y_small, p_demo)
    print(f"∂L/∂w = {dw_demo.flatten().round(6)}")
    print(f"∂L/∂b = {db_demo.flatten().round(6)}")
    print()
    print("链式法则回顾：")
    print("  p = σ(Xw+b),  L = BCE(y,p)")
    print("  → ∂L/∂z = p - y     （关键简化！）")
    print("  → ∂L/∂w = X.T @ (p-y) / N,  ∂L/∂b = mean(p-y)")

    # --- 训练 200 epoch ---
    section("3. 用解析梯度下降训练 200 个 epoch")
    w_star, b_star, losses = train_by_gradient_descent(
        X, y, D=D, epochs=200, lr=0.5, seed=1
    )
    print("\n训练完毕。前 5 个 loss：", [round(l, 4) for l in losses[:5]])
    print("          后 5 个 loss：", [round(l, 4) for l in losses[-5:]])
    print(f"最终 w* = {w_star.flatten().round(4)},  b* = {b_star.flatten().round(4)}")
    p_final = forward(X, w_star, b_star)
    print(f"最终准确率  = {((p_final >= 0.5).astype(int) == y).mean():.4f}")

    # --- 数值梯度校验 ---
    section("4. 数值梯度 vs 解析梯度（有限差分校验）")
    # 用小一点的数据做数值梯度（数值梯度 O(d) 次前向，慢）
    small_idx = np.random.default_rng(7).choice(X.shape[0], size=20, replace=False)
    check_gradient(X[small_idx], y[small_idx], seed=3)

    print()
    print("=" * 60)
    print('  Demo 结束：理解"反向传播 = 链式法则下的梯度计算"')
    print("=" * 60)
