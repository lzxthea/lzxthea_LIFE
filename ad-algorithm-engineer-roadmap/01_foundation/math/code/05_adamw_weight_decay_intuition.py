"""
05_adamw_weight_decay_intuition.py
====================================
主题：AdamW —— 权重衰减与 L2 正则化的差异（广告/推荐算法常用）
====================================

任务：
    y = 2 * x1 - 3 * x2 + noise,  N = 200

对比三种训练方式：
    1) Baseline   ：不加任何正则
    2) Adam + L2  ：把 L2 正则放进 loss 中 —— loss = MSE + 0.5*lambda*||w||^2
    3) AdamW      ：把"权重衰减"从优化器里独立出来 —— w = w - lr*(grad + lambda*w)

关键直觉：
    在 SGD（固定 lr）下，"L2 正则"等价于"权重衰减"：
        L2:   grad_total = grad + lambda * w      =>  w ← w - lr*(grad + lambda*w)
        WD:   w ← (1 - lr*lambda)*w - lr*grad     （与上式相同）

    但在 Adam 下，L2 正则不再等价于权重衰减：
        Adam 实际步长是 lr / sqrt(v_hat)   （自适应）
        所以 L2 项的有效"衰减系数"会随 sqrt(v_hat) 变化：
            w ← w - (lr / sqrt(v_hat)) * (grad + lambda*w)
        梯度大的参数衰减慢，梯度小的参数反而衰减快 —— 这并不合理

    AdamW（Loshchilov & Hutter, 2019）把权重衰减从梯度中剥离出来：
        grad = grad_of_MSE
        m, v 更新照常
        w ← w - lr * (m_hat / (sqrt(v_hat) + eps) + lambda * w)
    这样权重衰减对每个参数以同样比例生效，与 Adam 的自适应 lr 解耦。

为什么在 Transformer / CTR 模型中推荐 AdamW：
    - Embedding 层参数极多（词表 * 维度），且梯度稀疏
    - 稀疏梯度让 v 很小 → Adam 里 L2 正则对 Embedding 层衰减过强，权重被过度压制
    - AdamW 的解耦衰减更稳定，Embedding 权重保持在合理范围
"""

import numpy as np


# ============================================================
# 1. 生成数据：y = 2*x1 - 3*x2 + noise
# ============================================================
def generate_data(N=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 2))
    w_true = np.array([2.0, -3.0], dtype=np.float64)
    y = X @ w_true + rng.normal(0, 0.5, size=N)  # 加噪声
    return X, y, w_true


# ============================================================
# 2. 通用工具：MSE loss、MSE 梯度
# ============================================================
def mse_loss(X, y, w, b):
    """L = mean((X @ w + b - y)^2)"""
    residual = X @ w + b - y
    return float(np.mean(residual ** 2))


def mse_grad(X, y, w, b):
    """dL/dw = (2/N) * X^T @ (Xw + b - y)
       dL/db = (2/N) * sum(Xw + b - y)"""
    N = X.shape[0]
    residual = X @ w + b - y
    grad_w = (2.0 / N) * X.T @ residual
    grad_b = (2.0 / N) * np.sum(residual)
    return grad_w, grad_b


# ============================================================
# 3. 三种训练方式
# ============================================================

def train_baseline(X, y, steps=200, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, seed=1):
    """
    3-1) Baseline：纯粹的 Adam 训练，不加任何正则。
    预期：权重最大，loss 最低（因为没有衰减压力）。
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(2) * 2.0
    b = 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = 0.0
    v_b = 0.0

    history = []
    for t in range(1, steps + 1):
        g_w, g_b = mse_grad(X, y, w, b)

        m_w = b1 * m_w + (1 - b1) * g_w
        v_w = b2 * v_w + (1 - b2) * g_w ** 2
        m_b = b1 * m_b + (1 - b1) * g_b
        v_b = b2 * v_b + (1 - b2) * g_b ** 2

        m_hat_w = m_w / (1 - b1 ** t)
        v_hat_w = v_w / (1 - b2 ** t)
        m_hat_b = m_b / (1 - b1 ** t)
        v_hat_b = v_b / (1 - b2 ** t)

        w = w - lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
        b = b - lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        history.append((t, mse_loss(X, y, w, b), float(np.linalg.norm(w))))
    return w, b, history


def train_adam_l2(X, y, steps=200, lr=0.1, lam=0.1, b1=0.9, b2=0.999, eps=1e-8, seed=1):
    """
    3-2) Adam + L2 正则：把 L2 放进 loss 里。
        loss = MSE + 0.5 * lam * ||w||^2
        grad_w (total) = grad_w (MSE) + lam * w

    关键问题：
        在 Adam 下"grad_w + lam * w" 被除以 sqrt(v_hat_w)，
        即 L2 项的有效系数是 lam / sqrt(v_hat_w)。
        - 对于梯度大的参数（v_hat 大）→ lam / sqrt(v_hat) 小 → 衰减弱
        - 对于梯度小的参数（v_hat 小）→ lam / sqrt(v_hat) 大 → 衰减强
        衰减强度与梯度大小挂钩，**不稳定**。
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(2) * 2.0
    b = 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = 0.0
    v_b = 0.0

    history = []
    for t in range(1, steps + 1):
        g_w, g_b = mse_grad(X, y, w, b)
        g_w = g_w + lam * w  # <- L2 正则项加进梯度里

        m_w = b1 * m_w + (1 - b1) * g_w
        v_w = b2 * v_w + (1 - b2) * g_w ** 2
        m_b = b1 * m_b + (1 - b1) * g_b
        v_b = b2 * v_b + (1 - b2) * g_b ** 2

        m_hat_w = m_w / (1 - b1 ** t)
        v_hat_w = v_w / (1 - b2 ** t)
        m_hat_b = m_b / (1 - b1 ** t)
        v_hat_b = v_b / (1 - b2 ** t)

        w = w - lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
        b = b - lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        # 记录"纯 MSE loss"（不算正则项），便于公平对比
        history.append((t, mse_loss(X, y, w, b), float(np.linalg.norm(w))))
    return w, b, history


def train_adamw(X, y, steps=200, lr=0.1, lam=0.1, b1=0.9, b2=0.999, eps=1e-8, seed=1):
    """
    3-3) AdamW：把权重衰减从梯度中"解耦"出来。
        梯度 grad = grad_of_MSE（不含正则）
        m, v 的更新只用纯 MSE 梯度
        更新权重时额外多一项权重衰减：
            w ← w - lr * ( m_hat / (sqrt(v_hat) + eps) + lam * w )

    这样：
        - 权重衰减项 lam * w 直接按 lr 的比例缩放 w
        - 不会被 sqrt(v_hat) 再缩放一次
        - 每个参数的衰减强度是相同的、稳定的
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(2) * 2.0
    b = 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = 0.0
    v_b = 0.0

    history = []
    for t in range(1, steps + 1):
        g_w, g_b = mse_grad(X, y, w, b)  # 纯 MSE 梯度，不含正则

        # m, v 只积累"任务梯度"（不含正则项）
        m_w = b1 * m_w + (1 - b1) * g_w
        v_w = b2 * v_w + (1 - b2) * g_w ** 2
        m_b = b1 * m_b + (1 - b1) * g_b
        v_b = b2 * v_b + (1 - b2) * g_b ** 2

        m_hat_w = m_w / (1 - b1 ** t)
        v_hat_w = v_w / (1 - b2 ** t)
        m_hat_b = m_b / (1 - b1 ** t)
        v_hat_b = v_b / (1 - b2 ** t)

        # 解耦权重衰减：w 的更新多一项 lr * lam * w
        w = w - lr * (m_hat_w / (np.sqrt(v_hat_w) + eps) + lam * w)
        b = b - lr * m_hat_b / (np.sqrt(v_hat_b) + eps)  # 通常 bias 不衰减

        history.append((t, mse_loss(X, y, w, b), float(np.linalg.norm(w))))
    return w, b, history


# ============================================================
# 4. 主流程
# ============================================================

def print_sep(title=None):
    line = "=" * 78
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


if __name__ == "__main__":

    X, y, w_true = generate_data(N=200, seed=42)

    print_sep("数据概况")
    print(f"  y = 2*x1 - 3*x2 + noise,   N = {X.shape[0]}")
    print(f"  真实权重 w_true = [{w_true[0]:>+.2f}, {w_true[1]:>+.2f}],  ||w_true|| = {np.linalg.norm(w_true):.4f}")

    STEPS = 300
    LR = 0.15
    LAM = 0.05

    w0_b, b0_b, hist_b = train_baseline(X, y, steps=STEPS, lr=LR, seed=1)
    w0_l2, b0_l2, hist_l2 = train_adam_l2(X, y, steps=STEPS, lr=LR, lam=LAM, seed=1)
    w0_w, b0_w, hist_w = train_adamw(X, y, steps=STEPS, lr=LR, lam=LAM, seed=1)

    # --- 每 50 步打印 ---
    print_sep(f"训练过程（每 50 步打印，lambda = {LAM}）")
    header = f"{'Step':>6s}"
    for label in ["Baseline loss", "||w||", "Adam+L2 loss", "||w||", "AdamW loss", "||w||"]:
        header += f"  {label:>12s}"
    print(header)
    for i in range(0, STEPS):
        if (i + 1) % 50 == 0:
            t_b, l_b, n_b = hist_b[i]
            t_l, l_l, n_l = hist_l2[i]
            t_w, l_w, n_w = hist_w[i]
            assert t_b == t_l == t_w == (i + 1)
            print(
                f"{t_b:>6d}  "
                f"{l_b:>12.6f}  {n_b:>12.4f}  "
                f"{l_l:>12.6f}  {n_l:>12.4f}  "
                f"{l_w:>12.6f}  {n_w:>12.4f}"
            )

    # --- 最终权重对比 ---
    print_sep("最终权重对比")
    print(f"{'方法':<16s}  {'w1':>10s}  {'w2':>10s}  {'b':>10s}  "
          f"{'||w||_2':>10s}  {'MSE':>10s}")
    print("-" * 78)

    def row(name, w, b, loss):
        return (f"{name:<16s}  {w[0]:>+10.6f}  {w[1]:>+10.6f}  "
                f"{b:>+10.6f}  {np.linalg.norm(w):>10.6f}  {loss:>10.6f}")

    print(row("Baseline",      w0_b, b0_b, mse_loss(X, y, w0_b, b0_b)))
    print(row("Adam + L2",     w0_l2, b0_l2, mse_loss(X, y, w0_l2, b0_l2)))
    print(row("AdamW",         w0_w, b0_w, mse_loss(X, y, w0_w, b0_w)))
    print(row("True w",        w_true, 0.0, mse_loss(X, y, w_true, 0.0)))

    # --- 关键解释 ---
    print_sep("核心机制：为什么 Adam 中 L2 ≠ weight decay")
    print(
        "  SGD（固定 lr）：\n"
        "      w ← w - lr*(grad + lam*w)   ≡   w ← (1 - lr*lam)*w - lr*grad\n"
        "      两种写法完全等价，L2 == weight decay。\n"
    )
    print(
        "  Adam（自适应 lr）：\n"
        "      实际步长 = lr / sqrt(v_hat)，会随着每个参数的历史梯度平方变化。\n"
        "      Adam + L2：w ← w - (lr/sqrt(v_hat))*(grad + lam*w)\n"
        "          → 对梯度小的参数（v_hat 小）衰减系数被放大 → 过度衰减\n"
        "          → 对梯度大的参数（v_hat 大）衰减系数被压缩 → 衰减不足\n"
        "      AdamW：   w ← w - lr*(grad_hat + lam*w)      （grad_hat 是自适应项，lam*w 独立）\n"
        "          → 权重衰减独立生效，不受自适应 lr 影响 → 更稳定"
    )
    print_sep()

    # --- 为什么推荐 AdamW（针对广告 / 推荐模型） ---
    print_sep("为什么 Transformer / CTR 模型推荐用 AdamW")
    print(
        "  1) Embedding 层参数极多：词表(1e5~1e7) × 维度(64~512)，占总参数大头。\n"
        "  2) Embedding 梯度稀疏：每次只有少数 token 出现，多数 token 这一步梯度为 0。\n"
        "  3) 在 Adam 中，v_hat 会很小 → L2 项被 / sqrt(v_hat) 放大 → Embedding 权重被过度压制。\n"
        "  4) AdamW 解耦了权重衰减，lam * w 直接生效，Embedding 权重衰减稳定、可控。\n"
        "  5) 实证：BERT / GPT / DIN / DIEN 等主流模型都采用 AdamW，泛化更好。\n"
        "     （PyTorch 官方：torch.optim.AdamW；注意把 weight_decay 当作超参来调）"
    )
    print_sep()
