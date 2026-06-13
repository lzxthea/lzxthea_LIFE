"""
04_gradient_descent_variants.py
================================
主题：凸优化 —— GD vs SGD vs Momentum vs RMSProp vs Adam 的收敛对比
================================

本脚本在一个 2D 凸函数上对比 5 种常见优化器的收敛行为：
    f(w) = 0.5 * (w1^2 + 10 * w2^2)

- 最优点 w* = (0, 0)，最优损失 f* = 0
- w2 方向的曲率是 w1 的 10 倍，这是一个典型的「病态条件数」问题
  用来突出不同优化器在不同曲率方向上的表现差异

只用 NumPy 实现，便于理解内部原理。

对应 PyTorch API：
    - torch.optim.SGD, torch.optim.SGD(momentum=...)
    - torch.optim.RMSprop
    - torch.optim.Adam / AdamW
"""

import numpy as np


# ============================================================
# 1. 定义目标函数 f(w) 及其梯度
# ============================================================
# f(w) = 0.5 * (w1^2 + 10 * w2^2)
# df/dw1 = w1
# df/dw2 = 10 * w2
# ============================================================
def f(w):
    """2D 二次凸函数。"""
    w1, w2 = w[0], w[1]
    return 0.5 * (w1 ** 2 + 10 * w2 ** 2)


def grad_f(w, stochastic=False):
    """解析梯度；当 stochastic=True 时加入高斯噪声模拟随机梯度。"""
    g = np.array([w[0], 10 * w[1]], dtype=np.float64)
    if stochastic:
        # 模拟"样本噪声"：N(0, 0.3^2)
        g = g + np.random.randn(2) * 0.3
    return g


# ============================================================
# 2. 定义 5 个优化器
# ============================================================

def optimize_vanilla_gd(w0, lr, steps, rng):
    """
    1) Vanilla GD（批量梯度下降）
    ------------------------------------------------------------
    更新公式：
        w_{t+1} = w_t - lr * grad_t

    解决的问题：无，这是最朴素的一阶算法
    - 优点：稳定，在凸问题上单调下降
    - 缺点：① 需要遍历整个数据集才能算一次 grad，大数据时内存大、速度慢
            ② 对不同曲率方向使用相同 lr，病态条件数下收敛慢
    对应 PyTorch API：torch.optim.SGD（无 momentum）
    """
    w = np.array(w0, dtype=np.float64)
    traj = [w.copy()]
    losses = [f(w)]
    for _ in range(steps):
        g = grad_f(w, stochastic=False)
        w = w - lr * g
        traj.append(w.copy())
        losses.append(f(w))
    return traj, losses


def optimize_sgd(w0, lr, steps, rng):
    """
    2) SGD（随机梯度下降）
    ------------------------------------------------------------
    更新公式：
        w_{t+1} = w_t - lr * grad_stochastic_t
    （这里通过在真实梯度上加高斯噪声模拟 "单个样本/小批量"）

    解决的问题：大数据集下 batch-GD 的内存和速度瓶颈
    - 每次只用 1 个 / 一小批样本计算梯度，内存小、迭代快
    - 缺点：梯度有噪声，最终解周围抖动，不能精确收敛
    对应 PyTorch API：torch.optim.SGD
    """
    w = np.array(w0, dtype=np.float64)
    traj = [w.copy()]
    losses = [f(w)]
    for _ in range(steps):
        g = grad_f(w, stochastic=True)  # 带噪声的随机梯度
        w = w - lr * g
        traj.append(w.copy())
        losses.append(f(w))
    return traj, losses


def optimize_momentum(w0, lr, steps, rng, beta=0.9):
    """
    3) Momentum（动量）
    ------------------------------------------------------------
    更新公式：
        v_{t+1} = beta * v_t + grad_t
        w_{t+1} = w_t - lr * v_{t+1}

    解决的问题：① 高曲率方向上 GD 震荡；② 鞍点附近梯度接近于 0 时被卡住
    - 引入"速度" v：对历史梯度做指数滑动平均
    - 在曲率大的方向上 v 会"互相抵消"，收敛更平滑
    - 在平坦/鞍点方向上 v 积累的历史梯度提供"惯性"，帮助冲出
    对应 PyTorch API：torch.optim.SGD(momentum=beta)
    """
    w = np.array(w0, dtype=np.float64)
    v = np.zeros(2, dtype=np.float64)
    traj = [w.copy()]
    losses = [f(w)]
    for _ in range(steps):
        g = grad_f(w, stochastic=False)
        v = beta * v + g
        w = w - lr * v
        traj.append(w.copy())
        losses.append(f(w))
    return traj, losses


def optimize_rmsprop(w0, lr, steps, rng, beta=0.9, eps=1e-6):
    """
    4) RMSProp
    ------------------------------------------------------------
    更新公式：
        s_{t+1} = beta * s_t + (1 - beta) * grad_t^2   (逐元素平方)
        w_{t+1} = w_t - lr * grad_t / (sqrt(s_{t+1}) + eps)

    解决的问题：不同参数方向上使用相同 lr（自适应学习率）
    - s 记录每个参数方向上梯度平方的指数平均
    - 梯度大（曲率大）的方向上 sqrt(s) 大 → 实际步长小
    - 梯度小（曲率小）的方向上 sqrt(s) 小 → 实际步长大
    - 实现了"每个方向自适应 lr"，解决病态条件数问题
    对应 PyTorch API：torch.optim.RMSprop
    """
    w = np.array(w0, dtype=np.float64)
    s = np.zeros(2, dtype=np.float64)
    traj = [w.copy()]
    losses = [f(w)]
    for _ in range(steps):
        g = grad_f(w, stochastic=False)
        s = beta * s + (1 - beta) * g ** 2
        w = w - lr * g / (np.sqrt(s) + eps)
        traj.append(w.copy())
        losses.append(f(w))
    return traj, losses


def optimize_adam(w0, lr, steps, rng, b1=0.9, b2=0.999, eps=1e-8):
    """
    5) Adam（Adaptive Moment Estimation）
    ------------------------------------------------------------
    更新公式：
        m_{t+1} = b1 * m_t + (1 - b1) * grad_t     (一阶矩，即动量)
        v_{t+1} = b2 * v_t + (1 - b2) * grad_t^2   (二阶矩)

        m_hat = m / (1 - b1^t)      (偏置校正，因为 m、v 初始为 0 时前几轮偏小)
        v_hat = v / (1 - b2^t)

        w_{t+1} = w_t - lr * m_hat / (sqrt(v_hat) + eps)

    解决的问题：结合 Momentum 和 RMSProp 的优点
    - m 提供动量（历史梯度的滑动平均）
    - v 提供每个方向自适应 lr
    - 额外加上 bias correction：修正 m 和 v 的初始偏置
    - 在训练早期 m、v 都很小，校正后早期不会过慢
    对应 PyTorch API：torch.optim.Adam
    """
    w = np.array(w0, dtype=np.float64)
    m = np.zeros(2, dtype=np.float64)
    v = np.zeros(2, dtype=np.float64)
    traj = [w.copy()]
    losses = [f(w)]
    for t in range(1, steps + 1):
        g = grad_f(w, stochastic=False)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        # 偏置校正：m、v 初始为 0，早期估计有偏
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(w.copy())
        losses.append(f(w))
    return traj, losses


# ============================================================
# 3. 主流程
# ============================================================

def print_sep(title=None):
    line = "=" * 70
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


if __name__ == "__main__":

    rng = np.random.default_rng(42)

    # 所有优化器从同一个远离最优点的位置出发
    W0 = [10.0, 10.0]
    STEPS = 80

    # 学习率：为每个优化器手工调，保证可以稳定收敛
    optimizers = {
        "Vanilla GD": (optimize_vanilla_gd, 0.09),
        "SGD(噪声)": (optimize_sgd, 0.05),
        "Momentum": (optimize_momentum, 0.05),
        "RMSProp": (optimize_rmsprop, 0.3),
        "Adam": (optimize_adam, 0.3),
    }

    print_sep("优化器对比：2D 凸函数 f(w) = 0.5*(w1^2 + 10*w2^2)，起点 w0=[10,10]")
    print(f"{'Step':<8s}  ", end="")
    for name in optimizers:
        print(f"{name:>14s}", end="")
    print()

    histories = {}
    for name, (fn, lr) in optimizers.items():
        traj, losses = fn(W0, lr, STEPS, rng)
        histories[name] = (traj, losses)

    # 每 10 步打印一次 loss
    for step in range(0, STEPS + 1):
        if step % 10 == 0:
            print(f"step {step:<4d}", end="")
            for name in optimizers:
                _, losses = histories[name]
                print(f"{losses[step]:>14.6f}", end="")
            print()

    # 最终结果
    print_sep("最终结果（经过 80 步后）")
    print(f"{'优化器':<18s}  {'最终 loss':>12s}  {'w1':>10s}  {'w2':>10s}  {'||w||_2':>10s}")
    print("-" * 70)
    for name in optimizers:
        traj, losses = histories[name]
        w_final = traj[-1]
        print(
            f"{name:<18s}  {losses[-1]:>12.6f}  "
            f"{w_final[0]:>+10.6f}  {w_final[1]:>+10.6f}  "
            f"{np.linalg.norm(w_final):>10.6f}"
        )

    print()
    print_sep("结论（肉眼观察：")
    print("  - Vanilla GD：稳定但在 w2 方向（曲率大）收敛慢，每步只走 0.09*w2 方向走得慢")
    print("  - SGD：有噪声，loss 抖动，最终未精确收敛到 0")
    print("  - Momentum：借助历史梯度积累在 w2 方向走得更快，整体收敛比 GD 更快")
    print("  - RMSProp：自适应 lr，w2 方向梯度大 → 实际步长被压缩，w1 方向步长放大")
    print("  - Adam：Momentum + RMSProp + bias-correction，综合表现最好")
    print("  - 在广告/推荐训练中常用 Adam / AdamW，因收敛快且稳定")
    print_sep()
