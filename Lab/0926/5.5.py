import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# ---------------------- 1. 定义损失函数 ----------------------
def J0(theta0, theta1):
    """原损失函数：J₀(θ) = (θ₀ - 2)² + (θ₁ - 2)²（中心在(2,2)的椭圆）"""
    return (theta0 - 2) ** 2 + (theta1 - 2) ** 2

def J_L2(theta0, theta1, lambda2=0.1):
    """L2正则化损失：J₀(θ) + λ₂||θ||²"""
    return J0(theta0, theta1) + lambda2 * (theta0**2 + theta1**2)

def J_L1(theta0, theta1, lambda1=0.5):
    """L1正则化损失：J₀(θ) + λ₁||θ||₁"""
    return J0(theta0, theta1) + lambda1 * (np.abs(theta0) + np.abs(theta1))

# 用于数值优化的向量形式损失函数
def l2_loss(theta, lambda2):
    theta0, theta1 = theta
    return (theta0 - 2)**2 + (theta1 - 2)**2 + lambda2 * (theta0**2 + theta1**2)

def l1_loss(theta, lambda1):
    theta0, theta1 = theta
    return (theta0 - 2)**2 + (theta1 - 2)**2 + lambda1 * (np.abs(theta0) + np.abs(theta1))


# ---------------------- 2. 生成网格数据 ----------------------
theta0 = np.linspace(-0.5, 3.5, 200)  # 提高分辨率
theta1 = np.linspace(-0.5, 3.5, 200)
Theta0, Theta1 = np.meshgrid(theta0, theta1)


# ---------------------- 3. 计算各损失函数的值 ----------------------
J0_vals = J0(Theta0, Theta1)
lambda2 = 1.2      # L2 正则强度
lambda1 = 2.5 # L1 正则强度
J_L2_vals = J_L2(Theta0, Theta1, lambda2)
J_L1_vals = J_L1(Theta0, Theta1, lambda1)


# ---------------------- 4. 解析求解 L2 正则最优参数 ----------------------
theta_l2_0 = 2 / (1 + lambda2)  # λ2=1 → 2/(1+1)=1.0
theta_l2_1 = 2 / (1 + lambda2)


# ---------------------- 5. 数值求解 L1 正则最优参数 ----------------------
res_l1 = minimize(
    fun=lambda theta: l1_loss(theta, lambda1),
    x0=np.array([2.0, 2.0]),
    method='TNC',
    bounds=[(-3, 3), (-3, 3)],
    options={'maxiter': 1000}
)

if res_l1.success:
    theta_l1_0, theta_l1_1 = res_l1.x
else:
    print("L1 正则优化失败:", res_l1.message)
    theta_l1_0, theta_l1_1 = 2.0, 2.0

print(f"L2 正则最优解 (解析): θ₀={theta_l2_0:.4f}, θ₁={theta_l2_1:.4f}")
print(f"L1 正则最优解 (数值): θ₀={theta_l1_0:.4f}, θ₁={theta_l1_1:.4f}")


# ---------------------- 6. 绘制等高线图（增强版） ----------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 共用设置
label_fontsize = 12   # 增大标签字体
title_fontsize = 14
levels_dense = 15     # 增加等高线数量（更密集）

# ---------- 子图1：无正则化 ----------
ax = axes[0]
contour = ax.contour(Theta0, Theta1, J0_vals, levels=levels_dense, cmap='coolwarm', alpha=0.8, linewidths=1.2)
ax.clabel(contour, inline=True, fontsize=label_fontsize, fmt="%.2f", colors='black')  # 更大更清晰
ax.plot(2, 2, 'k*', markersize=14, label=r'$\theta^* = (2,2)$')
ax.set_title(r'$J_0(\theta)$ 无正则化', fontsize=title_fontsize)
ax.set_xlabel(r'$\theta_0$', fontsize=12)
ax.set_ylabel(r'$\theta_1$', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# ---------- 子图2：L2正则化 ----------
ax = axes[1]
# 原损失等高线（蓝色，更密集）
contour_J0 = ax.contour(Theta0, Theta1, J0_vals, levels=levels_dense, colors='blue', alpha=0.5, linestyles='--', linewidths=1)
ax.clabel(contour_J0, inline=True, fontsize=label_fontsize-2, fmt="%.2f", colors='blue')
# L2 正则项等值线（红色虚线圆）
L2_norm = Theta0**2 + Theta1**2
contour_L2 = ax.contour(Theta0, Theta1, L2_norm, levels=np.arange(0.5, 3.1, 0.5),  # 更密的正则等值线
                        colors='red', linestyles='--', alpha=0.8, linewidths=1.5)
ax.clabel(contour_L2, inline=True, fontsize=label_fontsize-2, colors='red', fmt="%.1f")
# 最优解
ax.plot(theta_l2_0, theta_l2_1, 'r*', markersize=14, label=fr'$\theta^*_\lambda = ({theta_l2_0:.2f}, {theta_l2_1:.2f})$')
ax.plot(2, 2, 'k*', markersize=14, label=r'$\theta^* = (2,2)$')
ax.set_title(r'L2 正则化', fontsize=title_fontsize)
ax.set_xlabel(r'$\theta_0$', fontsize=12)
ax.set_ylabel(r'$\theta_1$', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# ---------- 子图3：L1正则化 ----------
ax = axes[2]
# 原损失等高线（蓝色）
contour_J0 = ax.contour(Theta0, Theta1, J0_vals, levels=levels_dense, colors='blue', alpha=0.5, linestyles='--', linewidths=1)
ax.clabel(contour_J0, inline=True, fontsize=label_fontsize-2, fmt="%.2f", colors='blue')
# L1 正则项等值线（菱形）
L1_norm = np.abs(Theta0) + np.abs(Theta1)
contour_L1 = ax.contour(Theta0, Theta1, L1_norm, levels=np.arange(0.5, 3.1, 0.5),
                        colors='red', linestyles='--', alpha=0.8, linewidths=1.5)
ax.clabel(contour_L1, inline=True, fontsize=label_fontsize-2, colors='red', fmt="%.1f")
# 最优解
ax.plot(theta_l1_0, theta_l1_1, 'r*', markersize=14, label=fr'$\theta^*_\lambda = ({theta_l1_0:.2f}, {theta_l1_1:.2f})$')
ax.plot(2, 2, 'k*', markersize=14, label=r'$\theta^* = (2,2)$')
ax.set_title(r'L1 正则化', fontsize=title_fontsize)
ax.set_xlabel(r'$\theta_0$', fontsize=12)
ax.set_ylabel(r'$\theta_1$', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 整体调整
plt.tight_layout(pad=2.0)  # 增加子图间距，避免拥挤
plt.savefig('regularization_comparison_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()