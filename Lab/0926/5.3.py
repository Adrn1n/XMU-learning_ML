import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# 定义真实函数: x³ - 3x² + 2x + 1
def true_function(x):
    return x ** 3 - 3 * x ** 2 + 2 * x + 1


# 生成原始数据
np.random.seed(42)  # 设置随机种子，保证结果可复现
x_all = np.linspace(0, 2.5, 100)  # 生成更多基础点
y_true_all = true_function(x_all)
noise_all = np.random.normal(0, 0.2, size=x_all.shape)  # 生成噪声
y_all = y_true_all + noise_all  # 添加噪声的观测值

# 分割为训练集和测试集（随机采样）
train_indices = np.random.choice(len(x_all), 30, replace=False)  # 随机选择30个点作为训练集
test_indices = np.setdiff1d(np.arange(len(x_all)), train_indices)  # 剩余点作为测试集

x_train = x_all[train_indices]
y_train = y_all[train_indices]
x_test = x_all[test_indices]
y_test = y_all[test_indices]


# 定义最小二乘法解析解函数
def least_squares_fit(x, y, degree):
    # 构建设计矩阵X，每一列是x的i次方
    X = np.vander(x, degree + 1, increasing=True)

    # 计算解析解: w = (X^T X)^(-1) X^T y
    w = np.linalg.inv(X.T @ X) @ X.T @ y

    # 返回系数和拟合函数
    def fit_func(x_new):
        X_new = np.vander(x_new, degree + 1, increasing=True)
        return X_new @ w

    return w, fit_func


# 分别进行1阶(线性)、3阶和9阶多项式拟合（仅使用训练集）
_, linear_fit = least_squares_fit(x_train, y_train, 1)
_, poly3_fit = least_squares_fit(x_train, y_train, 3)
_, poly9_fit = least_squares_fit(x_train, y_train, 9)

# 生成用于绘制拟合曲线的密集点
x_plot = np.linspace(0, 2.5, 200)

# 创建图形
plt.figure(figsize=(18, 5))

# 绘制线性拟合
plt.subplot(1, 3, 1)
plt.scatter(x_train, y_train, c='blue', alpha=0.6, label='训练数据点')
plt.scatter(x_test, y_test, c='green', alpha=0.6, label='测试数据点')
plt.plot(x_plot, linear_fit(x_plot), 'r-', label='线性拟合')
plt.plot(x_plot, true_function(x_plot), 'k--', label='真实函数')
plt.title('线性拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0, 3.5)
plt.grid(True, alpha=0.3)
plt.legend()

# 绘制3阶多项式拟合
plt.subplot(1, 3, 2)
plt.scatter(x_train, y_train, c='blue', alpha=0.6, label='训练数据点')
plt.scatter(x_test, y_test, c='green', alpha=0.6, label='测试数据点')
plt.plot(x_plot, poly3_fit(x_plot), 'r-', label='3阶多项式拟合')
plt.plot(x_plot, true_function(x_plot), 'k--', label='真实函数')
plt.title('3阶多项式拟合')
plt.xlabel('x')
plt.ylim(0, 3.5)
plt.grid(True, alpha=0.3)
plt.legend()

# 绘制9阶多项式拟合 - 过拟合示例
plt.subplot(1, 3, 3)
plt.scatter(x_train, y_train, c='blue', alpha=0.6, label='训练数据点')
plt.scatter(x_test, y_test, c='green', alpha=0.6, label='测试数据点')
plt.plot(x_plot, poly9_fit(x_plot), 'r-', label='9阶多项式拟合')
plt.plot(x_plot, true_function(x_plot), 'k--', label='真实函数')
plt.title('9阶多项式拟合（过拟合）')
plt.xlabel('x')
plt.ylim(0, 3.5)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# 打印3阶多项式的拟合系数，理论上应该接近[1, 2, -3, 1]
print("3阶多项式的拟合系数(应接近[1, 2, -3, 1]):")
w3, _ = least_squares_fit(x_train, y_train, 3)
print(w3)


# 计算并比较不同模型的误差
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


linear_train_mse = calculate_mse(y_train, linear_fit(x_train))
linear_test_mse = calculate_mse(y_test, linear_fit(x_test))

poly3_train_mse = calculate_mse(y_train, poly3_fit(x_train))
poly3_test_mse = calculate_mse(y_test, poly3_fit(x_test))

poly9_train_mse = calculate_mse(y_train, poly9_fit(x_train))
poly9_test_mse = calculate_mse(y_test, poly9_fit(x_test))

print("\n模型误差比较:")
print(f"线性拟合 - 训练集MSE: {linear_train_mse:.4f}, 测试集MSE: {linear_test_mse:.4f}")
print(f"3阶多项式 - 训练集MSE: {poly3_train_mse:.4f}, 测试集MSE: {poly3_test_mse:.4f}")
print(f"9阶多项式 - 训练集MSE: {poly9_train_mse:.4f}, 测试集MSE: {poly9_test_mse:.4f}")
