import numpy as np
import time
import pandas as pd

# 目标函数 f(x)
def f(x):
    return np.sum([(1 - x[i])**2 + 100*(x[i+1] - x[i]**2)**2 for i in range(9)])

# 梯度计算
def gradient(x):
    grad = np.zeros_like(x)
    for i in range(9):
        grad[i] = -2*(1 - x[i]) + 200 * (x[i+1] - x[i]**2) * (-2*x[i])
    return grad

# 海森矩阵计算
def hessian(x):
    H = np.zeros((10, 10))
    for i in range(9):
        H[i, i] = 2 + 400 * x[i]**2 - 400 * x[i+1]
        H[i, i+1] = -200 * x[i]
        H[i+1, i] = -200 * x[i]
    H[9, 9] = 200
    return H

# 最速下降法（Gradient Descent）
def gradient_descent(x0, max_iter=10, alpha=0.001):
    x = x0
    history = []
    for i in range(max_iter):
        grad = gradient(x)
        # 更新参数
        x_new = x - alpha * grad
        f_val = f(x_new)
        history.append((i+1, f_val))
        x = x_new
        print(f"GD Iteration {i+1}: f(x) = {f_val}")
    return x, history

# 牛顿法（Newton's Method）
def newton_method(x0, max_iter=10, tol=1e-6):
    x = x0
    history = []
    for i in range(max_iter):
        grad = gradient(x)
        H = hessian(x)
        
        # 计算搜索方向
        p = np.linalg.solve(H, -grad)
        
        # 更新参数
        x_new = x + p
        f_val = f(x_new)
        history.append((i+1, f_val))
        
        x = x_new
        
        # 检查收敛
        if np.linalg.norm(grad) < tol:
            break
        
        print(f"Newton Iteration {i+1}: f(x) = {f_val}")
    
    return x, history

# 初始点
x0 = np.zeros(10)

# 记录最速下降法结果
start_time = time.time()
x_gd, history_gd = gradient_descent(x0)
gd_time = time.time() - start_time

# 记录牛顿法结果
start_time = time.time()
x_newton, history_newton = newton_method(x0)
newton_time = time.time() - start_time

# 输出最速下降法结果
print("\n最速下降法最终结果: f(x) = {:.6f}, CPU 时间 = {:.4f}秒".format(f(x_gd), gd_time))

# 输出牛顿法结果
print("\n牛顿法最终结果: f(x) = {:.6f}, CPU 时间 = {:.4f}秒".format(f(x_newton), newton_time))

# 记录表格形式的结果
gd_df = pd.DataFrame(history_gd, columns=["Iteration", "f(x)_GD"])
newton_df = pd.DataFrame(history_newton, columns=["Iteration", "f(x)_Newton"])

# 合并结果
result_df = pd.merge(gd_df, newton_df, on="Iteration", how="outer")
print("\n迭代结果对比：")
print(result_df)
