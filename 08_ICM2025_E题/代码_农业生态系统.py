# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:33:28 2026

@author: kk070
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib

# === 自动适配中文字体（跨平台兼容）===
def set_chinese_font():
    import platform
    system = platform.system()
    if system == "Windows":
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows 黑体
    elif system == "Darwin":  # macOS
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:  # Linux 或其他
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

set_chinese_font()

# 定义参数（可调整）
params = {
    'r_max': 1.0,       # 作物最大生长率
    'K': 100,           # 作物环境承载力
    'alpha': 0.01,      # 害虫取食率
    'beta': 0.008,      # 害虫转化效率
    'gamma': 0.02,      # 天敌捕食率
    'delta': 0.015,     # 天敌转化效率
    'mu': 0.1,          # 天敌自然死亡率
    'lambda_d': 0.005,  # 分解者增长系数
    'nu': 0.05,         # 分解者衰减率
    'k_pest': 0.8,      # 农药对害虫杀伤力
    'eta': 0.3,         # 农药对天敌伤害
    'theta': 0.2,       # 农药对分解者抑制
    'omega': 2 * np.pi / 365,  # 年周期（时间单位：天）
}

# 农药使用函数：每年喷洒两次（第100天和第250天附近）
def pesticide_use(t):
    # 使用模运算实现多年周期性
    t_mod = t % 365
    pulse1 = np.exp(-((t_mod - 100) ** 2) / (2 * 10**2))
    pulse2 = np.exp(-((t_mod - 250) ** 2) / (2 * 10**2))
    return 0.8 * (pulse1 + pulse2)

# 作物季节性生长率（春季高，冬季低）
def crop_growth_rate(t):
    return params['r_max'] * (1 + np.sin(params['omega'] * t)) / 2

# 微分方程系统
def ecosystem_model(t, y):
    C, P, B, D = y
    
    U = pesticide_use(t)
    r_crop = crop_growth_rate(t)
    
    dCdt = r_crop * C * (1 - C / params['K']) - params['alpha'] * C * P
    dPdt = params['beta'] * C * P - params['gamma'] * P * B - params['k_pest'] * U * P
    dBdt = params['delta'] * P * B - params['mu'] * B - params['eta'] * U * B
    dDdt = params['lambda_d'] * (C + P + B) - params['nu'] * D - params['theta'] * U * D
    
    return [dCdt, dPdt, dBdt, dDdt]

# 初始条件（刚开垦农田）
y0 = [20, 5, 2, 10]  # [作物, 害虫, 天敌, 分解者]

# 时间范围：3年（1095天）
t_span = (0, 1095)
t_eval = np.linspace(0, 1095, 2000)

# 求解微分方程
sol = solve_ivp(ecosystem_model, t_span, y0, t_eval=t_eval, method='RK45')

# 绘图
plt.figure(figsize=(12, 8))
plt.plot(sol.t, sol.y[0], label='作物 (生产者)', color='green', linewidth=2)
plt.plot(sol.t, sol.y[1], label='害虫 (初级消费者)', color='red')
plt.plot(sol.t, sol.y[2], label='天敌 (次级消费者)', color='blue')
plt.plot(sol.t, sol.y[3], label='分解者', color='purple')

plt.xlabel('时间 (天)')
plt.ylabel('生物量 / 数量（相对单位）')
plt.title('农业生态系统动态演化（森林转农田后3年）')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()