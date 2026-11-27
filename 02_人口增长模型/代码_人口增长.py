# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:07:54 2026

@author: kk070
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------
# 1. 原始数据
# ----------------------------
years = np.array([1820,1830,1840,1850,1860,1870,1880,
                  1890,1900,1910,1920,1930,1940,1950,
                  1960,1970,1980,1990,2000,2010,2020])
population = np.array([9.7,15.8,19.9,20.1,26.1,35.2,38.5,
                       50.6,60.1,78.2,92.3,108.2,128.5,151.9,
                       174.6,192.1,232.6,251.2,263.6,275.7,285.9])

# 以1820年为 t=0，提高数值稳定性
t = years - 1820
P = population
t_2030 = 2030 - 1820  # 用于预测


# ----------------------------
# 2. 模型定义与拟合
# ----------------------------

# (1) 指数增长模型: P(t) = P0 * exp(r * t)
def exp_model(t, P0, r):
    return P0 * np.exp(r * t)

p0_exp = [10, 0.02]
params_exp, _ = curve_fit(exp_model, t, P, p0=p0_exp)
P0_exp, r_exp = params_exp
P_pred_exp = exp_model(t, P0_exp, r_exp)
SSE_exp = np.sum((P - P_pred_exp)**2)
P_2030_exp = exp_model(t_2030, P0_exp, r_exp)


# (2) 改进指数模型: P(t) = P0 * exp(a*t - 0.5*b*t^2)
def improved_exp_model(t, P0, a, b):
    return P0 * np.exp(a * t - 0.5 * b * t**2)

p0_imp = [10, 0.03, 0.0001]
params_imp, _ = curve_fit(improved_exp_model, t, P, p0=p0_imp, maxfev=10000)
P0_imp, a_imp, b_imp = params_imp
P_pred_imp = improved_exp_model(t, P0_imp, a_imp, b_imp)
SSE_imp = np.sum((P - P_pred_imp)**2)
P_2030_imp = improved_exp_model(t_2030, P0_imp, a_imp, b_imp)


# (3) Logistic 模型: P(t) = K / (1 + ((K - P0)/P0) * exp(-r*t))
def logistic_model(t, K, P0, r):
    return K / (1 + (K - P0) / P0 * np.exp(-r * t))

p0_log = [300, 10, 0.03]
params_log, _ = curve_fit(logistic_model, t, P, p0=p0_log, maxfev=10000)
K_log, P0_log, r_log = params_log
P_pred_log = logistic_model(t, K_log, P0_log, r_log)
SSE_log = np.sum((P - P_pred_log)**2)
P_2030_log = logistic_model(t_2030, K_log, P0_log, r_log)


# ----------------------------
# 3. 绘图
# ----------------------------
t_smooth = np.linspace(0, 210, 500)
years_smooth = t_smooth + 1820

plt.figure(figsize=(12, 8))
plt.scatter(years, P, color='black', label='原始数据', zorder=5, s=30)

plt.plot(years_smooth, exp_model(t_smooth, P0_exp, r_exp),
         label=f'指数模型 (SSE={SSE_exp:.1f})', linestyle='--')
plt.plot(years_smooth, improved_exp_model(t_smooth, P0_imp, a_imp, b_imp),
         label=f'改进指数模型 (SSE={SSE_imp:.1f})', linestyle='-.')
plt.plot(years_smooth, logistic_model(t_smooth, K_log, P0_log, r_log),
         label=f'Logistic模型 (SSE={SSE_log:.1f})')

plt.xlabel('年份', fontsize=12)
plt.ylabel('人口（百万）', fontsize=12)
plt.title('美国历史人口数据与三种增长模型拟合', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(1820, 2040)
plt.tight_layout()
plt.show()


# ----------------------------
# 4. 输出结果表格
# ----------------------------
results = pd.DataFrame({
    '模型': ['指数增长模型', '改进指数模型', 'Logistic模型'],
    '2030年人口预测（百万）': [P_2030_exp, P_2030_imp, P_2030_log],
    '误差平方和 (SSE)': [SSE_exp, SSE_imp, SSE_log]
})

print("\n【模型拟合结果汇总】")
print(results.round(4).to_string(index=False))