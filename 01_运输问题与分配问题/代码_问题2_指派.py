# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:04:36 2026

@author: kk070
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

# 效益矩阵（用 -1e6 表示不可行，避免被选中）
cost = np.array([
    [18, 15, 20,  5,  4,  7],
    [17, 15, 33, 12,  8,  6],
    [39, 12, 18, 16, 30, 13],
    [12,  8, 11, 27, 19, 14],
    [-1e6, 7, 10, 21, 10, 32],   # 第5人不能做工作1
    [-1e6, -1e6, -1e6, 6, 11, 13] # 第6人只能做4,5,6
], dtype=float)

# 转换为最小化问题：取负值
profit_matrix = -cost

# 求解
row_ind, col_ind = linear_sum_assignment(profit_matrix)

# 计算总效益
total_profit = cost[row_ind, col_ind].sum()

# 输出结果
print("[OK] 最优指派方案（人 -> 工作）：")
for i, j in zip(row_ind, col_ind):
    print(f"  人 {i+1} -> 工作 {j+1}（效益 = {int(cost[i, j])}）")

print(f"\n 最大总效益 = {int(total_profit)}")
