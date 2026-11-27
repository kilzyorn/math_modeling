# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:01:46 2026

@author: kk070
"""

# 储蓄所排班问题：暴力搜索最优解（无 pulp）
import math

# 各时段需求 [9-10, 10-11, ..., 16-17]
demand = [3, 5, 4, 6, 5, 6, 8, 8]

# 初始化最优解
best_cost = float('inf')
best_solution = None

# 全时服务员人数上限估计：最大需求为8，所以 x1+x2 <= 8 足够
# 半时最多3人，每种班次最多3人
for x1 in range(0, 9):          # 12-13休息
    for x2 in range(0, 9):      # 13-14休息
        for y1 in range(0, 4):
            for y2 in range(0, 4):
                for y3 in range(0, 4):
                    for y4 in range(0, 4):
                        for y5 in range(0, 4):
                            y_total = y1 + y2 + y3 + y4 + y5
                            if y_total > 3:
                                continue  # 半时最多3人

                            # 计算每个时段实际人数
                            staff = [
                                x1 + x2 + y1,                     # 9-10
                                x1 + x2 + y1 + y2,                # 10-11
                                x1 + x2 + y1 + y2 + y3,           # 11-12
                                x2 + y1 + y2 + y3 + y4,           # 12-13 (x1休息)
                                x1 + y2 + y3 + y4 + y5,           # 13-14 (x2休息)
                                x1 + x2 + y3 + y4 + y5,           # 14-15
                                x1 + x2 + y4 + y5,                # 15-16
                                x1 + x2 + y5                      # 16-17
                            ]

                            # 检查是否满足所有时段需求
                            if all(staff[i] >= demand[i] for i in range(8)):
                                cost = 100 * (x1 + x2) + 40 * y_total
                                if cost < best_cost:
                                    best_cost = cost
                                    best_solution = (x1, x2, y1, y2, y3, y4, y5, staff)

# 输出结果
if best_solution:
    x1, x2, y1, y2, y3, y4, y5, staff = best_solution
    print("[OK] 找到最优排班方案：")
    print(f"全时服务员（12-13休息）x1 = {x1}")
    print(f"全时服务员（13-14休息）x2 = {x2}")
    print(f"半时班次：y1={y1}, y2={y2}, y3={y3}, y4={y4}, y5={y5}")
    print(f"各时段实际人数：{staff}")
    print(f"总成本 = {best_cost} 元")
else:
    print("[X] 无可行解")
    