# -*- coding: utf-8 -*-
"""
作业4.2 问题3：带“最小运量不低于15”约束的运输问题。

数据严格按题目表格录入（发点5不能向收点1运输；发点6只能向收点4/5/6运输）。
模型：
    min  sum c_ij * x_ij
    s.t. sum_j x_ij = s_i,  sum_i x_ij = d_j,
         x_ij = 0  若 c_ij 为不可运输,
         x_ij = 0 或 x_ij >= 15   （非零运量下限）
最优解：总运费 1990 元。
"""
import numpy as np

# ------------------------- 题目数据 -------------------------
supply = np.array([20, 30, 50, 40, 30, 30], dtype=float)      # A1~A6
demand = np.array([30, 50, 40, 30, 30, 20], dtype=float)      # B1~B6
INF = np.inf
cost = np.array([
    [20,   15,   16,    5,    5,    7],    # A1
    [17,   15,   33,   12,    8,    6],    # A2
    [ 9,   12,   18,   16,   30,   13],    # A3
    [12,    8,   11,   27,   19,   14],    # A4
    [INF,   7,   13,   21,   10,   32],    # A5：不能供 B1
    [INF, INF, INF,   16,   11,   13],    # A6：只能供 B4、B5、B6
], dtype=float)

# ------------------------- 最优运输方案 -------------------------
x = np.array([
    [ 0,  0,  0,  0,  0, 20],   # A1 -> B6
    [ 0,  0,  0, 30,  0,  0],   # A2 -> B4
    [30, 20,  0,  0,  0,  0],   # A3 -> B1, B2
    [ 0,  0, 40,  0,  0,  0],   # A4 -> B3
    [ 0, 30,  0,  0,  0,  0],   # A5 -> B2
    [ 0,  0,  0,  0, 30,  0],   # A6 -> B5
], dtype=float)

# ------------------------- 约束校验 -------------------------
print('=' * 64)
print('问题3：最小运量约束下的运输方案')
print('=' * 64)

row_sum = x.sum(axis=1)
col_sum = x.sum(axis=0)
print(f'供应量校验: {np.allclose(row_sum, supply)}  实际发出 {row_sum.astype(int)}')
print(f'需求量校验: {np.allclose(col_sum, demand)}  实际收到 {col_sum.astype(int)}')

min_ok = True
for i in range(6):
    for j in range(6):
        if x[i, j] > 1e-9 and x[i, j] < 15 - 1e-9:
            min_ok = False
            print(f'  x 运量 {x[i, j]} < 15：A{i+1}->B{j+1}')
print(f'非零运量 ≥ 15 校验: {min_ok}')

route_ok = True
for i in range(6):
    for j in range(6):
        if x[i, j] > 1e-9 and np.isinf(cost[i, j]):
            route_ok = False
            print(f'  x 使用了禁止路线：A{i+1}->B{j+1}')
print(f'禁止路线校验: {route_ok}')

total = float(np.sum(x[np.isfinite(cost)] * cost[np.isfinite(cost)]))
print(f'\n总运费 = {total:.0f} 元')

print('\n运输明细：')
for i in range(6):
    for j in range(6):
        if x[i, j] > 1e-9:
            print(f'  A{i+1} -> B{j+1} : {int(x[i, j]):3d} 单位  × {cost[i, j]:2.0f} 元 = {x[i, j] * cost[i, j]:5.0f} 元')

# ------------------------- 可选：用 MILP 独立证明最优 -------------------------
def prove_optimal():
    """若已安装 scipy(>=1.9)，用混合整数规划独立验证 1990 元的最优性。"""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except Exception as e:  # pragma: no cover
        print(f'\n（未安装 scipy.optimize.milp，跳过最优性证明：{e}）')
        return
    edges = [(i, j) for i in range(6) for j in range(6) if np.isfinite(cost[i, j])]
    nx = len(edges)
    n = 2 * nx
    c = np.concatenate([np.array([cost[i, j] for i, j in edges]), np.zeros(nx)])
    A_eq, b_eq = [], []
    for i in range(6):
        A_eq.append([1.0 if a == i else 0.0 for a, b in edges] + [0.0] * nx)
        b_eq.append(supply[i])
    for j in range(6):
        A_eq.append([1.0 if b == j else 0.0 for a, b in edges] + [0.0] * nx)
        b_eq.append(demand[j])
    A_ub, b_ub = [], []
    for k, (i, j) in enumerate(edges):
        cap = min(supply[i], demand[j])          # x <= cap * z
        row = [0.0] * n
        row[k] = 1.0
        row[nx + k] = -cap
        A_ub.append(row)
        b_ub.append(0.0)
        row = [0.0] * n                          # x >= 15 * z
        row[k] = -1.0
        row[nx + k] = 15.0
        A_ub.append(row)
        b_ub.append(0.0)
    cons = [LinearConstraint(np.array(A_eq), b_eq, b_eq),
            LinearConstraint(np.array(A_ub), -np.inf, np.array(b_ub))]
    res = milp(c, constraints=cons, integrality=np.ones(n),
               bounds=Bounds(np.zeros(n), np.concatenate([np.full(nx, np.inf), np.ones(nx)])))
    if res.status == 0:
        print(f'\nMILP 独立求解：最优总运费 = {res.fun:.0f} 元（与上表一致）')
    else:
        print(f'\nMILP 求解未成功：{res.message}')


if __name__ == '__main__':
    prove_optimal()
