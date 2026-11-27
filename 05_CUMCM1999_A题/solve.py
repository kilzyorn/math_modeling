# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:54:20 2026

@author: kk070
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    459, 362, 624, 542, 509, 584, 433, 748, 815, 505,
    612, 452, 434, 982, 640, 742, 565, 706, 593, 680,
    926, 653, 164, 487, 734, 608, 428, 1153, 593, 844,
    527, 552, 513, 781, 474, 388, 824, 538, 862, 659,
    775, 859, 755, 649, 697, 515, 628, 954, 771, 609,
    402, 960, 885, 610, 292, 837, 473, 677, 358, 638,
    699, 634, 555, 570, 84, 416, 606, 1062, 484, 120,
    447, 654, 564, 339, 280, 246, 687, 539, 790, 581,
    621, 724, 531, 512, 577, 496, 468, 499, 544, 645,
    764, 558, 378, 765, 666, 763, 217, 715, 310, 851
])

n = len(data)
f, t, d, k = 200, 10, 3000, 1000  

def solve_p1():
    best_N, min_cost = 0, float('inf')
    Ns = np.arange(100, 1201)
    costs = []
    for N in Ns:
        failed = data[data <= N]
        p_fail = len(failed) / n
        if p_fail > 0:
            waste = N + 1 - failed.mean()
            cost = t + (1 - p_fail) * k + p_fail * (d + f * waste)
        else:
            cost = t + k
        unit = cost / N
        costs.append(unit)
        if unit < min_cost:
            min_cost, best_N = unit, N
    return best_N, min_cost, Ns, costs

def solve_p2():
    pg_n, pg_f = 0.98, 0.40
    L_fa = 1500
    best_N, min_cost = 0, float('inf')
    Ns = np.arange(100, 1201)
    costs = []
    for N in Ns:
        p_fault = np.mean(data <= N)
        p_normal = 1 - p_fault
        p_def = p_normal * (1 - pg_n) + p_fault * (1 - pg_f)
        if p_def == 0:
            costs.append(float('inf'))
            continue
        p_f_given_def = p_fault * (1 - pg_f) / p_def
        cost_if_def = t + p_f_given_def * d + (1 - p_f_given_def) * L_fa
        waste_ok = N * (p_normal * (1 - pg_n) + p_fault * (1 - pg_f))
        cost_if_ok = t + k + f * waste_ok
        total_cost = p_def * cost_if_def + (1 - p_def) * cost_if_ok
        unit = total_cost / N
        costs.append(unit)
        if unit < min_cost:
            min_cost, best_N = unit, N
    return best_N, min_cost, Ns, costs

def solve_p3():
    pg_n, pg_f = 0.98, 0.40
    L_fa = 1500
    T = 1200
    ms = np.arange(1, 21)
    costs = []
    best_m, min_cost = 0, float('inf')
    for m in ms:
        p_fault = np.mean(data <= T)
        p_normal = 1 - p_fault
        p_all_good_n = pg_n ** m
        p_all_good_f = pg_f ** m
        p_judge_ok = p_normal * p_all_good_n + p_fault * p_all_good_f
        p_judge_bad = 1 - p_judge_ok
        if p_judge_bad == 0:
            costs.append(float('inf'))
            continue
        p_f_given_bad = (p_fault * (1 - p_all_good_f)) / p_judge_bad
        cost_bad = t * m + p_f_given_bad * d + (1 - p_f_given_bad) * L_fa
        waste = T * (p_normal * (1 - pg_n) + p_fault * (1 - pg_f))
        cost_ok = t * m + k + f * waste
        total_cost = p_judge_bad * cost_bad + p_judge_ok * cost_ok
        unit = total_cost / T
        costs.append(unit)
        if unit < min_cost:
            min_cost, best_m = unit, m
    return T, best_m, min_cost, ms, costs

if __name__ == "__main__":
    N1, c1, Ns1, costs1 = solve_p1()
    N2, c2, Ns2, costs2 = solve_p2()
    T3, m3, c3, ms3, costs3 = solve_p3()

    plt.figure(figsize=(8, 5))
    plt.plot(Ns1, costs1, 'b-', linewidth=1.5)
    plt.axvline(N1, color='red', linestyle='--', label=f'Optimal N={N1}')
    plt.xlabel('Inspection Interval N (units)')
    plt.ylabel('Unit Cost (CNY/unit)')
    plt.title('Problem 1: Unit Cost vs N')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('Figure_1.png', dpi=300)

    plt.figure(figsize=(8, 5))
    plt.plot(Ns2, costs2, 'r-', linewidth=1.5)
    plt.axvline(N2, color='blue', linestyle='--', label=f'Optimal N={N2}')
    plt.xlabel('Inspection Interval N (units)')
    plt.ylabel('Unit Cost (CNY/unit)')
    plt.title('Problem 2: Unit Cost vs N')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('Figure_2.png', dpi=300)

    plt.figure(figsize=(8, 5))
    plt.plot(ms3, costs3, 'g-o', markersize=4)
    plt.axvline(m3, color='purple', linestyle='--', label=f'Optimal m={m3}')
    plt.xlabel('Sample Size m')
    plt.ylabel('Unit Cost (CNY/unit)')
    plt.title('Problem 3: Unit Cost vs m (T=1200)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('Figure_3.png', dpi=300)