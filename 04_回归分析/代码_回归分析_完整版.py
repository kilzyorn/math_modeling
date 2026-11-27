# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:11:16 2026

@author: kk070
"""

import numpy as np
import pickle
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# ==================== 数据录入 ====================
# 根据题目表1录入数据
data = {
    'y': [44.6, 45.3, 54.3, 59.6, 49.9, 44.8, 45.7, 49.1, 39.4, 60.1,
          50.5, 37.4, 44.8, 47.2, 51.9, 49.2, 40.9, 46.7, 46.8, 50.4,
          39.4, 46.1, 45.4, 54.7],
    'x1': [44, 40, 44, 42, 38, 47, 40, 43, 44, 38,
           44, 45, 45, 47, 54, 49, 51, 51, 48, 47,
           57, 54, 52, 50],
    'x2': [89.5, 75.1, 85.8, 68.2, 89.0, 77.5, 76.0, 81.2, 81.4, 81.9,
           73.0, 87.7, 66.5, 79.2, 83.1, 81.4, 69.6, 77.9, 91.6, 73.4,
           73.4, 79.4, 76.3, 70.9],
    'x3': [6.82, 6.04, 5.19, 4.90, 5.53, 6.98, 7.17, 6.51, 7.85, 5.18,
           6.08, 8.42, 6.67, 6.36, 6.20, 5.37, 6.57, 6.00, 6.15, 6.05,
           7.58, 6.70, 5.78, 5.35],
    'x4': [62, 62, 45, 40, 55, 58, 70, 64, 63, 48,
           45, 56, 51, 47, 50, 44, 57, 48, 48, 67,
           58, 62, 48, 48],
    'x5': [178, 185, 156, 166, 178, 176, 176, 162, 174, 170,
           168, 186, 176, 162, 166, 180, 168, 162, 162, 168,
           174, 156, 164, 146]
}

df = pd.DataFrame(data)

# ==================== 单变量模型评价 ====================
print("========== 各变量与y的相关系数 ==========")
for col in ['x1','x2','x3','x4','x5']:
    corr = df['y'].corr(df[col])
    print(f"{col}: 相关系数 = {corr:.4f}")

print("\n========== 一元线性回归结果（R²） ==========")
best_var = None
best_r2 = -np.inf
best_model = None

for col in ['x1','x2','x3','x4','x5']:
    X = sm.add_constant(df[col])   # 添加截距项
    model = sm.OLS(df['y'], X).fit()
    r2 = model.rsquared
    print(f"{col}: R² = {r2:.4f}, 斜率 = {model.params[col]:.4f}, p值 = {model.pvalues[col]:.4e}")
    if r2 > best_r2:
        best_r2 = r2
        best_var = col
        best_model = model

# ==================== 输出最佳模型 ====================
print("\n========== 最佳单变量模型 ==========")
print(f"选择的变量: {best_var}")
print(f"R² = {best_r2:.4f}")
print(f"模型表达式: y = {best_model.params['const']:.4f} + {best_model.params[best_var]:.4f} * {best_var}")
print("\n模型详细摘要:")
print(best_model.summary())

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 图表1：各变量与y的散点图 + 回归线 ====================
variables = ['x1', 'x2', 'x3', 'x4', 'x5']
names = ['年龄 x1', '体重 x2', '1500m时间 x3', '静止心率 x4', '跑步后心率 x5']

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, (var, name) in enumerate(zip(variables, names)):
    ax = axes[i]
    ax.scatter(df[var], df['y'], alpha=0.7, edgecolors='k', s=70)
    slope, intercept = np.polyfit(df[var], df['y'], 1)
    x_line = np.linspace(df[var].min(), df[var].max(), 100)
    ax.plot(x_line, intercept + slope * x_line, 'r-', lw=2)
    r2 = np.corrcoef(df[var], df['y'])[0,1]**2
    ax.set_xlabel(name, fontsize=11)
    ax.set_ylabel('耗氧能力 y', fontsize=11)
    ax.set_title(f'{name}  (R² = {r2:.3f})', fontsize=12)
    ax.grid(alpha=0.3)
axes[5].set_visible(False)
plt.tight_layout()
plt.savefig('fig1_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 图表2：R²柱状图（附p值） ====================
corrs = [df['y'].corr(df[var]) for var in variables]
r2s = [c**2 for c in corrs]
pvals = []
for var in variables:
    X = sm.add_constant(df[var])
    pvals.append(sm.OLS(df['y'], X).fit().pvalues[var])

plt.figure(figsize=(8, 5))
bars = plt.bar(names, r2s, color='steelblue', alpha=0.8)
plt.ylabel('决定系数 $R^2$', fontsize=12)
plt.xlabel('自变量', fontsize=12)
plt.title('单变量线性回归模型的 $R^2$ 对比', fontsize=13)
for bar, r2, p in zip(bars, r2s, pvals):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'$R^2$={r2:.3f}\np={p:.2e}', ha='center', fontsize=9)
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_r2_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 第二问：二元组合搜索 ====================
from itertools import combinations

variables = ['x1', 'x2', 'x3', 'x4', 'x5']
names = ['年龄', '体重', '1500m时间', '静止心率', '跑步后心率']

results = []
for (var1, var2) in combinations(variables, 2):
    X = sm.add_constant(df[[var1, var2]])
    model = sm.OLS(df['y'], X).fit()
    results.append({
        '变量组合': f"{var1}+{var2}",
        'R²': model.rsquared,
        '调整R²': model.rsquared_adj,
        'AIC': model.aic,
        'BIC': model.bic,
        'p值(F检验)': model.f_pvalue,
        '模型': model
    })

# 转换为DataFrame并排序（按调整R²降序）
df_results = pd.DataFrame(results).sort_values('调整R²', ascending=False)

print("\n========== 第二问：所有二元模型评价指标 ==========")
print(df_results[['变量组合', 'R²', '调整R²', 'AIC', 'BIC', 'p值(F检验)']].to_string(float_format='%.4f'))

best = df_results.iloc[0]
best_model = best['模型']
best_vars = best['变量组合']

print("\n========== 最佳二元模型 ==========")
print(f"最佳变量组合：{best_vars}")
print(f"调整R² = {best['调整R²']:.4f}")
print(f"模型表达式：")
print(f"y = {best_model.params['const']:.4f} + {best_model.params[best_vars.split('+')[0]]:.4f} * {best_vars.split('+')[0]} + {best_model.params[best_vars.split('+')[1]]:.4f} * {best_vars.split('+')[1]}")
print("\n模型详细摘要：")
print(best_model.summary())

    # ==================== 第二问图表 ====================
# 图1：所有二元模型的调整R²柱状图（按降序排列）
df_plot = df_results.copy()
df_plot['组合标签'] = df_plot['变量组合'].str.replace('x', 'x')
df_plot = df_plot.sort_values('调整R²', ascending=False)

plt.figure(figsize=(10, 5))
bars = plt.bar(df_plot['组合标签'], df_plot['调整R²'], color='steelblue', alpha=0.8)
plt.ylabel('调整 $R^2$', fontsize=12)
plt.xlabel('变量组合', fontsize=12)
plt.title('二元线性回归模型的调整 $R^2$ 对比', fontsize=13)
plt.ylim(0, 1)
for bar, adj_r2 in zip(bars, df_plot['调整R²']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{adj_r2:.3f}', ha='center', fontsize=9)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_adjr2_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 图2：最佳模型 (x1+x3) 的残差诊断图
resid = best_model.resid
fitted = best_model.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 子图1：残差 vs 拟合值
axes[0].scatter(fitted, resid, alpha=0.7, edgecolors='k', s=60)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1.5)
axes[0].set_xlabel(r'拟合值 $\hat{y}$', fontsize=11)
axes[0].set_ylabel('残差', fontsize=11)
axes[0].set_title('残差 vs 拟合值', fontsize=12)
axes[0].grid(alpha=0.3)

# 子图2：Q-Q 正态性图
stats.probplot(resid, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q 图（正态性检验）', fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig2_best_model_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()
# ==================== 第三问：全模型与变量选择 ====================
import itertools
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 全模型
X_all = sm.add_constant(df[['x1','x2','x3','x4','x5']])
model_all = sm.OLS(df['y'], X_all).fit()
print("\n========== 全模型摘要 ==========")
print(model_all.summary())

# 方法1：全子集回归（按 AIC 选择最佳模型）
def subset_selection(X_df, y, criterion='AIC'):
    """
    枚举所有子集，返回按指定准则排序的结果
    criterion: 'AIC', 'BIC', 'adjR2'
    """
    variables = X_df.columns.tolist()
    best_score = np.inf if criterion in ['AIC','BIC'] else -np.inf
    best_model = None
    best_vars = None
    results = []
    for k in range(0, len(variables)+1):
        for combo in itertools.combinations(variables, k):
            if k == 0:
                # 空模型：仅截距
                X = sm.add_constant(pd.DataFrame(index=df.index))
                model = sm.OLS(y, X).fit()
            else:
                X = sm.add_constant(df[list(combo)])
                model = sm.OLS(y, X).fit()
            if criterion == 'AIC':
                score = model.aic
                better = score < best_score
            elif criterion == 'BIC':
                score = model.bic
                better = score < best_score
            elif criterion == 'adjR2':
                score = model.rsquared_adj
                better = score > best_score
            else:
                raise ValueError("criterion must be 'AIC', 'BIC' or 'adjR2'")
            results.append({
                'vars': combo,
                'n_var': k,
                'R2': model.rsquared,
                'adjR2': model.rsquared_adj,
                'AIC': model.aic,
                'BIC': model.bic
            })
            if better:
                best_score = score
                best_model = model
                best_vars = combo
    results_df = pd.DataFrame(results).sort_values(criterion, ascending=(criterion in ['AIC','BIC']))
    return best_model, best_vars, results_df

print("\n========== 全子集回归（按 AIC 最小选择） ==========")
best_model_subset, best_vars_subset, subset_results = subset_selection(df[['x1','x2','x3','x4','x5']], df['y'], criterion='AIC')
print(f"最佳变量组合：{best_vars_subset if best_vars_subset else '(仅截距)'}")
print(f"AIC = {best_model_subset.aic:.2f}")
print(f"调整 R² = {best_model_subset.rsquared_adj:.4f}")
print("\n该模型摘要：")
print(best_model_subset.summary())

# 方法2：后向逐步回归（基于 p 值）
def backward_elimination(X_df, y, alpha=0.05):
    """
    后向逐步回归：从全模型开始，反复剔除 p 值 > alpha 的最大 p 值变量
    """
    current_vars = X_df.columns.tolist()
    while True:
        X = sm.add_constant(df[current_vars])
        model = sm.OLS(y, X).fit()
        pvals = model.pvalues.drop('const')
        max_pval = pvals.max()
        if max_pval > alpha and len(current_vars) > 1:
            worst_var = pvals.idxmax()
            print(f"剔除 {worst_var} (p={max_pval:.4f})")
            current_vars.remove(worst_var)
        else:
            break
    final_X = sm.add_constant(df[current_vars])
    final_model = sm.OLS(y, final_X).fit()
    return final_model, current_vars

print("\n========== 后向逐步回归（p值阈值0.05） ==========")
back_model, back_vars = backward_elimination(df[['x1','x2','x3','x4','x5']], df['y'])
print(f"最终保留变量：{back_vars}")
print(back_model.summary())

# 选择最终模型（这里以 AIC 最小的全子集结果作为最终模型）
final_model = best_model_subset
final_vars = best_vars_subset

# 输出最终模型的 VIF（多重共线性诊断）
if len(final_vars) > 1:
    print("\n========== 最终模型的多重共线性诊断（VIF） ==========")
    X_final = df[list(final_vars)]
    vif_data = pd.DataFrame()
    vif_data["变量"] = X_final.columns
    vif_data["VIF"] = [variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])]
    print(vif_data)
else:
    print("\n最终模型仅含一个变量，无需 VIF 诊断。")

# 保存最终模型供第四问使用
import pickle
with open('final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("\n最终模型已保存为 final_model.pkl，供第四问使用。")

# ==================== 第三问：候选模型表格 ====================
# 从全子集回归结果中提取并排序
subset_results_sorted = subset_results.sort_values('AIC').reset_index(drop=True)
subset_results_sorted['变量组合'] = subset_results_sorted['vars'].apply(lambda x: ', '.join(x) if x else '(截距仅)')

# 选取前10个AIC最小的模型
table_data = subset_results_sorted[['变量组合', 'n_var', 'R2', 'adjR2', 'AIC', 'BIC']].head(10)
table_data.columns = ['变量组合', '变量个数', 'R²', '调整R²', 'AIC', 'BIC']

print("\n========== 全子集回归（按AIC排序，前10个模型） ==========")
print(table_data.to_string(index=False, float_format='%.4f'))

# 保存为CSV，方便插入论文
table_data.to_csv('full_subset_models.csv', index=False, encoding='utf-8-sig')
print("\n表格已保存为 full_subset_models.csv")
# ==================== 第四问：手动计算预测与预测区间 ====================
import pickle
import numpy as np
from scipy import stats

# 加载模型
with open('final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)

# 提取参数和残差方差
params = final_model.params          # Series: const, x1, x3, x5
resid_var = final_model.mse_resid    # 残差方差
n = final_model.nobs                 # 24
p = len(params) - 1                  # 3
dof = n - p - 1                      # 20

# 新样本（题目给定 5 个变量；只用进入最终模型的变量）
new_sample = {'x1': 41, 'x2': 73, 'x3': 5.32, 'x4': 60, 'x5': 170}
model_vars = [v for v in params.index if v != 'const']
print(f"最终模型变量：{model_vars}")

# 点预测
y_pred = params['const'] + sum(params[v] * new_sample[v] for v in model_vars)

# 设计矩阵 (包含常数项)
X_design = final_model.model.exog
x0 = np.array([1.0] + [new_sample[v] for v in model_vars], dtype=float)

# 计算杠杆值 h = x0 (X'X)^{-1} x0'
try:
    XTX_inv = np.linalg.inv(X_design.T @ X_design)
except np.linalg.LinAlgError:
    XTX_inv = np.linalg.pinv(X_design.T @ X_design)
h = x0 @ XTX_inv @ x0

# 预测区间
t_crit = stats.t.ppf(0.975, dof)
se_pred = np.sqrt(resid_var * (1 + h))
lower = y_pred - t_crit * se_pred
upper = y_pred + t_crit * se_pred

# 置信区间（均值）
se_mean = np.sqrt(resid_var * h)
mean_lower = y_pred - t_crit * se_mean
mean_upper = y_pred + t_crit * se_mean

print("\n========== 第四问：预测结果 ==========")
print("新样本：" + ", ".join(f"{k}={v}" for k, v in new_sample.items()))
print(f"进入模型的变量：{', '.join(model_vars)}")
print(f"点预测耗氧能力 ŷ = {y_pred:.4f}")
print(f"95% 预测区间：({lower:.4f}, {upper:.4f})")
print(f"95% 置信区间（均值）：({mean_lower:.4f}, {mean_upper:.4f})")