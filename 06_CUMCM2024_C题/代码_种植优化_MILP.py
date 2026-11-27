# -*- coding: utf-8 -*-
"""
CUMCM 2024 C 题 问题1：农作物种植策略（2024-2030，滞销情形）
混合整数线性规划（MILP），基于 附件1 / 附件2 实际数据，用 scipy.optimize.milp 求解。

建模口径（显式假设）：
1. 每个地块每季只种一种作物（题目允许合种；本模型不拆分地块，便于田间管理）。
2. 亩产量、种植成本取 2023 年统计值；销售单价取 2023 年所给区间的中点。
3. 预期销售量 = 2023 年该作物的总产量；超出部分完全滞销，收入为 0（问题1口径）。
4. 同一地块同一季次不得连续两年种植同一作物（重茬）。
5. 每个地块（含大棚）在任意连续三年内至少种植一次豆类（粮食豆类或蔬菜豆类）。
6. 水浇地每年二选一：单季水稻，或“第一季蔬菜 + 第二季大白菜/白萝卜/红萝卜”。
7. 不考虑作物间可替代性。

依赖：numpy pandas scipy(>=1.9) openpyxl
输出：控制台报告 + 种植方案_2024_2030.xlsx
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

BASE = Path(__file__).resolve().parent
F1 = BASE / '数据_附件1_耕地与作物.xlsx'
F2 = BASE / '数据_附件2_2023种植与统计.xlsx'
OUT = BASE / '产出_种植方案_2024_2030.xlsx'

YEARS = list(range(2024, 2031))
LEGUME = {1, 2, 3, 4, 5, 17, 18, 19}
VEG2_ONLY = {35, 36, 37}
RICE = 16

# ------------------------------------------------------------------ 数据
plots = pd.read_excel(F1, sheet_name='乡村的现有耕地').dropna(subset=['地块名称'])
plots = plots[['地块名称', '地块类型', '地块面积/亩']]
plots.columns = ['pname', 'ptype', 'area']
plots['ptype'] = plots['ptype'].astype(str).str.strip()

crops = pd.read_excel(F1, sheet_name='乡村种植的农作物')
crops = crops[['作物编号', '作物名称', '作物类型']]
crops.columns = ['cid', 'cname', 'ctype']
crops['cid'] = pd.to_numeric(crops['cid'], errors='coerce')
crops = crops.dropna(subset=['cid'])
crops['cid'] = crops['cid'].astype(int)
crops['ctype'] = crops['ctype'].astype(str).str.strip()
CNAME = dict(zip(crops.cid, crops.cname))
CTYPE = dict(zip(crops.cid, crops.ctype))

stats = pd.read_excel(F2, sheet_name='2023年统计的相关数据')
stats = stats[['作物编号', '地块类型', '种植季次', '亩产量/斤',
               '种植成本/(元/亩)', '销售单价/(元/斤)']]
stats.columns = ['cid', 'ptype', 'season', 'yld', 'cost', 'price']
stats['cid'] = pd.to_numeric(stats['cid'], errors='coerce')
stats = stats.dropna(subset=['cid', 'ptype', 'season'])
stats['cid'] = stats['cid'].astype(int)
stats['ptype'] = stats['ptype'].astype(str).str.strip()
stats['season'] = stats['season'].astype(str).str.strip()
lo = stats['price'].astype(str).str.split('-').str[0]
hi = stats['price'].astype(str).str.split('-').str[1]
stats['pmid'] = (lo.astype(float) + hi.astype(float)) / 2

plant23 = pd.read_excel(F2, sheet_name='2023年的农作物种植情况')
plant23['种植地块'] = plant23['种植地块'].ffill()   # 合并单元格：同一地块的季次行补全
plant23['作物编号'] = pd.to_numeric(plant23['作物编号'], errors='coerce')
plant23 = plant23.dropna(subset=['种植地块', '作物编号'])
plant23 = plant23[['种植地块', '作物编号', '种植面积/亩', '种植季次']]
plant23.columns = ['pname', 'cid', 'area', 'season']
plant23['cid'] = plant23['cid'].astype(int)
plant23['season'] = plant23['season'].astype(str).str.strip()

PTYPE = dict(zip(plots.pname, plots.ptype))
AREA = dict(zip(plots.pname, plots.area))

Y, K, PR, = {}, {}, {}
for _, r in stats.iterrows():
    key = (r.cid, r.ptype, r.season)
    Y[key], K[key], PR[key] = float(r.yld), float(r.cost), float(r.pmid)
for cid in stats.cid.unique():
    src = (cid, '普通大棚', '第一季')
    if src in Y:
        Y[(cid, '智慧大棚', '第一季')] = Y[src]
        K[(cid, '智慧大棚', '第一季')] = K[src]
        PR[(cid, '智慧大棚', '第一季')] = PR[src]

PRICE = stats.groupby('cid')['pmid'].mean().to_dict()

# 2023 年各作物总产量 = 问题1口径下的预期销售量
DEMAND = {}
for _, r in plant23.iterrows():
    key = (r.cid, PTYPE[r.pname], r.season)
    if key in Y:
        DEMAND[r.cid] = DEMAND.get(r.cid, 0.0) + r.area * Y[key]

# ------------------------------------------------------------ 决策变量集合
allowed = []
for _, p in plots.iterrows():
    pl, pt = p.pname, p.ptype
    grain = [c for c in crops.cid if CTYPE[c].startswith('粮食') and c != RICE]
    veg = [c for c in crops.cid if CTYPE[c].startswith('蔬菜')]
    veg1 = [c for c in veg if c not in VEG2_ONLY]
    veg2 = [c for c in veg if c in VEG2_ONLY]
    fungi = [c for c in crops.cid if CTYPE[c].startswith('食用菌')]
    if pt in ('平旱地', '梯田', '山坡地'):
        allowed += [(pl, '单季', c) for c in grain]
    elif pt == '水浇地':
        allowed += [(pl, '单季', RICE)]
        allowed += [(pl, '第一季', c) for c in veg1]
        allowed += [(pl, '第二季', c) for c in veg2]
    elif pt == '普通大棚':
        allowed += [(pl, '第一季', c) for c in veg1]
        allowed += [(pl, '第二季', c) for c in fungi]
    elif pt == '智慧大棚':
        allowed += [(pl, '第一季', c) for c in veg1]
        allowed += [(pl, '第二季', c) for c in veg1]

# 只保留“2023 年有销量”或“豆类（轮作必须）”的作物，压缩 0 收益变量
allowed = [(pl, s, c) for (pl, s, c) in allowed
           if (c, PTYPE[pl], s) in Y and (DEMAND.get(c, 0.0) > 0 or c in LEGUME)]
vars_ = [(pl, s, c, y) for y in YEARS for (pl, s, c) in allowed]
IDX = {v: i for i, v in enumerate(vars_)}
NV = len(vars_)

crop_ids = sorted(PRICE)
SELL = {(c, y): NV + i * len(YEARS) + j
        for i, c in enumerate(crop_ids) for j, y in enumerate(YEARS)}
NSELL = len(crop_ids) * len(YEARS)

rice_plots = sorted({pl for pl, s, c in allowed if c == RICE})
Z = {(pl, y): NV + NSELL + i * len(YEARS) + j
     for i, pl in enumerate(rice_plots) for j, y in enumerate(YEARS)}
NVAR = NV + NSELL + len(rice_plots) * len(YEARS)

# 2023 基准
base23 = {}
for _, r in plant23.iterrows():
    base23.setdefault((r.pname, r.season), r.cid)
base_legume = {p: int(any(c in LEGUME for c in
                          [base23.get((p, s)) for s in ('单季', '第一季', '第二季')] if c))
               for p in plots.pname}

# ------------------------------------------------------------ 约束
eq_rows, eq_rhs = [], []


def add_eq(items, rhs):
    eq_rows.append(items)
    eq_rhs.append(rhs)


for _, p in plots.iterrows():
    pl, pt = p.pname, p.ptype
    for y in YEARS:
        if pt in ('平旱地', '梯田', '山坡地'):
            add_eq([(IDX[(pl, '单季', c, y)], 1) for c in crops.cid
                    if (pl, '单季', c, y) in IDX], 1)
        elif pt == '水浇地':
            x1 = [(IDX[(pl, '第一季', c, y)], 1) for c in crops.cid
                  if (pl, '第一季', c, y) in IDX]
            x2 = [(IDX[(pl, '第二季', c, y)], 1) for c in crops.cid
                  if (pl, '第二季', c, y) in IDX]
            xr = [(IDX[(pl, '单季', RICE, y)], 1)]
            add_eq(x1 + xr, 1)                      # 第一季：蔬菜 或 水稻
            add_eq(xr + [(Z[(pl, y)], -1)], 0)      # 种水稻 <=> 水稻模式
            add_eq(x2 + [(Z[(pl, y)], 1)], 1)       # 第二季 = 1 - 水稻模式
        else:
            add_eq([(IDX[(pl, '第一季', c, y)], 1) for c in crops.cid
                    if (pl, '第一季', c, y) in IDX], 1)
            add_eq([(IDX[(pl, '第二季', c, y)], 1) for c in crops.cid
                    if (pl, '第二季', c, y) in IDX], 1)

ub_rows, ub_rhs = [], []


def add_ub(items, rhs):
    ub_rows.append(items)
    ub_rhs.append(rhs)


# 重茬：同一地块同一季次相邻两年不同作物
for (pl, s, c) in allowed:
    for y in YEARS[:-1]:
        a, b = IDX[(pl, s, c, y)], IDX[(pl, s, c, y + 1)]
        add_ub([(a, 1), (b, 1)], 1)
    if base23.get((pl, s)) == c:            # 2023 种过，2024 不能重茬
        add_ub([(IDX[(pl, s, c, 2024)], 1)], 0)

# 三年豆类窗口
for pl in plots.pname:
    for t in range(2023, 2029):
        items = []
        for y in range(t, t + 3):
            if y == 2023:
                continue
            for (pp, s, c) in allowed:
                if pp == pl and c in LEGUME and (pp, s, c, y) in IDX:
                    items.append((IDX[(pp, s, c, y)], -1))
        need = 1 - (base_legume[pl] if t == 2023 else 0)
        if need > 0:
            if not items:
                print('  NO LEGUME VARS:', pl, PTYPE[pl], 'window', t)
            add_ub(items, -need)

# 销售上限与产量耦合
for (c, y) in SELL:
    items = [(SELL[(c, y)], 1)]
    for (pl, s, cc, yy) in vars_:
        if yy == y and cc == c:
            items.append((IDX[(pl, s, cc, yy)], -AREA[pl] * Y[(c, PTYPE[pl], s)]))
    add_ub(items, 0)

# ------------------------------------------------------------ 组装矩阵
print(f'变量数: {NVAR} (种植 {NV}, 销售 {NSELL}, 水稻模式 {NVAR - NV - NSELL})')
print(f'约束规模: 等式 {len(eq_rows)} 行，不等式 {len(ub_rows)} 行')
A_eq = lil_matrix((len(eq_rows), NVAR))
for i, items in enumerate(eq_rows):
    for j, v in items:
        A_eq[i, j] += v
A_ub = lil_matrix((len(ub_rows), NVAR))
for i, items in enumerate(ub_rows):
    for j, v in items:
        A_ub[i, j] += v

c = np.zeros(NVAR)
for (pl, s, cc, y) in vars_:
    c[IDX[(pl, s, cc, y)]] = K[(cc, PTYPE[pl], s)]
for (cc, y) in SELL:
    c[SELL[(cc, y)]] = -PRICE.get(cc, 0.0)

lb = np.zeros(NVAR)
ub = np.ones(NVAR)
for (cc, y) in SELL:
    ub[SELL[(cc, y)]] = max(0.0, DEMAND.get(cc, 0.0))
for (pl, y) in Z:
    ub[Z[(pl, y)]] = 1.0

integrality = np.zeros(NVAR)
for i in range(NV):
    integrality[i] = 1
for key in Z:
    integrality[Z[key]] = 1

cons = [LinearConstraint(A_eq.tocsr(), eq_rhs, eq_rhs),
        LinearConstraint(A_ub.tocsr(), -np.inf, ub_rhs)]
res = milp(c, constraints=cons, integrality=integrality,
           bounds=Bounds(lb, ub), options={'time_limit': 180, 'mip_rel_gap': 0.01})

print('求解状态:', res.status, res.message)
if res.x is None:
    raise SystemExit('未找到可行解')

slack = np.abs(A_eq @ res.x - np.array(eq_rhs)).max()
print(f'等式约束最大残差: {slack:.2e}')
gap = getattr(res, 'mip_gap', None)
if gap is not None:
    print(f'当前解与下界的相对间隙: {gap:.2%}（≤1% 即视为近优）')
print(f'7 年总利润 = {(-res.fun) / 1e8:.4f} 亿元')

# ------------------------------------------------------------ 结果整理
X = res.x
plan = {}
for (pl, s, cc, y) in vars_:
    if X[IDX[(pl, s, cc, y)]] > 0.5:
        plan[(y, pl)] = plan.get((y, pl), '') + (('' if s == '单季' else s + ':') + CNAME[cc] + ' ')

annual_profit = {}
for y in YEARS:
    rev = sum(PRICE.get(cc, 0.0) * X[SELL[(cc, y)]] for cc in crop_ids)
    cost = sum(K[(cc, PTYPE[pl], s)] * X[IDX[(pl, s, cc, yy)]]
               for (pl, s, cc, yy) in vars_ if yy == y)
    annual_profit[y] = rev - cost

print('\n年度利润（万元）:')
for y in YEARS:
    print(f'  {y}: {annual_profit[y] / 1e4:10.2f}')

print('\n典型地块种植序列:')
for pl in ['A1', 'B3', 'D1', 'E1', 'F1']:
    print(f'  {pl} ({PTYPE[pl]}): ' + ' | '.join(plan.get((y, pl), '—') for y in YEARS))

# 导出
rows = []
for _, p in plots.iterrows():
    for y in YEARS:
        rows.append({'年份': y, '地块': p.pname, '地块类型': p.ptype,
                     '种植方案': plan.get((y, p.pname), '')})
df = pd.DataFrame(rows)
summary = pd.DataFrame({'年份': YEARS,
                        '利润/元': [annual_profit[y] for y in YEARS]})
with pd.ExcelWriter(OUT) as w:
    df.to_excel(w, sheet_name='种植方案', index=False)
    summary.to_excel(w, sheet_name='年度利润', index=False)
print(f'\n结果已保存: {OUT}')
