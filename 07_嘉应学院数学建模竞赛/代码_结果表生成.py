# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:00:19 2026

@author: kk070
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpInteger, value
import traceback

# ====================== 参数配置（根据题目合理设定） ======================
# 人工最大流向数（每个分拣中心最多允许的人工建包方向数，假设为50）
MANUAL_MAX_FLOWS = 50
# 人工建包每个流向每天成本（每人每天90元，每人管理5个格口）
MANUAL_COST_PER_FLOW = 90 / 5   # 18 元/流向/天
# 机器建包每个格口每天成本（现有机器不产生额外变动成本，设为0）
MACHINE_COST_PER_SLOT = 0.0
YEAR_DAYS = 365
GROWTH_RATE = 0.2               # 未来一年货量增长20%

# ====================== 问题1：货量预测 ======================
def forecast_volume(hist_file, pred_days=7):
    print("  读取附件表2...")
    df = pd.read_csv(hist_file, encoding='gbk')
    df = df.rename(columns={'日期':'日期', '始发分拣':'始发分拣中心', '目的分拣':'目的分拣中心', '包裹量':'货量'})
    df = df.sort_values(['始发分拣中心', '目的分拣中心', '日期'])
    predictions = []
    groups = df.groupby(['始发分拣中心', '目的分拣中心'])
    print(f"  共有 {len(groups)} 个流向需要预测")
    for idx, ((orig, dest), group) in enumerate(groups):
        if idx % 100 == 0:
            print(f"  预测进度: {idx}/{len(groups)}")
        ts = group.set_index('日期')['货量'].asfreq('D', fill_value=0)
        try:
            model = ExponentialSmoothing(ts, seasonal_periods=7, trend='add', seasonal='add')
            fitted = model.fit()
            forecast = fitted.forecast(pred_days)
        except:
            forecast = [ts.iloc[-1]] * pred_days
        last_date = ts.index[-1]
        for i, val in enumerate(forecast):
            pred_date = last_date + pd.Timedelta(days=i+1)
            predictions.append([orig, dest, pred_date, max(0, val)])
    pred_df = pd.DataFrame(predictions, columns=['始发分拣中心', '目的分拣中心', '日期', '预测货量'])
    print(f"  预测完成，共 {len(pred_df)} 条记录")
    return pred_df

# ====================== 处理表3 ======================
def process_site_table(site_file):
    df = pd.read_csv(site_file, encoding='gbk')
    grouped = df.groupby('分拣名称').agg({'格口数量':'sum', '产能上限':'sum'}).reset_index()
    grouped = grouped.rename(columns={'分拣名称':'分拣中心', '格口数量':'现有机器总格口数', '产能上限':'机器总产能'})
    # 计算单格口容量
    grouped['机器单格口容量'] = (grouped['机器总产能'] / grouped['现有机器总格口数']).astype(int)
    # 处理异常值（如果总格口数为0或产能为0，则单格口容量设为0，代表无机器能力）
    grouped.loc[grouped['现有机器总格口数'] == 0, '机器单格口容量'] = 0
    grouped['机器单格口容量'] = grouped['机器单格口容量'].fillna(0)
    return grouped

# ====================== 问题2 ======================
def optimize_routing(volume_df, route_df, site_df):
    route_df = route_df.rename(columns={'始发分拣':'首分拣', '目的分拣':'末分拣', '走货路由':'路由序列'})
    print("  计算边货量...")
    flow_on_edge = {}
    for _, row in route_df.iterrows():
        orig = row['首分拣']
        dest = row['末分拣']
        vol = volume_df[(volume_df['始发分拣中心']==orig) & (volume_df['目的分拣中心']==dest)]['预测货量'].sum()
        if vol == 0:
            continue
        route = str(row['路由序列']).split('-')
        for k in range(len(route)-1):
            j = route[k]
            j_next = route[k+1]
            flow_on_edge[(j, j_next)] = flow_on_edge.get((j, j_next), 0) + vol
    print(f"  共有 {len(flow_on_edge)} 条有向边")
    sites = site_df['分拣中心'].unique()
    results = []
    for idx, j in enumerate(sites):
        if idx % 20 == 0:
            print(f"  场地优化进度: {idx}/{len(sites)}")
        out_edges = [(j_next, vol) for (j_from, j_next), vol in flow_on_edge.items() if j_from == j]
        if not out_edges:
            continue
        row = site_df[site_df['分拣中心']==j].iloc[0]
        a_j = row['机器单格口容量']
        # 如果机器单格口容量 <= 0，则该场地无机器能力，全部人工
        if a_j <= 0:
            for e, vol in out_edges:
                results.append({
                    '分拣中心': j,
                    '下一分拣': e,
                    '日货量': vol,
                    '处理方式': '人工',
                    '占用格口数': 0
                })
            continue
        # 正常情况：有机器能力
        L_max = MANUAL_MAX_FLOWS
        cost_man = MANUAL_COST_PER_FLOW
        cost_mach = MACHINE_COST_PER_SLOT
        total_machine_slots = row['现有机器总格口数']
        prob = LpProblem(f"site_{j}", LpMinimize)
        edges = [j_next for j_next, vol in out_edges]
        x = {e: LpVariable(f"x_{j}_{e}", cat=LpBinary) for e in edges}
        prob += lpSum(x[e]*cost_man + (1-x[e])*cost_mach*np.ceil(vol/a_j) for e,vol in out_edges)
        prob += lpSum(x[e] for e in edges) <= L_max
        prob += lpSum((1-x[e])*np.ceil(vol/a_j) for e,vol in out_edges) <= total_machine_slots
        prob.solve()
        for e, vol in out_edges:
            results.append({
                '分拣中心': j,
                '下一分拣': e,
                '日货量': vol,
                '处理方式': '人工' if x[e].value()>0.5 else '机器',
                '占用格口数': np.ceil(vol/a_j) if x[e].value()<0.5 else 0
            })
    print(f"  优化完成，共 {len(results)} 条结果")
    return pd.DataFrame(results)

# ====================== 问题3 ======================
def equipment_optimization(volume_df, route_df, site_df, equip_df):
    route_df = route_df.rename(columns={'始发分拣':'首分拣', '目的分拣':'末分拣', '走货路由':'路由序列'})
    equip_df = equip_df.rename(columns={
        '分拣机名称':'设备类型', '格口数量':'格口数', '设计产能':'设备产能',
        '折旧年限':'折旧年限', '设备成本':'成本'
    })
    # 未来货量（增长20%）
    volume_df_future = volume_df.copy()
    volume_df_future['预测货量'] = volume_df_future['预测货量'] * (1 + GROWTH_RATE)
    print("  计算边货量（未来）...")
    flow_on_edge = {}
    for _, row in route_df.iterrows():
        orig = row['首分拣']
        dest = row['末分拣']
        vol = volume_df_future[(volume_df_future['始发分拣中心']==orig) & (volume_df_future['目的分拣中心']==dest)]['预测货量'].sum()
        if vol == 0:
            continue
        route = str(row['路由序列']).split('-')
        for k in range(len(route)-1):
            j = route[k]
            j_next = route[k+1]
            flow_on_edge[(j, j_next)] = flow_on_edge.get((j, j_next), 0) + vol
    sites = site_df['分拣中心'].unique()
    equip_plans = []
    routing_results = []
    for idx, j in enumerate(sites):
        if idx % 20 == 0:
            print(f"  设备优化进度: {idx}/{len(sites)}")
        out_edges = [(j_next, vol) for (j_from, j_next), vol in flow_on_edge.items() if j_from == j]
        if not out_edges:
            continue
        row = site_df[site_df['分拣中心']==j].iloc[0]
        a_j = row['机器单格口容量']
        # 如果机器能力为0，则全部人工（不买设备）
        if a_j <= 0:
            for e, vol in out_edges:
                routing_results.append({
                    '分拣中心': j,
                    '下一分拣': e,
                    '处理方式': '人工'
                })
            continue
        L_max = MANUAL_MAX_FLOWS
        cost_man = MANUAL_COST_PER_FLOW
        existing_slots = row['现有机器总格口数']
        existing_cap = row['机器总产能']
        equip_types = equip_df['设备类型'].tolist()
        g_d = equip_df['格口数'].tolist()
        cap_d = equip_df['设备产能'].tolist()
        years_d = equip_df['折旧年限'].tolist()
        cost_d = equip_df['成本'].tolist()
        ann_cost_d = [c/y for c,y in zip(cost_d, years_d)]   # 年分摊成本
        prob = LpProblem(f"site_{j}_equip", LpMinimize)
        edges = [j_next for j_next, vol in out_edges]
        x = {e: LpVariable(f"x_{j}_{e}", cat=LpBinary) for e in edges}
        n = {d_idx: LpVariable(f"n_{j}_{d_idx}", lowBound=0, cat=LpInteger) for d_idx in range(len(equip_types))}
        p = LpVariable(f"p_{j}", lowBound=0, cat=LpInteger)   # 人工管理人员数
        total_machine_slots = existing_slots + lpSum(n[d_idx]*g_d[d_idx] for d_idx in range(len(equip_types)))
        total_machine_cap = existing_cap + lpSum(n[d_idx]*cap_d[d_idx] for d_idx in range(len(equip_types)))
        machine_volume = lpSum((1-x[e])*vol for e,vol in out_edges)
        machine_used_slots = lpSum((1-x[e])*np.ceil(vol/a_j) for e,vol in out_edges)
        manual_dir_count = lpSum(x[e] for e in edges)
        # 目标：设备年分摊成本 + 人工建包年成本 + 人工管理员工资
        prob += (lpSum(n[d_idx]*ann_cost_d[d_idx] for d_idx in range(len(equip_types))) +
                 YEAR_DAYS * (manual_dir_count * cost_man + p * 90))
        prob += manual_dir_count <= L_max
        prob += machine_used_slots <= total_machine_slots
        prob += machine_volume <= total_machine_cap
        prob += p >= manual_dir_count / 5
        prob.solve()
        for d_idx in range(len(equip_types)):
            val = n[d_idx].value()
            if val is not None and val > 0:
                equip_plans.append({
                    '分拣中心': j,
                    '设备类型': equip_types[d_idx],
                    '购买数量': int(val)
                })
        for e, vol in out_edges:
            routing_results.append({
                '分拣中心': j,
                '下一分拣': e,
                '处理方式': '人工' if x[e].value()>0.5 else '机器'
            })
    print(f"  设备优化完成，设备计划 {len(equip_plans)} 条，集包规则 {len(routing_results)} 条")
    return pd.DataFrame(equip_plans), pd.DataFrame(routing_results)

# ====================== 主程序 ======================
if __name__ == "__main__":
    base = r"C:\Users\kk070\Desktop\math_modeling\附件"
    hist_file = f"{base}\\附件表2.csv"
    route_file = f"{base}\\附件表1.csv"
    site_file = f"{base}\\附件表3.csv"
    equip_file = f"{base}\\附件表4.xlsx"
    try:
        print("===== 预处理：附件表3 =====")
        site_info = process_site_table(site_file)
        print(f"  共 {len(site_info)} 个分拣中心")
        print(site_info.head())
        
        print("===== 问题1：货量预测 =====")
        try:
            pred_vol = pd.read_excel(f"{base}\\结果表1.xlsx")
            print(f"结果表1读取成功，行数: {len(pred_vol)}")
        except:
            pred_vol = forecast_volume(hist_file)
            pred_vol.to_excel(f"{base}\\结果表1.xlsx", index=False)
            print("结果表1已生成")
        
        print("===== 问题2：集包规则优化 =====")
        route_info = pd.read_csv(route_file, encoding='gbk')
        routing_result = optimize_routing(pred_vol, route_info, site_info)
        routing_result.to_excel(f"{base}\\结果表2.xlsx", index=False)
        print("结果表2已保存")
        
        print("===== 问题3：设备购置决策 =====")
        equip_info = pd.read_excel(equip_file)
        equip_plan, routing_result3 = equipment_optimization(pred_vol, route_info, site_info, equip_info)
        with pd.ExcelWriter(f"{base}\\结果表3.xlsx") as writer:
            equip_plan.to_excel(writer, sheet_name="设备购置情况", index=False)
            routing_result3.to_excel(writer, sheet_name="集包规则", index=False)
        print("结果表3已保存")
        print("全部完成！")
    except Exception as e:
        print("发生错误：", str(e))
        traceback.print_exc()