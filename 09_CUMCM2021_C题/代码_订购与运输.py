# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:40:55 2026

@author: kk070
"""

# =============================
# 导入库
# =============================
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).resolve().parent

# =============================
# 读取附件1：供应商订货与供货数据
# =============================
file1 = BASE / '数据_附件1_供应商.xlsx'
raw_order = pd.read_excel(file1, sheet_name='企业的订货量（m³）')
raw_supply = pd.read_excel(file1, sheet_name='供应商的供货量（m³）')

# 提取供应商ID和材料类型
supplier_ids = raw_order.iloc[:, 0]          # 第1列：供应商ID
material_type = raw_order.iloc[:, 1]         # 第2列：材料分类（A/B/C）
order_data = raw_order.iloc[:, 2:]           # 第3列起：240周订货量
supply_data = raw_supply.iloc[:, 2:]         # 第3列起：240周供货量

# 设置索引
order_data.index = supplier_ids
supply_data.index = supplier_ids
material_type.index = supplier_ids

print("✅ 附件1加载成功！")
print("订货量数据形状:", order_data.shape)
print("供货量数据形状:", supply_data.shape)

# =============================
# 读取附件2：转运商损耗率
# =============================
file2 = BASE / '数据_附件2_转运商.xlsx'
loss_raw = pd.read_excel(file2)

transporter_ids = loss_raw.iloc[:, 0]        # 第1列：转运商ID（T1~T8）
loss_data = loss_raw.iloc[:, 1:]             # 第2列起：240周损耗率（%）

# 转换为小数（如 1.5539% → 0.015539）
loss_data = loss_data / 100.0
loss_data.index = transporter_ids

print("✅ 附件2加载成功！")
print("转运商损耗率数据形状:", loss_data.shape)

# =============================
# 问题1：构建四个指标
# =============================
metrics = pd.DataFrame(index=supplier_ids)
eps = 1e-6  # 防止除零

# 指标1：平均供货量
metrics['avg_supply'] = supply_data.mean(axis=1)

# 指标2：供货满足率 = min(订货, 供货) 的总和 / 订货总和
order_sum = order_data.sum(axis=1)
supply_clipped = np.minimum(order_data, supply_data)
satisfied_sum = supply_clipped.sum(axis=1)
metrics['fulfill_rate'] = satisfied_sum / (order_sum + eps)

# 指标3：稳定性 = 平均供货量 / 供货标准差（越大越稳定）
supply_std = supply_data.std(axis=1)
metrics['stability'] = metrics['avg_supply'] / (supply_std + eps)

# 指标4：缺货频率倒数 = 1 - 缺货周数/总周数
is_shortage = (supply_data < order_data) & (order_data > 0)
shortage_freq = is_shortage.sum(axis=1) / 240
metrics['shortage_inv'] = 1 - shortage_freq

# 填充 NaN（如从未供货的供应商）
metrics = metrics.fillna(0)

print("\n✅ 四个指标计算完成！")
print("前5行指标：")
print(metrics.head())

# =============================
# 熵权法计算权重
# =============================
def entropy_weight(data):
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    data_norm = np.maximum(data_norm, 1e-6)  # 避免 log(0)
    p = data_norm / data_norm.sum(axis=0)
    e = -np.sum(p * np.log(p), axis=0) / np.log(data.shape[0])
    d = 1 - e
    w = d / d.sum()
    return w, data_norm

# 提取四个指标
X = metrics[['avg_supply', 'fulfill_rate', 'stability', 'shortage_inv']].values
weights, X_norm = entropy_weight(X)

# 打印权重
weight_names = ['平均供货量', '供货满足率', '稳定性', '缺货频率倒数']
for name, w in zip(weight_names, weights):
    print(f"{name}: {w:.4f}")

# 计算综合得分
metrics['score'] = X_norm @ weights
metrics = metrics.sort_values('score', ascending=False)

# 取前50
top50 = metrics.head(50).reset_index()
top50.rename(columns={'index': '供应商ID'}, inplace=True)
top50['排名'] = range(1, 51)
top50 = top50[['排名', '供应商ID', 'score']]
top50.rename(columns={'score': '综合得分'}, inplace=True)

print("\n✅ 前10名供应商：")
print(top50.head(10))

# =============================
# 分析前50的材料类型分布
# =============================
top50_ids = top50['供应商ID'].tolist()
top50_materials = material_type[top50_ids]

print("\n🔍 前50供应商的材料类型分布：")
print(top50_materials.value_counts())

# 按材料类型分组（用于问题2）
c_suppliers = top50_materials[top50_materials == 'C'].index.tolist()
b_suppliers = top50_materials[top50_materials == 'B'].index.tolist()
a_suppliers = top50_materials[top50_materials == 'A']. index.tolist()

print(f"\nC类供应商数量: {len(c_suppliers)}")
print(f"B类供应商数量: {len(b_suppliers)}")
print(f"A类供应商数量: {len(a_suppliers)}")

# =============================
# 问题2：前8周订购与转运方案
# =============================
# 参数设置
weekly_demand = 28200 / 0.72   # 全用C类原料，39166.67 m³/周
safety_stock = 2 * weekly_demand  # 安全库存 = 78333.33
initial_inventory = safety_stock
weeks = 8

# =============================
# 改进：将C类供应商均匀分配到8家转运商（按损耗排序）
# =============================
# 先按平均损耗对转运商排序
avg_loss_per_transporter = loss_data.mean(axis=1).sort_values()
sorted_transporters = avg_loss_per_transporter.index.tolist()  # T3, T6, T1, ...

assigned_transporter = {}
transporter_avg_loss = {}

# C类：均匀分配到转运商（轮询）
for i, supplier in enumerate(c_suppliers):
    transporter = sorted_transporters[i % len(sorted_transporters)]
    assigned_transporter[supplier] = transporter
    transporter_avg_loss[supplier] = avg_loss_per_transporter[transporter]

# B类和A类：同样处理（或沿用原逻辑）
for supplier_list in [b_suppliers, a_suppliers]:
    for i, supplier in enumerate(supplier_list):
        transporter = sorted_transporters[i % len(sorted_transporters)]
        assigned_transporter[supplier] = transporter
        transporter_avg_loss[supplier] = avg_loss_per_transporter[transporter]

print("\n✅ 改进版转运商分配完成！C类前5家：")
for sid in c_suppliers[:5]:
    print(f"{sid} → {assigned_transporter[sid]} (损耗: {transporter_avg_loss[sid]:.4f})")
    # =============================
# 初始化订货与转运计划表
# =============================
weeks = 8
order_plan = pd.DataFrame(0.0, index=top50_ids, columns=[f'第{w+1}周' for w in range(weeks)])
transport_plan = pd.DataFrame(0.0, index=loss_data.index, columns=[f'第{w+1}周' for w in range(weeks)])

# 供应商供货能力（设为足够大）
supply_capacity = pd.Series(1e5, index=top50_ids)

# 按优先级排序：C > B > A，每类内部按综合得分降序
def get_sorted_suppliers(mat_list, score_series):
    return sorted(mat_list, key=lambda x: score_series[x], reverse=True)

c_sorted = get_sorted_suppliers(c_suppliers, metrics['score'])
b_sorted = get_sorted_suppliers(b_suppliers, metrics['score'])
a_sorted = get_sorted_suppliers(a_suppliers, metrics['score'])

# =============================
# 模拟8周订货过程
# =============================
weekly_demand = 28200 / 0.72   # ≈39166.67 m³/周
safety_stock = 2 * weekly_demand
initial_inventory = safety_stock
inventory = initial_inventory

for week in range(weeks):
    # 计算本周需补充量
    after_consume = inventory - weekly_demand
    need_to_order = max(0, safety_stock - after_consume)
    remaining_need = need_to_order

    # 按 C → B → A 顺序满足需求
    for supplier_list in [c_sorted, b_sorted, a_sorted]:
        for supplier in supplier_list:
            if remaining_need <= 1e-3:
                break
            transporter = assigned_transporter[supplier]
            current_transported = transport_plan.loc[transporter, f'第{week+1}周']
            available_cap = 6000 - current_transported
            
            if available_cap <= 0:
                continue
                
            order_amount = min(remaining_need, supply_capacity[supplier], available_cap)
            if order_amount <= 0:
                continue
                
            # 记录订货和转运
            order_plan.loc[supplier, f'第{week+1}周'] = order_amount
            transport_plan.loc[transporter, f'第{week+1}周'] += order_amount
            remaining_need -= order_amount

    # 计算净收货（考虑损耗）
    net_receipt = 0.0
    for supplier in top50_ids:
        amount = order_plan.loc[supplier, f'第{week+1}周']
        loss_rate = transporter_avg_loss[supplier]
        net_receipt += amount * (1 - loss_rate)
    
    inventory = inventory - weekly_demand + net_receipt

    print(f"第{week+1}周: 需补充 {need_to_order:.1f}, 实际订货 {need_to_order - remaining_need:.1f}, "
          f"未满足 {remaining_need:.1f}, 净收货 {net_receipt:.1f}, 期末库存 {inventory:.1f}")

# =============================
# 保存结果并统计
# =============================
order_plan_nonzero = order_plan.loc[(order_plan > 0).any(axis=1)]
transport_plan_nonzero = transport_plan.loc[(transport_plan > 0).any(axis=1)]

order_plan_nonzero.to_csv(BASE / "ordering_plan.csv", encoding='utf-8-sig')
transport_plan_nonzero.to_csv(BASE / "transport_plan.csv", encoding='utf-8-sig')

print(f"\n✅ 有订货的供应商数量: {len(order_plan_nonzero)}")

# 安全统计各材料使用量
ordered_suppliers = order_plan_nonzero.index.tolist()
c_used = [s for s in ordered_suppliers if s in c_suppliers]
b_used = [s for s in ordered_suppliers if s in b_suppliers]
a_used = [s for s in ordered_suppliers if s in a_suppliers]

total_c = order_plan_nonzero.loc[c_used].sum().sum() if c_used else 0
total_b = order_plan_nonzero.loc[b_used].sum().sum() if b_used else 0
total_a = order_plan_nonzero.loc[a_used].sum().sum() if a_used else 0

print("\n📊 各材料类型总订货量：")
print(f"C类: {total_c:,.0f} m³")
print(f"B类: {total_b:,.0f} m³")
print(f"A类: {total_a:,.0f} m³")
print(f"总计: {total_c + total_b + total_a:,.0f} m³")

if inventory >= safety_stock - 500:
    print("\n🎉 库存始终满足安全要求！")
else:
    print("\n⚠️ 库存不足，请检查方案！")