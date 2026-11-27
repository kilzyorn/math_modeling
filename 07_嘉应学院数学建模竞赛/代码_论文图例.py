# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:34:21 2026

@author: kk070
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

base = r"C:\Users\kk070\Desktop\math_modeling\附件"

# 读取数据
df2 = pd.read_excel(f"{base}\\结果表2.xlsx")

# 图1：整体流向数量占比（人工 vs 机器）
manual_cnt = df2[df2['处理方式']=='人工'].shape[0]
machine_cnt = df2[df2['处理方式']=='机器'].shape[0]
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie([manual_cnt, machine_cnt], labels=['人工', '机器'], autopct='%1.1f%%',
        colors=['#ff9999','#66b3ff'], startangle=90)
ax1.set_title('集包规则处理方式占比（按流向数量）')
plt.savefig(f"{base}\\图1_流向数量占比.png", dpi=300, bbox_inches='tight')
plt.close()

# 图2：整体货量占比（人工 vs 机器）
manual_vol = df2[df2['处理方式']=='人工']['日货量'].sum()
machine_vol = df2[df2['处理方式']=='机器']['日货量'].sum()
fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.pie([manual_vol, machine_vol], labels=['人工', '机器'], autopct='%1.1f%%',
        colors=['#ff9999','#66b3ff'], startangle=90)
ax2.set_title('集包规则处理货量占比（日总量）')
plt.savefig(f"{base}\\图2_货量占比.png", dpi=300, bbox_inches='tight')
plt.close()

# 图3：货量分布直方图（验证人工处理小货量流向的合理性）
volumes = df2['日货量']
fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.hist(volumes, bins=50, color='gray', edgecolor='black', alpha=0.7)
ax3.set_title(f'流向日货量分布 (共{len(volumes)}个流向)')
ax3.set_xlabel('日货量 (件)')
ax3.set_ylabel('频数')
ax3.set_yscale('log')
ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.savefig(f"{base}\\图3_货量分布直方图.png", dpi=300, bbox_inches='tight')
plt.close()

# 图4：人工处理流向数最多的前10个分拣中心（展示是否接近上限）
manual_top10 = df2[df2['处理方式']=='人工'].groupby('分拣中心').size().sort_values(ascending=False).head(10)
fig4, ax4 = plt.subplots(figsize=(10,6))
manual_top10.plot(kind='barh', ax=ax4, color='#ff9999')
ax4.set_title('人工处理流向数最多的前10个分拣中心')
ax4.set_xlabel('人工流向数')
ax4.set_ylabel('分拣中心')
ax4.axvline(x=50, color='red', linestyle='--', label='假设上限(50)')
ax4.legend()
plt.tight_layout()
plt.savefig(f"{base}\\图4_人工流向使用情况.png", dpi=300)
plt.close()

print("精炼图表已生成，共4张：流向占比、货量占比、货量分布、人工流向使用情况。")