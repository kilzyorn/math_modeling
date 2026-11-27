# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:17:42 2026

@author: kk070
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# === 自动设置支持中文的字体 ===
def set_chinese_font():
    system = platform.system()
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 常见中文字体列表（按系统分类）
    chinese_fonts = {
        "Windows": ["SimHei", "Microsoft YaHei", "KaiTi", "SimSun"],
        "Darwin": ["PingFang HK", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB", "Heiti TC"],
        "Linux": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Droid Sans Fallback"]
    }
    
    font_found = False
    for font in chinese_fonts.get(system, []):
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            font_found = True
            print(f"✅ 使用中文字体: {font}")
            break
    
    if not font_found:
        # 如果没有找到中文字体，回退到 DejaVu Sans 并改用英文（避免方框）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("⚠️ 未找到中文字体，将使用英文标签以避免乱码。")
        return False  # 表示应使用英文
    
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    return True  # 表示可使用中文

# 应用字体设置
use_chinese = set_chinese_font()

# === 模拟函数 ===
def simulate_predator_prey(M0, O0, r=0.2, a=0.001, s=0.3, b=0.0005, years=50):
    M = np.zeros(years + 1)
    O = np.zeros(years + 1)
    M[0] = M0
    O[0] = O0

    for t in range(years):
        if M[t] <= 0:
            M[t] = 0
        if O[t] <= 0:
            O[t] = 0

        M[t+1] = M[t] * (1 + r - a * O[t])
        O[t+1] = O[t] * (1 - s + b * M[t])

        M[t+1] = max(M[t+1], 0)
        O[t+1] = max(O[t+1], 0)

    return M, O

# === 根据是否支持中文决定标签语言 ===
if use_chinese:
    label_mice = '田鼠'
    label_owls = '猫头鹰'
    ylabel = '数量'
    xlabel = '年份'
    title1 = '情况 (1): 初始 100 田鼠, 50 猫头鹰'
    title2 = '情况 (2): 初始 100 田鼠, 200 猫头鹰'
    title3 = '情况 (3): 调整参数 (r=0.25, a=0.0008, s=0.2, b=0.001)'
else:
    label_mice = 'Mice'
    label_owls = 'Owls'
    ylabel = 'Population'
    xlabel = 'Year'
    title1 = 'Case (1): Initial 100 Mice, 50 Owls'
    title2 = 'Case (2): Initial 100 Mice, 200 Owls'
    title3 = 'Case (3): Adjusted Parameters (r=0.25, a=0.0008, s=0.2, b=0.001)'

# === 绘图 ===
plt.figure(figsize=(14, 10))

# 情况 1
M1, O1 = simulate_predator_prey(M0=100, O0=50)
plt.subplot(3, 1, 1)
plt.plot(M1, label=label_mice, color='green')
plt.plot(O1, label=label_owls, color='red')
plt.title(title1)
plt.ylabel(ylabel)
plt.legend()
plt.grid(True)

# 情况 2
M2, O2 = simulate_predator_prey(M0=100, O0=200)
plt.subplot(3, 1, 2)
plt.plot(M2, label=label_mice, color='green')
plt.plot(O2, label=label_owls, color='red')
plt.title(title2)
plt.ylabel(ylabel)
plt.legend()
plt.grid(True)

# 情况 3：调整参数
M3, O3 = simulate_predator_prey(M0=100, O0=50, r=0.25, a=0.0008, s=0.2, b=0.001)
plt.subplot(3, 1, 3)
plt.plot(M3, label=label_mice, color='green')
plt.plot(O3, label=label_owls, color='red')
plt.title(title3)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
