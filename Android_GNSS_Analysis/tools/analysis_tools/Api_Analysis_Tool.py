"""
Android GNSS API Uncertainty Analysis Tool (v6 - Combined & Subplots)
功能：
1. 【Subplots模式】：分系统绘制，避免重叠，查看各系统细节。
2. 【Combined模式 (新增)】：不分系统（合绘），查看整体规律和模型一致性。
3. 核心算法：P98 自动 Y 轴缩放、纯白文献风格。

Date: 2026-01-22
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
CLIGHT = 299792458.0  # 光速

# 颜色映射
COLOR_MAP = {
    'GPS': '#1f77b4',      # 蓝
    'GLONASS': '#ff7f0e',  # 橙
    'BDS': '#d62728',      # 红
    'Galileo': '#2ca02c',  # 绿
    'QZSS': '#9467bd',     # 紫
    'UNKNOWN': '#7f7f7f'   # 灰
}

# 卫星系统映射
SYS_MAP = {1: 'GPS', 3: 'GLONASS', 4: 'QZSS', 5: 'BDS', 6: 'Galileo', 0: 'UNKNOWN'}

def setup_publication_style():
    """设置纯白文献风格"""
    plt.rcdefaults()
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
    })

# =============================================================================
# 数据解析类
# =============================================================================
class GnssSat:
    def __init__(self):
        self.time_nano = 0
        self.svid = 0
        self.constellation_type = 0
        self.cn0_dbhz = 0.0
        self.received_sv_time_uncertainty_nano = 0
        self.accumulated_delta_range_uncertainty_meter = 0.0
        self.pseudorange_rate_uncertainty_meter_per_second = 0.0

    def parse_from(self, line):
        parts = line.strip().split(',')
        parts = [p if p != '' else '0' for p in parts]
        try:
            self.time_nano = int(parts[2])
            self.svid = int(parts[11])
            self.received_sv_time_uncertainty_nano = float(parts[15])
            self.cn0_dbhz = float(parts[16])
            self.pseudorange_rate_uncertainty_meter_per_second = float(parts[18])
            self.accumulated_delta_range_uncertainty_meter = float(parts[21])
            self.constellation_type = int(parts[28])
        except ValueError:
            pass

# =============================================================================
# 核心处理逻辑
# =============================================================================
def analyze_file(file_path):
    logger.info(f"处理文件: {os.path.basename(file_path)}")
    data = {}
    start_time = None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if "Raw," in line and "#" not in line:
                sat = GnssSat()
                sat.parse_from(line)

                sys_name = SYS_MAP.get(sat.constellation_type, 'UNKNOWN')
                if sys_name not in data:
                    data[sys_name] = {'epoch': [], 'cn0': [], 'unc_time_m': [], 'unc_adr_m': [], 'unc_prr_ms': []}

                if start_time is None: start_time = sat.time_nano
                rel_time = (sat.time_nano - start_time) * 1e-9

                unc_time_m = sat.received_sv_time_uncertainty_nano * 1e-9 * CLIGHT

                if sat.cn0_dbhz <= 0: continue

                data[sys_name]['epoch'].append(rel_time)
                data[sys_name]['cn0'].append(sat.cn0_dbhz)
                data[sys_name]['unc_time_m'].append(unc_time_m)
                data[sys_name]['unc_adr_m'].append(sat.accumulated_delta_range_uncertainty_meter)
                data[sys_name]['unc_prr_ms'].append(sat.pseudorange_rate_uncertainty_meter_per_second)

        input_dir = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(input_dir, "Api Analysis", file_name)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        # 1. 绘制分系统子图 (Subplots)
        plot_metrics_subplots(data, output_dir)

        # 2. 绘制综合图 (Combined)
        plot_metrics_combined(data, output_dir)

        return True

    except Exception as e:
        logger.error(f"Error: {e}")
        return False

# =============================================================================
# 绘图逻辑 A: 分系统子图 (Subplots)
# =============================================================================
def plot_metrics_subplots(data, output_dir):
    active_systems = get_sorted_systems(data)
    if not active_systems: return

    metrics = get_metrics_config()

    for key, ylabel, fname_suffix, title_desc in metrics:

        # 计算全局 P98 范围 (所有系统共享同一个Y轴范围，方便对比)
        current_ylim = calculate_p98_ylim(data, active_systems, key)
        if current_ylim is None: continue

        def create_subplots(x_key, x_label, suffix):
            num = len(active_systems)
            fig, axes = plt.subplots(nrows=num, ncols=1, sharex=True, figsize=(8, 2 * num + 1))
            if num == 1: axes = [axes]

            for i, sys_name in enumerate(active_systems):
                ax = axes[i]
                d = data[sys_name]
                ax.scatter(d[x_key], d[key], s=4,
                           color=COLOR_MAP.get(sys_name, 'black'),
                           alpha=0.6, edgecolors='none')

                ax.set_ylim(current_ylim)
                # 标签放在图内
                ax.text(0.98, 0.85, sys_name, transform=ax.transAxes,
                        horizontalalignment='right', fontsize=12, fontweight='bold',
                        color=COLOR_MAP.get(sys_name, 'black'))
                ax.set_ylabel(ylabel, fontsize=9)

            axes[-1].set_xlabel(x_label)
            axes[0].set_title(f'{title_desc} vs {suffix} (Subplots)')

            plt.subplots_adjust(hspace=0.1)
            plt.savefig(os.path.join(output_dir, f"{fname_suffix}_vs_{suffix}_Subplots.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

        create_subplots('epoch', 'epoch', 'epoch')
        create_subplots('cn0', 'C/N0 (dB-Hz)', 'CN0')

# =============================================================================
# 绘图逻辑 B: 综合图 (Combined) - 新增功能
# =============================================================================
def plot_metrics_combined(data, output_dir):
    active_systems = get_sorted_systems(data)
    if not active_systems: return

    metrics = get_metrics_config()

    for key, ylabel, fname_suffix, title_desc in metrics:

        # 同样使用 P98 自动缩放
        current_ylim = calculate_p98_ylim(data, active_systems, key)
        if current_ylim is None: continue

        def create_combined(x_key, x_label, suffix):
            fig, ax = plt.subplots(figsize=(8, 6))

            # 遍历所有系统绘制在同一张图上
            for sys_name in active_systems:
                d = data[sys_name]
                if len(d[x_key]) > 0:
                    ax.scatter(d[x_key], d[key], s=4,
                               label=sys_name, # 用于生成图例
                               color=COLOR_MAP.get(sys_name, 'black'),
                               alpha=0.6, edgecolors='none')

            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title_desc} vs {suffix} (Combined)')
            ax.set_ylim(current_ylim)

            # 统一图例 (放在底部)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(active_systems), frameon=False)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{fname_suffix}_vs_{suffix}_Combined.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

        create_combined('epoch', 'epoch', 'epoch')
        create_combined('cn0', 'C/N0 (dB-Hz)', 'CN0')

# =============================================================================
# 通用工具函数
# =============================================================================
def get_metrics_config():
    return [
        ('unc_time_m', 'Pseudorange Unc. (m)', 'TimeUnc', 'Pseudorange Uncertainty'),
        ('unc_adr_m', 'ADR Unc. (m)', 'AdrUnc', 'Carrier Phase Uncertainty'),
        ('unc_prr_ms', 'Doppler Unc. (m/s)', 'DopplerUnc', 'Doppler Uncertainty')
    ]

def get_sorted_systems(data):
    active = [sys for sys in data.keys() if len(data[sys]['epoch']) > 0]
    priority = {'GPS':1, 'BDS':2, 'Galileo':3, 'GLONASS':4, 'QZSS':5, 'UNKNOWN':6}
    active.sort(key=lambda x: priority.get(x, 99))
    return active

def calculate_p98_ylim(data, active_systems, key):
    all_values = []
    for sys in active_systems:
        all_values.extend(data[sys][key])

    if not all_values: return None

    p98 = np.percentile(all_values, 98)
    y_max = p98 * 1.1 if p98 > 0 else 1.0
    return 0, y_max

# =============================================================================
# 主入口
# =============================================================================
def main():
    setup_publication_style()
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # 置顶窗口

    print("选择 Android GNSS Log 文件...")
    input_files = filedialog.askopenfilenames(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
    if not input_files:
        root.destroy()
        return

    count = 0
    for f in input_files:
        if analyze_file(f): count += 1

    if count > 0:
        messagebox.showinfo("完成", f"已处理 {count} 个文件。\n请检查 Api Analysis 文件夹。", parent=root)
    root.destroy()

if __name__ == "__main__":
    main()