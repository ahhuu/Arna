"""
GNSS数据可视化模块
负责生成各类分析图表
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, List, Optional
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class GNSSPlotter:
    """GNSS数据绘图器"""
    
    def __init__(self, config=None):
        from ..core.config import GNSSConfig
        self.config = config if config else GNSSConfig()
        self.setup_fonts()
    
    def setup_fonts(self):
        """设置中文字体"""
        try:
            chinese_fonts = []
            for font in fm.fontManager.ttflist:
                if 'SimHei' in font.name or 'Microsoft YaHei' in font.name or 'SimSun' in font.name:
                    chinese_fonts.append(font.name)
            
            if chinese_fonts:
                plt.rcParams['font.sans-serif'] = chinese_fonts[:3] + ['DejaVu Sans']
                print(f"使用中文字体: {chinese_fonts[:3]}")
            else:
                print("警告: 未找到中文字体，中文可能显示为方框")
        except Exception as e:
            print(f"字体设置警告: {e}")
    
    def plot_code_phase_differences(self, cp_differences: Dict, output_dir: str = None, 
                                  show: bool = True, save: bool = True) -> str:
        """绘制码相差分析图表"""
        
        if not cp_differences.get('differences'):
            print("没有码相差数据可绘制")
            return None
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('码相差分析', fontsize=16, fontweight='bold')
        
        # 收集所有数据用于统计
        all_differences = []
        sat_colors = {}
        color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for sat_idx, (sat_id, freqs) in enumerate(cp_differences['differences'].items()):
            sat_colors[sat_id] = color_cycle[sat_idx % 10]
            
            for freq, data in freqs.items():
                if data['values']:
                    all_differences.extend(data['values'])
        
        if not all_differences:
            plt.close(fig)
            print("没有有效的码相差数据")
            return None
        
        # 1. 时间序列图
        ax1 = axes[0, 0]
        for sat_id, freqs in cp_differences['differences'].items():
            for freq, data in freqs.items():
                if data['values'] and data['times']:
                    times = pd.to_datetime(data['times'])
                    ax1.plot(times, data['values'], 'o-', alpha=0.7, 
                            color=sat_colors[sat_id], label=f'{sat_id}-{freq}', markersize=3)
        
        ax1.set_xlabel('时间')
        ax1.set_ylabel('码相差 (m)')
        ax1.set_title('码相差时间序列')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. 分布直方图
        ax2 = axes[0, 1]
        ax2.hist(all_differences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('码相差 (m)')
        ax2.set_ylabel('频次')
        ax2.set_title('码相差分布')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = np.mean(all_differences)
        std_val = np.std(all_differences)
        ax2.axvline(x=mean_val, color='red', linestyle='--', 
                   label=f'均值: {mean_val:.3f}m')
        ax2.axvline(x=mean_val + 3*std_val, color='orange', linestyle='--', 
                   label=f'+3σ: {mean_val + 3*std_val:.3f}m')
        ax2.axvline(x=mean_val - 3*std_val, color='orange', linestyle='--', 
                   label=f'-3σ: {mean_val - 3*std_val:.3f}m')
        ax2.legend()
        
        # 3. 卫星统计对比
        ax3 = axes[1, 0]
        sat_stats = []
        sat_labels = []
        
        for sat_id in cp_differences['statistics']:
            for freq in cp_differences['statistics'][sat_id]:
                stats = cp_differences['statistics'][sat_id][freq]
                if stats['count'] > 0:
                    sat_stats.append([stats['mean'], stats['std'], stats['count']])
                    sat_labels.append(f'{sat_id}-{freq}')
        
        if sat_stats:
            sat_stats = np.array(sat_stats)
            x_pos = np.arange(len(sat_labels))
            
            bars = ax3.bar(x_pos, sat_stats[:, 0], yerr=sat_stats[:, 1], 
                          alpha=0.7, capsize=5)
            ax3.set_xlabel('卫星-频率')
            ax3.set_ylabel('码相差均值 (m)')
            ax3.set_title('各卫星码相差统计')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(sat_labels, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
        
        # 4. 异常值分布
        ax4 = axes[1, 1]
        outlier_counts = []
        outlier_labels = []
        
        for sat_id in cp_differences['outliers']:
            for freq in cp_differences['outliers'][sat_id]:
                outliers = cp_differences['outliers'][sat_id][freq]
                outlier_counts.append(outliers['count'])
                outlier_labels.append(f'{sat_id}-{freq}')
        
        if outlier_counts:
            colors = ['red' if count > 0 else 'green' for count in outlier_counts]
            ax4.bar(range(len(outlier_counts)), outlier_counts, color=colors, alpha=0.7)
            ax4.set_xlabel('卫星-频率')
            ax4.set_ylabel('异常值数量')
            ax4.set_title('异常值统计')
            ax4.set_xticks(range(len(outlier_labels)))
            ax4.set_xticklabels(outlier_labels, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = None
        if save and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'code_phase_differences.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"码相差分析图保存至: {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return output_file
    
    def plot_phase_stagnation(self, stagnation_results: Dict, output_dir: str = None, 
                            show: bool = True, save: bool = True) -> str:
        """绘制相位停滞分析图表"""
        
        if not stagnation_results.get('detected'):
            print("未检测到相位停滞现象")
            return None
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('相位停滞分析', fontsize=16, fontweight='bold')
        
        # 1. 停滞卫星统计
        ax1 = axes[0, 0]
        stagnant_sats = []
        stagnant_counts = []
        
        for sat_id, freqs in stagnation_results['details'].items():
            total_periods = 0
            for freq, details in freqs.items():
                if details['is_stagnant']:
                    total_periods += len(details['stagnation_periods'])
            if total_periods > 0:
                stagnant_sats.append(sat_id)
                stagnant_counts.append(total_periods)
        
        if stagnant_sats:
            ax1.bar(range(len(stagnant_sats)), stagnant_counts, alpha=0.7, color='red')
            ax1.set_xlabel('卫星')
            ax1.set_ylabel('停滞期数量')
            ax1.set_title('相位停滞卫星统计')
            ax1.set_xticks(range(len(stagnant_sats)))
            ax1.set_xticklabels(stagnant_sats, rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # 2. 停滞期持续时间分布
        ax2 = axes[0, 1]
        durations = []
        
        for sat_id, freqs in stagnation_results['details'].items():
            for freq, details in freqs.items():
                for period in details.get('stagnation_periods', []):
                    durations.append(period['duration_epochs'])
        
        if durations:
            ax2.hist(durations, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax2.set_xlabel('停滞期持续历元数')
            ax2.set_ylabel('频次')
            ax2.set_title('停滞期持续时间分布')
            ax2.grid(True, alpha=0.3)
        
        # 3. 停滞现象总结
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        summary = stagnation_results['summary']
        summary_text = f"""相位停滞检测结果:
        
总卫星数: {summary['total_satellites']}
受影响卫星数: {summary['affected_satellites']}
影响率: {summary['affected_satellites']/max(summary['total_satellites'], 1)*100:.1f}%

停滞期总数: {len(durations)}
平均持续时间: {np.mean(durations):.1f} 历元
最长持续时间: {max(durations) if durations else 0} 历元"""
        
        ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # 4. 频率对比
        ax4 = axes[1, 1]
        freq_stats = defaultdict(int)
        
        for sat_id, freqs in stagnation_results['details'].items():
            for freq, details in freqs.items():
                if details['is_stagnant']:
                    freq_stats[freq] += len(details['stagnation_periods'])
        
        if freq_stats:
            freqs = list(freq_stats.keys())
            counts = list(freq_stats.values())
            ax4.pie(counts, labels=freqs, autopct='%1.1f%%', alpha=0.7)
            ax4.set_title('各频率停滞现象分布')
        
        plt.tight_layout()
        
        # 保存图表
        output_file = None
        if save and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'phase_stagnation.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"相位停滞分析图保存至: {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return output_file
    
    def plot_isb_analysis(self, isb_results: Dict, output_dir: str = None, 
                         show: bool = True, save: bool = True) -> str:
        """绘制ISB分析图表"""
        
        if not isb_results or 'isb_estimates' not in isb_results:
            print("没有ISB分析结果可绘制")
            return None
        
        # 创建图表
        fig = plt.figure(figsize=(16, 12))
        
        # 主标题
        fig.suptitle('系统间偏差(ISB)分析结果', fontsize=16, fontweight='bold')
        
        # 创建子图布局
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222) 
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        # ISB时间序列
        isb_values = np.array(isb_results['isb_estimates'])
        epochs = isb_results['isb_epochs']
        
        ax1.plot(epochs, isb_values, 'b-', linewidth=1, alpha=0.7, label='ISB时间序列')
        ax1.axhline(y=isb_results['isb_mean'], color='r', linestyle='--',
                   label=f"均值: {isb_results['isb_mean']:.3f}m")
        ax1.set_xlabel('时间')
        ax1.set_ylabel('ISB (m)')
        ax1.set_title('ISB时间序列')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ISB直方图
        ax2.hist(isb_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=isb_results['isb_mean'], color='r', linestyle='--', linewidth=2,
                   label=f"均值: {isb_results['isb_mean']:.3f}m")
        ax2.set_xlabel('ISB (m)')
        ax2.set_ylabel('频次')
        ax2.set_title('ISB分布直方图')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ISB统计信息
        ax3.axis('off')
        stats_text = f"""ISB统计信息:
        
均值: {isb_results['isb_mean']:.3f} m
标准差: {isb_results['isb_std']:.3f} m
最小值: {np.min(isb_values):.3f} m
最大值: {np.max(isb_values):.3f} m
有效历元数: {len(isb_values)}
基准卫星: {isb_results['reference_satellite']}"""
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8,
                         edgecolor='black', linewidth=1))
        
        # ISB残差分析
        ax4.plot(epochs, isb_values - isb_results['isb_mean'], 'g-', linewidth=1, alpha=0.7)
        ax4.axhline(y=0, color='r', linestyle='-', linewidth=1)
        ax4.set_xlabel('时间')
        ax4.set_ylabel('ISB 残差 (m)')
        ax4.set_title('ISB 残差时间序列')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = None
        if save and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'isb_analysis.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"ISB分析图保存至: {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return output_file
    
    def plot_satellite_sky_plot(self, observations: Dict, output_dir: str = None, 
                              show: bool = True, save: bool = True) -> str:
        """绘制卫星天空图"""
        
        # 这里需要卫星的方位角和仰角数据
        # 由于原始数据中可能没有这些信息，这里提供一个框架
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 示例：随机生成一些卫星位置用于演示
        for sat_id in list(observations.keys())[:10]:  # 限制显示前10颗卫星
            # 这里应该从观测数据中计算或读取卫星的方位角和仰角
            # 现在使用随机值作为示例
            azimuth = np.random.uniform(0, 2*np.pi)
            elevation = np.random.uniform(10, 90)  # 仰角度数
            
            # 极坐标中，半径对应天顶角(90-仰角)
            radius = 90 - elevation
            
            ax.plot(azimuth, radius, 'o', markersize=8, label=sat_id)
            ax.text(azimuth, radius-5, sat_id, ha='center', va='center', fontsize=8)
        
        ax.set_ylim(0, 90)
        ax.set_yticks(np.arange(0, 91, 30))
        ax.set_yticklabels(['90°', '60°', '30°', '0°'])
        ax.set_title('卫星天空图', pad=20)
        ax.grid(True)
        
        # 保存图表
        output_file = None
        if save and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'satellite_skyplot.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"卫星天空图保存至: {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return output_file
    
    def close_all_figures(self):
        """关闭所有图表"""
        plt.close('all')