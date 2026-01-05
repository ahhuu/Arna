import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI线程问题
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import os
import sys


# 定义模型函数
def elevation_model(elevation_rad, a, b):
    """高度角模型：σ² = a + b / sin²(el)"""
    sin_el = np.sin(elevation_rad)
    sin_el = np.clip(sin_el, 0.1, 1.0)  # 避免sin(el)过小导致数值问题
    return a + b / (sin_el ** 2)


def snr_model(snr, a, b):
    """信噪比模型：σ² = a + b × 10^(-SNR/10)"""
    return a + b * 10 ** (-snr / 10)


def linear_snr_model(snr, a, b):
    """线性信噪比模型：σ² = a × SNR + b"""
    return a * snr + b


def combined_model(params, elevation_rad, snr):
    """联合模型：σ² = a + b / sin²(el) + c × 10^(-SNR/10)"""
    a, b, c = params
    sin_el = np.sin(elevation_rad)
    sin_el = np.clip(sin_el, 0.1, 1.0)
    return a + b / (sin_el ** 2) + c * 10 ** (-snr / 10)


def error_function(params, model, x, y):
    """联合模型的误差函数"""
    return model(params, *x) - y ** 2


# 模型拟合函数
def fit_elevation_model(x, y):
    p0 = [1, 1]
    return curve_fit(elevation_model, x, y ** 2, p0=p0)[0]


def fit_snr_model(x, y):
    p0 = [1, 100]
    return curve_fit(snr_model, x, y ** 2, p0=p0)[0]


def fit_linear_snr_model(x, y):
    p0 = [0.1, 10]
    return curve_fit(linear_snr_model, x, y ** 2, p0=p0)[0]


def fit_combined_model(x, y):
    x_comb = (x[0], x[1])
    p0 = [1, 1, 100]
    return least_squares(error_function, p0, args=(combined_model, x_comb, y)).x


# 生成模型信息文本函数
def generate_model_info(system, freq, data_count,
                        elev_params, snr_params, linear_snr_params, combined_params):
    model_info = f"""
卫星系统: {system}, 频率: {freq}, 数据量: {data_count}

高度角模型：
σ² = {elev_params[0]:.4f}+{elev_params[1]:.4f}./sin(e).^2

指数信噪比模型：
σ² = {snr_params[0]:.4f}+{snr_params[1]:.4f}.*10.^(-SNR/10)

线性信噪比模型：
σ² = {linear_snr_params[0]:.4f}.*SNR+{linear_snr_params[1]:.4f}

联合模型：
σ² = {combined_params[0]:.4f}+{combined_params[1]:.4f}./sin(e).^2+{combined_params[2]:.4f}.*10.^(-SNR/10)
"""
    return model_info


# 可视化函数
def visualize_elevation_model(ax, subset, a_elev, b_elev, system, freq, data_count):
    ax.scatter(subset['ele'], subset['residual'].values ** 2, alpha=0.5, s=10, label='观测值')
    elev_range = np.linspace(min(subset['ele']), max(subset['ele']), 100)
    elev_range_rad = np.radians(elev_range)
    # 绘制方差拟合曲线
    variance_pred = elevation_model(elev_range_rad, a_elev, b_elev)
    ax.plot(elev_range, variance_pred, 'r-', linewidth=2, label='拟合曲线')
    ax.set_xlabel('高度角 (°)')
    ax.set_ylabel('伪距残差平方 (m²)')
    ax.set_title('高度角模型')
    ax.legend()


def visualize_snr_model(ax, subset, a_snr, b_snr, system, freq, data_count):
    ax.scatter(subset['snr'], subset['residual'].values ** 2, alpha=0.5, s=10, label='观测值')
    snr_range = np.linspace(min(subset['snr']), max(subset['snr']), 100)
    # 绘制方差拟合曲线
    variance_pred = snr_model(snr_range, a_snr, b_snr)
    ax.plot(snr_range, variance_pred, 'g-', linewidth=2, label='拟合曲线')
    ax.set_xlabel('信噪比 (dB-Hz)')
    ax.set_ylabel('伪距残差平方 (m²)')
    ax.set_title('指数信噪比模型')
    ax.legend()


def visualize_linear_snr_model(ax, subset, a_linear_snr, b_linear_snr, system, freq, data_count):
    ax.scatter(subset['snr'], subset['residual'].values ** 2, alpha=0.5, s=10, label='观测值')
    snr_range = np.linspace(min(subset['snr']), max(subset['snr']), 100)
    # 绘制方差拟合曲线
    variance_pred = linear_snr_model(snr_range, a_linear_snr, b_linear_snr)
    ax.plot(snr_range, variance_pred, 'm-', linewidth=2, label='拟合曲线')
    ax.set_xlabel('信噪比 (dB-Hz)')
    ax.set_ylabel('伪距残差平方 (m²)')
    ax.set_title('线性信噪比模型')
    ax.legend()


def visualize_combined_model(ax, subset, a_comb, b_comb, c_comb, system, freq, data_count):
    # 按历元排序数据
    subset_sorted = subset.sort_values('epoch')
    
    # 创建历元（从1开始）
    subset_sorted = subset_sorted.copy()
    subset_sorted['epoch_num'] = range(1, len(subset_sorted) + 1)
    
    # 创建双y轴
    ax2 = ax.twinx()
    
    # 绘制伪距残差平方随历元变化 - 显示所有数据点，使用更小的点
    ax.scatter(subset_sorted['epoch_num'], subset_sorted['residual'] ** 2, 
              alpha=0.4, s=6, color='red', label='伪距残差平方')
    ax.set_xlabel('历元')
    ax.set_ylabel('伪距残差平方（m²）', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    
    # 设置伪距残差平方的y轴范围
    residual_squared = subset_sorted['residual'] ** 2
    residual_min, residual_max = residual_squared.min(), residual_squared.max()
    residual_range = residual_max - residual_min
    if residual_range > 0:
        # 添加30%的边距
        margin = residual_range * 0.3
        ax.set_ylim(residual_min - margin, residual_max + margin)
    
    # 绘制高度角随历元变化 - 使用散点
    ax2.scatter(subset_sorted['epoch_num'], subset_sorted['ele'], 
               alpha=0.6, s=6, color='blue', label='高度角')
    ax2.set_ylabel('高度角（°）', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # 设置高度角的y轴范围
    ele_min, ele_max = subset_sorted['ele'].min(), subset_sorted['ele'].max()
    ele_range = ele_max - ele_min
    if ele_range > 0:
        # 添加30%的边距
        margin = ele_range * 0.3
        ax2.set_ylim(max(0, ele_min - margin), min(90, ele_max + margin))
    else:
        ax2.set_ylim(0, 90)
    
    # 如果有信噪比数据，也绘制信噪比 - 使用散点
    if 'snr' in subset_sorted.columns and not subset_sorted['snr'].isna().all():
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.scatter(subset_sorted['epoch_num'], subset_sorted['snr'], 
                   alpha=0.6, s=6, color='green', label='信噪比')
        ax3.set_ylabel('信噪比（dB-Hz）', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        
        # 设置信噪比的y轴范围
        snr_min, snr_max = subset_sorted['snr'].min(), subset_sorted['snr'].max()
        snr_range = snr_max - snr_min
        if snr_range > 0:
            # 添加30%的边距
            margin = snr_range * 0.3
            ax3.set_ylim(max(0, snr_min - margin), snr_max + margin)
        else:
            ax3.set_ylim(0, 50)
    
    ax.set_title('联合模型 - 伪距残差平方随历元变化')
    ax.grid(True, alpha=0.3)
    
    # 优化x轴刻度显示
    max_epoch = len(subset_sorted)
    if max_epoch > 20:
        step = max(1, max_epoch // 10)  # 最多显示10个刻度
        ax.set_xticks(range(1, max_epoch + 1, step))
    else:
        ax.set_xticks(range(1, max_epoch + 1))
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if 'snr' in subset_sorted.columns and not subset_sorted['snr'].isna().all():
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
    else:
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')


def visualize_individual_combined_models(subset, system, freq, output_dir):
    """为每个卫星每个频率绘制单独的联合模型图"""
    # 按PRN分组
    prns = subset['prn'].unique()
    
    if len(prns) == 0:
        return
    
    # 计算子图布局
    n_prns = len(prns)
    n_cols = min(3, n_prns)  # 最多3列
    n_rows = (n_prns + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # 确保axes是二维数组
    if n_prns == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = np.array(axes)
    
    for i, prn in enumerate(prns):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        prn_data = subset[subset['prn'] == prn].sort_values('epoch')
        
        if len(prn_data) > 0:
            # 创建历元（从1开始）
            prn_data = prn_data.copy()
            prn_data['epoch_num'] = range(1, len(prn_data) + 1)
            
            # 创建双y轴
            ax2 = ax.twinx()
            
            # 绘制伪距残差随历元变化 - 显示所有数据点，使用更小的点
            ax.scatter(prn_data['epoch_num'], prn_data['residual'], 
                      alpha=0.4, s=6, color='red', label='伪距残差')
            ax.set_xlabel('历元')
            ax.set_ylabel('伪距残差（m）', color='red')
            ax.tick_params(axis='y', labelcolor='red')
            
            # 设置伪距残差的y轴范围
            residual_min, residual_max = prn_data['residual'].min(), prn_data['residual'].max()
            residual_range = residual_max - residual_min
            if residual_range > 0:
                # 添加30%的边距
                margin = residual_range * 0.3
                ax.set_ylim(residual_min - margin, residual_max + margin)
            
            # 绘制高度角随历元变化 - 使用散点
            ax2.scatter(prn_data['epoch_num'], prn_data['ele'], 
                       alpha=0.6, s=6, color='blue', label='高度角')
            ax2.set_ylabel('高度角（°）', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            # 设置高度角的y轴范围
            ele_min, ele_max = prn_data['ele'].min(), prn_data['ele'].max()
            ele_range = ele_max - ele_min
            if ele_range > 0:
                # 添加30%的边距
                margin = ele_range * 0.3
                ax2.set_ylim(max(0, ele_min - margin), min(90, ele_max + margin))
            else:
                ax2.set_ylim(0, 90)
            
            # 如果有信噪比数据，也绘制信噪比 - 使用散点
            if 'snr' in prn_data.columns and not prn_data['snr'].isna().all():
                ax3 = ax.twinx()
                ax3.spines['right'].set_position(('outward', 60))
                ax3.scatter(prn_data['epoch_num'], prn_data['snr'], 
                           alpha=0.6, s=6, color='green', label='信噪比')
                ax3.set_ylabel('信噪比（dB-Hz）', color='green')
                ax3.tick_params(axis='y', labelcolor='green')
                
                # 设置信噪比的y轴范围
                snr_min, snr_max = prn_data['snr'].min(), prn_data['snr'].max()
                snr_range = snr_max - snr_min
                if snr_range > 0:
                    # 添加30%的边距
                    margin = snr_range * 0.3
                    ax3.set_ylim(max(0, snr_min - margin), snr_max + margin)
                else:
                    ax3.set_ylim(0, 50)
            
            ax.set_title(f'PRN {prn} - 频率 {freq}')
            ax.grid(True, alpha=0.3)
            
            # 优化x轴刻度显示
            max_epoch = len(prn_data)
            if max_epoch > 20:
                step = max(1, max_epoch // 10)  # 最多显示10个刻度
                ax.set_xticks(range(1, max_epoch + 1, step))
            else:
                ax.set_xticks(range(1, max_epoch + 1))
    
    # 隐藏多余的子图
    for i in range(n_prns, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    individual_combined_file = os.path.join(output_dir, f'individual_combined_{system}_{freq}.png')
    fig.savefig(individual_combined_file, dpi=300, bbox_inches='tight')
    plt.close(fig)




def visualize_epoch_residuals(subset, system, freq, output_dir):
    """绘制各卫星各频率伪距残差随历元变化图"""
    # 按PRN分组
    prns = subset['prn'].unique()
    
    if len(prns) == 0:
        return
    
    # 计算子图布局
    n_prns = len(prns)
    n_cols = min(3, n_prns)  # 最多3列
    n_rows = (n_prns + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # 确保axes是二维数组
    if n_prns == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = np.array(axes)
    
    for i, prn in enumerate(prns):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        prn_data = subset[subset['prn'] == prn].sort_values('epoch')
        
        if len(prn_data) > 0:
            # 创建历元（从1开始）
            prn_data = prn_data.copy()
            prn_data['epoch_num'] = range(1, len(prn_data) + 1)
            
            ax.scatter(prn_data['epoch_num'], prn_data['residual'], alpha=0.7, s=15, color='red')
            ax.set_title(f'PRN {prn} - 频率 {freq}')
            ax.set_xlabel('历元')
            ax.set_ylabel('伪距残差 (m)')
            ax.grid(True, alpha=0.3)
            
            # 优化x轴刻度显示
            max_epoch = len(prn_data)
            if max_epoch > 20:
                step = max(1, max_epoch // 10)  # 最多显示10个刻度
                ax.set_xticks(range(1, max_epoch + 1, step))
            else:
                ax.set_xticks(range(1, max_epoch + 1))
    
    # 隐藏多余的子图
    for i in range(n_prns, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片 - 文件名表示这是某个卫星系统-频率组合下的所有卫星
    epoch_file = os.path.join(output_dir, f'epoch_residuals_{system}_{freq}.png')
    fig.savefig(epoch_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_global_residuals(df, output_dir):
    """绘制手机所有伪距残差的全局分析图"""
    if len(df) == 0:
        return
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('手机伪距残差全局分析', fontsize=16, fontweight='bold')
    
    # 1. 伪距残差随时间变化图
    ax1 = axes[0, 0]
    
    # 按卫星-频率组合分组，为每个组合分配不同颜色
    df_sorted = df.sort_values('epoch')
    df_sorted = df_sorted.copy()
    
    # 创建卫星-频率组合标识
    df_sorted['prn_freq'] = df_sorted['prn'] + ' ' + df_sorted['frequency']
    prn_freq_combinations = df_sorted['prn_freq'].unique()
    
    # 为每个组合分配颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(prn_freq_combinations)))
    prn_freq_color_map = dict(zip(prn_freq_combinations, colors))
    
    # 创建时间轴（从第一个历元开始，以分钟:秒格式显示）
    time_minutes = []
    for epoch in df_sorted['epoch']:
        if hasattr(epoch, 'minute') and hasattr(epoch, 'second'):
            time_str = f"{epoch.minute:02d}:{epoch.second:02d}"
        else:
            # 如果是字符串或其他格式，尝试解析
            try:
                if isinstance(epoch, str):
                    from datetime import datetime
                    dt = datetime.strptime(epoch, '%Y-%m-%d %H:%M:%S')
                    time_str = f"{dt.minute:02d}:{dt.second:02d}"
                else:
                    time_str = str(epoch)
            except:
                time_str = str(epoch)
        time_minutes.append(time_str)
    
    df_sorted['time_display'] = time_minutes
    
    # 为每个卫星-频率组合绘制散点图
    for prn_freq in prn_freq_combinations:
        combo_data = df_sorted[df_sorted['prn_freq'] == prn_freq]
        if len(combo_data) > 0:
            ax1.scatter(combo_data['time_display'], combo_data['residual'], 
                       alpha=0.6, s=8, color=prn_freq_color_map[prn_freq])
    
    ax1.set_xlabel('时间 (分:秒)')
    ax1.set_ylabel('伪距残差 (m)')
    ax1.set_title('伪距残差随时间变化')
    ax1.grid(True, alpha=0.3)
    
    # 设置纵坐标范围正负平均
    residual_max = np.max(np.abs(df_sorted['residual']))
    ax1.set_ylim(-residual_max, residual_max)
    
    # 优化x轴刻度显示
    unique_times = df_sorted['time_display'].unique()
    if len(unique_times) > 10:
        step = max(1, len(unique_times) // 10)
        selected_times = unique_times[::step]
        ax1.set_xticks(selected_times)
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. 伪距残差分布直方图
    ax2 = axes[0, 1]
    
    residuals = df['residual'].values
    ax2.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax2.set_xlabel('伪距残差 (m)')
    ax2.set_ylabel('密度')
    ax2.set_title('伪距残差分布')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax2.axvline(mean_residual, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_residual:.2f}m')
    ax2.axvline(mean_residual + std_residual, color='orange', linestyle='--', linewidth=2, label=f'±1σ: {std_residual:.2f}m')
    ax2.axvline(mean_residual - std_residual, color='orange', linestyle='--', linewidth=2)
    ax2.legend()
    
    # 设置横坐标范围正负平均
    residual_max = np.max(np.abs(residuals))
    ax2.set_xlim(-residual_max, residual_max)
    
    # 3. 伪距残差与高度角关系
    ax3 = axes[1, 0]
    
    ax3.scatter(df['ele'], df['residual'], alpha=0.6, s=8, color='red')
    ax3.set_xlabel('高度角 (°)')
    ax3.set_ylabel('伪距残差 (m)')
    ax3.set_title('伪距残差与高度角关系')
    ax3.grid(True, alpha=0.3)
    
    # 设置纵坐标范围正负平均
    residual_max = np.max(np.abs(df['residual']))
    ax3.set_ylim(-residual_max, residual_max)
    
    # 4. 伪距残差与信噪比关系
    ax4 = axes[1, 1]
    
    ax4.scatter(df['snr'], df['residual'], alpha=0.6, s=8, color='red')
    ax4.set_xlabel('信噪比 (dB-Hz)')
    ax4.set_ylabel('伪距残差 (m)')
    ax4.set_title('伪距残差与信噪比关系')
    ax4.grid(True, alpha=0.3)
    
    # 设置纵坐标范围正负平均
    residual_max = np.max(np.abs(df['residual']))
    ax4.set_ylim(-residual_max, residual_max)
    
    plt.tight_layout()
    
    # 保存图片
    global_file = os.path.join(output_dir, 'global_residuals_analysis.png')
    fig.savefig(global_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def fit_and_visualize(subset, system, freq, output_dir):
    y = subset['residual'].values
    x_elev = subset['elevation_rad'].values
    x_snr = subset['snr'].values

    # 模型拟合
    a_elev, b_elev = fit_elevation_model(x_elev, y)
    a_snr, b_snr = fit_snr_model(x_snr, y)
    a_linear_snr, b_linear_snr = fit_linear_snr_model(x_snr, y)
    a_comb, b_comb, c_comb = fit_combined_model((x_elev, x_snr), y)

    # 保存结果
    result = {
        'system': system,
        'frequency': freq,
        'elevation_model': (a_elev, b_elev),
        'snr_model': (a_snr, b_snr),
        'linear_snr_model': (a_linear_snr, b_linear_snr),
        'combined_model': (a_comb, b_comb, c_comb),
        'data_count': len(subset)
    }

    # 生成模型信息文本
    model_info = generate_model_info(system, freq, len(subset),
                                     (a_elev, b_elev), (a_snr, b_snr), (a_linear_snr, b_linear_snr),
                                     (a_comb, b_comb, c_comb))
    model_file = os.path.join(output_dir, f'model_{system}_{freq}.txt')
    with open(model_file, 'w', encoding='utf-8') as f:
        f.write(model_info)

    # 可视化拟合效果
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'卫星系统: {system}, 频率: {freq}, 数据量: {len(subset)}')

    visualize_elevation_model(axs[0], subset, a_elev, b_elev, system, freq, len(subset))
    visualize_snr_model(axs[1], subset, a_snr, b_snr, system, freq, len(subset))
    visualize_linear_snr_model(axs[2], subset, a_linear_snr, b_linear_snr, system, freq, len(subset))
    visualize_combined_model(axs[3], subset, a_comb, b_comb, c_comb, system, freq, len(subset))

    plt.tight_layout()

    image_file = os.path.join(output_dir, f'fitting_{system}_{freq}.png')
    fig.savefig(image_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 绘制各卫星各频率伪距残差随历元变化图
    visualize_epoch_residuals(subset, system, freq, output_dir)
    
    # 绘制每个卫星每个频率的单独联合模型图
    visualize_individual_combined_models(subset, system, freq, output_dir)

    return result


def main():
    # 输入文件路径
    input_file = 'results/K70128H/pseudorange_residuals.csv'

    # 创建输出文件夹
    output_dir = os.path.join(os.path.dirname(input_file), 'Weighting')
    os.makedirs(output_dir, exist_ok=True)

    # 配置 matplotlib
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 读取数据
    df = pd.read_csv(input_file, parse_dates=['epoch'])
    df['prn'] = df['prn'].astype(str)  # 确保PRN列为字符串类型
    df['system'] = df['prn'].str[0]  # 提取卫星系统信息

    # 数据预处理
    df = df.dropna()
    df = df[df['residual'] != 0]
    df['ele'] = pd.to_numeric(df['ele'], errors='coerce')
    df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
    df['residual'] = pd.to_numeric(df['residual'], errors='coerce')
    df = df.dropna()
    df['elevation_rad'] = np.radians(df['ele'])

    # 绘制全局伪距残差分析图
    print("开始绘制全局伪距残差分析图...")
    visualize_global_residuals(df, output_dir)

    # 按卫星系统和频率分组拟合
    systems = df['system'].unique()
    frequencies = df['frequency'].unique()
    results = []
    print("开始分组参数拟合...")
    for system in systems:
        for freq in frequencies:
            subset = df[(df['system'] == system) & (df['frequency'] == freq)]
            if len(subset) < 10:  # 数据量太少则不拟合
                continue
            result = fit_and_visualize(subset, system, freq, output_dir)
            results.append(result)

    # 所有卫星系统和频率一起进行全局拟合
    global_subset = df.copy()
    if len(global_subset) >= 50:  # 确保数据量足够
        print("开始全局参数拟合...")
        result = fit_and_visualize(global_subset, 'ALL', 'ALL', output_dir)
        result['system'] = 'ALL'
        result['frequency'] = 'ALL'
        results.append(result)
    else:
        print("数据量不足，无法进行全局拟合", file=sys.stderr)

    # 保存所有拟合结果到CSV
    if results:
        results_df = pd.DataFrame(results)
        results_file = os.path.join(output_dir, 'fitting_results.csv')
        results_df.to_csv(results_file, index=False)

    print(f"所有拟合已完成，结果已保存到 {output_dir}")


if __name__ == "__main__":
    main()
