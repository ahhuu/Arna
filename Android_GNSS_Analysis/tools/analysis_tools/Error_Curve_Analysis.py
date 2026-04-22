import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

# ================= 1. 坐标转换工具 =================
A_EARTH = 6378137.0
F_EARTH = 1.0 / 298.257223563
E2_EARTH = 2 * F_EARTH - F_EARTH ** 2


def ecef_to_lla(x, y, z):
    p = math.sqrt(x ** 2 + y ** 2)
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1 - E2_EARTH))
    return math.degrees(lat), math.degrees(lon)


def ecef_to_enu(ref_lat, ref_lon, dx, dy, dz):
    lat_rad = math.radians(ref_lat)
    lon_rad = math.radians(ref_lon)

    slat, clat = math.sin(lat_rad), math.cos(lat_rad)
    slon, clon = math.sin(lon_rad), math.cos(lon_rad)

    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    u = clat * clon * dx + clat * slon * dy + slat * dz
    return e, n, u


# ================= 2. 数据加载器 =================
def load_ppp_data(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == '.pos':
        df = pd.read_csv(path, sep=r'\s+', comment='%', header=None)

        def time_to_sow(d_str, t_str):
            dt = datetime.strptime(f"{d_str} {t_str}", "%Y/%m/%d %H:%M:%S.%f")
            return (dt - datetime(1980, 1, 6)).total_seconds() % 604800

        df['sow'] = df.apply(lambda r: time_to_sow(r[0], r[1]), axis=1)

        return df[['sow', 2, 3, 4]].rename(
            columns={2: 'x', 3: 'y', 4: 'z'}
        )
    else:
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None)

        return df[[2, 3, 4, 5]].rename(
            columns={2: 'sow', 3: 'x', 4: 'y', 5: 'z'}
        )


# ================= 3. 主程序 =================
def main():
    root = tk.Tk()
    root.withdraw()

    true_path = filedialog.askopenfilename(
        title="选择真值文件",
        filetypes=[("Text Files", "*.txt")]
    )
    ppp_path = filedialog.askopenfilename(
        title="选择 PPP 结果",
        filetypes=[("GNSS Results", "*.pos *.txt")]
    )

    if not true_path or not ppp_path:
        return

    # ===== 读取数据 =====
    df_true = pd.read_csv(
        true_path,
        sep=r'\s+',
        header=None,
        names=['sow', 'xt', 'yt', 'zt']
    )

    df_ppp = load_ppp_data(ppp_path)

    # ===== 排序（最近邻匹配必须）=====
    df_true = df_true.sort_values('sow').reset_index(drop=True)
    df_ppp = df_ppp.sort_values('sow').reset_index(drop=True)

    print("\n--- 时间轴诊断 ---")
    print(f"真值范围: {df_true['sow'].min():.2f} - {df_true['sow'].max():.2f}")
    print(f"PPP范围: {df_ppp['sow'].min():.2f} - {df_ppp['sow'].max():.2f}")

    # ===== 时间补偿（可选）=====
    time_offset = 0
    df_ppp['sow'] = df_ppp['sow'] + time_offset

    # ===== 限制PPP在真值时间范围内 =====
    mask = (df_ppp['sow'] >= df_true['sow'].min()) & (df_ppp['sow'] <= df_true['sow'].max())
    df_ppp = df_ppp[mask].reset_index(drop=True)

    if df_ppp.empty:
        print("[错误] PPP时间不在真值范围内，无法匹配")
        return

    # ===== 最近邻匹配（核心）=====
    true_sow = df_true['sow'].values
    true_x = df_true['xt'].values
    true_y = df_true['yt'].values
    true_z = df_true['zt'].values

    ppp_sow = df_ppp['sow'].values

    # 使用 searchsorted 查找每个 ppp_sow 在 true_sow 中的插入位置
    idx = np.searchsorted(true_sow, ppp_sow, side='left')
    # 避免越界：若插入位置为0，只能选第一个；若为len，只能选最后一个
    idx = np.clip(idx, 1, len(true_sow) - 1)

    # 计算与左侧邻居的时间差
    left_diff = np.abs(ppp_sow - true_sow[idx - 1])
    # 计算与右侧邻居的时间差
    right_diff = np.abs(true_sow[idx] - ppp_sow)

    # 选择时间差较小的邻居索引
    use_left = left_diff <= right_diff
    matched_idx = np.where(use_left, idx - 1, idx)

    # 提取匹配的真值坐标
    xt_match = true_x[matched_idx]
    yt_match = true_y[matched_idx]
    zt_match = true_z[matched_idx]

    # 计算时间偏差（用于诊断）
    time_diff = ppp_sow - true_sow[matched_idx]
    mean_abs_diff = np.mean(np.abs(time_diff))
    max_abs_diff = np.max(np.abs(time_diff))
    print(f"\n最近邻匹配时间偏差统计:")
    print(f"  平均绝对偏差: {mean_abs_diff:.3f} 秒")
    print(f"  最大绝对偏差: {max_abs_diff:.3f} 秒")

    merged = pd.DataFrame({
        'sow': ppp_sow,
        'x': df_ppp['x'].values,
        'y': df_ppp['y'].values,
        'z': df_ppp['z'].values,
        'xt': xt_match,
        'yt': yt_match,
        'zt': zt_match,
        'time_diff': time_diff          # 保留时间偏差，便于后续分析
    })

    # ===== ENU转换 =====
    ref_x = merged['xt'].iloc[0]
    ref_y = merged['yt'].iloc[0]
    ref_z = merged['zt'].iloc[0]

    ref_lat, ref_lon = ecef_to_lla(ref_x, ref_y, ref_z)

    true_enu = []
    ppp_enu = []

    for r in merged.itertuples():
        true_enu.append(
            ecef_to_enu(ref_lat, ref_lon,
                        r.xt - ref_x, r.yt - ref_y, r.zt - ref_z)
        )
        ppp_enu.append(
            ecef_to_enu(ref_lat, ref_lon,
                        r.x - ref_x, r.y - ref_y, r.z - ref_z)
        )

    merged[['Et', 'Nt', 'Ut']] = pd.DataFrame(true_enu, index=merged.index)
    merged[['Ep', 'Np', 'Up']] = pd.DataFrame(ppp_enu, index=merged.index)

    # ===== 误差（cm）=====
    merged['dE'] = (merged['Ep'] - merged['Et']) * 100
    merged['dN'] = (merged['Np'] - merged['Nt']) * 100
    merged['dU'] = (merged['Up'] - merged['Ut']) * 100

    # ===== 绘图 =====
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(15, 10))

    ax_traj = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    ax_traj.plot(merged['Et'], merged['Nt'], 'k-', label='真值')
    ax_traj.plot(merged['Ep'], merged['Np'], 'r--', label='PPP')
    ax_traj.set_title(f"轨迹对比")
    ax_traj.axis('equal')
    ax_traj.legend()

    for i, col in enumerate(['dE', 'dN', 'dU']):
        ax = plt.subplot2grid((3, 2), (i, 1))
        rms = np.sqrt(np.mean(merged[col] ** 2))
        ax.plot(merged['sow'], merged[col])
        ax.set_title(f"{col} RMS: {rms:.2f} cm")

    plt.tight_layout()

    # ===== 自动保存 =====
    true_dir = os.path.dirname(true_path)
    base_name = os.path.splitext(os.path.basename(true_path))[0]
    save_name = base_name.replace("TruePos", "Dynamicresult") + ".png"
    save_path = os.path.join(true_dir, save_name)

    plt.savefig(save_path, dpi=300)
    print(f"\n图片已保存至: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()