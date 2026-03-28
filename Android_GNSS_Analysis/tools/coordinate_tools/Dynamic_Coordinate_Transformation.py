import math
import re
import os
import tkinter as tk
from tkinter import filedialog

# =================================================================
# 1. 常量与参数配置 (WGS84 椭球体参数)
# =================================================================
A_EARTH = 6378137.0  # 长半轴 (单位: m)
F_EARTH = 1.0 / 298.257223563  # 扁率
E2_EARTH = 2 * F_EARTH - F_EARTH ** 2  # 第一偏心率平方

# 【关键配置】杆臂值 (Lever Arm): 手机相对于 RTK 中心的偏移量
# 坐标系定义: Y轴向前(行进方向), X轴向右, Z轴向上
LEVER_ARMS = {
    'Phone_K_Left': {'dx': -0.075, 'dy': 0.0, 'dz': 0.09},
    'Phone_F_Front': {'dx': 0.0, 'dy': 0.075, 'dz': 0.09},
    'Phone_A_Right': {'dx': 0.075, 'dy': 0.0, 'dz': 0.09}
}


# =================================================================
# 2. 核心数学转换函数
# =================================================================

def dms_to_decimal(deg, min, sec):
    """将度分秒(DMS)转换为十进制度(DD)"""
    return float(deg) + float(min) / 60.0 + float(sec) / 3600.0


def parse_line(line):
    """解析原始字符串: Pt号, 纬度(DMS), 经度(DMS), 高程"""
    pattern = r"(Pt\d+),(\d+)°(\d+)′([\d.]+)″,(\d+)°(\d+)′([\d.]+)″,([\d.]+)"
    match = re.search(pattern, line)
    if not match: return None
    return {
        'id': match.group(1),
        'lat': dms_to_decimal(match.group(2), match.group(3), match.group(4)),
        'lon': dms_to_decimal(match.group(5), match.group(6), match.group(7)),
        'alt': float(match.group(8))
    }


def lla_to_ecef(lat_deg, lon_deg, alt):
    """大地坐标 (LLA) 转 地心直角坐标 (ECEF XYZ)"""
    lat_rad, lon_rad = math.radians(lat_deg), math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
    N = A_EARTH / math.sqrt(1 - E2_EARTH * sin_lat ** 2)
    X = (N + alt) * cos_lat * cos_lon
    Y = (N + alt) * cos_lat * sin_lon
    Z = (N * (1 - E2_EARTH) + alt) * sin_lat
    return X, Y, Z


def ecef_delta_to_enu(X, Y, Z, dX, dY, dZ):
    """将 ECEF 下的增量矢量投影到站心坐标系(ENU)以计算方位角"""
    p = math.sqrt(X ** 2 + Y ** 2)
    lon_rad = math.atan2(Y, X)
    lat_rad = math.atan2(Z, p * (1 - E2_EARTH))
    slat, clat = math.sin(lat_rad), math.cos(lat_rad)
    slon, clon = math.sin(lon_rad), math.cos(lon_rad)
    dE = -slon * dX + clon * dY
    dN = -slat * clon * dX - slat * slon * dY + clat * dZ
    return dE, dN


def apply_lever_arm(base_X, base_Y, base_Z, lat_deg, lon_deg, heading_rad, offset):
    """根据航向角和杆臂值推算偏置后的 ECEF 坐标"""
    dx, dy, dz = offset['dx'], offset['dy'], offset['dz']
    # 机体转 ENU
    dE = dx * math.cos(heading_rad) + dy * math.sin(heading_rad)
    dN = -dx * math.sin(heading_rad) + dy * math.cos(heading_rad)
    # ENU 转 ECEF
    lat_rad, lon_rad = math.radians(lat_deg), math.radians(lon_deg)
    slat, clat = math.sin(lat_rad), math.cos(lat_rad)
    slon, clon = math.sin(lon_rad), math.cos(lon_rad)
    dX = -slon * dE - slat * clon * dN + clat * clon * dz
    dY = clon * dE - slat * slon * dN + clat * slon * dz
    dZ = clat * dN + slat * dz
    return base_X + dX, base_Y + dY, base_Z + dZ


# =================================================================
# 3. 主程序逻辑
# =================================================================

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="选择 RTK 数据文件", filetypes=[("Data Files", "*.dat *.txt")])
    if not file_path: return

    all_points = []
    # 尝试不同编码读取文件
    for enc in ['gbk', 'utf-8']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                for line in f:
                    res = parse_line(line)
                    if res: all_points.append(res)
            if all_points: break
        except UnicodeDecodeError:
            continue

    if not all_points:
        print("错误: 未能解析到有效数据。")
        return

    # 1. 交互选择区间
    print(f"\n[文件读取成功] 共有 {len(all_points)} 个历元。")
    try:
        start_idx = int(input(f"请输入起始索引 (0~{len(all_points) - 1}, 默认0): ") or 0)
        end_idx = int(input(f"请输入结束索引 ({start_idx}~{len(all_points) - 1}, 默认末尾): ") or len(all_points) - 1)
    except ValueError:
        start_idx, end_idx = 0, len(all_points) - 1

    # 截取选定区间
    selected = all_points[start_idx: end_idx + 1]

    # 2. 【修复关键点】预先计算所有选定点的 ECEF 坐标
    # 这样在后续计算方位角引用前后点时，'X','Y','Z' 键才一定存在
    for pt in selected:
        pt['X'], pt['Y'], pt['Z'] = lla_to_ecef(pt['lat'], pt['lon'], pt['alt'])

    # 3. 创建输出目录
    output_dir = os.path.join(os.path.dirname(file_path), "Phone_coordinates")
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 4. 初始化输出文件
    handlers = {name: open(os.path.join(output_dir, f"{name}_TruePos.dat"), 'w', encoding='gbk') for name in LEVER_ARMS}

    print("-> 正在解算轨迹...")
    for i in range(len(selected)):
        pt = selected[i]

        # 5. 计算航向 (Heading)
        # 使用中心差分或端点差分推算运动方向
        if i == 0 and len(selected) > 1:
            p1, p2 = selected[i], selected[i + 1]  # 起始点
        elif i == len(selected) - 1 and len(selected) > 1:
            p1, p2 = selected[i - 1], selected[i]  # 结束点
        elif len(selected) > 2:
            p1, p2 = selected[i - 1], selected[i + 1]  # 中间点使用前后邻点
        else:
            p1 = p2 = pt

        dX, dY, dZ = p2['X'] - p1['X'], p2['Y'] - p1['Y'], p2['Z'] - p1['Z']

        if dX == 0 and dY == 0:
            heading = 0.0  # 静止状态
        else:
            dE, dN = ecef_delta_to_enu(pt['X'], pt['Y'], pt['Z'], dX, dY, dZ)
            heading = math.atan2(dE, dN)

        # 6. 应用杆臂并保存
        for name, offset in LEVER_ARMS.items():
            tx, ty, tz = apply_lever_arm(pt['X'], pt['Y'], pt['Z'], pt['lat'], pt['lon'], heading, offset)
            handlers[name].write(f"{pt['id']},{tx:.4f},{ty:.4f},{tz:.4f}\n")

    for h in handlers.values(): h.close()
    print(f"成功！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()