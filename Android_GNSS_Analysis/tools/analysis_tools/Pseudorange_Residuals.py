import numpy as np
import pandas as pd
from datetime import datetime
import re
import time
import os
from pathlib import Path

# 1. 频率映射关系
FREQ_MAPPING = {
    'G': (['C1C', 'C5Q'], ['S1C', 'S5Q']),  # GPS
    'R': (['C1C'], ['S1C']),  # GLONASS
    'E': (['C1C', 'C5Q', 'C7Q'], ['S1C', 'S5Q', 'S7Q']),  # Galileo
    'C': (['C1P', 'C2I', 'C5P'], ['S1P', 'S2I', 'S5P'])  # BDS
}


# 新增的独立高度角计算函数
def calculate_elevation(receiver_xyz, satellite_xyz):
    """计算卫星高度角（单位：度）

    参数:
        receiver_xyz: 接收机的ECEF坐标 [X, Y, Z] (米)
        satellite_xyz: 卫星的ECEF坐标 [X, Y, Z] (米)

    返回:
        高度角（单位：度，0-90°）
    """
    # WGS84椭球参数
    a = 6378137.0  # 长半轴 (米)
    f = 1 / 298.257223563  # 扁率
    b = a * (1 - f)  # 短半轴

    # 计算接收机到卫星的向量
    vector = satellite_xyz - receiver_xyz

    # 将笛卡尔坐标转换为大地坐标
    x, y, z = receiver_xyz
    p = np.sqrt(x ** 2 + y ** 2)

    # 辅助计算
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + (a ** 2 - b ** 2) / b * np.sin(theta) ** 3,
                     p - (a ** 2 - b ** 2) / a * np.cos(theta) ** 3)
    lon = np.arctan2(y, x)

    # 计算东北天坐标系(ENU)的旋转矩阵
    slat = np.sin(lat)
    clat = np.cos(lat)
    slon = np.sin(lon)
    clon = np.cos(lon)

    R = np.array([
        [-slon, clon, 0],
        [-slat * clon, -slat * slon, clat],
        [clat * clon, clat * slon, slat]
    ])

    # 将卫星向量转换到ENU坐标系
    enu = R @ vector
    e, n, u = enu

    # 计算高度角并转换为度数
    elevation = np.arctan2(u, np.sqrt(e ** 2 + n ** 2))
    return np.degrees(elevation)


# 2. RINEX解析函数
def parse_rinex_header(lines):
    """严格基于SYS / # / OBS TYPES标记的RINEX表头解析"""
    header = {
        'obs_types': {},  # 系统: 观测类型列表（按顺序）
        'end_header': 0,
        'interval': 1.0
    }

    current_sys = None
    current_obs_count = 0
    obs_types = []

    for i, line in enumerate(lines):
        line = line.strip()
        if 'END OF HEADER' in line:
            header['end_header'] = i + 1
            break

        if 'INTERVAL' in line:
            header['interval'] = float(line.split()[0])

        # 查找SYS / # / OBS TYPES标记
        if 'SYS / # / OBS TYPES' in line:
            # 提取系统标识和观测类型数量
            parts = line.split()
            if len(parts) >= 3:
                current_sys = parts[0]
                try:
                    current_obs_count = int(parts[1])
                except (ValueError, IndexError):
                    current_obs_count = 0

                # 提取标记前的观测类型
                sys_idx = line.index('SYS')
                obs_part = line[:sys_idx].strip()
                obs_types = obs_part.split()[2:]  # 跳过系统和数量

                # 检查是否需要继续读取下一行
                if len(obs_types) < current_obs_count and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith(('G', 'R', 'C', 'E', 'S', 'J')):
                        # 处理续行（如L5Q D5Q S5Q行）
                        next_sys_idx = next_line.find('SYS') if 'SYS' in next_line else len(next_line)
                        next_obs_part = next_line[:next_sys_idx].strip()
                        obs_types.extend(next_obs_part.split())

            # 保存当前系统的观测类型
            if current_sys and obs_types:
                header['obs_types'][current_sys] = obs_types[:current_obs_count]
                obs_types = []

    # 清理无效系统标识
    valid_systems = ['G', 'R', 'C', 'E', 'S', 'J']
    header['obs_types'] = {k: v for k, v in header['obs_types'].items() if k in valid_systems}

    return header


def read_rinex_obs(file_path, is_mobile, debug_list):
    """读取RINEX数据"""
    with open(file_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]

    header = parse_rinex_header(lines)
    obs_data = {}
    device_type = "mobile" if is_mobile else "base"

    # # 打印表头信息用于调试
    # print(f"解析RINEX文件: {file_path}")
    # print(f"观测类型定义: {header['obs_types']}")

    # 时间秒数四舍五入
    for line in lines[header['end_header']:]:
        if not line:
            continue

        if line.startswith('>'):  # 历元行
            match = re.search(r'(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}\.\d+)', line)
            if match:
                year, month, day, hour, minute, second = match.groups()
                sec_part = int(second.split('.')[0])
                frac_part = float('0.' + second.split('.')[1])
                sec_rounded = sec_part + (1 if frac_part >= 0.5 else 0)
                if sec_rounded >= 60:
                    sec_rounded = 0
                    minute = str(int(minute) + 1)
                epoch = datetime.strptime(
                    f"{year}-{month}-{day} {hour}:{minute}:{sec_rounded:02d}",
                    '%Y-%m-%d %H:%M:%S'
                )
                obs_data[epoch] = {}
            continue

        # 卫星数据行解析
        prn = line[:3].strip()
        sys = prn[0]

        if sys not in FREQ_MAPPING or sys not in header['obs_types']:
            debug_list.append({
                'device': device_type,
                'epoch': epoch,
                'prn': prn,
                'error': f'卫星系统 {sys} 不在支持列表中或表头未定义观测类型'
            })
            continue

        obs_types = header['obs_types'][sys]
        pseudorange_types, snr_types = FREQ_MAPPING[sys]
        prn_data = {}

        # 动态选择解析
        if is_mobile:
            # 手机数据：固定宽度16字符解析
            obs_values = []
            for j in range(3, len(line), 16):  # PRN占3字符，之后每16字符一个观测值
                val_str = line[j:j + 16].strip()  # 取16字符宽度
                obs_values.append(val_str if val_str else None)
            data_values = obs_values
            values_per_obs = 1
        else:
            # 接收机数据：使用固定宽度字段解析，与Rinex_analysis.py保持一致
            sat_data = line[3:]  # 跳过PRN标识符
            field_width = 16  # 每个字段16字符
            expected_fields = len(obs_types)
            actual_fields = (len(sat_data) + field_width - 1) // field_width
            
            def read_field(k):
                """读取第k个字段的值（前14字符为数值）"""
                if k >= expected_fields or k >= actual_fields:
                    return None
                start = k * field_width
                end = start + field_width
                field = sat_data[start:end]
                value_str = field[:14].strip()  # 前14列为数值
                try:
                    return float(value_str) if value_str else None
                except ValueError:
                    return None

        for pr_type, snr_type in zip(pseudorange_types, snr_types):
            if pr_type not in obs_types or snr_type not in obs_types:
                continue

            if is_mobile:
                # 手机数据：使用原来的逻辑
                pr_idx = obs_types.index(pr_type)
                snr_idx = obs_types.index(snr_type)
                
                pr_data_pos = pr_idx
                snr_data_pos = snr_idx
                
                # 伪距提取
                pr_val = None
                if pr_data_pos < len(data_values):
                    pr_val_str = data_values[pr_data_pos]
                    if pr_val_str:
                        try:
                            pr_val = float(pr_val_str)
                        except ValueError:
                            debug_list.append({
                                'device': device_type,
                                'epoch': epoch,
                                'prn': prn,
                                'frequency': pr_type,
                                'error': f'无法解析伪距值: {pr_val_str}'
                            })

                # 信噪比提取
                snr_val = None
                if snr_data_pos < len(data_values):
                    snr_val_str = data_values[snr_data_pos]
                    if snr_val_str:
                        try:
                            snr_val = float(snr_val_str)
                        except ValueError:
                            debug_list.append({
                                'device': device_type,
                                'epoch': epoch,
                                'prn': prn,
                                'frequency': pr_type,
                                'error': f'无法解析SNR值: {snr_val_str}'
                            })
            else:
                # 接收机数据：使用固定宽度字段解析
                pr_idx = obs_types.index(pr_type)
                snr_idx = obs_types.index(snr_type)
                
                pr_val = read_field(pr_idx)
                snr_val = read_field(snr_idx)

            # 调试信息
            if pr_val is not None and snr_val is not None:
                if pr_val > 0 and 10 < snr_val < 60:
                    prn_data[pr_type] = {'pseudorange': pr_val, 'snr': snr_val}
                    debug_list.append({
                        'device': device_type,
                        'epoch': epoch,
                        'prn': prn,
                        'frequency': pr_type,
                        'pseudorange': pr_val,
                        'snr': snr_val,
                        'status': '有效数据'
                    })
                else:
                    debug_list.append({
                        'device': device_type,
                        'epoch': epoch,
                        'prn': prn,
                        'frequency': pr_type,
                        'pseudorange': pr_val,
                        'snr': snr_val,
                        'error': '伪距或SNR超出有效范围'
                    })
            else:
                error_msg = ('伪距和SNR均无效' if (pr_val is None and snr_val is None)
                             else '伪距无效' if pr_val is None else 'SNR无效')
                debug_list.append({
                    'device': device_type,
                    'epoch': epoch,
                    'prn': prn,
                    'frequency': pr_type,
                    'error': error_msg
                })

        if prn_data:
            obs_data[epoch][prn] = prn_data
        else:
            debug_list.append({
                'device': device_type,
                'epoch': epoch,
                'prn': prn,
                'error': '该卫星所有频率数据均无效'
            })

    return obs_data


# 3. 卫星位置读取
def read_satellite_pos(file_path, mobile_epochs, output_dir):
    """读取卫星位置，并将original_epoch映射到实际观测历元"""
    data = []
    debug_sat_list = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = re.split(r'[\t\s]+', line)
            if len(parts) >= 6:
                try:
                    epoch = int(parts[0])
                    prn = parts[1]
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])

                    # 转换PRN格式
                    def convert_prn(prn_str):
                        try:
                            prn_int = int(prn_str)
                            if 1 <= prn_int <= 100:
                                return f"G{prn_int:02d}"
                            elif 101 <= prn_int <= 199:
                                return f"R{prn_int - 100:02d}"
                            elif 201 <= prn_int <= 299:
                                return f"E{prn_int - 200:02d}"
                            elif 301 <= prn_int <= 399:
                                return f"C{prn_int - 300:02d}"
                            elif 401 <= prn_int <= 410:
                                return f"J{prn_int - 400:02d}"
                            return prn_str
                        except ValueError:
                            return prn_str

                    converted_prn = convert_prn(prn)

                    # 将original_epoch映射到实际历元时间
                    actual_epoch = None
                    if 1 <= epoch <= len(mobile_epochs):
                        actual_epoch = mobile_epochs[epoch - 1]

                    data.append({
                        'Epoch': epoch,
                        'ActualEpoch': actual_epoch,
                        'PRN': converted_prn,
                        'X': x,
                        'Y': y,
                        'Z': z
                    })

                    debug_sat_list.append({
                        'original_epoch': epoch,
                        'actual_epoch': actual_epoch,
                        'original_prn': prn,
                        'converted_prn': converted_prn,
                        'X': x,
                        'Y': y,
                        'Z': z
                    })
                except (ValueError, IndexError):
                    continue

    # 保存卫星位置调试信息到指定目录
    pd.DataFrame(debug_sat_list).to_csv(os.path.join(output_dir, 'debug_sat_pos.csv'), index=False)
    print(f"卫星位置调试数据已保存至 {os.path.join(output_dir, 'debug_sat_pos.csv')}（共{len(debug_sat_list)}条）")

    pos_df = pd.DataFrame(data)
    return pos_df


# 4. 伪距残差计算（严格历元匹配）
def calculate_single_difference(mobile_obs, base_obs, sat_pos_df, mobile_coords, base_coords):
    """计算单差残差，仅保留三方（手机、基准站、卫星位置）均存在的历元"""
    residuals = {}
    mobile_xyz = np.array(mobile_coords)
    base_xyz = np.array(base_coords)

    # 获取手机和基准站的历元集合
    mobile_epochs = set(sorted(mobile_obs.keys()))
    base_epochs = set(sorted(base_obs.keys()))

    # 计算两者共有的历元（精确匹配，精确到秒）
    common_epochs = mobile_epochs & base_epochs
    print(f"手机历元数: {len(mobile_epochs)}, 基准站历元数: {len(base_epochs)}, 共同历元数: {len(common_epochs)}")

    # 建立实际历元与卫星数据的映射（仅保留有卫星数据的历元）
    epoch_to_sat = {row['ActualEpoch']: row for _, row in sat_pos_df.iterrows()
                    if row['ActualEpoch'] is not None}
    sat_epochs = set(epoch_to_sat.keys())

    # 最终保留的历元：三方均存在
    valid_epochs = common_epochs & sat_epochs
    print(f"卫星位置有效历元数: {len(sat_epochs)}, 三方共同有效历元数: {len(valid_epochs)}")

    for epoch in sorted(valid_epochs):
        # 从字典中直接获取对应历元的数据（已确保存在）
        mobile_data = mobile_obs[epoch]
        base_data = base_obs[epoch]

        residuals[epoch] = {}
        # 遍历共同卫星
        common_prns = set(mobile_data.keys()) & set(base_data.keys())
        for prn in common_prns:
            sys = prn[0]
            if sys not in FREQ_MAPPING:
                continue

            # 遍历共同伪距类型
            mobile_types = set(mobile_data[prn].keys())
            base_types = set(base_data[prn].keys())
            common_types = mobile_types & base_types

            # 获取当前卫星的位置（精确匹配历元和PRN）
            sat_pos = sat_pos_df[(sat_pos_df['ActualEpoch'] == epoch) &
                                 (sat_pos_df['PRN'] == prn)]
            if sat_pos.empty:
                continue  # 跳过没有卫星位置的卫星
            sat_xyz = sat_pos[['X', 'Y', 'Z']].values[0]

            # 计算手机位置的高度角
            elevation = calculate_elevation(mobile_xyz, sat_xyz)

            for pr_type in common_types:
                # 提取伪距
                p_mobile = mobile_data[prn][pr_type]['pseudorange']
                p_base = base_data[prn][pr_type]['pseudorange']

                # 计算几何距离差
                rho_mobile = np.linalg.norm(sat_xyz - mobile_xyz)
                rho_base = np.linalg.norm(sat_xyz - base_xyz)
                delta_rho = rho_mobile - rho_base

                # 计算残差
                delta_p = p_mobile - p_base
                residual = delta_p - delta_rho

                # 保存所有残差（无过滤）
                residuals[epoch][f"{prn}_{pr_type}"] = {
                    'residual': residual,
                    'snr': mobile_data[prn][pr_type]['snr'],
                    'ele': elevation  # 新增：存储高度角
                }
    return residuals


# 5. 钟差估计
def remove_clock_bias(residuals):
    for epoch in residuals:
        valid_res = [residuals[epoch][key]['residual'] for key in residuals[epoch]
                     if not np.isnan(residuals[epoch][key]['residual'])]
        if len(valid_res) >= 3:
            clock_bias = np.mean(valid_res)
            for key in residuals[epoch]:
                residuals[epoch][key]['residual'] -= clock_bias
    return residuals


# 6. 主函数
def main(mobile_rinex, base_rinex, sat_pos_path, mobile_coords, base_coords, output_dir=None):
    start_time = time.time()

    # 1. 创建输出目录：results/手机RINEX文件名（不含扩展名）
    mobile_filename = os.path.basename(mobile_rinex)
    mobile_basename = os.path.splitext(mobile_filename)[0]  # 去除扩展名
    if output_dir is None:
        output_dir = os.path.join('results', mobile_basename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)  # 递归创建目录

    # 2. 初始化调试列表
    debug_obs_list = []

    # 3. 读取手机RINEX数据
    mobile_obs = read_rinex_obs(mobile_rinex, is_mobile=True, debug_list=debug_obs_list)

    # 4. 读取卫星位置数据（输出到指定目录）
    mobile_epochs_sorted = sorted(mobile_obs.keys())
    sat_pos = read_satellite_pos(sat_pos_path, mobile_epochs_sorted, output_dir)

    # 5. 读取基准站RINEX数据
    base_obs = read_rinex_obs(base_rinex, is_mobile=False, debug_list=debug_obs_list)

    # 6. 保存观测数据调试文件到指定目录
    debug_obs_df = pd.DataFrame(debug_obs_list)
    debug_obs_df.to_csv(os.path.join(output_dir, 'debug_obs_data.csv'), index=False)
    print(f"观测数据调试信息已保存至 {os.path.join(output_dir, 'debug_obs_data.csv')}（共{len(debug_obs_df)}条）")

    # 7. 计算残差（严格历元匹配）
    raw_res = calculate_single_difference(mobile_obs, base_obs, sat_pos, mobile_coords, base_coords)
    final_res = remove_clock_bias(raw_res)

    # 8. 保存结果到指定目录（修改排序逻辑）
    result = []
    for epoch, data in final_res.items():
        for key, val in data.items():
            prn, pr_type = key.split('_')
            result.append({
                'epoch': epoch,
                'prn': prn,
                'frequency': pr_type,
                'residual': val['residual'],
                'snr': val['snr'],
                'ele': val['ele']
            })

    # 新增：对结果按历元、系统和PRN号排序
    def prn_sort_key(item):
        prn = item.get('prn', '')
        if len(prn) >= 2:
            sys = prn[0]
            try:
                num = int(prn[1:])
            except ValueError:
                num = 0
            # 定义系统排序顺序：G(1), R(2), E(3), C(4)，其他(5)
            sys_order = {'G': 1, 'R': 2, 'E': 3, 'C': 4}
            return sys_order.get(sys, 5), num
        return 5, 0  # 默认排序值

    # 按历元排序，同一历元内按系统和PRN排序
    sorted_result = sorted(
        result,
        key=lambda x: (x.get('epoch', datetime.min), prn_sort_key(x))
    )

    # 使用排序后的结果保存CSV
    pd.DataFrame(sorted_result).to_csv(os.path.join(output_dir, 'pseudorange_residuals.csv'), index=False)
    print(
        f"处理完成，残差结果文件保存至 {os.path.join(output_dir, 'pseudorange_residuals.csv')}（共{len(sorted_result)}条）")

    print(f"总计算耗时: {time.time() - start_time:.2f}秒")


if __name__ == "__main__":
    MOBILE_RINEX = "data/K70128H.25o"
    BASE_RINEX = "data/p03000000_R_20251281514_000_01S.25O"
    SAT_POS = "data/satellite_positions-128-k7.txt"
    MOBILE_COORDS = [-1324698.041159006, 5323031.038016253, 3244602.006945656]
    BASE_COORDS = [-1324698.104573897, 5323031.050568524, 3244601.728187757]
    main(MOBILE_RINEX, BASE_RINEX, SAT_POS, MOBILE_COORDS, BASE_COORDS)
