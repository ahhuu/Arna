import tkinter as tk
from tkinter import filedialog
import math
import os


def dms_to_degrees(degrees, minutes, seconds):
    """将度分秒格式转换为十进制度格式"""
    return degrees + minutes / 60 + seconds / 3600


def degrees_to_dms(degrees):
    """将十进制度格式转换为度分秒格式"""
    d = int(degrees)
    md = abs(degrees - d) * 60
    m = int(md)
    s = (md - m) * 60
    return d, m, s


def deg_to_xyz(lat_deg, lon_deg, height):
    """
    将经纬度(度)和大地高转换为WGS84坐标系下的XYZ坐标
    参数:
        lat_deg: 纬度(度)
        lon_deg: 经度(度)
        height: 大地高(米)
    返回:
        X, Y, Z: WGS84坐标系下的XYZ坐标(米)
    """
    # WGS84椭球体参数
    a = 6378137.0  # 长半轴(米)
    f = 1 / 298.257223563  # 扁率
    e_sq = 2 * f - f ** 2  # 第一偏心率的平方

    # 转换为弧度
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    # 计算卯酉圈曲率半径
    N = a / math.sqrt(1 - e_sq * math.sin(lat_rad) ** 2)

    # 计算XYZ坐标
    X = (N + height) * math.cos(lat_rad) * math.cos(lon_rad)
    Y = (N + height) * math.cos(lat_rad) * math.sin(lon_rad)
    Z = (N * (1 - e_sq) + height) * math.sin(lat_rad)

    return X, Y, Z


def convert_coordinates(lat_dms, lon_dms, height):
    """
    转换经纬度和大地高
    返回:
        lat_deg: 纬度(度)
        lon_deg: 经度(度)
        height: 大地高(米)
        X, Y, Z: WGS84坐标系下的XYZ坐标(米)
    """
    lat_deg = dms_to_degrees(*lat_dms)
    lon_deg = dms_to_degrees(*lon_dms)
    X, Y, Z = deg_to_xyz(lat_deg, lon_deg, height)
    return lat_deg, lon_deg, height, X, Y, Z


def average_coordinates(data):
    """
    计算多次测量数据转换后的平均值
    返回:
        avg_lat: 平均纬度(度)
        avg_lon: 平均经度(度)
        avg_height: 平均大地高(米)
        avg_X, avg_Y, avg_Z: 平均XYZ坐标(米)
    """
    total_lat = 0.0
    total_lon = 0.0
    total_height = 0.0
    total_X = 0.0
    total_Y = 0.0
    total_Z = 0.0
    num_measurements = len(data)

    for lat_dms, lon_dms, height in data:
        lat_deg, lon_deg, height, X, Y, Z = convert_coordinates(lat_dms, lon_dms, height)
        total_lat += lat_deg
        total_lon += lon_deg
        total_height += height
        total_X += X
        total_Y += Y
        total_Z += Z

    if num_measurements == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    avg_lat = total_lat / num_measurements
    avg_lon = total_lon / num_measurements
    avg_height = total_height / num_measurements
    avg_X = total_X / num_measurements
    avg_Y = total_Y / num_measurements
    avg_Z = total_Z / num_measurements

    return avg_lat, avg_lon, avg_height, avg_X, avg_Y, avg_Z


def read_data_from_file(file_path):
    """
    从文件读取测量数据
    支持的格式: 点号,纬度(度分秒),经度(度分秒),大地高
    返回:
        数据列表，每个元素为 (纬度DMS元组, 经度DMS元组, 大地高)
    """
    data = []
    encodings = ['gbk', 'utf-8']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                print(f"成功以 {encoding} 编码打开文件: {file_path}")
                for i, line in enumerate(file, 1):
                    line = line.strip().rstrip(',')
                    print(f"第{i}行数据: {line}")
                    if line:
                        parts = line.split(',')
                        if len(parts) == 4:
                            try:
                                lat_parts = list(
                                    map(float, parts[1].replace('°', ' ').replace('′', ' ').replace('″', ' ').split()))
                                lon_parts = list(
                                    map(float, parts[2].replace('°', ' ').replace('′', ' ').replace('″', ' ').split()))
                                height = float(parts[3])
                                if len(lat_parts) == 3 and len(lon_parts) == 3:
                                    data.append((tuple(lat_parts), tuple(lon_parts), height))
                            except ValueError as e:
                                print(f"数据格式错误-文件:{file_path}, 行:{i}, 错误信息: {e}")
            break
        except UnicodeDecodeError:
            print(f"无法以 {encoding} 编码打开文件: {file_path}, 尝试其他编码...")
        except Exception as e:
            print(f"读取文件时出现错误-文件:{file_path}, 错误信息: {e}")
    return data


def save_results(file_path, avg_lat, avg_lon, avg_height, avg_X, avg_Y, avg_Z):
    """保存计算结果到文件"""
    try:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)
        output_file_name = f"Avg-{base_name}{ext}"
        output_file_path = os.path.join(file_dir, output_file_name)

        lat_dms = degrees_to_dms(avg_lat)
        lon_dms = degrees_to_dms(avg_lon)

        with open(output_file_path, 'w', encoding='gbk') as file:
            file.write(f"Avg-WGS84-{base_name}\n")
            file.write(f"---------------------------------------------\n")
            file.write(f"Lat(dms)：{lat_dms[0]}°{lat_dms[1]}′{lat_dms[2]:.6f}″\n")
            file.write(f"Lon(dms)：{lon_dms[0]}°{lon_dms[1]}′{lon_dms[2]:.6f}″\n")
            file.write(f"Height(m)：{avg_height:.4f}\n")
            file.write(f"---------------------------------------------\n")
            file.write(f"Lat(deg)：{avg_lat:.6f}\n")
            file.write(f"Lon(deg)：{avg_lon:.6f}\n")
            file.write(f"Height(m)：{avg_height:.4f}\n")
            file.write(f"---------------------------------------------\n")
            file.write(f"X(ECEF/m)：{avg_X:.6f}\n")
            file.write(f"Y(ECEF/m)：{avg_Y:.6f}\n")
            file.write(f"Z(ECEF/m)：{avg_Z:.6f}\n")

        print(f"结果已保存至: {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"保存结果时出现错误-文件:{file_path}, 错误信息: {e}")
        return None


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 允许选择多个文件
    file_paths = filedialog.askopenfilenames(title="选择数据文件",
                                             filetypes=(("文本文件", "*.dat *.txt"), ("所有文件", "*.*")))

    if file_paths:
        for file_path in file_paths:
            print(f"\n处理文件: {file_path}")
            measurements = read_data_from_file(file_path)
            if measurements:
                avg_lat, avg_lon, avg_height, avg_X, avg_Y, avg_Z = average_coordinates(measurements)

                # 转换为度分秒格式用于显示
                lat_dms = degrees_to_dms(avg_lat)
                lon_dms = degrees_to_dms(avg_lon)

                print("--------------------------------------------------------")
                print(f"文件: {file_path} 处理结果:")
                print(f"平均纬度(dms): {lat_dms[0]}°{lat_dms[1]}′{lat_dms[2]:.6f}″")
                print(f"平均经度(dms): {lon_dms[0]}°{lon_dms[1]}′{lon_dms[2]:.6f}″")
                print(f"平均纬度(deg): {avg_lat:.6f}°")
                print(f"平均经度(deg): {avg_lon:.6f}°")
                print(f"平均大地高: {avg_height:.4f}m")
                print(f"平均X坐标: {avg_X:.6f}m")
                print(f"平均Y坐标: {avg_Y:.6f}m")
                print(f"平均Z坐标: {avg_Z:.6f}m")

                save_results(file_path, avg_lat, avg_lon, avg_height, avg_X, avg_Y, avg_Z)
            else:
                print(f"文件: {file_path} 中未找到有效的数据")
    else:
        print("未选择任何文件")