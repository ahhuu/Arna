import math
import re
import os
from datetime import datetime, timezone, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox

# =================================================================
# 1. 常量与参数配置 (WGS84 椭球体参数)
# =================================================================
A_EARTH = 6378137.0  # 长半轴 (单位: m)
F_EARTH = 1.0 / 298.257223563  # 扁率
E2_EARTH = 2 * F_EARTH - F_EARTH ** 2  # 第一偏心率平方

# GPS 与 UTC 的闰秒差（截至当前通常为 18s）
GPS_UTC_LEAP_SECONDS = 18


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


def utc_to_gps_week_sow(utc_dt):
    """UTC datetime -> (GPS week, seconds of week)"""
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
    gps_dt = utc_dt + timedelta(seconds=GPS_UTC_LEAP_SECONDS)
    delta_sec = (gps_dt - gps_epoch).total_seconds()
    gps_week = int(delta_sec // 604800)
    sow = delta_sec - gps_week * 604800
    return gps_week, sow


def utc_to_gpst_str(utc_dt):
    """UTC datetime -> GPST格式字符串 (YYYY/MM/DD HH:MM:SS.sss)"""
    gpst_dt = utc_dt + timedelta(seconds=GPS_UTC_LEAP_SECONDS)
    return gpst_dt.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


# =================================================================
# 3. 集成式 GUI 应用程序类
# =================================================================

class RTKProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GNSS 动态坐标与杆臂解算工具")
        self.root.geometry("550x650")
        self.root.resizable(False, False)

        # 将窗口居中
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_reqwidth()) // 2
        y = (self.root.winfo_screenheight() - self.root.winfo_reqheight()) // 2
        self.root.geometry(f"+{x}+{y}")

        self.all_points = []
        self.file_path = ""
        self.lever_arm_entries = {}

        self.create_widgets()

    def create_widgets(self):
        # --- 1. 文件选择区 ---
        frame_file = tk.LabelFrame(self.root, text=" 1. 数据文件 ", padx=10, pady=10)
        frame_file.pack(fill="x", padx=15, pady=10)

        self.btn_select_file = tk.Button(frame_file, text="选择 RTK 数据文件", command=self.load_file, bg="#E8F0FE")
        self.btn_select_file.pack(side="left")

        self.lbl_file_info = tk.Label(frame_file, text="未选择文件", fg="gray")
        self.lbl_file_info.pack(side="left", padx=15)

        self.lbl_points_count = tk.Label(frame_file, text="历元总数: 0", font=("Arial", 9, "bold"), fg="blue")
        self.lbl_points_count.pack(side="right")

        # --- 2. 轨迹参数区 ---
        frame_params = tk.LabelFrame(self.root, text=" 2. 轨迹与时间参数 ", padx=10, pady=10)
        frame_params.pack(fill="x", padx=15, pady=5)

        tk.Label(frame_params, text="UTC日期 (YYYY-MM-DD):").grid(row=0, column=0, sticky="e", pady=5)
        self.ent_date = tk.Entry(frame_params, width=15)
        self.ent_date.insert(0, "2026-03-21")
        self.ent_date.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(frame_params, text="UTC时间 (HH:MM:SS.sss):").grid(row=0, column=2, sticky="e", pady=5)
        self.ent_time = tk.Entry(frame_params, width=15)
        self.ent_time.insert(0, "03:26:00.000")
        self.ent_time.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(frame_params, text="采样间隔 (秒):").grid(row=1, column=0, sticky="e", pady=5)
        self.ent_interval = tk.Entry(frame_params, width=15)
        self.ent_interval.insert(0, "1.0")
        self.ent_interval.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(frame_params, text="起始索引 (默认0):").grid(row=2, column=0, sticky="e", pady=5)
        self.ent_start_idx = tk.Entry(frame_params, width=15)
        self.ent_start_idx.insert(0, "0")
        self.ent_start_idx.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(frame_params, text="结束索引:").grid(row=2, column=2, sticky="e", pady=5)
        self.ent_end_idx = tk.Entry(frame_params, width=15)
        self.ent_end_idx.grid(row=2, column=3, padx=5, pady=5)

        # --- 3. 杆臂值区 ---
        frame_lever = tk.LabelFrame(self.root, text=" 3. 杆臂值设置 (相对于RTK中心, 单位: 米) ", padx=10, pady=10)
        frame_lever.pack(fill="x", padx=15, pady=10)

        tk.Label(frame_lever, text="手机名称", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=5, pady=5)
        tk.Label(frame_lever, text="dX (向右为正)", font=("Arial", 9, "bold")).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(frame_lever, text="dY (向前为正)", font=("Arial", 9, "bold")).grid(row=0, column=2, padx=5, pady=5)
        tk.Label(frame_lever, text="dZ (向上为正)", font=("Arial", 9, "bold")).grid(row=0, column=3, padx=5, pady=5)

        # 默认配置
        phones = [
            ("Phone_Left", -0.13, 0.356, 0.02),
            ("Phone_Center", 0.00, 0.356, 0.02),
            ("Phone_Right", 0.14, 0.356, 0.02)
        ]

        for i, (name, def_dx, def_dy, def_dz) in enumerate(phones):
            tk.Label(frame_lever, text=name).grid(row=i + 1, column=0, padx=5, pady=5)
            ent_dx = tk.Entry(frame_lever, width=10)
            ent_dx.insert(0, str(def_dx))
            ent_dx.grid(row=i + 1, column=1, padx=5, pady=5)

            ent_dy = tk.Entry(frame_lever, width=10)
            ent_dy.insert(0, str(def_dy))
            ent_dy.grid(row=i + 1, column=2, padx=5, pady=5)

            ent_dz = tk.Entry(frame_lever, width=10)
            ent_dz.insert(0, str(def_dz))
            ent_dz.grid(row=i + 1, column=3, padx=5, pady=5)

            self.lever_arm_entries[name] = (ent_dx, ent_dy, ent_dz)

        # --- 4. 运行按钮 ---
        self.btn_run = tk.Button(self.root, text="▶ 开 始 解 算", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white",
                                 height=2, command=self.process_data)
        self.btn_run.pack(fill="x", padx=15, pady=20)

    def load_file(self):
        """选择并解析文件，更新UI状态"""
        file_path = filedialog.askopenfilename(title="选择 RTK 数据文件", filetypes=[("Data Files", "*.dat *.txt")])
        if not file_path: return

        self.file_path = file_path
        filename = os.path.basename(file_path)
        self.lbl_file_info.config(text=filename, fg="black")

        self.all_points = []
        for enc in ['gbk', 'utf-8']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    for line in f:
                        res = parse_line(line)
                        if res: self.all_points.append(res)
                if self.all_points: break
            except UnicodeDecodeError:
                continue

        if not self.all_points:
            messagebox.showerror("文件错误", "未能解析到有效数据，请检查文件格式。")
            self.lbl_points_count.config(text="历元总数: 0", fg="red")
            return

        total_pts = len(self.all_points)
        self.lbl_points_count.config(text=f"历元总数: {total_pts}", fg="green")

        # 自动填充默认索引
        self.ent_start_idx.delete(0, tk.END)
        self.ent_start_idx.insert(0, "0")
        self.ent_end_idx.delete(0, tk.END)
        self.ent_end_idx.insert(0, str(total_pts - 1))

    def process_data(self):
        """核心处理逻辑"""
        if not self.all_points:
            messagebox.showwarning("提示", "请先选择并加载有效的数据文件！")
            return

        # 1. 读取并校验输入参数
        try:
            start_idx = int(self.ent_start_idx.get())
            end_idx = int(self.ent_end_idx.get())
            if start_idx < 0 or end_idx >= len(self.all_points) or start_idx > end_idx:
                raise ValueError("索引越界")

            utc_date_str = self.ent_date.get().strip()
            utc_time_str = self.ent_time.get().strip()
            interval_sec = float(self.ent_interval.get())
            utc_start = datetime.fromisoformat(f"{utc_date_str}T{utc_time_str}").replace(tzinfo=timezone.utc)

            lever_arms = {}
            for name, (edx, edy, edz) in self.lever_arm_entries.items():
                lever_arms[name] = {
                    'dx': float(edx.get()),
                    'dy': float(edy.get()),
                    'dz': float(edz.get())
                }
        except ValueError as e:
            messagebox.showerror("输入错误", "参数格式或数值有误，请检查！(注意日期时间格式和数字)")
            return
        except Exception as e:
            messagebox.showerror("时间格式错误", "UTC时间解析失败，请确保格式如: 2026-03-21 和 03:26:00.000")
            return

        # 2. 截取并预计算
        selected = self.all_points[start_idx: end_idx + 1]
        for pt in selected:
            pt['X'], pt['Y'], pt['Z'] = lla_to_ecef(pt['lat'], pt['lon'], pt['alt'])

        # 3. 创建输出环境
        output_dir = os.path.join(os.path.dirname(self.file_path), "Phone_dynamic_coordinates")
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        handlers = {name: open(os.path.join(output_dir, f"{name}_TruePos.txt"), 'w', encoding='utf-8') for name in
                    lever_arms}
        pos_handlers = {}
        for name in lever_arms:
            pos_file = open(os.path.join(output_dir, f"{name}_TruePos.pos"), 'w', encoding='utf-8')
            pos_file.write("%  GPST                      x-ecef(m)      y-ecef(m)      z-ecef(m)\n")
            pos_handlers[name] = pos_file

        # 4. 循环解算
        try:
            for i in range(len(selected)):
                pt = selected[i]
                utc_i = utc_start + timedelta(seconds=i * interval_sec)
                _, sow_i = utc_to_gps_week_sow(utc_i)
                gpst_str = utc_to_gpst_str(utc_i)

                # 计算航向 (Heading)
                if i == 0 and len(selected) > 1:
                    p1, p2 = selected[i], selected[i + 1]
                elif i == len(selected) - 1 and len(selected) > 1:
                    p1, p2 = selected[i - 1], selected[i]
                elif len(selected) > 2:
                    p1, p2 = selected[i - 1], selected[i + 1]
                else:
                    p1 = p2 = pt

                dX, dY, dZ = p2['X'] - p1['X'], p2['Y'] - p1['Y'], p2['Z'] - p1['Z']

                if dX == 0 and dY == 0:
                    heading = 0.0
                else:
                    dE, dN = ecef_delta_to_enu(pt['X'], pt['Y'], pt['Z'], dX, dY, dZ)
                    heading = math.atan2(dE, dN)

                # 投影及保存
                for name, offset in lever_arms.items():
                    tx, ty, tz = apply_lever_arm(pt['X'], pt['Y'], pt['Z'], pt['lat'], pt['lon'], heading, offset)
                    handlers[name].write(f"{sow_i:.3f} {tx:.6f} {ty:.6f} {tz:.6f}\n")
                    pos_handlers[name].write(f"{gpst_str}  {tx:.4f}  {ty:.4f}  {tz:.4f}\n")

            messagebox.showinfo("解算完成", f"成功处理 {len(selected)} 个历元！\n\n文件已保存至:\n{output_dir}")

        except Exception as e:
            messagebox.showerror("解算异常", f"处理过程中发生错误: {str(e)}")

        finally:
            for h in handlers.values(): h.close()
            for h in pos_handlers.values(): h.close()


if __name__ == "__main__":
    root = tk.Tk()
    app = RTKProcessorApp(root)
    root.mainloop()