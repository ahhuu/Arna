import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import math
import os
import tempfile

# =================================================================
# 1. 常量与 WGS84 椭球体参数
# =================================================================
A_EARTH = 6378137.0
F_EARTH = 1 / 298.257223563
# 修正此处：统一使用大写变量名
E_SQ = 2 * F_EARTH - F_EARTH ** 2


# =================================================================
# 2. 核心数学转换函数
# =================================================================

def dms_to_degrees(degrees, minutes, seconds):
    return degrees + minutes / 60 + seconds / 3600


def degrees_to_dms(degrees):
    d = int(degrees)
    md = abs(degrees - d) * 60
    m = int(md)
    s = (md - m) * 60
    return d, m, s


def deg_to_xyz(lat_deg, lon_deg, height):
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    N = A_EARTH / math.sqrt(1 - E_SQ * math.sin(lat_rad) ** 2)
    X = (N + height) * math.cos(lat_rad) * math.cos(lon_rad)
    Y = (N + height) * math.cos(lat_rad) * math.sin(lon_rad)
    Z = (N * (1 - E_SQ) + height) * math.sin(lat_rad)
    return X, Y, Z


def xyz_to_lla(X, Y, Z):
    lon = math.atan2(Y, X)
    p = math.sqrt(X ** 2 + Y ** 2)
    lat = math.atan2(Z, p * (1 - E_SQ))
    for _ in range(10):
        N = A_EARTH / math.sqrt(1 - E_SQ * math.sin(lat) ** 2)
        lat = math.atan2(Z + E_SQ * N * math.sin(lat), p)
    height = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), height


def get_ecef_offset(avg_lat, avg_lon, dx, dy, dz, heading_deg):
    heading_rad = math.radians(heading_deg)
    lat_rad = math.radians(avg_lat)
    lon_rad = math.radians(avg_lon)

    # 机体坐标系 -> 站心坐标系 (ENU)
    dE = dx * math.cos(heading_rad) + dy * math.sin(heading_rad)
    dN = -dx * math.sin(heading_rad) + dy * math.cos(heading_rad)
    dU = dz

    # ENU -> ECEF 偏移量
    slat, clat = math.sin(lat_rad), math.cos(lat_rad)
    slon, clon = math.sin(lon_rad), math.cos(lon_rad)

    dX_ecef = -slon * dE - slat * clon * dN + clat * clon * dU
    dY_ecef = clon * dE - slat * slon * dN + clat * slon * dU
    dZ_ecef = clat * dN + slat * dU

    return dX_ecef, dY_ecef, dZ_ecef


# =================================================================
# 3. 逻辑处理
# =================================================================

def read_data_from_file(file_path):
    data = []
    encodings = ['gbk', 'utf-8']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                for line in file:
                    line = line.strip().rstrip(',')
                    if not line: continue
                    parts = line.split(',')
                    if len(parts) == 4:
                        try:
                            # 清洗经纬度中的符号
                            lats = parts[1].replace('°', ' ').replace('′', ' ').replace('″', ' ').split()
                            lons = parts[2].replace('°', ' ').replace('′', ' ').replace('″', ' ').split()
                            lat_p = list(map(float, lats))
                            lon_p = list(map(float, lons))
                            height = float(parts[3])
                            if len(lat_p) == 3 and len(lon_p) == 3:
                                data.append((tuple(lat_p), tuple(lon_p), height))
                        except:
                            continue
            break
        except:
            continue
    return data


def average_coordinates(data):
    total_lat = total_lon = total_height = total_X = total_Y = total_Z = 0.0
    for lat_dms, lon_dms, h in data:
        lat_deg = dms_to_degrees(*lat_dms)
        lon_deg = dms_to_degrees(*lon_dms)
        X, Y, Z = deg_to_xyz(lat_deg, lon_deg, h)
        total_lat += lat_deg
        total_lon += lon_deg
        total_height += h
        total_X += X
        total_Y += Y
        total_Z += Z
    n = len(data)
    return total_lat / n, total_lon / n, total_height / n, total_X / n, total_Y / n, total_Z / n


def save_results_as_original_format(file_path, avg_lat, avg_lon, avg_height, avg_X, avg_Y, avg_Z, prefix="Avg"):
    try:
        file_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_name = f"{prefix}-{base_name}.txt"
        output_file_path = os.path.join(file_dir, output_file_name)
        lat_dms = degrees_to_dms(avg_lat)
        lon_dms = degrees_to_dms(avg_lon)
        with open(output_file_path, 'w', encoding='gbk') as file:
            file.write(f"{prefix}-WGS84-{base_name}\n")
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
        return output_file_path
    except:
        return None


def show_line_selection_dialog(lines):
    selected_indices = []

    def on_ok():
        nonlocal selected_indices
        selected_indices = list(listbox.curselection())
        win.destroy()

    win = tk.Toplevel()
    win.title("数据行过滤")
    win.geometry("700x500")
    frame = tk.Frame(win)
    frame.pack(fill="both", expand=True, padx=10, pady=5)
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")
    listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, width=100, height=20, yscrollcommand=scrollbar.set)
    for idx, line in enumerate(lines):
        listbox.insert(tk.END, f"{idx + 1}: {line.strip()}")
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)
    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="全选", command=lambda: listbox.select_set(0, tk.END), width=10).pack(side="left",
                                                                                                    padx=20)
    tk.Button(btn_frame, text="确认解算", command=on_ok, width=10, bg="#4CAF50", fg="white").pack(side="left", padx=20)
    win.grab_set()
    win.wait_window()
    return selected_indices


# =================================================================
# 4. 集成主界面
# =================================================================

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RTK 静态坐标转换与多手机平差工具")
        self.root.geometry("650x650")

        # 1. 顶部控制栏
        top_frame = tk.Frame(root, pady=10)
        top_frame.pack(fill='x', padx=20)

        tk.Label(top_frame, text="全局航向角 (Heading):").pack(side='left')
        self.ent_heading = tk.Entry(top_frame, width=10)
        self.ent_heading.insert(0, "138.0")
        self.ent_heading.pack(side='left', padx=10)

        btn_select = tk.Button(top_frame, text="选择原始数据文件", command=self.select_files, bg="#2196F3", fg="white")
        btn_select.pack(side='right')

        self.selected_files = []
        self.file_label = tk.Label(root, text="未加载文件", fg="gray")
        self.file_label.pack()

        # 2. 手机配置区 (带滚动条)
        self.config_frame = tk.LabelFrame(root, text="手机位置参数设置 (机体系：X右+ / Y前+ / Z上+)")
        self.config_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # 表头布局
        header = tk.Frame(self.config_frame)
        header.pack(fill='x', padx=5, pady=5)
        headers = [("手机标识", 18), ("dX (m)", 12), ("dY (m)", 12), ("dZ (m)", 12), ("操作", 10)]
        for text, w in headers:
            tk.Label(header, text=text, width=w, font=('Arial', 9, 'bold')).pack(side='left', padx=5)

        # 滚动区域
        self.canvas = tk.Canvas(self.config_frame, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.config_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.phone_rows = []
        # 初始化默认三个手机 (与动态场景一致)
        self.add_phone_row("Left_Phone", "-0.13", "0.356", "0.02")
        self.add_phone_row("Middle_Phone", "0.0", "0.356", "0.02")
        self.add_phone_row("Right_Phone", "0.14", "0.356", "0.02")

        # 3. 底部操作栏
        bottom_frame = tk.Frame(root, pady=20)
        bottom_frame.pack(fill='x')

        tk.Button(bottom_frame, text="+ 添加新手机配置",
                  command=lambda: self.add_phone_row("New_Phone", "0.0", "0.0", "0.0"), width=18).pack(side='left',
                                                                                                       padx=50)
        tk.Button(bottom_frame, text="▶ 执行批量转换", command=self.run_process, bg="#4CAF50", fg="white",
                  font=("Arial", 11, "bold"), width=20).pack(side='right', padx=50)

    def add_phone_row(self, name, dx, dy, dz):
        row = tk.Frame(self.scrollable_frame)
        row.pack(fill='x', pady=2)

        e_name = tk.Entry(row, width=18)
        e_name.insert(0, name)
        e_name.pack(side='left', padx=5)
        e_dx = tk.Entry(row, width=12)
        e_dx.insert(0, dx)
        e_dx.pack(side='left', padx=5)
        e_dy = tk.Entry(row, width=12)
        e_dy.insert(0, dy)
        e_dy.pack(side='left', padx=5)
        e_dz = tk.Entry(row, width=12)
        e_dz.insert(0, dz)
        e_dz.pack(side='left', padx=5)

        btn_del = tk.Button(row, text="移除", command=lambda r=row: self.remove_row(r), fg="red", font=('Arial', 8))
        btn_del.pack(side='left', padx=15)

        self.phone_rows.append({'frame': row, 'entries': [e_name, e_dx, e_dy, e_dz]})

    def remove_row(self, row_frame):
        for i, row in enumerate(self.phone_rows):
            if row['frame'] == row_frame:
                row['frame'].destroy()
                self.phone_rows.pop(i)
                break

    def select_files(self):
        files = filedialog.askopenfilenames(title="选择数据文件",
                                            filetypes=(("文本/数据", "*.dat *.txt"), ("所有文件", "*.*")))
        if files:
            self.selected_files = list(files)
            self.file_label.config(text=f"已载入 {len(files)} 个文件", fg="black")

    def run_process(self):
        if not self.selected_files:
            messagebox.showwarning("提示", "请先选择数据文件！")
            return
        try:
            heading = float(self.ent_heading.get())
            configs = []
            for row in self.phone_rows:
                configs.append((row['entries'][0].get(), float(row['entries'][1].get()), float(row['entries'][2].get()),
                                float(row['entries'][3].get())))
        except:
            messagebox.showerror("参数错误", "请确保所有偏移坐标和航向角均为有效数字！")
            return

        for f_path in self.selected_files:
            self.process_file(f_path, heading, configs)
        messagebox.showinfo("成功", "所有文件处理完成！")

    def process_file(self, file_path, heading, configs):
        # 读取内容
        lines = []
        for enc in ['gbk', 'utf-8']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    lines = f.readlines()
                break
            except:
                continue

        if not lines: return

        # 弹出行选择器
        selected_idx = show_line_selection_dialog(lines)
        if not selected_idx: return

        selected_lines = [lines[i] for i in selected_idx]
        with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as tmp:
            tmp.writelines(selected_lines)
            tmp_path = tmp.name

        measurements = read_data_from_file(tmp_path)
        os.remove(tmp_path)

        if measurements:
            # 1. 计算 RTK 平均位置
            r_lat, r_lon, r_h, r_X, r_Y, r_Z = average_coordinates(measurements)
            save_results_as_original_format(file_path, r_lat, r_lon, r_h, r_X, r_Y, r_Z, prefix="Avg-RTK")

            # 2. 依次计算不同手机位置
            for name, dx, dy, dz in configs:
                dX_e, dY_e, dZ_e = get_ecef_offset(r_lat, r_lon, dx, dy, dz, heading)
                p_X, p_Y, p_Z = r_X + dX_e, r_Y + dY_e, r_Z + dZ_e
                p_lat, p_lon, p_h = xyz_to_lla(p_X, p_Y, p_Z)

                save_results_as_original_format(file_path, p_lat, p_lon, p_h, p_X, p_Y, p_Z, prefix=f"Avg-{name}")
                print(f"[日志] 文件 {os.path.basename(file_path)}: 手机 {name} 已处理。")


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()