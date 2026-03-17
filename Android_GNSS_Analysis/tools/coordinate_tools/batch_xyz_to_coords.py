import os
import sys
from tkinter import (
    Tk, Toplevel, Label, Entry, Button, filedialog, messagebox
)

# =========================
# 固定 Coords.txt 路径
# =========================
COORDS_FILE = r"D:\code\matlab\raPPPid\raPPPid\DATA\COORDS\Coords.txt"

# =========================
# 批量选择输入文件
# =========================
def select_input_files(root):
    files = filedialog.askopenfilenames(
        title="请选择包含 XYZ 的 dat/txt 文件（可多选）",
        filetypes=[("Text files", "*.txt *.dat"), ("All files", "*.*")],
        parent=root
    )
    return list(files)

# =========================
# 解析 XYZ
# =========================
def parse_xyz(file_path):
    # ---------- 1. 多编码安全读取 ----------
    for enc in ("gbk", "gb2312", "utf-8"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            lines = None
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

    # ---------- 2. 逐行查找并提取 ----------
    def extract_value(axis):
        for line in lines:
            if axis in line and "ECEF" in line:
                # 支持全角/半角冒号，括号，空格
                import re
                m = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line)
                if m:
                    return float(m.group(1))
        # 调试输出：显示前几行内容
        preview = '\n'.join([l.strip() for l in lines[:10]])
        raise ValueError(f"{os.path.basename(file_path)} 中未找到 {axis} 坐标。请检查文件格式。\n文件内容预览:\n{preview}")

    x = extract_value("X")
    y = extract_value("Y")
    z = extract_value("Z")
    return x, y, z



# =========================
# 读取已存在 station
# =========================
def get_existing_stations(coords_file):
    stations = set()
    with open(coords_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("%"):
                stations.add(line.split()[0])
    return stations

# =========================
# 格式化一行
# =========================
def format_line(station, x, y, z):
    return f"{station:<10s}00000.000          {x:>15.6f}     {y:>15.6f}    {z:>15.6f}\n"

# =========================
# 写入 Coords.txt（��部）
# =========================
def insert_lines(coords_file, new_lines):
    with open(coords_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith("%"):
            insert_idx = i
            break

    for line in reversed(new_lines):
        lines.insert(insert_idx, line)

    with open(coords_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

# =========================
# GUI：输入 station
# =========================
def station_input_gui(root, file_xyz_map):
    gui = Toplevel(root)
    gui.title("输入 Station（每个文件一个）")

    entries = {}

    Label(gui, text="文件名", width=40, anchor="w").grid(row=0, column=0)
    Label(gui, text="Station", width=20).grid(row=0, column=1)

    for i, file in enumerate(file_xyz_map.keys(), start=1):
        Label(gui, text=os.path.basename(file), anchor="w").grid(row=i, column=0)
        e = Entry(gui, width=20)
        e.grid(row=i, column=1)
        entries[file] = e

    def submit():
        station_map = {}
        for file, entry in entries.items():
            station = entry.get().strip()
            if not station:
                messagebox.showerror("错误", "Station 不能为空", parent=gui)
                return
            station_map[file] = station
        gui.station_map = station_map
        gui.destroy()

    def on_close():
        gui.station_map = None
        gui.destroy()

    gui.protocol("WM_DELETE_WINDOW", on_close)

    Button(gui, text="确认写入", command=submit).grid(
        row=len(entries) + 1, columnspan=2, pady=10
    )

    gui.grab_set()
    gui.focus_set()
    gui.wait_window()
    return getattr(gui, "station_map", None)


# =========================
# 主流程
# =========================
def main():
    root = Tk()
    root.withdraw()

    files = select_input_files(root)
    if not files:
        root.destroy()
        sys.exit()

    file_xyz = {}
    for f in files:
        file_xyz[f] = parse_xyz(f)

    station_map = station_input_gui(root, file_xyz)
    if not station_map:
        root.destroy()
        sys.exit()

    existing = get_existing_stations(COORDS_FILE)

    # 检查重复
    for station in station_map.values():
        if station in existing:
            messagebox.showerror(
                "Station 重复",
                f"Station '{station}' 已存在于 Coords.txt 中\n操作已取消",
                parent=root
            )
            root.destroy()
            sys.exit()
            return

    new_lines = []
    for file, station in station_map.items():
        x, y, z = file_xyz[file]
        new_lines.append(format_line(station, x, y, z))

    insert_lines(COORDS_FILE, new_lines)

    messagebox.showinfo("完成", "所有坐标已成功写入 Coords.txt", parent=root)
    root.destroy()
    sys.exit()

# =========================
# 入口
# =========================
if __name__ == "__main__":
    main()
