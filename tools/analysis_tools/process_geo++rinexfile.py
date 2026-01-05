import os
import tkinter as tk
from tkinter import filedialog


def modify_rinex_file(input_file):
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}-mod{ext}"

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        in_header = True

        for line in infile:
            # 处理文件头
            if in_header:
                if line.startswith("E ") and "SYS / # / OBS TYPES" in line:
                    # line = "E    8 C1C L1C D1C S1C C5Q L5Q D5Q S5Q                      SYS / # / OBS TYPES\n"
                    line = "E    8 C1C L1C D1C S1C C7Q L7Q D7Q S7Q                      SYS / # / OBS TYPES\n"
                outfile.write(line)
                if "END OF HEADER" in line:
                    in_header = False
                continue

            # 处理数据部分
            if line.startswith('>'):
                outfile.write(line)
            elif line.strip().startswith('E'):
                # 分割卫星编号和观测数据
                stripped = line.strip()
                sat_id = stripped[:3]  # 如E16
                obs_data = stripped[3:].strip()  # 删除观测数据前多余空格

                # 重建行，确保卫星编号后只有2个空格
                # 保留观测数据内部原有的空格
                new_line = f"{sat_id}  {obs_data}\n"
                outfile.write(new_line)
            else:
                outfile.write(line)

    return output_file


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择RINEX观测文件",
        filetypes=[("RINEX观测文件", "*.??o"), ("所有文件", "*.*")]
    )

    if not file_path:
        print("操作取消")
        return

    try:
        output_path = modify_rinex_file(file_path)
        print(f"处理成功，结果已保存到:\n{output_path}")
    except Exception as e:
        print(f"处理失败: {str(e)}")


if __name__ == "__main__":
    main()
