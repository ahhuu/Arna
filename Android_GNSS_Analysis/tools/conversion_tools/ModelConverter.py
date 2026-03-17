import tkinter as tk
from tkinter import filedialog, messagebox
import re
import os

class ModelConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("raPPPid 随机模型转化工具 v1.0")
        self.root.geometry("800x650")

        # 频率顺序定义 (用户指定)
        self.FREQ_MAP = {
            'G': {'L1': 0, 'L5': 1},
            'R': {'G1': 0},
            'E': {'E1': 0, 'E5a': 1, 'E5b': 2},
            'C': {'B1': 0, 'B1AC': 1, 'B2a': 2},
            'J': {'L1': 0}
        }
        
        # 系统名称展示映射
        self.SYS_NAME_MAP = {
            'G': 'GPS',
            'R': 'GLO',
            'E': 'GAL',
            'C': 'BDS',
            'J': 'QZSS'
        }

        self.setup_ui()

    def setup_ui(self):
        # 顶部操作区
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill=tk.X)
        
        btn_select = tk.Button(top_frame, text="第一步：选择 model_*.txt 文件 (支持多选)", 
                              command=self.select_files, bg="#e1f5fe", padx=20)
        btn_select.pack(side=tk.LEFT, padx=20)

        # 结果显示区 (分为四个模型)
        self.results = {}
        model_types = [
            ("Elevation", "高度角模型 (a + b ./ sin(e).^2)"),
            ("ExpSNR", "指数信噪比模型 (a + b .* 10.^(-SNR/10))"),
            ("LinearSNR", "线性信噪比模型 (a .* SNR + b)"),
            ("Combined", "联合模型 (a + b ./ sin(e).^2 + c .* 10.^(-SNR/10))")
        ]

        main_frame = tk.Frame(self.root, padx=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        for key, title in model_types:
            label = tk.Label(main_frame, text=title, font=("Microsoft YaHei", 10, "bold"), pady=5)
            label.pack(anchor=tk.W)
            
            frame = tk.Frame(main_frame)
            frame.pack(fill=tk.X, pady=2)
            
            text_area = tk.Entry(frame, font=("Consolas", 10))
            text_area.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            btn_copy = tk.Button(frame, text="复制", command=lambda t=text_area: self.copy_to_clipboard(t.get()))
            btn_copy.pack(side=tk.RIGHT, padx=5)
            
            self.results[key] = text_area

        # 状态栏
        self.status = tk.Label(self.root, text="就绪。请选择拟合结果文件。", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="选择拟合结果文件",
            filetypes=[("Text files", "model_*.txt"), ("All files", "*.*")]
        )
        if not files:
            return
        
        self.process_files(files)

    def process_files(self, file_paths):
        # 初始化存储结构: {System: [f1_data, f2_data, f3_data]}
        # data 格式: {'Elev': (a,b), 'ExpSNR': (a,b), 'LinearSNR': (a,b), 'Comb': (a,b,c)}
        data_store = {}
        global_model = None

        for path in file_paths:
            filename = os.path.basename(path)
            # 解析系统和频率
            match = re.match(r"model_(.*)_(.*)\.txt", filename)
            if not match: continue
            
            sys_code, freq_name = match.groups()
            
            # 读取并解析文件内容
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            params = self.parse_model_params(content)
            
            if sys_code == 'ALL' and freq_name == 'ALL':
                global_model = params
                continue
            
            if sys_code not in data_store:
                data_store[sys_code] = [None, None, None]
            
            # 确定频率索引
            idx = self.FREQ_MAP.get(sys_code, {}).get(freq_name, -1)
            if idx != -1 and idx < 3:
                data_store[sys_code][idx] = params

        # 生成最终字符串
        self.update_ui_strings(data_store, global_model)
        self.status.config(text=f"已处理 {len(file_paths)} 个文件。")

    def parse_model_params(self, content):
        """极其精确地解析文件中的 a, b, c 系数，避免误抓公式中的常数(如^2)"""
        res = {
            'Elev': [0.0, 0.0],
            'ExpSNR': [0.0, 0.0],
            'LinearSNR': [0.0, 0.0],
            'Comb': [0.0, 0.0, 0.0]
        }

        # 1. 高度角模型: σ² = a+b./sin(e).^2
        m = re.search(r"高度角模型：\s*σ² = ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\./sin", content)
        if m: res['Elev'] = [float(m.group(1)), float(m.group(2))]

        # 2. 指数信噪比模型: σ² = a+b.*10.^(-SNR/10)
        m = re.search(r"指数信噪比模型：\s*σ² = ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.\*10", content)
        if m: res['ExpSNR'] = [float(m.group(1)), float(m.group(2))]

        # 3. 线性信噪比模型: σ² = (a).*SNR+b
        m = re.search(r"线性信噪比模型：\s*σ² = \(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)\.\*SNR\+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", content)
        if m: res['LinearSNR'] = [float(m.group(1)), float(m.group(2))]

        # 4. 联合模型: σ² = a+b./sin(e).^2+c.*10.^(-SNR/10)
        m = re.search(r"联合模型：\s*σ² = ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\./sin\(e\)\.\^2\+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.\*10", content)
        if m: res['Comb'] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]

        return res

    def update_ui_strings(self, data_store, global_model):
        final_strs = {'Elev': [], 'ExpSNR': [], 'LinearSNR': [], 'Comb': []}

        def fmt_eq(type, p):
            if type == 'Elev':
                return f"1./({p[0]:.4f} + {p[1]:.4f}./sin(e).^2)"
            elif type == 'ExpSNR':
                return f"1./({p[0]:.4f} + {p[1]:.4f}.*10.^(-SNR/10))"
            elif type == 'LinearSNR':
                return f"1./(({p[0]:.4f}).*SNR + {p[1]:.4f})"
            elif type == 'Comb':
                return f"1./({p[0]:.4f} + {p[1]:.4f}./sin(e).^2 + {p[2]:.4f}.*10.^(-SNR/10))"
            return ""

        # 1. 如果有全局模型，先放全局模型 (无需前缀)
        if global_model:
            for k in ['Elev', 'ExpSNR', 'LinearSNR', 'Comb']:
                final_strs[k].append(fmt_eq(k, global_model[k]))

        # 2. 遍历各系统
        for sys_code in sorted(data_store.keys()):
            sys_name = self.SYS_NAME_MAP.get(sys_code, sys_code)
            sys_strs = {'Elev': [], 'ExpSNR': [], 'LinearSNR': [], 'Comb': []}
            
            for params in data_store[sys_code]:
                if params is None:
                    for k in sys_strs: sys_strs[k].append("")
                else:
                    # 修正：传递具体的模型系数列表 params[k]，而不是整个字典 params
                    for k in sys_strs: sys_strs[k].append(fmt_eq(k, params[k]))

            # 组合该系统字符串: "SYS: f1; f2; f3"
            for k in final_strs:
                s_list = sys_strs[k]
                while s_list and s_list[-1] == "": s_list.pop()
                content = "; ".join(s_list)
                if content:
                    final_strs[k].append(f"{sys_name}: {content}")

        # 更新到 UI 文本框
        self.results['Elevation'].delete(0, tk.END)
        self.results['Elevation'].insert(0, "; ".join(final_strs['Elev']))
        
        self.results['ExpSNR'].delete(0, tk.END)
        self.results['ExpSNR'].insert(0, "; ".join(final_strs['ExpSNR']))
        
        self.results['LinearSNR'].delete(0, tk.END)
        self.results['LinearSNR'].insert(0, "; ".join(final_strs['LinearSNR']))
        
        self.results['Combined'].delete(0, tk.END)
        self.results['Combined'].insert(0, "; ".join(final_strs['Comb']))

    def copy_to_clipboard(self, text):
        if text:
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.root.update() # 现在内容已在剪贴板
                self.status.config(text="已复制到剪贴板！")
            except Exception as e:
                messagebox.showerror("错误", f"复制失败: {e}")
        else:
            messagebox.showwarning("警告", "内容为空！")

if __name__ == "__main__":
    root = tk.Tk()
    # 尝试设置更大的中文字体
    try:
        from tkinter import font
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=10)
    except:
        pass
        
    app = ModelConverterApp(root)
    root.mainloop()
