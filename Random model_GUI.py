import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import time
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io

# 导入分析脚本
import sys
import importlib.util
from pathlib import Path

# 动态导入 Pseudorange_Residuals
pr_path = Path(__file__).parent / 'tools' / 'analysis_tools' / 'Pseudorange_Residuals.py'
pr_spec = importlib.util.spec_from_file_location("Pseudorange_Residuals", pr_path)
Pseudorange_Residuals = importlib.util.module_from_spec(pr_spec)
sys.modules["Pseudorange_Residuals"] = Pseudorange_Residuals
pr_spec.loader.exec_module(Pseudorange_Residuals)

# 动态导入 SNR_Weighting
snr_path = Path(__file__).parent / 'tools' / 'analysis_tools' / 'SNR_Weighting.py'
snr_spec = importlib.util.spec_from_file_location("SNR_Weighting", snr_path)
SNR_Weighting = importlib.util.module_from_spec(snr_spec)
sys.modules["SNR_Weighting"] = SNR_Weighting
snr_spec.loader.exec_module(SNR_Weighting)


class TextRedirector(io.StringIO):
    """用于重定向stdout和stderr到GUI"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update()


class GNSSAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GNSS 数据分析工具")
        self.root.geometry("800x900")
        self.root.resizable(True, True)

        # 先初始化变量
        self.mobile_rinex_file = tk.StringVar()
        self.base_rinex_file = tk.StringVar()
        self.sat_pos_file = tk.StringVar()
        self.residuals_file = tk.StringVar()
        
        # 坐标变量
        self.mobile_x = tk.DoubleVar(value=-1324698.041159006)
        self.mobile_y = tk.DoubleVar(value=5323031.038016253)
        self.mobile_z = tk.DoubleVar(value=3244602.006945656)
        self.base_x = tk.DoubleVar(value=-1324698.104573897)
        self.base_y = tk.DoubleVar(value=5323031.050568524)
        self.base_z = tk.DoubleVar(value=3244601.728187757)
        
        # 然后创建界面
        self.create_widgets()

    def create_widgets(self):
        # 创建笔记本控件（标签页）
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建三个标签页
        self.create_complete_tab(notebook)
        self.create_residuals_tab(notebook)
        self.create_weighting_tab(notebook)

    def create_complete_tab(self, notebook):
        """创建完整流程标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="完整流程")

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        # 手机RINEX文件
        ttk.Label(file_frame, text="手机RINEX文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.mobile_rinex_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.mobile_rinex_file, "RINEX")).grid(row=0, column=2, padx=5, pady=2)

        # 基准站RINEX文件
        ttk.Label(file_frame, text="基准站RINEX文件:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.base_rinex_file, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.base_rinex_file, "RINEX")).grid(row=1, column=2, padx=5, pady=2)

        # 卫星位置文件
        ttk.Label(file_frame, text="卫星位置文件:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.sat_pos_file, width=60).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.sat_pos_file, "TXT")).grid(row=2, column=2, padx=5, pady=2)

        # 坐标输入区域
        coord_frame = ttk.LabelFrame(frame, text="坐标输入 (ECEF, 单位:米)", padding=10)
        coord_frame.pack(fill=tk.X, padx=10, pady=5)

        # 手机坐标
        ttk.Label(coord_frame, text="手机坐标:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(coord_frame, text="X:").grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_x, width=15).grid(row=0, column=2, padx=5)
        ttk.Label(coord_frame, text="Y:").grid(row=0, column=3, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_y, width=15).grid(row=0, column=4, padx=5)
        ttk.Label(coord_frame, text="Z:").grid(row=0, column=5, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_z, width=15).grid(row=0, column=6, padx=5)

        # 基准站坐标
        ttk.Label(coord_frame, text="基准站坐标:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(coord_frame, text="X:").grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_x, width=15).grid(row=1, column=2, padx=5)
        ttk.Label(coord_frame, text="Y:").grid(row=1, column=3, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_y, width=15).grid(row=1, column=4, padx=5)
        ttk.Label(coord_frame, text="Z:").grid(row=1, column=5, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_z, width=15).grid(row=1, column=6, padx=5)

        # 控制按钮
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="运行完整流程", command=self.run_complete_analysis, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress_complete = ttk.Progressbar(frame, mode='indeterminate')
        self.progress_complete.pack(fill=tk.X, padx=10, pady=5)

        # 日志输出区域
        log_frame = ttk.LabelFrame(frame, text="运行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text_complete = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text_complete.pack(fill=tk.BOTH, expand=True)

    def create_residuals_tab(self, notebook):
        """创建伪距残差分析标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="伪距残差分析")

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        # 复用完整流程的文件选择变量
        ttk.Label(file_frame, text="手机RINEX文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.mobile_rinex_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.mobile_rinex_file, "RINEX")).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(file_frame, text="基准站RINEX文件:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.base_rinex_file, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.base_rinex_file, "RINEX")).grid(row=1, column=2, padx=5, pady=2)

        ttk.Label(file_frame, text="卫星位置文件:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.sat_pos_file, width=60).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.sat_pos_file, "TXT")).grid(row=2, column=2, padx=5, pady=2)

        # 坐标输入区域（复用变量）
        coord_frame = ttk.LabelFrame(frame, text="坐标输入 (ECEF, 单位:米)", padding=10)
        coord_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(coord_frame, text="手机坐标:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(coord_frame, text="X:").grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_x, width=15).grid(row=0, column=2, padx=5)
        ttk.Label(coord_frame, text="Y:").grid(row=0, column=3, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_y, width=15).grid(row=0, column=4, padx=5)
        ttk.Label(coord_frame, text="Z:").grid(row=0, column=5, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_z, width=15).grid(row=0, column=6, padx=5)

        ttk.Label(coord_frame, text="基准站坐标:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(coord_frame, text="X:").grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_x, width=15).grid(row=1, column=2, padx=5)
        ttk.Label(coord_frame, text="Y:").grid(row=1, column=3, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_y, width=15).grid(row=1, column=4, padx=5)
        ttk.Label(coord_frame, text="Z:").grid(row=1, column=5, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_z, width=15).grid(row=1, column=6, padx=5)

        # 控制按钮
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="计算伪距残差", command=self.run_residuals_analysis, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空日志", command=lambda: self.log_text_residuals.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress_residuals = ttk.Progressbar(frame, mode='indeterminate')
        self.progress_residuals.pack(fill=tk.X, padx=10, pady=5)

        # 日志输出区域
        log_frame = ttk.LabelFrame(frame, text="运行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text_residuals = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text_residuals.pack(fill=tk.BOTH, expand=True)

    def create_weighting_tab(self, notebook):
        """创建随机模型拟合标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="随机模型拟合")

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="伪距残差文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.residuals_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.residuals_file, "CSV")).grid(row=0, column=2, padx=5, pady=2)

        # 控制按钮
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="运行随机模型拟合", command=self.run_weighting_analysis, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空日志", command=lambda: self.log_text_weighting.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress_weighting = ttk.Progressbar(frame, mode='indeterminate')
        self.progress_weighting.pack(fill=tk.X, padx=10, pady=5)

        # 日志输出区域
        log_frame = ttk.LabelFrame(frame, text="运行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text_weighting = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text_weighting.pack(fill=tk.BOTH, expand=True)

    def browse_file(self, var, file_type):
        """文件浏览对话框"""
        if file_type == "RINEX":
            filetypes = [("RINEX files", "*.25o *.25O *.obs *.rnx"), ("All files", "*.*")]
        elif file_type == "TXT":
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
        elif file_type == "CSV":
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        else:
            filetypes = [("All files", "*.*")]

        filename = filedialog.askopenfilename(
            title=f"选择{file_type}文件",
            filetypes=filetypes
        )
        if filename:
            var.set(filename)

    def validate_inputs_residuals(self):
        """验证伪距残差分析输入"""
        if not self.mobile_rinex_file.get():
            messagebox.showerror("输入错误", "请选择手机RINEX文件")
            return False
        if not self.base_rinex_file.get():
            messagebox.showerror("输入错误", "请选择基准站RINEX文件")
            return False
        if not self.sat_pos_file.get():
            messagebox.showerror("输入错误", "请选择卫星位置文件")
            return False
        
        # 检查文件是否存在
        for file_path in [self.mobile_rinex_file.get(), self.base_rinex_file.get(), self.sat_pos_file.get()]:
            if not os.path.exists(file_path):
                messagebox.showerror("文件错误", f"文件不存在: {file_path}")
                return False
        
        return True

    def validate_inputs_weighting(self):
        """验证随机模型拟合输入"""
        if not self.residuals_file.get():
            messagebox.showerror("输入错误", "请选择伪距残差文件")
            return False
        
        if not os.path.exists(self.residuals_file.get()):
            messagebox.showerror("文件错误", f"文件不存在: {self.residuals_file.get()}")
            return False
        
        return True

    def run_residuals_analysis(self):
        """运行伪距残差分析"""
        if not self.validate_inputs_residuals():
            return

        def run_analysis():
            try:
                self.progress_residuals.start()
                
                # 重定向输出到日志窗口
                redirector = TextRedirector(self.log_text_residuals)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    self.log_text_residuals.insert(tk.END, "开始计算伪距残差...\n")
                    self.log_text_residuals.see(tk.END)
                    
                    # 准备参数
                    mobile_coords = [self.mobile_x.get(), self.mobile_y.get(), self.mobile_z.get()]
                    base_coords = [self.base_x.get(), self.base_y.get(), self.base_z.get()]
                    
                    # 调用伪距残差分析函数
                    Pseudorange_Residuals.main(
                        self.mobile_rinex_file.get(),
                        self.base_rinex_file.get(),
                        self.sat_pos_file.get(),
                        mobile_coords,
                        base_coords
                    )
                    
                    self.log_text_residuals.insert(tk.END, "\n伪距残差分析完成！\n")
                    self.log_text_residuals.see(tk.END)
                    
                    # 自动设置残差文件路径用于下一步分析
                    mobile_filename = os.path.basename(self.mobile_rinex_file.get())
                    mobile_basename = os.path.splitext(mobile_filename)[0]
                    residuals_path = os.path.join('results', mobile_basename, 'pseudorange_residuals.csv')
                    if os.path.exists(residuals_path):
                        self.residuals_file.set(residuals_path)
                    
            except Exception as e:
                self.log_text_residuals.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_residuals.see(tk.END)
                messagebox.showerror("运行错误", f"伪距残差分析失败: {str(e)}")
            finally:
                self.progress_residuals.stop()

        # 在新线程中运行以避免界面冻结
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()

    def run_weighting_analysis(self):
        """运行随机模型拟合分析"""
        if not self.validate_inputs_weighting():
            return

        def run_analysis():
            try:
                self.progress_weighting.start()
                
                # 重定向输出到日志窗口
                redirector = TextRedirector(self.log_text_weighting)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    self.log_text_weighting.insert(tk.END, "开始随机模型拟合...\n")
                    self.log_text_weighting.see(tk.END)
                    
                    # 临时修改SNR_Weighting的输入文件路径
                    original_input_file = SNR_Weighting.main.__globals__.get('input_file')
                    
                    # 创建一个临时的main函数来处理我们的输入文件
                    def temp_snr_main(input_file_path):
                        # 临时修改模块的输入文件路径
                        original_code = SNR_Weighting.main.__code__
                        
                        # 直接调用修改后的逻辑
                        import pandas as pd
                        import numpy as np
                        import matplotlib.pyplot as plt
                        from scipy.optimize import curve_fit, least_squares
                        
                        # 创建输出文件夹
                        output_dir = os.path.join(os.path.dirname(input_file_path), 'Weighting')
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 配置 matplotlib
                        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        # 读取数据
                        df = pd.read_csv(input_file_path, parse_dates=['epoch'])
                        df['prn'] = df['prn'].astype(str)
                        df['system'] = df['prn'].str[0]
                        
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
                        SNR_Weighting.visualize_global_residuals(df, output_dir)
                        
                        # 按卫星系统和频率分组拟合
                        systems = df['system'].unique()
                        frequencies = df['frequency'].unique()
                        results = []
                        print("开始分组参数拟合...")
                        for system in systems:
                            for freq in frequencies:
                                subset = df[(df['system'] == system) & (df['frequency'] == freq)]
                                if len(subset) < 10:
                                    continue
                                result = SNR_Weighting.fit_and_visualize(subset, system, freq, output_dir)
                                results.append(result)
                        
                        # 全局拟合
                        global_subset = df.copy()
                        if len(global_subset) >= 50:
                            print("开始全局参数拟合...")
                            result = SNR_Weighting.fit_and_visualize(global_subset, 'ALL', 'ALL', output_dir)
                            result['system'] = 'ALL'
                            result['frequency'] = 'ALL'
                            results.append(result)
                        else:
                            print("数据量不足，无法进行全局拟合")
                        
                        # 保存所有拟合结果到CSV
                        if results:
                            results_df = pd.DataFrame(results)
                            results_file = os.path.join(output_dir, 'fitting_results.csv')
                            results_df.to_csv(results_file, index=False)
                        
                        print(f"所有拟合已完成，结果已保存到 {output_dir}")
                    
                    # 调用修改后的分析函数
                    temp_snr_main(self.residuals_file.get())
                    
                    self.log_text_weighting.insert(tk.END, "\n随机模型拟合完成！\n")
                    self.log_text_weighting.see(tk.END)
                    
            except Exception as e:
                self.log_text_weighting.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_weighting.see(tk.END)
                messagebox.showerror("运行错误", f"随机模型拟合失败: {str(e)}")
            finally:
                self.progress_weighting.stop()

        # 在新线程中运行以避免界面冻结
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()

    def run_complete_analysis(self):
        """运行完整分析流程"""
        if not self.validate_inputs_residuals():
            return

        def run_analysis():
            try:
                self.progress_complete.start()
                
                # 重定向输出到日志窗口
                redirector = TextRedirector(self.log_text_complete)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    self.log_text_complete.insert(tk.END, "开始完整分析流程...\n")
                    self.log_text_complete.insert(tk.END, "第一步：计算伪距残差...\n")
                    self.log_text_complete.see(tk.END)
                    
                    # 第一步：伪距残差分析
                    mobile_coords = [self.mobile_x.get(), self.mobile_y.get(), self.mobile_z.get()]
                    base_coords = [self.base_x.get(), self.base_y.get(), self.base_z.get()]
                    
                    Pseudorange_Residuals.main(
                        self.mobile_rinex_file.get(),
                        self.base_rinex_file.get(),
                        self.sat_pos_file.get(),
                        mobile_coords,
                        base_coords
                    )
                    
                    self.log_text_complete.insert(tk.END, "\n第一步完成！\n")
                    self.log_text_complete.insert(tk.END, "第二步：随机模型拟合...\n")
                    self.log_text_complete.see(tk.END)
                    
                    # 确定残差文件路径
                    mobile_filename = os.path.basename(self.mobile_rinex_file.get())
                    mobile_basename = os.path.splitext(mobile_filename)[0]
                    residuals_path = os.path.join('results', mobile_basename, 'pseudorange_residuals.csv')
                    
                    if not os.path.exists(residuals_path):
                        raise FileNotFoundError(f"残差文件未找到: {residuals_path}")
                    
                    # 第二步：随机模型拟合
                    # 直接调用SNR_Weighting的逻辑
                    import pandas as pd
                    import numpy as np
                    import matplotlib.pyplot as plt
                    
                    # 创建输出文件夹
                    output_dir = os.path.join(os.path.dirname(residuals_path), 'Weighting')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 配置 matplotlib
                    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    # 读取数据
                    df = pd.read_csv(residuals_path, parse_dates=['epoch'])
                    df['prn'] = df['prn'].astype(str)
                    df['system'] = df['prn'].str[0]
                    
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
                    SNR_Weighting.visualize_global_residuals(df, output_dir)
                    
                    # 按卫星系统和频率分组拟合
                    systems = df['system'].unique()
                    frequencies = df['frequency'].unique()
                    results = []
                    print("开始分组参数拟合...")
                    for system in systems:
                        for freq in frequencies:
                            subset = df[(df['system'] == system) & (df['frequency'] == freq)]
                            if len(subset) < 10:
                                continue
                            result = SNR_Weighting.fit_and_visualize(subset, system, freq, output_dir)
                            results.append(result)
                    
                    # 全局拟合
                    global_subset = df.copy()
                    if len(global_subset) >= 50:
                        print("开始全局参数拟合...")
                        result = SNR_Weighting.fit_and_visualize(global_subset, 'ALL', 'ALL', output_dir)
                        result['system'] = 'ALL'
                        result['frequency'] = 'ALL'
                        results.append(result)
                    else:
                        print("数据量不足，无法进行全局拟合")
                    
                    # 保存所有拟合结果到CSV
                    if results:
                        results_df = pd.DataFrame(results)
                        results_file = os.path.join(output_dir, 'fitting_results.csv')
                        results_df.to_csv(results_file, index=False)
                    
                    print(f"所有拟合已完成，结果已保存到 {output_dir}")
                    
                    self.log_text_complete.insert(tk.END, "\n完整分析流程全部完成！\n")
                    self.log_text_complete.see(tk.END)
                    
                    # 更新残差文件路径
                    self.residuals_file.set(residuals_path)
                    
            except Exception as e:
                self.log_text_complete.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_complete.see(tk.END)
                messagebox.showerror("运行错误", f"完整流程分析失败: {str(e)}")
            finally:
                self.progress_complete.stop()

        # 在新线程中运行以避免界面冻结
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()

    def clear_log(self):
        """清空完整流程的日志"""
        self.log_text_complete.delete(1.0, tk.END)


def main():
    root = tk.Tk()
    
    # 设置现代化主题
    style = ttk.Style()
    try:
        # 尝试使用现代主题
        style.theme_use('clam')
    except:
        pass
    
    # 创建自定义样式
    style.configure('Accent.TButton', foreground='white', background='#0078d4')
    
    app = GNSSAnalysisGUI(root)
    
    # 设置窗口图标（如果有的话）
    try:
        if os.path.exists('icon.ico'):
            root.iconbitmap('icon.ico')
    except:
        pass
    
    root.mainloop()


if __name__ == "__main__":
    main()
