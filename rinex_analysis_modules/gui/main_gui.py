"""
主GUI界面模块
提供图形用户界面
"""

import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
from typing import Optional

from ..core.analyzer import GNSSAnalyzer
from ..io.rinex_reader import RinexReader
from ..io.rinex_writer import RinexWriter
from ..visualization.plotter import GNSSPlotter


class MainGUI:
    """主GUI界面类"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GNSS数据分析器")
        self.root.geometry("800x500")
        self.root.resizable(True, True)
        
        # 组件初始化
        self.analyzer = None
        self.reader = None
        self.writer = None
        self.plotter = None
        
        # 数据存储
        self.phone_data = None
        self.receiver_data = None
        
        # 创建界面
        self.create_widgets()
        self.center_window()
    
    def center_window(self):
        """窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """创建界面组件"""
        
        # 创建主菜单栏
        self.create_menu()
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 欢迎信息
        welcome_label = ttk.Label(main_frame, text="ANDROID RINEX数据分析器",
                                  font=("Microsoft YaHei", 16, "bold"))
        welcome_label.pack(pady=20)
        
        # 功能说明
        desc_frame = ttk.LabelFrame(main_frame, text="功能说明", padding="20")
        desc_frame.pack(fill=tk.X, pady=20)
        
        ttk.Label(desc_frame, text="• 预处理：码相不一致性(CCI)建模校正→CMC变化阈值剔除→历元间双差剔除→ISB分析校正",
                  font=("Microsoft YaHei", 10)).pack(anchor=tk.W, pady=2)
        ttk.Label(desc_frame, text="• BDS2/3 ISB分析：单独进行北斗二号与三号系统间偏差分析", 
                  font=("Microsoft YaHei", 10)).pack(anchor=tk.W, pady=2)
        ttk.Label(desc_frame, text="• 可视化：生成各类分析图表，支持单独保存和批量保存",
                  font=("Microsoft YaHei", 10)).pack(anchor=tk.W, pady=2)
        ttk.Label(desc_frame, text="• 报告：生成完整的分析报告，包含所有预处理分析结果",
                  font=("Microsoft YaHei", 10)).pack(anchor=tk.W, pady=2)
        
        # 快速操作按钮
        quick_frame = ttk.LabelFrame(main_frame, text="快速操作", padding="20")
        quick_frame.pack(fill=tk.X, pady=20)
        
        quick_btn_frame = ttk.Frame(quick_frame)
        quick_btn_frame.pack()
        
        ttk.Button(quick_btn_frame, text="开始预处理",
                   command=self.show_cleaning_window).pack(side=tk.LEFT, padx=10)
        ttk.Button(quick_btn_frame, text="数据可视化", 
                   command=self.show_charts_window).pack(side=tk.LEFT, padx=10)
        ttk.Button(quick_btn_frame, text="生成报告",
                   command=self.show_report_window).pack(side=tk.LEFT, padx=10)
        
        # 版权信息
        copyright_frame = ttk.Frame(main_frame)
        copyright_frame.pack(fill=tk.X, pady=(20, 10))
        
        ttk.Label(copyright_frame, text="© 2025 cz",
                  font=("Microsoft YaHei", 9),
                  foreground="gray").pack(anchor=tk.CENTER)
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 预处理菜单
        cleaning_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="预处理", menu=cleaning_menu)
        cleaning_menu.add_command(label="执行预处理", command=self.show_cleaning_window)
        
        # 可视化菜单
        charts_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="可视化", menu=charts_menu)
        charts_menu.add_command(label="选择图表类型", command=self.show_charts_window)
        
        # 报告菜单
        report_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="报告", menu=report_menu)
        report_menu.add_command(label="生成分析报告", command=self.show_report_window)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
    
    def show_cleaning_window(self):
        """显示数据预处理功能窗口"""
        cleaning_window = tk.Toplevel(self.root)
        cleaning_window.title("数据预处理")
        cleaning_window.geometry("700x800")
        cleaning_window.resizable(True, True)
        cleaning_window.transient(self.root)
        cleaning_window.grab_set()
        
        # 居中显示窗口
        self.center_child_window(cleaning_window, 700, 800)
        
        # 主框架
        main_frame = ttk.Frame(cleaning_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 文件选择
        file_frame = ttk.LabelFrame(main_frame, text="选择数据文件", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        # 手机RINEX文件选择
        phone_frame = ttk.Frame(file_frame)
        phone_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(phone_frame, text="手机RINEX文件:").pack(side=tk.LEFT)
        phone_file_var = tk.StringVar()
        phone_file_entry = ttk.Entry(phone_frame, textvariable=phone_file_var, width=50)
        phone_file_entry.pack(side=tk.LEFT, padx=(10, 10), fill=tk.X, expand=True)
        
        def select_phone_file():
            file_types = [
                ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
                ("All Files", "*.*")
            ]
            filename = filedialog.askopenfilename(
                title="选择手机RINEX观测文件",
                filetypes=file_types
            )
            if filename:
                phone_file_var.set(filename)
        
        ttk.Button(phone_frame, text="浏览", command=select_phone_file).pack(side=tk.RIGHT)
        
        # 接收机RINEX文件选择
        receiver_frame = ttk.Frame(file_frame)
        receiver_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(receiver_frame, text="接收机RINEX文件(CCI建模和ISB分析必需):").pack(side=tk.LEFT)
        receiver_file_var = tk.StringVar()
        receiver_file_entry = ttk.Entry(receiver_frame, textvariable=receiver_file_var, width=50)
        receiver_file_entry.pack(side=tk.LEFT, padx=(10, 10), fill=tk.X, expand=True)
        
        def select_receiver_file():
            file_types = [
                ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
                ("All Files", "*.*")
            ]
            filename = filedialog.askopenfilename(
                title="选择接收机RINEX观测文件",
                filetypes=file_types
            )
            if filename:
                receiver_file_var.set(filename)
        
        ttk.Button(receiver_frame, text="浏览", command=select_receiver_file).pack(side=tk.RIGHT)
        
        # 参数设置
        param_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="10")
        param_frame.pack(fill=tk.X, pady=10)
        
        # 粗差处理参数设置
        outlier_frame = ttk.LabelFrame(param_frame, text="粗差处理", padding="5")
        outlier_frame.pack(fill=tk.X, pady=5)
        
        # 历元间双差最大阈值设置
        double_diff_frame = ttk.Frame(outlier_frame)
        double_diff_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(double_diff_frame, text="历元间双差最大阈值:").pack(side=tk.LEFT)
        
        # 伪距阈值
        ttk.Label(double_diff_frame, text="伪距(米):").pack(side=tk.LEFT, padx=(10, 0))
        code_threshold_var = tk.DoubleVar(value=10.0)
        code_threshold_entry = ttk.Entry(double_diff_frame, textvariable=code_threshold_var, width=8)
        code_threshold_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # 相位阈值
        ttk.Label(double_diff_frame, text="相位(米):").pack(side=tk.LEFT, padx=(10, 0))
        phase_threshold_var = tk.DoubleVar(value=1.5)
        phase_threshold_entry = ttk.Entry(double_diff_frame, textvariable=phase_threshold_var, width=8)
        phase_threshold_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # 多普勒阈值
        ttk.Label(double_diff_frame, text="多普勒(米/秒):").pack(side=tk.LEFT, padx=(10, 0))
        doppler_threshold_var = tk.DoubleVar(value=5.0)
        doppler_threshold_entry = ttk.Entry(double_diff_frame, textvariable=doppler_threshold_var, width=8)
        doppler_threshold_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # CMC变化阈值设置
        threshold_frame = ttk.Frame(outlier_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(threshold_frame, text="CMC变化阈值(米):").pack(side=tk.LEFT)
        threshold_var = tk.DoubleVar(value=4.0)
        threshold_entry = ttk.Entry(threshold_frame, textvariable=threshold_var, width=10)
        threshold_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # 可选处理设置
        option_frame = ttk.LabelFrame(param_frame, text="可选处理", padding="5")
        option_frame.pack(fill=tk.X, pady=5)
        
        # 码相不一致性处理选项
        cci_enable_var = tk.BooleanVar(value=True)
        cci_enable_checkbox = ttk.Checkbutton(option_frame, text="启用码相不一致性(CCI)处理",
                                              variable=cci_enable_var)
        cci_enable_checkbox.pack(side=tk.LEFT)
        
        ttk.Label(option_frame, text="(需要接收机文件作为基准，校正载波相位观测值)").pack(side=tk.LEFT, padx=(10, 20))
        
        # ISB处理选项
        isb_enable_var = tk.BooleanVar(value=True)
        isb_enable_checkbox = ttk.Checkbutton(option_frame, text="启用ISB处理",
                                              variable=isb_enable_var)
        isb_enable_checkbox.pack(side=tk.LEFT)
        
        ttk.Label(option_frame, text="(需要接收机文件作为基准，校正BDS系统间偏差)").pack(side=tk.LEFT, padx=(10, 0))
        
        # 码相不一致性处理参数设置
        cci_frame = ttk.LabelFrame(param_frame, text="码相不一致性处理参数", padding="5")
        cci_frame.pack(fill=tk.X, pady=5)
        
        # R方阈值和CV值阈值设置
        threshold_row_frame = ttk.Frame(cci_frame)
        threshold_row_frame.pack(fill=tk.X, pady=2)
        
        # R方阈值
        ttk.Label(threshold_row_frame, text="R方阈值:").pack(side=tk.LEFT)
        r_squared_var = tk.DoubleVar(value=0.5)
        r_squared_entry = ttk.Entry(threshold_row_frame, textvariable=r_squared_var, width=10)
        r_squared_entry.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(threshold_row_frame, text="(默认: 0.5, 线性漂移判断)").pack(side=tk.LEFT, padx=(5, 0))
        
        # CV值阈值
        ttk.Label(threshold_row_frame, text="CV阈值:").pack(side=tk.LEFT, padx=(20, 0))
        cv_threshold_var = tk.DoubleVar(value=0.6)
        cv_threshold_entry = ttk.Entry(threshold_row_frame, textvariable=cv_threshold_var, width=10)
        cv_threshold_entry.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(threshold_row_frame, text="(默认: 0.6, ROC模型选择)").pack(side=tk.LEFT, padx=(5, 0))
        
        # 手机独有卫星分析设置
        phone_only_frame = ttk.Frame(cci_frame)
        phone_only_frame.pack(fill=tk.X, pady=2)
        
        phone_only_var = tk.BooleanVar(value=False)
        phone_only_checkbox = ttk.Checkbutton(phone_only_frame, text="启用手机独有卫星分析",
                                              variable=phone_only_var)
        phone_only_checkbox.pack(side=tk.LEFT)
        
        ttk.Label(phone_only_frame, text="(检测手机独有卫星的码相不一致性)").pack(side=tk.LEFT, padx=(10, 0))
        
        # BDS-2/3 ISB处理参数设置
        isb_frame = ttk.LabelFrame(param_frame, text="BDS-2/3 ISB处理参数", padding="5")
        isb_frame.pack(fill=tk.X, pady=5)
        
        # ISB分析说明
        ttk.Label(isb_frame, text="使用动态基准卫星选择，自动选择质量最好的BDS-2卫星作为基准",
                  font=("Microsoft YaHei", 9)).pack(pady=5)
        
        # 进度显示
        progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL,
                                       variable=progress_var, mode='determinate')
        progress_bar.pack(fill=tk.X, pady=5)
        
        status_var = tk.StringVar(value="等待开始...")
        status_label = ttk.Label(progress_frame, textvariable=status_var)
        status_label.pack()
        
        # 操作按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        def start_cleaning():
            """开始数据预处理"""
            if not phone_file_var.get():
                messagebox.showerror("错误", "请先选择手机RINEX文件")
                return
            
            # 验证可选功能的选择
            if cci_enable_var.get() and not receiver_file_var.get():
                messagebox.showerror("错误", "启用码相不一致性(CCI)处理需要选择接收机RINEX文件作为基准")
                return
            
            if isb_enable_var.get() and not receiver_file_var.get():
                messagebox.showerror("错误", "启用ISB处理需要选择接收机RINEX文件作为基准")
                return
            
            # 这里应该调用实际的预处理功能
            messagebox.showinfo("提示", "预处理功能正在开发中...")
        
        def start_bds_analysis_only():
            """只进行BDS2/3 ISB分析"""
            if not phone_file_var.get():
                messagebox.showerror("错误", "请先选择手机RINEX文件")
                return
            
            if not receiver_file_var.get():
                messagebox.showerror("错误", "BDS2/3 ISB分析需要接收机RINEX文件作为基准站")
                return
            
            # 这里应该调用实际的BDS ISB分析功能
            messagebox.showinfo("提示", "BDS2/3 ISB分析功能正在开发中...")
        
        start_btn = ttk.Button(button_frame, text="开始预处理", command=start_cleaning)
        start_btn.pack(side=tk.LEFT, padx=10)
        
        select_btn = ttk.Button(button_frame, text="选择性预处理", command=start_cleaning)
        select_btn.pack(side=tk.LEFT, padx=10)
        
        bds_only_btn = ttk.Button(button_frame, text="BDS2/3 ISB分析", 
                                 command=start_bds_analysis_only, state='disabled')
        bds_only_btn.pack(side=tk.LEFT, padx=10)
        
        # 监听接收机文件变化来启用BDS按钮
        def on_receiver_file_change(*args):
            if receiver_file_var.get().strip():
                bds_only_btn.config(state='normal')
            else:
                bds_only_btn.config(state='disabled')
        
        receiver_file_var.trace('w', on_receiver_file_change)
        
        def close_cleaning_window():
            """关闭预处理窗口"""
            cleaning_window.destroy()
        
        ttk.Button(button_frame, text="关闭", command=close_cleaning_window).pack(side=tk.LEFT, padx=10)
    
    def show_charts_window(self):
        """显示图表功能窗口"""
        charts_window = tk.Toplevel(self.root)
        charts_window.title("图表生成")
        charts_window.geometry("700x800")
        charts_window.resizable(True, True)
        charts_window.transient(self.root)
        charts_window.grab_set()
        
        # 居中显示窗口
        self.center_child_window(charts_window, 700, 800)
        
        # 主框架
        main_frame = ttk.Frame(charts_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 文件选择
        file_frame = ttk.LabelFrame(main_frame, text="选择手机RINEX文件", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=file_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        def select_file():
            file_types = [
                ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
                ("All Files", "*.*")
            ]
            filename = filedialog.askopenfilename(
                title="选择RINEX观测文件",
                filetypes=file_types
            )
            if filename:
                file_var.set(filename)
                load_satellite_info()
        
        ttk.Button(file_frame, text="浏览", command=select_file).pack(side=tk.RIGHT)
        
        # 图表类型选择
        chart_frame = ttk.LabelFrame(main_frame, text="图表类型", padding="10")
        chart_frame.pack(fill=tk.X, pady=10)
        
        chart_types = [
            ("原始观测值", "raw_observations"),
            ("观测值一阶差分", "derivatives"),
            ("伪距相位差值之差", "code_phase_diffs"),
            ("伪距相位原始差值", "code_phase_diff_raw"),
            ("相位预测误差", "phase_pred_errors"),
            ("历元间双差", "double_differences"),
            ("ISB分析", "isb_analysis"),
            ("接收机CMC", "receiver_cmc")
        ]
        
        chart_var = tk.StringVar(value="raw_observations")
        for text, value in chart_types:
            ttk.Radiobutton(chart_frame, text=text, variable=chart_var,
                            value=value).pack(anchor=tk.W, pady=2)
        
        # 接收机RINEX文件选择
        rx_file_frame = ttk.LabelFrame(main_frame, text="选择接收机RINEX文件(接收机CMC、ISB分析)", padding="10")
        rx_file_frame.pack(fill=tk.X, pady=10)
        
        rx_file_var = tk.StringVar()
        rx_file_entry = ttk.Entry(rx_file_frame, textvariable=rx_file_var, width=50)
        rx_file_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        def select_rx_file():
            file_types = [
                ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
                ("All Files", "*.*")
            ]
            filename = filedialog.askopenfilename(
                title="选择接收机RINEX观测文件",
                filetypes=file_types
            )
            if filename:
                rx_file_var.set(filename)
                try:
                    load_receiver_satellite_info()
                except Exception as e:
                    messagebox.showwarning("警告", f"加载接收机RINEX信息失败:{str(e)}")
        
        ttk.Button(rx_file_frame, text="浏览", command=select_rx_file).pack(side=tk.RIGHT)
        
        # 卫星系统、PRN和频率选择
        sat_frame = ttk.LabelFrame(main_frame, text="卫星系统、PRN和频率选择", padding="10")
        sat_frame.pack(fill=tk.X, pady=10)
        
        sat_frame_inner = ttk.Frame(sat_frame)
        sat_frame_inner.pack(fill=tk.X)
        
        ttk.Label(sat_frame_inner, text="卫星系统:").pack(side=tk.LEFT)
        sat_system_var = tk.StringVar()
        sat_system_combo = ttk.Combobox(sat_frame_inner, textvariable=sat_system_var, width=15)
        sat_system_combo.pack(side=tk.LEFT, padx=(10, 20))
        
        ttk.Label(sat_frame_inner, text="卫星PRN:").pack(side=tk.LEFT)
        sat_prn_var = tk.StringVar()
        sat_prn_combo = ttk.Combobox(sat_frame_inner, textvariable=sat_prn_var, width=15)
        sat_prn_combo.pack(side=tk.LEFT, padx=(10, 20))
        
        ttk.Label(sat_frame_inner, text="频率:").pack(side=tk.LEFT)
        freq_var = tk.StringVar()
        freq_combo = ttk.Combobox(sat_frame_inner, textvariable=freq_var, width=15)
        freq_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # 进度显示
        progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL,
                                       variable=progress_var, mode='determinate')
        progress_bar.pack(fill=tk.X, pady=5)
        
        status_var = tk.StringVar(value="等待开始...")
        status_label = ttk.Label(progress_frame, textvariable=status_var)
        status_label.pack()
        
        def load_satellite_info():
            """从输入文件中加载实际的卫星和频率信息"""
            # 这里应该实现实际的卫星信息加载逻辑
            sat_system_combo['values'] = ['G', 'R', 'E', 'C']
            sat_prn_combo['values'] = ['01', '02', '03', '04', '05']
            freq_combo['values'] = ['L1C', 'L5Q']
            
            if sat_system_combo['values']:
                sat_system_var.set(sat_system_combo['values'][0])
            if sat_prn_combo['values']:
                sat_prn_var.set(sat_prn_combo['values'][0])
            if freq_combo['values']:
                freq_var.set(freq_combo['values'][0])
        
        def load_receiver_satellite_info():
            """加载接收机卫星信息"""
            # 这里应该实现接收机卫星信息加载逻辑
            pass
        
        # 操作按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        def generate_chart():
            """生成图表"""
            if not file_var.get():
                messagebox.showerror("错误", "请先选择数据文件")
                return
            
            chart_type = chart_var.get()
            messagebox.showinfo("提示", f"生成 {chart_type} 图表功能正在开发中...")
        
        def batch_generate():
            """批量生成图表"""
            if not file_var.get():
                messagebox.showerror("错误", "请先选择数据文件")
                return
            
            messagebox.showinfo("提示", "批量生成图表功能正在开发中...")
        
        ttk.Button(button_frame, text="生成图表", command=generate_chart).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="批量生成", command=batch_generate).pack(side=tk.LEFT, padx=10)
        
        def close_charts_window():
            """关闭图表窗口"""
            charts_window.destroy()
        
        ttk.Button(button_frame, text="关闭", command=close_charts_window).pack(side=tk.LEFT, padx=10)
    
    def show_report_window(self):
        """显示报告功能窗口"""
        report_window = tk.Toplevel(self.root)
        report_window.title("分析报告")
        report_window.geometry("600x500")
        report_window.resizable(True, True)
        report_window.transient(self.root)
        report_window.grab_set()
        
        # 居中显示窗口
        self.center_child_window(report_window, 600, 500)
        
        # 主框架
        main_frame = ttk.Frame(report_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 文件选择
        file_frame = ttk.LabelFrame(main_frame, text="选择数据文件", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=file_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        def select_file():
            file_types = [
                ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
                ("All Files", "*.*")
            ]
            filename = filedialog.askopenfilename(
                title="选择RINEX观测文件",
                filetypes=file_types
            )
            if filename:
                file_var.set(filename)
        
        ttk.Button(file_frame, text="浏览", command=select_file).pack(side=tk.RIGHT)
        
        # 报告选项
        report_frame = ttk.LabelFrame(main_frame, text="报告选项", padding="10")
        report_frame.pack(fill=tk.X, pady=10)
        
        include_plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(report_frame, text="包含图表", variable=include_plots_var).pack(anchor=tk.W)
        
        include_stats_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(report_frame, text="包含统计信息", variable=include_stats_var).pack(anchor=tk.W)
        
        # 进度显示
        progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL,
                                       variable=progress_var, mode='determinate')
        progress_bar.pack(fill=tk.X, pady=5)
        
        status_var = tk.StringVar(value="等待开始...")
        status_label = ttk.Label(progress_frame, textvariable=status_var)
        status_label.pack()
        
        # 操作按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        def generate_report():
            """生成报告"""
            if not file_var.get():
                messagebox.showerror("错误", "请先选择数据文件")
                return
            
            messagebox.showinfo("提示", "生成报告功能正在开发中...")
        
        ttk.Button(button_frame, text="生成报告", command=generate_report).pack(side=tk.LEFT, padx=10)
        
        def close_report_window():
            """关闭报告窗口"""
            try:
                # 关闭所有Matplotlib图表窗口
                import matplotlib.pyplot as plt
                plt.close('all')
            except Exception as e:
                print(f"关闭报告窗口时清理图表出错: {str(e)}")
            finally:
                report_window.destroy()
        
        ttk.Button(button_frame, text="关闭", command=close_report_window).pack(side=tk.LEFT, padx=10)
    
    def center_child_window(self, window, width, height):
        """子窗口居中显示"""
        window.update_idletasks()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')
    
    def initialize_components(self):
        """初始化分析组件"""
        if self.analyzer is None:
            self.analyzer = GNSSAnalyzer()
            self.reader = RinexReader()
            self.writer = RinexWriter()
            self.plotter = GNSSPlotter()
            
            # 设置进度回调
            self.analyzer.set_progress_callback(self.update_progress)

    def update_progress(self, value):
        """更新进度条"""
        # 这个方法在有实际进度条的窗口中会被重写
        pass

    def update_status(self, message):
        """更新状态信息"""
        # 这个方法在有状态显示的窗口中会被重写
        pass
    
    def show_help(self):
        """显示使用说明"""
        help_text = """GNSS数据分析器使用说明

1. 预处理：
   - 选择手机RINEX观测文件（必需）
   - 选择接收机RINEX观测文件（码相不一致性和ISB分析需要）
   - 设置处理参数：阈值、启用的处理功能等
   - 点击开始预处理执行完整处理或选择性预处理

2. 数据可视化：
   - 选择手机RINEX文件
   - 选择图表类型：原始观测值、码相差分析、相位停滞等
   - 设置卫星系统和频率（可选）
   - 生成单个或批量图表

3. 分析报告：
   - 选择分析数据文件
   - 设置报告选项：是否包含图表、统计信息等
   - 生成完整分析报告

4. 主要功能：
   - 码相不一致性(CCI)建模校正
   - CMC变化阈值剔除
   - 历元间双差剔除
   - BDS-2/3 ISB分析校正
   - 相位停滞检测
   - 多种图表可视化

注意：某些功能需要同时提供手机和接收机RINEX文件。"""
        
        messagebox.showinfo("使用说明", help_text)
    
    def show_about(self):
        """显示关于信息"""
        about_text = """GNSS数据分析器 - 模块化版本

版本：1.0.0
作者：cz

这是一个模块化的GNSS数据分析工具，
将原始的单体脚本拆分为多个功能模块，
便于维护和扩展。

主要特性：
- 模块化架构设计
- 完整的GNSS数据预处理流程
- 丰富的可视化功能
- 详细的分析报告生成
- 用户友好的图形界面

© 2025 cz. All rights reserved."""
        
        messagebox.showinfo("关于", about_text)
    
    def run(self):
        """运行GUI"""
        def on_closing():
            """程序关闭时的清理函数"""
            try:
                if self.plotter:
                    self.plotter.close_all_figures()
                self.root.destroy()
            except Exception as e:
                print(f"关闭程序时出现错误: {str(e)}")
                self.root.quit()
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # 启动主循环
        self.root.mainloop()