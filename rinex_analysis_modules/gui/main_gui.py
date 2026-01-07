"""
GUI主界面模块
完整保留原始Rinex_analysis.py中的所有GUI函数
"""

import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import datetime
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..core.analyzer import GNSSAnalyzer


def center_window(window, width=None, height=None):
    """将窗口居中显示在屏幕上"""
    window.update_idletasks()

    # 获取窗口尺寸
    if width is None or height is None:
        window_width = window.winfo_width()
        window_height = window.winfo_height()
    else:
        window_width = width
        window_height = height

    # 获取屏幕尺寸
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # 计算居中位置
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # 设置窗口位置
    window.geometry(f"{window_width}x{window_height}+{x}+{y}")


def main():
    root = tk.Tk()
    root.title("GNSS数据分析器")
    root.geometry("750x500")
    root.resizable(True, True)

    # 创建主菜单栏
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # 预处理菜单
    cleaning_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="预处理", menu=cleaning_menu)
    cleaning_menu.add_command(label="执行预处理", command=lambda: show_cleaning_window(root))

    # 可视化菜单
    charts_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="可视化", menu=charts_menu)
    charts_menu.add_command(label="选择图表类型", command=lambda: show_charts_window(root))

    # 报告菜单
    report_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="报告", menu=report_menu)
    report_menu.add_command(label="生成分析报告", command=lambda: show_report_window(root))

    # 主界面
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 欢迎信息
    welcome_label = ttk.Label(main_frame, text="ANDROID RINEX数据分析器",
                              font=("Microsoft YaHei", 16, "bold"))
    welcome_label.pack(pady=20)

    # 功能说明
    desc_frame = ttk.LabelFrame(main_frame, text="功能说明", padding="20")
    desc_frame.pack(fill=tk.X, pady=20)

    ttk.Label(desc_frame, text="• 预处理：多普勒预测相位→码相不一致性建模校正→CMC变化阈值剔除→历元间双差剔除→BDS2/3 ISB分析校正",
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
               command=lambda: show_cleaning_window(root)).pack(side=tk.LEFT, padx=10)
    ttk.Button(quick_btn_frame, text="数据可视化",
               command=lambda: show_charts_window(root)).pack(side=tk.LEFT, padx=10)
    ttk.Button(quick_btn_frame, text="生成报告",
               command=lambda: show_report_window(root)).pack(side=tk.LEFT, padx=10)

    # 版权信息
    copyright_frame = ttk.Frame(main_frame)
    copyright_frame.pack(fill=tk.X, pady=(20, 10))

    ttk.Label(copyright_frame, text="© 2025 cz",
              font=("Microsoft YaHei", 9),
              foreground="gray").pack(anchor=tk.CENTER)

    def on_closing():
        """程序关闭时的清理函数"""
        try:
            # 关闭所有Matplotlib图表窗口
            plt.close('all')
            # 清理Matplotlib资源
            import matplotlib
            matplotlib.pyplot.close('all')
            # 销毁主窗口
            root.destroy()
        except Exception as e:
            print(f"关闭程序时出现错误: {str(e)}")
            # 强制退出
            root.quit()

    # 绑定关闭事件
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 居中显示主窗口
    center_window(root, 750, 500)

    root.mainloop()


def show_cleaning_window(parent):
    """显示数据预处理功能窗口"""
    cleaning_window = tk.Toplevel(parent)
    cleaning_window.title("数据预处理")
    cleaning_window.geometry("760x850")
    cleaning_window.resizable(True, True)
    cleaning_window.transient(parent)
    cleaning_window.grab_set()

    # 居中显示窗口
    center_window(cleaning_window, 760, 850)

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

    # 接收机RINEX文件选择（可选，用于码相不一致性建模）
    receiver_frame = ttk.Frame(file_frame)
    receiver_frame.pack(fill=tk.X, pady=5)

    ttk.Label(receiver_frame, text="接收机RINEX文件(CCI建模和ISB分析必需):").pack(side=tk.LEFT)
    receiver_file_var = tk.StringVar()
    receiver_file_entry = ttk.Entry(receiver_frame, textvariable=receiver_file_var, width=50)
    receiver_file_entry.pack(side=tk.LEFT, padx=(10, 10), fill=tk.X, expand=True)

    # 添加监听器，当接收机文件路径改变时更新按钮状态
    def on_receiver_file_change(*args):
        if receiver_file_var.get().strip():
            bds_only_btn.config(state='normal')
        else:
            bds_only_btn.config(state='disabled')

    receiver_file_var.trace('w', on_receiver_file_change)

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
            # 启用BDS2/3分析按钮
            bds_only_btn.config(state='normal')
        else:
            # 如果没有选择文件，禁用BDS2/3分析按钮
            bds_only_btn.config(state='disabled')

    ttk.Button(receiver_frame, text="浏览", command=select_receiver_file).pack(side=tk.RIGHT)

    # 兼容性：保持原有的file_var变量
    file_var = phone_file_var

    # 参数设置
    param_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="10")
    param_frame.pack(fill=tk.X, pady=10)

    # 参数模板选择
    template_frame = ttk.LabelFrame(param_frame, text="参数模板", padding="5")
    template_frame.pack(fill=tk.X, pady=5)

    template_row = ttk.Frame(template_frame)
    template_row.pack(fill=tk.X, pady=2)

    ttk.Label(template_row, text="选择预设模板:").pack(side=tk.LEFT)
    template_var = tk.StringVar(value="自定义")
    template_combo = ttk.Combobox(template_row, textvariable=template_var, width=15, state="readonly")
    template_combo['values'] = ["自定义", "开阔环境", "遮挡环境"]
    template_combo.pack(side=tk.LEFT, padx=(10, 10))

    def apply_template(*args):
        """应用参数模板"""
        template = template_var.get()
        if template == "开阔环境":
            # 开阔环境：信号质量好，多径干扰少，可以使用相对严格的阈值
            code_threshold_var.set(8.0)
            phase_threshold_var.set(3.0)
            doppler_threshold_var.set(4.0)
            threshold_var.set(3.0)
            r_squared_var.set(0.6)
            cv_threshold_var.set(0.6)
        elif template == "遮挡环境":
            # 遮挡环境：建筑物、树木等遮挡，多径严重，信号质量差，需要宽松参数
            code_threshold_var.set(10.0)
            phase_threshold_var.set(4.0)
            doppler_threshold_var.set(5.0)
            threshold_var.set(5.0)
            r_squared_var.set(0.5)
            cv_threshold_var.set(0.5)
        # "自定义"不做任何更改，保持用户设定值

    template_combo.bind('<<ComboboxSelected>>', apply_template)

    # 智能推荐按钮
    recommend_frame = ttk.Frame(template_frame)
    recommend_frame.pack(fill=tk.X, pady=2)

    def smart_recommend():
        """基于文件内容智能推荐参数"""
        if not phone_file_var.get():
            messagebox.showwarning("警告", "请先选择RINEX文件")
            return
        
        try:
            # 分析文件
            analyzer = GNSSAnalyzer()
            file_path = phone_file_var.get()
            data = analyzer.read_rinex_obs(file_path)
            
            # 评估数据质量
            total_sats = len(analyzer.observations_meters)
            total_epochs = len(data.get('epochs', []))
            
            # 统计观测值质量
            valid_obs_ratio = 0
            total_possible = 0
            total_valid = 0
            
            for sat_id, freq_data in analyzer.observations_meters.items():
                for freq, obs_data in freq_data.items():
                    if 'code' in obs_data:
                        code_values = obs_data['code']
                        total_possible += len(code_values)
                        total_valid += sum(1 for v in code_values if v is not None)
            
            if total_possible > 0:
                valid_obs_ratio = total_valid / total_possible
            
            # 基于统计结果推荐参数
            if valid_obs_ratio < 0.75:
                # 数据质量差，推荐遮挡环境参数
                template_var.set("遮挡环境")
                recommended_env = "遮挡环境"
            else:
                # 数据质量好，推荐开阔环境参数
                template_var.set("开阔环境")
                recommended_env = "开阔环境"
            
            # 应用推荐的模板
            apply_template()
            
            # 显示推荐结果
            messagebox.showinfo("智能推荐", 
                f"手机GNSS文件分析结果:\n"
                f"- 总卫星数: {total_sats}\n"
                f"- 总历元数: {total_epochs}\n"
                f"- 观测值完整率: {valid_obs_ratio:.1%}\n\n"
                f"推荐参数模板: {recommended_env}\n"
                f"建议: {'信号环境较差，适用遮挡环境参数' if valid_obs_ratio < 0.75 else '信号环境较好，适用开阔环境参数'}")
                
        except Exception as e:
            messagebox.showerror("错误", f"文件分析失败:\n{str(e)}")

    ttk.Button(recommend_frame, text="智能推荐参数", command=smart_recommend).pack(side=tk.LEFT, padx=(10, 10))

    # 参数保存/加载
    def save_params():
        """保存当前参数配置"""
        try:
            config = {
                'code_threshold': code_threshold_var.get(),
                'phase_threshold': phase_threshold_var.get(),
                'doppler_threshold': doppler_threshold_var.get(),
                'cmc_threshold': threshold_var.get(),
                'r_squared': r_squared_var.get(),
                'cv_threshold': cv_threshold_var.get(),
                'enable_doppler': doppler_enable_var.get(),
                'enable_cci': cci_enable_var.get(),
                'enable_isb': isb_enable_var.get(),
                'phone_only': phone_only_var.get()
            }
            
            import json
            filename = filedialog.asksaveasfilename(
                title="保存参数配置",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("成功", f"参数配置已保存到:\n{filename}")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败:\n{str(e)}")

    def load_params():
        """加载参数配置"""
        try:
            filename = filedialog.askopenfilename(
                title="加载参数配置",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            if filename:
                import json
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 应用配置
                code_threshold_var.set(config.get('code_threshold', 10.0))
                phase_threshold_var.set(config.get('phase_threshold', 1.5))
                doppler_threshold_var.set(config.get('doppler_threshold', 5.0))
                threshold_var.set(config.get('cmc_threshold', 4.0))
                r_squared_var.set(config.get('r_squared', 0.5))
                cv_threshold_var.set(config.get('cv_threshold', 0.6))
                doppler_enable_var.set(config.get('enable_doppler', False))
                cci_enable_var.set(config.get('enable_cci', True))
                isb_enable_var.set(config.get('enable_isb', True))
                phone_only_var.set(config.get('phone_only', False))
                
                template_var.set("自定义")
                messagebox.showinfo("成功", f"参数配置已从以下文件加载:\n{filename}")
        except Exception as e:
            messagebox.showerror("错误", f"加载配置失败:\n{str(e)}")

    config_buttons = ttk.Frame(recommend_frame)
    config_buttons.pack(side=tk.RIGHT)
    ttk.Button(config_buttons, text="保存配置", command=save_params).pack(side=tk.LEFT, padx=2)
    ttk.Button(config_buttons, text="加载配置", command=load_params).pack(side=tk.LEFT, padx=2)

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

    # 多普勒预测相位处理选项（第一行）
    doppler_frame = ttk.Frame(option_frame)
    doppler_frame.pack(fill=tk.X, pady=2)

    doppler_enable_var = tk.BooleanVar(value=False)
    doppler_enable_checkbox = ttk.Checkbutton(doppler_frame, text="启用多普勒预测相位",
                                              variable=doppler_enable_var)
    doppler_enable_checkbox.pack(side=tk.LEFT)

    ttk.Label(doppler_frame, text="(基于多普勒观测值预测并填补缺失的载波相位观测值)").pack(side=tk.LEFT, padx=(10, 0))

    # 码相不一致性处理选项（第二行）
    cci_frame_option = ttk.Frame(option_frame)
    cci_frame_option.pack(fill=tk.X, pady=2)

    cci_enable_var = tk.BooleanVar(value=True)
    cci_enable_checkbox = ttk.Checkbutton(cci_frame_option, text="启用码相不一致性(CCI)处理",
                                          variable=cci_enable_var)
    cci_enable_checkbox.pack(side=tk.LEFT)

    ttk.Label(cci_frame_option, text="(需要接收机文件作为基准，校正载波相位观测值)").pack(side=tk.LEFT, padx=(10, 0))

    # ISB处理选项（第三行）
    isb_frame_option = ttk.Frame(option_frame)
    isb_frame_option.pack(fill=tk.X, pady=2)

    isb_enable_var = tk.BooleanVar(value=True)
    isb_enable_checkbox = ttk.Checkbutton(isb_frame_option, text="启用ISB处理",
                                          variable=isb_enable_var)
    isb_enable_checkbox.pack(side=tk.LEFT)

    ttk.Label(isb_frame_option, text="(需要接收机文件作为基准，校正BDS系统间偏差)").pack(side=tk.LEFT, padx=(10, 0))

    # 码相不一致性处理
    cci_frame = ttk.LabelFrame(param_frame, text="码相不一致性处理", padding="5")
    cci_frame.pack(fill=tk.X, pady=5)

    # R方阈值和CV值阈值设置（同一行）
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

    def start_bds_analysis_only():
        """只进行BDS2/3 ISB分析，直接读取原始文件"""
        if not phone_file_var.get():
            tk.messagebox.showerror("错误", "请先选择手机RINEX文件")
            return

        if not receiver_file_var.get():
            tk.messagebox.showerror("错误", "BDS2/3 ISB分析需要接收机RINEX文件作为基准站")
            return

        # 禁用按钮
        start_btn.config(state='disabled')
        select_btn.config(state='disabled')
        bds_only_btn.config(state='disabled')

        def bds_analysis_process():
            try:
                analyzer = GNSSAnalyzer()

                # 设置文件路径
                file_path = phone_file_var.get()
                analyzer.input_file_path = file_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(file_path), "analysis_results")

                # 更新状态
                status_var.set("正在读取原始RINEX文件...")
                progress_var.set(10)
                cleaning_window.update_idletasks()

                # 直接读取原始文件
                data = analyzer.read_rinex_obs(file_path)

                # 进行BDS2/3 ISB分析（必须有接收机文件）
                status_var.set("正在进行BDS2/3 ISB分析...")
                progress_var.set(50)
                cleaning_window.update_idletasks()

                isb_results = analyzer.perform_complete_isb_analysis(receiver_file_var.get())
                print("BDS2/3 ISB分析完成")

                # 完成
                status_var.set("BDS2/3 ISB分析完成!")
                progress_var.set(100)
                cleaning_window.update_idletasks()

                # 显示结果
                phone_file_name = os.path.basename(file_path)
                phone_file_name_no_ext = os.path.splitext(phone_file_name)[0]
                phone_result_dir = os.path.join("results", phone_file_name_no_ext)
                message = f"BDS2/3 ISB分析完成！\n结果保存在：{phone_result_dir}"

                # 显示ISB分析结果
                if isb_results:
                    isb_mean = isb_results.get('isb_mean', 0)
                    isb_std = isb_results.get('isb_std', 0)
                    message += f"\n\nISB分析结果:"
                    message += f"\nISB均值: {isb_mean:.3f} m"
                    message += f"\nISB标准差: {isb_std:.3f} m"

                    if 'corrected_rinex_path' in isb_results:
                        isb_corrected_file = isb_results['corrected_rinex_path']
                        message += f"\nISB校正后的RINEX文件：{isb_corrected_file}"

                # 在主线程中显示消息框
                parent.after(0, lambda: tk.messagebox.showinfo("完成", message))

            except Exception as e:
                # 在主线程中显示错误消息
                parent.after(0, lambda: tk.messagebox.showerror("错误", f"BDS2/3 ISB分析过程中出现错误：\n{str(e)}"))
                status_var.set("分析失败")
            finally:
                # 在主线程中恢复按钮
                parent.after(0, lambda: start_btn.config(state='normal'))
                parent.after(0, lambda: select_btn.config(state='normal'))
                parent.after(0, lambda: bds_only_btn.config(state='normal'))

        # 在新线程中执行
        import threading
        thread = threading.Thread(target=bds_analysis_process)
        thread.daemon = True
        thread.start()

    def start_cleaning():
        if not file_var.get():
            tk.messagebox.showerror("错误", "请先选择数据文件")
            return

        # 验证可选功能的选择
        if cci_enable_var.get() and not receiver_file_var.get():
            tk.messagebox.showerror("错误", "启用码相不一致性(CCI)处理需要选择接收机RINEX文件作为基准")
            return
        
        if isb_enable_var.get() and not receiver_file_var.get():
            tk.messagebox.showerror("错误", "启用ISB处理需要选择接收机RINEX文件作为基准")
            return

        # 禁用按钮
        start_btn.config(state='disabled')
        select_btn.config(state='disabled')
        bds_only_btn.config(state='disabled')

        def cleaning_process():
            try:
                analyzer = GNSSAnalyzer()

                # 设置文件路径
                file_path = file_var.get()
                analyzer.input_file_path = file_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(file_path), "analysis_results")

                # 设置R方阈值
                analyzer.r_squared_threshold = r_squared_var.get()

                # 设置CV值阈值
                analyzer.cv_threshold = cv_threshold_var.get()

                # 设置历元间双差最大阈值
                analyzer.max_threshold_limits = {
                    'code': code_threshold_var.get(),
                    'phase': phase_threshold_var.get(),
                    'doppler': doppler_threshold_var.get()
                }

                # 设置手机独有卫星分析
                analyzer.enable_phone_only_analysis = phone_only_var.get()

                # 获取用户选择
                enable_doppler = doppler_enable_var.get()
                enable_cci = cci_enable_var.get()
                enable_isb = isb_enable_var.get()

                # 设置是否启用CCI处理
                analyzer.enable_cci_processing = enable_cci

                # 更新状态
                status_var.set("正在读取RINEX文件...")
                progress_var.set(5)
                cleaning_window.update_idletasks()

                # 读取文件
                data = analyzer.read_rinex_obs(file_path)

                # 步骤1: 多普勒预测相位（如果启用）
                doppler_results = None
                if enable_doppler:
                    status_var.set("正在进行多普勒预测相位...")
                    progress_var.set(15)
                    cleaning_window.update_idletasks()

                    try:
                        # 执行多普勒预测相位
                        doppler_results = analyzer.doppler_phase_prediction(data)
                        
                        # 生成修补后的RINEX文件
                        corrected_rinex_path = analyzer.generate_doppler_corrected_rinex_file()
                        
                        # 保存详细日志
                        log_path = analyzer.save_doppler_correction_log()
                        
                        print("多普勒预测相位完成")
                    except Exception as e:
                        print(f"多普勒预测相位失败: {e}")
                        # 继续执行其他步骤，不中断整个流程

                # 步骤2: 码相不一致性建模和校正（如果启用且提供了接收机文件）
                cci_results = None
                if enable_cci and receiver_file_var.get():
                    status_var.set("正在进行码相不一致性建模和校正...")
                    progress_var.set(25)
                    cleaning_window.update_idletasks()

                    try:
                        # 如果进行了多普勒预测，使用修补后的文件作为输入
                        input_file = file_path
                        if enable_doppler and doppler_results and analyzer.results.get('doppler_rinex_generation'):
                            doppler_corrected_path = analyzer.results['doppler_rinex_generation']['output_path']
                            if os.path.exists(doppler_corrected_path):
                                # 重新读取修补后的文件
                                analyzer.input_file_path = doppler_corrected_path
                                data = analyzer.read_rinex_obs(doppler_corrected_path)
                                print(f"使用多普勒修补后的文件进行CCI处理: {os.path.basename(doppler_corrected_path)}")

                        # 执行码相不一致性建模和校正
                        cci_results = analyzer.perform_code_phase_inconsistency_modeling(
                            receiver_rinex_path=receiver_file_var.get()
                        )
                        print("码相不一致性建模和校正完成")
                    except Exception as e:
                        print(f"码相不一致性建模和校正失败: {e}")
                        # 继续执行其他步骤，不中断整个流程

                # 步骤3: 计算伪距相位差值
                status_var.set("正在计算伪距相位差值...")
                progress_var.set(45)
                cleaning_window.update_idletasks()

                # 确保使用正确的数据文件计算差值
                current_input_file = file_path  # 默认使用原始文件
                if enable_cci and receiver_file_var.get() and cci_results:
                    # 如果CCI处理完成，使用CCI处理后的数据
                    current_input_file = analyzer.input_file_path
                    print(f"CCI处理完成，使用处理后的文件: {os.path.basename(current_input_file)}")
                elif enable_doppler and doppler_results and analyzer.results.get('doppler_rinex_generation'):
                    # 否则如果多普勒预测完成，确保使用预测后的文件
                    doppler_corrected_path = analyzer.results['doppler_rinex_generation']['output_path']
                    if os.path.exists(doppler_corrected_path):
                        current_input_file = doppler_corrected_path
                        analyzer.input_file_path = doppler_corrected_path
                        data = analyzer.read_rinex_obs(doppler_corrected_path)
                        print(f"切换到多普勒修补后的文件: {os.path.basename(doppler_corrected_path)}")
                
                # 确保使用正确的文件进行差值计算
                if current_input_file != file_path:
                    # 当切换到处理后的文件时
                    if analyzer.input_file_path != current_input_file:
                        analyzer.input_file_path = current_input_file
                        data = analyzer.read_rinex_obs(current_input_file)
                        print(f"使用文件进行差值计算: {os.path.basename(current_input_file)}")

                code_phase_diffs = analyzer.calculate_code_phase_differences(data)

                # 步骤4: 第一阶段剔除（基于CMC变化阈值）
                status_var.set("正在执行第一阶段剔除...")
                progress_var.set(65)
                cleaning_window.update_idletasks()

                cleaned_file_path = analyzer.remove_code_phase_outliers(data, threshold_var.get())
                
                # 为第二阶段剔除准备正确的输入文件
                if cleaned_file_path and os.path.exists(cleaned_file_path):
                    # 使用第一阶段剔除后的文件作为第二阶段的输入
                    analyzer.input_file_path = cleaned_file_path
                    # 重新读取剔除后的文件数据
                    data = analyzer.read_rinex_obs(cleaned_file_path)
                    print(f"切换到第一阶段剔除后的文件: {os.path.basename(cleaned_file_path)}")

                # 步骤5: 计算历元间双差
                status_var.set("正在计算历元间双差...")
                progress_var.set(75)
                cleaning_window.update_idletasks()

                double_diffs = analyzer.calculate_epoch_double_differences()
                triple_errors = analyzer.calculate_triple_median_error(double_diffs)

                # 步骤6: 第二阶段剔除（基于双差）
                status_var.set("正在执行第二阶段剔除...")
                progress_var.set(85)
                cleaning_window.update_idletasks()

                analyzer.remove_outliers_and_save(double_diffs, triple_errors)

                # 步骤7: ISB分析（如果启用且提供了接收机文件）
                isb_results = None
                if enable_isb and receiver_file_var.get():
                    try:
                        status_var.set("正在进行ISB分析...")
                        progress_var.set(90)
                        cleaning_window.update_idletasks()

                        isb_results = analyzer.perform_complete_isb_analysis(receiver_file_var.get())
                        print("ISB分析完成")
                    except Exception as e:
                        print(f"ISB分析失败: {e}")
                        # 继续执行其他步骤，不中断整个流程

                # 完成
                status_var.set("处理完成!")
                progress_var.set(100)
                cleaning_window.update_idletasks()

                # 显示结果
                phone_file_name = os.path.basename(file_path)
                phone_file_name_no_ext = os.path.splitext(phone_file_name)[0]
                phone_result_dir = os.path.join("results", phone_file_name_no_ext)
                message = f"数据预处理完成！\n结果保存在：{phone_result_dir}"

                # 显示处理步骤完成情况
                message += "\n\n处理步骤完成情况："
                step_num = 1
                
                # 步骤1: 多普勒预测相位
                if enable_doppler and doppler_results:
                    message += f"\n{step_num}. ✓ 多普勒预测相位"
                    step_num += 1
                
                # 步骤2: 码相不一致性建模和校正
                if enable_cci and receiver_file_var.get() and cci_results:
                    message += f"\n{step_num}. ✓ 码相不一致性建模和校正"
                    step_num += 1
                
                # 步骤3: 伪距相位差值计算
                message += f"\n{step_num}. ✓ 伪距相位差值计算"
                step_num += 1
                
                # 步骤4: 第一阶段剔除（CMC变化阈值）
                message += f"\n{step_num}. ✓ 第一阶段剔除（CMC变化阈值）"
                step_num += 1
                
                # 步骤5: 历元间双差计算
                message += f"\n{step_num}. ✓ 历元间双差计算"
                step_num += 1
                
                # 步骤6: 第二阶段剔除（双差阈值）
                message += f"\n{step_num}. ✓ 第二阶段剔除（双差阈值）"
                step_num += 1
                
                # 步骤7: ISB分析
                if enable_isb and receiver_file_var.get() and isb_results:
                    message += f"\n{step_num}. ✓ ISB分析"
                    step_num += 1

                # 显示多普勒预测相位结果
                if enable_doppler and doppler_results:
                    total_missing = doppler_results.get('total_missing', 0)
                    total_predicted = doppler_results.get('total_predicted', 0)
                    success_rate = (total_predicted / total_missing * 100) if total_missing > 0 else 0
                    
                    message += f"\n\n多普勒预测相位已完成"
                    message += f"\n总缺失相位观测值: {total_missing} 个"
                    message += f"\n成功预测修补: {total_predicted} 个"
                    message += f"\n修补成功率: {success_rate:.2f}%"
                    
                    if analyzer.results.get('doppler_rinex_generation'):
                        doppler_corrected_path = analyzer.results['doppler_rinex_generation']['output_path']
                        message += f"\n修补后的RINEX文件：{doppler_corrected_path}"

                if enable_cci and receiver_file_var.get() and cci_results:
                    # 显示码相不一致性建模和校正的实际文件路径
                    corrected_file_path = cci_results.get('corrected_rinex_path', '')
                    if corrected_file_path:
                        message += f"\n\n码相不一致性建模和校正已完成\n校正后的RINEX文件：{corrected_file_path}"
                    else:
                        message += "\n\n码相不一致性建模和校正已完成"

                if enable_isb and isb_results:
                    # 显示ISB分析结果
                    isb_mean = isb_results.get('isb_mean', 0)
                    isb_std = isb_results.get('isb_std', 0)
                    message += f"\n\nISB分析已完成\nISB均值: {isb_mean:.3f} m\nISB标准差: {isb_std:.3f} m"

                    if 'corrected_rinex_path' in isb_results:
                        isb_corrected_file = isb_results['corrected_rinex_path']
                        message += f"\nISB校正后的RINEX文件：{isb_corrected_file}"

                    # 添加报告和日志文件信息
                    message += f"\n\nISB分析报告和日志文件已生成在：\n{analyzer.current_result_dir}"

                # 在主线程中显示消息框
                parent.after(0, lambda: tk.messagebox.showinfo("完成", message))

            except Exception as e:
                # 捕获异常信息
                error_msg = str(e)
                # 在主线程中显示错误消息
                parent.after(0, lambda: tk.messagebox.showerror("错误", f"处理过程中出现错误：\n{error_msg}"))
                status_var.set("处理失败")
            finally:
                # 在主线程中恢复按钮
                parent.after(0, lambda: start_btn.config(state='normal'))
                parent.after(0, lambda: select_btn.config(state='normal'))
                parent.after(0, lambda: bds_only_btn.config(state='normal'))

        # 在新线程中执行
        import threading
        thread = threading.Thread(target=cleaning_process)
        thread.daemon = True
        thread.start()

    start_btn = ttk.Button(button_frame, text="开始预处理", command=start_cleaning)
    start_btn.pack(side=tk.LEFT, padx=10)

    bds_only_btn = ttk.Button(button_frame, text="BDS2/3 ISB分析", command=start_bds_analysis_only, state='disabled')
    bds_only_btn.pack(side=tk.LEFT, padx=10)

    select_btn = ttk.Button(button_frame, text="选择文件", command=select_phone_file)
    select_btn.pack(side=tk.LEFT, padx=10)

    def close_cleaning_window():
        """关闭剔除窗口时的清理函数"""
        try:
            # 关闭所有Matplotlib图表窗口
            plt.close('all')
        except Exception as e:
            print(f"关闭剔除窗口时清理图表出错: {str(e)}")
        finally:
            cleaning_window.destroy()

    ttk.Button(button_frame, text="关闭",
               command=close_cleaning_window).pack(side=tk.LEFT, padx=10)


def show_charts_window(parent):
    """显示图表功能窗口"""
    charts_window = tk.Toplevel(parent)
    charts_window.title("图表生成")
    charts_window.geometry("700x800")
    charts_window.resizable(True, True)
    charts_window.transient(parent)
    charts_window.grab_set()

    # 居中显示窗口
    center_window(charts_window, 800, 850)

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
            # 自动加载预定义的卫星和频率信息
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

    # 接收机RINEX文件选择（用于接收机CMC）
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
            # 自动加载接收机RINEX的卫星与频率到下拉框
            try:
                load_receiver_satellite_info()
            except Exception as e:
                messagebox.showwarning("警告", f"加载接收机RINEX信息失败:\n{str(e)}")

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

    # 保存选项已移除 - 图表窗口自带保存功能，操作更方便

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

    def load_satellite_info():
        """从输入文件中加载实际的卫星和频率信息"""
        try:
            if not file_var.get():
                return

            # 创建分析器并读取文件
            analyzer = GNSSAnalyzer()
            file_path = file_var.get()
            analyzer.input_file_path = file_path

            # 更新状态
            status_var.set("正在读取文件...")
            charts_window.update_idletasks()

            # 读取文件
            data = analyzer.read_rinex_obs(file_path)

            # 从实际数据中获取卫星系统
            if hasattr(analyzer, 'observations_meters') and analyzer.observations_meters:
                # 获取所有可用的卫星PRN
                available_satellites = list(analyzer.observations_meters.keys())

                # 从PRN中提取卫星系统
                satellite_systems = set()
                for sat in available_satellites:
                    if sat and len(sat) > 0:
                        satellite_systems.add(sat[0])  # 第一个字符是卫星系统标识

                satellite_systems = list(satellite_systems)

                # 设置卫星系统列表
                sat_system_combo['values'] = satellite_systems
                if satellite_systems:
                    sat_system_var.set(satellite_systems[0])  # 默认选择第一个卫星系统

                # 设置卫星PRN列表（从实际数据中获取）
                sat_prn_combo['values'] = available_satellites
                if available_satellites:
                    sat_prn_var.set(available_satellites[0])  # 默认选择第一个PRN

                # 根据选择的卫星PRN设置对应的频率列表
                if available_satellites:
                    selected_prn = available_satellites[0]
                    if selected_prn in analyzer.observations_meters:
                        frequencies = list(analyzer.observations_meters[selected_prn].keys())
                        freq_combo['values'] = frequencies
                        if frequencies:
                            freq_var.set(frequencies[0])  # 默认选择第一个频率

                status_var.set("文件读取完成")
            else:
                messagebox.showwarning("警告", "文件中未找到观测数据")
                status_var.set("未找到数据")

        except Exception as e:
            messagebox.showerror("错误", f"加载卫星和频率信息失败：\n{str(e)}")
            status_var.set("加载失败")

    def load_receiver_satellite_info():
        """从接收机RINEX文件加载卫星与频率信息，填充PRN与频率下拉框"""
        try:
            if not rx_file_var.get():
                return
            analyzer = GNSSAnalyzer()
            rx_path = rx_file_var.get()
            analyzer.read_receiver_rinex_obs(rx_path)

            if analyzer.receiver_observations:
                available_satellites = list(analyzer.receiver_observations.keys())

                # 推断卫星系统集合
                satellite_systems = sorted({sid[0] for sid in available_satellites if sid})
                sat_system_combo['values'] = satellite_systems
                if satellite_systems:
                    sat_system_var.set(satellite_systems[0])

                # 所有卫星列表
                sat_prn_combo['values'] = available_satellites
                if available_satellites:
                    sat_prn_var.set(available_satellites[0])

                # 对应频率列表
                first = sat_prn_var.get()
                if first in analyzer.receiver_observations:
                    freqs = list(analyzer.receiver_observations[first].keys())
                    freq_combo['values'] = freqs
                    if freqs:
                        freq_var.set(freqs[0])
                status_var.set("接收机文件加载完成")
            else:
                messagebox.showwarning("警告", "接收机RINEX中未找到观测数据")
                status_var.set("未找到数据")
        except Exception as e:
            messagebox.showerror("错误", f"加载接收机RINEX信息失败：\n{str(e)}")
            status_var.set("加载失败")

    def get_prn_list_for_system(system):
        """根据卫星系统获取对应的PRN列表"""
        prn_mapping = {
            'G': [f'G{i:02d}' for i in range(1, 33)],  # G01-G32
            'R': [f'R{i:02d}' for i in range(2, 23)],  # R02-R22
            'E': [f'E{i:02d}' for i in range(2, 37)],  # E02-E36
            'C': [f'C{i:02d}' for i in range(1, 63)],  # C01-C62
            'J': [f'J{i:02d}' for i in range(2, 8)]  # J02-J07
        }
        return prn_mapping.get(system, [])

    def on_satellite_system_change(*args):
        """卫星系统改变时更新PRN和频率列表"""
        if sat_system_var.get():
            try:
                # 获取当前选择的卫星系统
                selected_system = sat_system_var.get()

                # 从实际数据中筛选属于该系统的卫星PRN
                # 需要重新读取文件来获取数据
                if rx_file_var.get():
                    # 优先处理接收机RINEX
                    analyzer = GNSSAnalyzer()
                    analyzer.read_receiver_rinex_obs(rx_file_var.get())
                    if analyzer.receiver_observations:
                        available_satellites = list(analyzer.receiver_observations.keys())
                        system_satellites = [sat for sat in available_satellites if sat.startswith(selected_system)]

                        # 更新PRN列表
                        sat_prn_combo['values'] = system_satellites
                        if system_satellites:
                            sat_prn_var.set(system_satellites[0])

                            # 更新频率列表
                            selected_prn = system_satellites[0]
                            if selected_prn in analyzer.receiver_observations:
                                frequencies = list(analyzer.receiver_observations[selected_prn].keys())
                                freq_combo['values'] = frequencies
                                if frequencies:
                                    freq_var.set(frequencies[0])
                        return

                if file_var.get():
                    analyzer = GNSSAnalyzer()
                    file_path = file_var.get()
                    analyzer.input_file_path = file_path
                    data = analyzer.read_rinex_obs(file_path)

                    if hasattr(analyzer, 'observations_meters') and analyzer.observations_meters:
                        available_satellites = list(analyzer.observations_meters.keys())
                        system_satellites = [sat for sat in available_satellites if sat.startswith(selected_system)]

                        # 更新PRN列表
                        sat_prn_combo['values'] = system_satellites
                        if system_satellites:
                            sat_prn_var.set(system_satellites[0])  # 选择第一个PRN

                            # 更新频率列表
                            selected_prn = system_satellites[0]
                            if selected_prn in analyzer.observations_meters:
                                frequencies = list(analyzer.observations_meters[selected_prn].keys())
                                freq_combo['values'] = frequencies
                                if frequencies:
                                    freq_var.set(frequencies[0])  # 选择第一个频率
            except Exception as e:
                print(f"更新PRN和频率列表时出错: {e}")

    def on_satellite_prn_change(*args):
        """卫星PRN改变时更新频率列表"""
        if sat_prn_var.get():
            try:
                selected_prn = sat_prn_var.get()

                # 从实际数据中获取该卫星的频率列表
                # 需要重新读取文件来获取数据
                if rx_file_var.get():
                    analyzer = GNSSAnalyzer()
                    analyzer.read_receiver_rinex_obs(rx_file_var.get())
                    if analyzer.receiver_observations:
                        if selected_prn in analyzer.receiver_observations:
                            frequencies = list(analyzer.receiver_observations[selected_prn].keys())
                            freq_combo['values'] = frequencies
                            if frequencies:
                                freq_var.set(frequencies[0])
                        return

                if file_var.get():
                    analyzer = GNSSAnalyzer()
                    file_path = file_var.get()
                    analyzer.input_file_path = file_path
                    data = analyzer.read_rinex_obs(file_path)

                    if hasattr(analyzer, 'observations_meters') and analyzer.observations_meters:
                        if selected_prn in analyzer.observations_meters:
                            frequencies = list(analyzer.observations_meters[selected_prn].keys())
                            freq_combo['values'] = frequencies
                            if frequencies:
                                freq_var.set(frequencies[0])  # 选择第一个频率
            except Exception as e:
                print(f"更新频率列表时出错: {e}")

    # 绑定卫星系统选择变化事件
    sat_system_var.trace('w', on_satellite_system_change)

    # 绑定卫星PRN选择变化事件
    sat_prn_var.trace('w', on_satellite_prn_change)

    def generate_chart():
        """生成选中的图表"""
        selected = chart_var.get()
        # 按图表类型判断所需输入文件
        if selected == "receiver_cmc":
            if not rx_file_var.get():
                messagebox.showerror("错误", "请先选择接收机RINEX文件")
                return
        else:
            if not file_var.get():
                messagebox.showerror("错误", "请先选择数据文件")
                return

        # 禁用按钮
        generate_btn.config(state='disabled')

        def chart_process():
            try:
                analyzer = GNSSAnalyzer()

                # 设置文件路径
                file_path = file_var.get()
                analyzer.input_file_path = file_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(file_path), "analysis_results")

                # 更新状态
                status_var.set("正在读取RINEX文件...")
                progress_var.set(20)
                charts_window.update_idletasks()

                # 读取文件（接收机CMC不需要手机文件解析）
                data = None
                if selected != "receiver_cmc":
                    data = analyzer.read_rinex_obs(file_path)

                # 检查选择的卫星PRN和频率是否在数据中存在（接收机CMC走接收机数据容器）
                sat_prn = sat_prn_var.get()
                freq = freq_var.get()

                if selected != "receiver_cmc":
                    if sat_prn not in analyzer.observations_meters:
                        messagebox.showerror("错误", f"在输入文件中未找到卫星 {sat_prn} 的观测数据")
                        status_var.set("生成失败")
                        return
                    if freq not in analyzer.observations_meters[sat_prn]:
                        messagebox.showerror("错误", f"卫星 {sat_prn} 在输入文件中未找到频率 {freq} 的观测数据")
                        status_var.set("生成失败")
                        return
                    obs_data = analyzer.observations_meters[sat_prn][freq]
                    if not obs_data or len(obs_data) < 2:
                        messagebox.showerror("错误", f"卫星 {sat_prn} 频率 {freq} 的观测数据不足，至少需要2个历元的数据")
                        status_var.set("生成失败")
                        return
                    valid_obs = [obs for obs in obs_data if obs is not None]
                    if len(valid_obs) < 2:
                        messagebox.showerror("错误",
                                             f"卫星 {sat_prn} 频率 {freq} 的有效观测数据不足，至少需要2个有效观测值")
                        status_var.set("生成失败")
                        return
                else:
                    # 接收机CMC路径：读取接收机并立即dump调试
                    analyzer.read_receiver_rinex_obs(rx_file_var.get())
                    analyzer.dump_receiver_observations_debug()

                # 根据图表类型生成相应的图表
                chart_type = chart_var.get()

                status_var.set("正在生成图表...")
                progress_var.set(60)
                charts_window.update_idletasks()

                # 使用after方法在主线程中执行绘图操作
                def plot_on_main_thread():
                    try:
                        # 直接调用相应的绘图函数 - 总是显示图表，不自动保存
                        if chart_type == "raw_observations":
                            analyzer.plot_raw_observations(sat_prn, False)
                        elif chart_type == "derivatives":
                            # 计算观测值一阶差分
                            derivatives = analyzer.calculate_observable_derivatives(data)
                            analyzer.plot_observable_derivatives(derivatives, sat_prn, freq, False)
                        elif chart_type == "code_phase_diffs":
                            # 计算伪距相位差值之差
                            code_phase_diffs = analyzer.calculate_code_phase_differences(data)
                            analyzer.plot_code_phase_differences(code_phase_diffs, sat_prn, freq, False)
                        elif chart_type == "code_phase_diff_raw":
                            # 计算伪距相位原始差值
                            code_phase_diffs = analyzer.calculate_code_phase_differences(data)
                            analyzer.plot_code_phase_raw_differences(code_phase_diffs, sat_prn, freq, False)
                        elif chart_type == "phase_pred_errors":
                            # 计算相位预测误差
                            phase_pred_errors = analyzer.calculate_phase_prediction_errors(data)
                            analyzer.plot_phase_prediction_errors(phase_pred_errors, sat_prn, freq, False)
                        elif chart_type == "double_differences":
                            # 计算历元间双差
                            double_diffs = analyzer.calculate_epoch_double_differences()
                            triple_errors = analyzer.calculate_triple_median_error(double_diffs)
                            analyzer.plot_double_differences(double_diffs, triple_errors, sat_prn, freq, False)
                        elif chart_type == "receiver_cmc":
                            # 读取接收机RINEX并绘制CMC
                            if not rx_file_var.get():
                                messagebox.showerror("错误", "请先选择接收机RINEX文件")
                                return
                            analyzer.read_receiver_rinex_obs(rx_file_var.get())
                            cmc_results = analyzer.calculate_receiver_cmc()
                            if sat_prn not in cmc_results or freq not in cmc_results.get(sat_prn, {}):
                                messagebox.showerror("错误",
                                                     f"接收机CMC无数据: {sat_prn} {freq}\n请确认RINEX包含所选卫星与频率")
                                return
                            # 轻量级绘制
                            vals = cmc_results[sat_prn][freq]['cmc_m']
                            epochs = list(range(1, len(vals) + 1))
                            plt.figure(figsize=(12, 6))
                            plt.plot(epochs, vals, 'b-', label='CMC (m)')
                            plt.axhline(0, color='k', linestyle='--', alpha=0.4)
                            plt.xlabel('历元')
                            plt.ylabel('CMC (m)')
                            plt.title(f'{sat_prn}-{freq} 接收机CMC')
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            plt.tight_layout()
                            plt.show()

                        elif chart_type == "isb_analysis":
                            # ISB分析（需要接收机文件）
                            if not rx_file_var.get():
                                messagebox.showerror("错误", "ISB分析需要选择接收机RINEX文件")
                                return

                            try:
                                isb_results = analyzer.perform_complete_isb_analysis(rx_file_var.get())
                                analyzer.plot_isb_analysis(isb_results, save=False)
                            except Exception as e:
                                messagebox.showerror("错误", f"ISB分析失败：\n{str(e)}")
                                return

                        # 完成
                        status_var.set("图表生成完成!")
                        progress_var.set(100)
                        charts_window.update_idletasks()
                    except Exception as e:
                        messagebox.showerror("错误", f"生成图表过程中出现错误：\n{str(e)}")
                        status_var.set("生成失败")
                    finally:
                        # 恢复按钮
                        generate_btn.config(state='normal')

                # 在主线程中执行绘图操作
                charts_window.after(100, plot_on_main_thread)

            except Exception as e:
                messagebox.showerror("错误", f"准备图表数据过程中出现错误：\n{str(e)}")
                status_var.set("生成失败")
                # 恢复按钮
                generate_btn.config(state='normal')

        # 在新线程中执行数据准备
        import threading
        thread = threading.Thread(target=chart_process)
        thread.daemon = True
        thread.start()

    generate_btn = ttk.Button(button_frame, text="生成图表", command=generate_chart)
    generate_btn.pack(side=tk.LEFT, padx=10)

    ttk.Button(button_frame, text="批量保存所有图表",
               command=lambda: batch_save_all_charts()).pack(side=tk.LEFT, padx=10)

    def batch_save_all_charts():
        """批量保存所有类型的图表"""
        # receiver_cmc 允许没有手机文件
        if chart_var.get() != "receiver_cmc" and not file_var.get():
            messagebox.showerror("错误", "请先选择数据文件")
            return

        # 禁用按钮
        batch_btn.config(state='disabled')

        def batch_process():
            """在后台线程中准备数据"""
            try:
                analyzer = GNSSAnalyzer()

                # 选择基准路径（接收机CMC走接收机文件，其它走手机文件）
                selected_type = chart_var.get()
                base_path = rx_file_var.get() if selected_type == "receiver_cmc" else file_var.get()
                if not base_path or not os.path.isfile(base_path):
                    charts_window.after(0, lambda: messagebox.showerror("错误", f"未找到文件: {base_path}"))
                    batch_btn.config(state='normal')
                    return
                analyzer.input_file_path = base_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(base_path), "analysis_results")
                os.makedirs(analyzer.current_result_dir, exist_ok=True)

                # 更新状态
                charts_window.after(0, lambda: status_var.set("正在读取RINEX文件..."))
                charts_window.after(0, lambda: progress_var.set(20))
                charts_window.after(0, lambda: charts_window.update_idletasks())

                # 读取文件（接收机CMC不依赖手机文件）
                data = None
                if selected_type != "receiver_cmc":
                    data = analyzer.read_rinex_obs(base_path)

                # 更新状态
                charts_window.after(0, lambda: status_var.set("正在计算分析数据..."))
                charts_window.after(0, lambda: progress_var.set(40))
                charts_window.after(0, lambda: charts_window.update_idletasks())

                # 计算所有必要的数据分析结果（非接收机CMC）
                if selected_type != "receiver_cmc":
                    derivatives = analyzer.calculate_observable_derivatives(data)
                    analyzer.results['observable_derivatives'] = derivatives
                    code_phase_diffs = analyzer.calculate_code_phase_differences(data)
                    analyzer.results['code_phase_diffs'] = code_phase_diffs
                    phase_pred_errors = analyzer.calculate_phase_prediction_errors(data)
                    analyzer.results['phase_prediction_errors'] = phase_pred_errors
                    analyzer.calculate_epoch_double_differences()
                    analyzer.calculate_triple_median_error(analyzer.results['double_differences'])
                else:
                    # 接收机CMC预先载入接收机RINEX
                    if not rx_file_var.get():
                        charts_window.after(0, lambda: messagebox.showerror("错误", "请先选择接收机RINEX文件"))
                        return
                    analyzer.read_receiver_rinex_obs(base_path)
                    analyzer.calculate_receiver_cmc()

                # 更新状态
                charts_window.after(0, lambda: status_var.set("正在生成所有图表..."))
                charts_window.after(0, lambda: progress_var.set(60))
                charts_window.after(0, lambda: charts_window.update_idletasks())

                # 在主线程中执行绘图操作
                def plot_on_main_thread():
                    try:
                        # 保存所有图表
                        analyzer.save_all_plots()

                        # 完成
                        status_var.set("批量保存完成!")
                        progress_var.set(100)
                        charts_window.update_idletasks()

                        # 显示结果
                        result_dir = analyzer.current_result_dir
                        messagebox.showinfo("完成", f"批量保存完成！\n结果保存在：{result_dir}")

                    except Exception as e:
                        messagebox.showerror("错误", f"生成图表过程中出现错误：\n{str(e)}")
                        status_var.set("生成失败")
                    finally:
                        # 恢复按钮
                        batch_btn.config(state='normal')

                # 在主线程中执行绘图操作
                charts_window.after(100, plot_on_main_thread)

            except Exception as e:
                messagebox.showerror("错误", f"准备图表数据过程中出现错误：\n{str(e)}")
                status_var.set("准备失败")
                # 恢复按钮
                batch_btn.config(state='normal')

        # 在新线程中执行数据准备
        import threading
        thread = threading.Thread(target=batch_process)
        thread.daemon = True
        thread.start()

    def batch_save_selected_chart_type():
        """批量保存选中的图表类型"""
        selected_chart_type = chart_var.get()
        if selected_chart_type != "receiver_cmc" and not file_var.get():
            messagebox.showerror("错误", "请先选择数据文件")
            return
        if selected_chart_type == "receiver_cmc" and not rx_file_var.get():
            messagebox.showerror("错误", "请先选择接收机RINEX文件")
            return

        # 获取用户选择的图表类型（已在前面拿到）
        if not selected_chart_type:
            messagebox.showerror("错误", "请先选择要保存的图表类型")
            return

        # 禁用按钮
        batch_btn.config(state='disabled')

        def batch_process():
            """在后台线程中准备数据"""
            try:
                analyzer = GNSSAnalyzer()

                # 选择基准路径（接收机CMC走接收机文件，其它走手机文件）
                base_path = rx_file_var.get() if selected_chart_type == "receiver_cmc" else file_var.get()
                if not base_path or not os.path.isfile(base_path):
                    charts_window.after(0, lambda: messagebox.showerror("错误", f"未找到文件: {base_path}"))
                    batch_btn.config(state='normal')
                    return
                analyzer.input_file_path = base_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(base_path), "analysis_results")
                os.makedirs(analyzer.current_result_dir, exist_ok=True)

                # 更新状态
                charts_window.after(0, lambda: status_var.set("正在读取RINEX文件..."))
                charts_window.after(0, lambda: progress_var.set(20))
                charts_window.after(0, lambda: charts_window.update_idletasks())

                # 读取文件（接收机CMC不依赖手机文件）
                data = None
                if selected_chart_type != "receiver_cmc":
                    data = analyzer.read_rinex_obs(base_path)

                # 更新状态
                charts_window.after(0, lambda: status_var.set(f"正在生成{selected_chart_type}图表..."))
                charts_window.after(0, lambda: progress_var.set(60))
                charts_window.after(0, lambda: charts_window.update_idletasks())

                # 准备数据
                if selected_chart_type == "raw_observations":
                    satellites = list(analyzer.observations_meters.keys())
                elif selected_chart_type == "derivatives":
                    derivatives = analyzer.calculate_observable_derivatives(data)
                    satellites = list(derivatives.keys())
                elif selected_chart_type == "code_phase_diff_raw":
                    differences = analyzer.calculate_code_phase_differences(data)
                    satellites = list(differences.keys())
                elif selected_chart_type == "code_phase_diffs":
                    differences = analyzer.calculate_code_phase_differences(data)
                    satellites = list(differences.keys())
                elif selected_chart_type == "phase_pred_errors":
                    errors = analyzer.calculate_phase_prediction_errors(data)
                    satellites = list(errors.keys())
                elif selected_chart_type == "receiver_cmc":
                    analyzer.read_receiver_rinex_obs(base_path)
                    cmc_results = analyzer.calculate_receiver_cmc()
                    satellites = list(cmc_results.keys())
                elif selected_chart_type == "double_differences":
                    analyzer.calculate_epoch_double_differences()
                    analyzer.calculate_triple_median_error(analyzer.results['double_differences'])
                    satellites = list(analyzer.observations_meters.keys())
                elif selected_chart_type == "isb_analysis":
                    # ISB分析需要接收机文件
                    if not rx_file_var.get():
                        messagebox.showerror("错误", "ISB分析需要选择接收机RINEX文件")
                        return
                    try:
                        isb_results = analyzer.perform_complete_isb_analysis(rx_file_var.get())
                        satellites = ["isb_analysis"]  # ISB分析不需要按卫星分别处理
                    except Exception as e:
                        messagebox.showerror("错误", f"ISB分析失败：\n{str(e)}")
                        return
                else:
                    satellites = []

                # 计算总图表数量
                total_charts = 0
                for sat_id in satellites:
                    if selected_chart_type == "raw_observations":
                        total_charts += len(analyzer.observations_meters[sat_id])
                    elif selected_chart_type in ["derivatives", "code_phase_diff_raw", "code_phase_diffs",
                                                 "phase_pred_errors"]:
                        if sat_id in locals().get(selected_chart_type.replace("_", ""), {}):
                            total_charts += len(locals()[selected_chart_type.replace("_", "")][sat_id])
                    elif selected_chart_type == "double_differences":
                        total_charts += len(analyzer.observations_meters[sat_id])
                    elif selected_chart_type == "isb_analysis":
                        total_charts = 1  # ISB分析只生成一组图表

                # 在主线程中执行绘图操作
                def plot_on_main_thread():
                    try:
                        saved_charts = 0

                        # 根据选择的图表类型保存对应的图表
                        if selected_chart_type == "raw_observations":
                            for sat_id in satellites:
                                for freq in analyzer.observations_meters[sat_id]:
                                    try:
                                        analyzer.plot_raw_observations(sat_id, save=True)
                                        saved_charts += 1
                                        # 更新进度
                                        progress = 60 + (saved_charts / total_charts) * 30
                                        progress_var.set(int(progress))
                                        charts_window.update_idletasks()
                                        # 强制清理内存
                                        plt.close('all')
                                        import gc
                                        gc.collect()
                                        # 添加短暂延迟，让系统有时间释放资源
                                        import time
                                        time.sleep(0.1)
                                    except Exception as e:
                                        print(f"保存{sat_id} {freq}原始观测值图表时出错: {str(e)}")
                                        continue

                        elif selected_chart_type == "derivatives":
                            for sat_id in satellites:
                                if sat_id in derivatives:
                                    for freq in derivatives[sat_id]:
                                        try:
                                            analyzer.plot_observable_derivatives(derivatives, sat_id, freq, save=True)
                                            saved_charts += 1
                                            progress = 60 + (saved_charts / total_charts) * 30
                                            progress_var.set(int(progress))
                                            charts_window.update_idletasks()
                                            plt.close('all')
                                            import gc
                                            gc.collect()
                                            import time
                                            time.sleep(0.1)
                                        except Exception as e:
                                            print(f"保存{sat_id} {freq}导数图表时出错: {str(e)}")
                                            continue

                        elif selected_chart_type == "code_phase_diff_raw":
                            for sat_id in satellites:
                                if sat_id in differences:
                                    for freq in differences[sat_id]:
                                        try:
                                            analyzer.plot_code_phase_raw_differences(differences, sat_id, freq,
                                                                                     save=True)
                                            saved_charts += 1
                                            progress = 60 + (saved_charts / total_charts) * 30
                                            progress_var.set(int(progress))
                                            charts_window.update_idletasks()
                                            plt.close('all')
                                            import gc
                                            gc.collect()
                                            import time
                                            time.sleep(0.1)
                                        except Exception as e:
                                            print(f"保存{sat_id} {freq}伪距相位原始差值图表时出错: {str(e)}")
                                            continue

                        elif selected_chart_type == "code_phase_diffs":
                            for sat_id in satellites:
                                if sat_id in differences:
                                    for freq in differences[sat_id]:
                                        try:
                                            analyzer.plot_code_phase_differences(differences, sat_id, freq, save=True)
                                            saved_charts += 1
                                            progress = 60 + (saved_charts / total_charts) * 30
                                            progress_var.set(int(progress))
                                            charts_window.update_idletasks()
                                            plt.close('all')
                                            import gc
                                            gc.collect()
                                            import time
                                            time.sleep(0.1)
                                        except Exception as e:
                                            print(f"保存{sat_id} {freq}伪距相位差值图表时出错: {str(e)}")
                                            continue

                        elif selected_chart_type == "phase_pred_errors":
                            for sat_id in satellites:
                                if sat_id in errors:
                                    for freq in errors[sat_id]:
                                        try:
                                            analyzer.plot_phase_prediction_errors(errors, sat_id, freq, save=True)
                                            saved_charts += 1
                                            progress = 60 + (saved_charts / total_charts) * 30
                                            progress_var.set(int(progress))
                                            charts_window.update_idletasks()
                                            plt.close('all')
                                            import gc
                                            gc.collect()
                                            import time
                                            time.sleep(0.1)
                                        except Exception as e:
                                            print(f"保存{sat_id} {freq}相位预测误差图表时出错: {str(e)}")
                                            continue

                        elif selected_chart_type == "double_differences":
                            for sat_id in satellites:
                                for freq in analyzer.observations_meters[sat_id]:
                                    try:
                                        analyzer.plot_double_differences(analyzer.results['double_differences'],
                                                                         analyzer.results['triple_median_errors'],
                                                                         sat_id, freq, save=True)
                                        saved_charts += 1
                                        progress = 60 + (saved_charts / total_charts) * 30
                                        progress_var.set(int(progress))
                                        charts_window.update_idletasks()
                                        plt.close('all')
                                        import gc
                                        gc.collect()
                                        import time
                                        time.sleep(0.1)
                                    except Exception as e:
                                        print(f"保存{sat_id} {freq}双差图表时出错: {str(e)}")
                                        continue

                        elif selected_chart_type == "receiver_cmc":
                            for sat_id in satellites:
                                for freq in cmc_results.get(sat_id, {}):
                                    try:
                                        vals = cmc_results[sat_id][freq]['cmc_m']
                                        epochs = list(range(1, len(vals) + 1))
                                        plt.figure(figsize=(12, 6))
                                        plt.plot(epochs, vals, 'b-', label='CMC (m)')
                                        plt.axhline(0, color='k', linestyle='--', alpha=0.4)
                                        plt.xlabel('历元')
                                        plt.ylabel('CMC (m)')
                                        plt.title(f'{sat_id}-{freq} 接收机CMC')
                                        plt.grid(True, alpha=0.3)
                                        plt.legend()
                                        plt.tight_layout()
                                        category_dir = os.path.join(analyzer.current_result_dir, 'receiver_cmc')
                                        os.makedirs(category_dir, exist_ok=True)
                                        out_path = os.path.join(category_dir, f"{sat_id}_{freq}_receiver_cmc.png")
                                        plt.savefig(out_path, dpi=300, bbox_inches='tight')
                                        plt.close()
                                        saved_charts += 1
                                        progress = 60 + (saved_charts / total_charts) * 30
                                        progress_var.set(int(progress))
                                        charts_window.update_idletasks()
                                        import gc
                                        gc.collect()
                                        import time
                                        time.sleep(0.05)
                                    except Exception as e:
                                        print(f"保存{sat_id} {freq} CMC图表时出错: {str(e)}")
                                        continue

                        elif selected_chart_type == "isb_analysis":
                            try:
                                analyzer.plot_isb_analysis(isb_results, save=True)
                                saved_charts += 1
                                progress = 60 + (saved_charts / total_charts) * 30
                                progress_var.set(int(progress))
                            except Exception as e:
                                print(f"保存ISB分析图表时出错: {str(e)}")
                                charts_window.update_idletasks()
                                plt.close('all')
                                import gc
                                gc.collect()
                                import time
                                time.sleep(0.1)

                        # 完成
                        status_var.set(f"{selected_chart_type}图表保存完成!")
                        progress_var.set(100)
                        charts_window.update_idletasks()

                        # 显示结果
                        result_dir = analyzer.current_result_dir
                        messagebox.showinfo("完成", f"{selected_chart_type}图表保存完成！\n结果保存在：{result_dir}")

                    except Exception as e:
                        messagebox.showerror("错误", f"生成图表过程中出现错误：\n{str(e)}")
                        status_var.set("生成失败")
                    finally:
                        # 恢复按钮
                        batch_btn.config(state='normal')

                # 在主线程中执行绘图操作
                charts_window.after(100, plot_on_main_thread)

            except Exception as e:
                messagebox.showerror("错误", f"准备图表数据过程中出现错误：\n{str(e)}")
                status_var.set("准备失败")
                # 恢复按钮
                batch_btn.config(state='normal')

        # 在新线程中执行数据准备
        import threading
        thread = threading.Thread(target=batch_process)
        thread.daemon = True
        thread.start()

    batch_btn = ttk.Button(button_frame, text="批量保存选中类型", command=batch_save_selected_chart_type)
    batch_btn.pack(side=tk.LEFT, padx=10)

    def close_charts_window():
        """关闭图表窗口时的清理函数"""
        try:
            # 关闭所有Matplotlib图表窗口
            plt.close('all')
        except Exception as e:
            print(f"关闭图表窗口时清理图表出错: {str(e)}")
        finally:
            charts_window.destroy()

    ttk.Button(button_frame, text="关闭", command=close_charts_window).pack(side=tk.LEFT, padx=10)

    # 初始化时加载预定义的卫星和频率信息
    load_satellite_info()


def show_chart_window(analyzer, data, chart_type, sat_prn, freq, auto_save):
    """显示图表窗口"""
    chart_window = tk.Toplevel()
    chart_window.title(f"图表显示 - {chart_type} - {sat_prn} - {freq}")
    chart_window.geometry("1000x700")
    chart_window.resizable(True, True)

    # 居中显示窗口
    center_window(chart_window, 1000, 700)

    # 主框架
    main_frame = ttk.Frame(chart_window, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 图表信息
    info_frame = ttk.LabelFrame(main_frame, text="图表信息", padding="10")
    info_frame.pack(fill=tk.X, pady=10)

    ttk.Label(info_frame, text=f"卫星PRN: {sat_prn}").pack(side=tk.LEFT, padx=(0, 20))
    ttk.Label(info_frame, text=f"频率: {freq}").pack(side=tk.LEFT, padx=(0, 20))
    ttk.Label(info_frame, text=f"图表类型: {chart_type}").pack(side=tk.LEFT, padx=(0, 20))

    # 图表显示区域
    chart_frame = ttk.LabelFrame(main_frame, text="图表显示", padding="10")
    chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    # 创建matplotlib图形
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 8))
    canvas = FigureCanvasTkAgg(fig, chart_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # 生成图表
    try:
        if chart_type == "raw_observations":
            generate_raw_observations_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "derivatives":
            generate_derivatives_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "code_phase_diffs":
            generate_code_phase_diffs_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "code_phase_diff_raw":
            generate_code_phase_raw_diffs_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "phase_pred_errors":
            generate_phase_pred_errors_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "double_differences":
            generate_double_differences_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "isb_analysis":
            # ISB分析图表
            generate_isb_analysis_plot(analyzer, data, fig)

        # 刷新画布
        canvas.draw()

    except Exception as e:
        # 显示错误信息
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"生成图表时出错：\n{str(e)}",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        canvas.draw()

    # 保存按钮
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=10)

    def save_chart():
        try:
            file_types = [
                ("PNG Files", "*.png"),
                ("PDF Files", "*.pdf"),
                ("SVG Files", "*.svg"),
                ("All Files", "*.*")
            ]
            filename = filedialog.asksaveasfilename(
                title="保存图表",
                filetypes=file_types,
                defaultextension=".png"
            )
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("成功", f"图表已保存到：{filename}")
        except Exception as e:
            messagebox.showerror("错误", f"保存图表失败：{str(e)}")

    ttk.Button(button_frame, text="保存图表", command=save_chart).pack(side=tk.LEFT, padx=10)

    def close_chart_window():
        """关闭图表窗口时的清理函数"""
        try:
            # 关闭当前图表
            plt.close(fig)
        except Exception as e:
            print(f"关闭图表窗口时清理图表出错: {str(e)}")
        finally:
            chart_window.destroy()

    ttk.Button(button_frame, text="关闭", command=close_chart_window).pack(side=tk.LEFT, padx=10)

    # 如果自动保存开启，则自动保存
    if auto_save:
        try:
            results_dir = analyzer.current_result_dir
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{chart_type}_{sat_prn}_{freq}_{timestamp}.png"
            filepath = os.path.join(results_dir, filename)

            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("自动保存", f"图表已自动保存到：{filepath}")
        except Exception as e:
            messagebox.showerror("自动保存失败", f"自动保存图表失败：{str(e)}")


def generate_raw_observations_plot(analyzer, data, sat_prn, freq, fig):
    """生成原始观测值图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 从处理后的观测数据中获取数据
    if sat_prn in analyzer.observations_meters and freq in analyzer.observations_meters[sat_prn]:
        obs_data = analyzer.observations_meters[sat_prn][freq]
        times = obs_data['times']
        code_values = obs_data['code']
        phase_values = obs_data['phase']
        doppler_values = obs_data['doppler']

        # 绘制图表
        ax.plot(times, code_values, 'b-', label='伪距 (Code)', alpha=0.7)
        ax.plot(times, phase_values, 'r-', label='载波相位 (Phase)', alpha=0.7)
        ax.plot(times, doppler_values, 'g-', label='多普勒 (Doppler)', alpha=0.7)

        ax.set_xlabel('时间')
        ax.set_ylabel('观测值')
        ax.set_title(f'原始观测值 - {sat_prn} - {freq}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        import matplotlib.pyplot as plt
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, f'未找到卫星 {sat_prn} 频率 {freq} 的数据',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('数据缺失')


def generate_derivatives_plot(analyzer, data, sat_prn, freq, fig):
    """生成观测值一阶差分图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 计算一阶差分
    derivatives = analyzer.calculate_observable_derivatives(data)

    if sat_prn in derivatives and freq in derivatives[sat_prn]:
        deriv_data = derivatives[sat_prn][freq]
        epochs = list(deriv_data.keys())

        # 提取差分值
        c_derivs = [deriv_data[epoch].get('C', []) for epoch in epochs]
        l_derivs = [deriv_data[epoch].get('L', []) for epoch in epochs]
        d_derivs = [deriv_data[epoch].get('D', []) for epoch in epochs]

        # 绘制图表
        ax.plot(epochs, c_derivs, 'b-', label='伪距差分', alpha=0.7)
        ax.plot(epochs, l_derivs, 'r-', label='载波相位差分', alpha=0.7)
        ax.plot(epochs, d_derivs, 'g-', label='多普勒差分', alpha=0.7)

        ax.set_xlabel('历元')
        ax.set_ylabel('一阶差分')
        ax.set_title(f'观测值一阶差分 - {sat_prn} - {freq}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的一阶差分数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def generate_code_phase_diffs_plot(analyzer, data, sat_prn, freq, fig):
    """生成伪距相位差值图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 计算伪距相位差值
    code_phase_diffs = analyzer.calculate_code_phase_differences(data)

    if sat_prn in code_phase_diffs and freq in code_phase_diffs[sat_prn]:
        diff_data = code_phase_diffs[sat_prn][freq]
        epochs = list(diff_data.keys())

        # 提取差值
        diff_values = [diff_data[epoch].get('diff', []) for epoch in epochs]

        # 绘制图表
        ax.plot(epochs, diff_values, 'b-', label='伪距相位差值', alpha=0.7)

        ax.set_xlabel('历元')
        ax.set_ylabel('差值 (米)')
        ax.set_title(f'伪距相位差值 - {sat_prn} - {freq}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的伪距相位差值数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def generate_code_phase_raw_diffs_plot(analyzer, data, sat_prn, freq, fig):
    """生成伪距相位原始差值图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 计算伪距相位差值
    code_phase_diffs = analyzer.calculate_code_phase_differences(data)

    if sat_prn in code_phase_diffs and freq in code_phase_diffs[sat_prn]:
        diff_data = code_phase_diffs[sat_prn][freq]
        epochs = list(diff_data.keys())

        # 提取原始差值
        raw_diff_values = [diff_data[epoch].get('raw_diff', []) for epoch in epochs]

        # 绘制图表
        ax.plot(epochs, raw_diff_values, 'r-', label='伪距相位原始差值', alpha=0.7)

        ax.set_xlabel('历元')
        ax.set_ylabel('原始差值 (米)')
        ax.set_title(f'伪距相位原始差值 - {sat_prn} - {freq}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的伪距相位原始差值数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def generate_phase_pred_errors_plot(analyzer, data, sat_prn, freq, fig):
    """生成相位预测误差图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 计算相位预测误差
    phase_pred_errors = analyzer.calculate_phase_prediction_errors(data)

    if sat_prn in phase_pred_errors and freq in phase_pred_errors[sat_prn]:
        error_data = phase_pred_errors[sat_prn][freq]

        # 检查数据结构
        if 'prediction_error' in error_data and len(error_data['prediction_error']) > 0:
            # 提取误差值和时间
            error_values = error_data['prediction_error']
            times = error_data['times']

            # 生成历元序号（从1开始）
            epochs = list(range(1, len(error_values) + 1))

            # 绘制图表
            ax.plot(epochs, error_values, 'm-', label='相位预测误差', alpha=0.7)

            ax.set_xlabel('历元')
            ax.set_ylabel('预测误差 (米)')
            ax.set_title(f'相位预测误差 - {sat_prn} - {freq}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的有效相位预测误差数据",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的相位预测误差数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def generate_double_differences_plot(analyzer, data, sat_prn, freq, fig):
    """生成历元间双差图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 计算历元间双差
    double_diffs = analyzer.calculate_epoch_double_differences()
    triple_errors = analyzer.calculate_triple_median_error(double_diffs)

    if sat_prn in double_diffs and freq in double_diffs[sat_prn]:
        diff_data = double_diffs[sat_prn][freq]
        epochs = list(diff_data.keys())

        # 提取双差值
        diff_values = [diff_data[epoch].get('double_diff', []) for epoch in epochs]

        # 绘制图表
        ax.plot(epochs, diff_values, 'c-', label='历元间双差', alpha=0.7)

        ax.set_xlabel('历元')
        ax.set_ylabel('双差值 (米)')
        ax.set_title(f'历元间双差 - {sat_prn} - {freq}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的历元间双差数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def generate_isb_analysis_plot(analyzer, data, fig):
    """生成ISB分析图表"""
    try:
        # 检查是否有ISB分析结果
        if 'isb_analysis' not in analyzer.results:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "没有ISB分析结果，请先进行ISB分析",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return

        isb_results = analyzer.results['isb_analysis']

        if not isb_results['isb_estimates']:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "没有有效的ISB估计值",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return

        # 创建子图
        fig.clear()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        # ISB时间序列
        isb_values = np.array(isb_results['isb_estimates'])
        epochs = isb_results['isb_epochs']

        ax1.plot(epochs, isb_values, 'b-', linewidth=1, alpha=0.7, label='ISB时间序列')
        ax1.axhline(y=isb_results['isb_mean'], color='r', linestyle='--',
                    label=f'均值: {isb_results["isb_mean"]:.3f}m')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('ISB (m)')
        ax1.set_title('ISB时间序列')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ISB直方图
        ax2.hist(isb_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=isb_results['isb_mean'], color='r', linestyle='--', linewidth=2,
                    label=f'均值: {isb_results["isb_mean"]:.3f}m')
        ax2.set_xlabel('ISB (m)')
        ax2.set_ylabel('频次')
        ax2.set_title('ISB分布直方图')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ISB统计信息
        ax3.axis('off')
        stats_text = f"""ISB统计信息:
        
均值: {isb_results['isb_mean']:.3f} m
标准差: {isb_results['isb_std']:.3f} m
最小值: {np.min(isb_values):.3f} m
最大值: {np.max(isb_values):.3f} m
有效历元数: {len(isb_values)}
基准卫星: {isb_results['reference_satellite']}"""

        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=12,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8,
                           edgecolor='black', linewidth=1))

        # ISB残差分析
        ax4.plot(epochs, isb_values - isb_results['isb_mean'], 'g-', linewidth=1, alpha=0.7)
        ax4.axhline(y=0, color='r', linestyle='-', linewidth=1)
        ax4.set_xlabel('时间')
        ax4.set_ylabel('ISB 残差 (m)')
        ax4.set_title('ISB 残差时间序列')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

    except Exception as e:
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"生成ISB分析图表时出错：\n{str(e)}",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def show_report_window(parent):
    """显示报告功能窗口"""
    report_window = tk.Toplevel(parent)
    report_window.title("分析报告")
    report_window.geometry("550x450")
    report_window.resizable(True, True)
    report_window.transient(parent)
    report_window.grab_set()

    # 居中显示窗口
    center_window(report_window, 550, 450)

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
        if not file_var.get():
            tk.messagebox.showerror("错误", "请先选择数据文件")
            return

        # 禁用按钮
        generate_btn.config(state='disabled')

        def report_process():
            try:
                analyzer = GNSSAnalyzer()

                # 设置文件路径
                file_path = file_var.get()
                analyzer.input_file_path = file_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(file_path), "analysis_results")

                # 更新状态
                status_var.set("正在读取RINEX文件...")
                progress_var.set(20)
                report_window.update_idletasks()

                # 读取文件
                data = analyzer.read_rinex_obs(file_path)

                # 更新状态
                status_var.set("正在生成分析报告...")
                progress_var.set(60)
                report_window.update_idletasks()

                # 生成报告
                analyzer.save_report()

                # 完成
                status_var.set("报告生成完成!")
                progress_var.set(100)
                report_window.update_idletasks()

                # 显示结果
                result_dir = analyzer.current_result_dir
                # 在主线程中显示消息框
                parent.after(0, lambda: tk.messagebox.showinfo("完成", f"分析报告生成完成！\n结果保存在：{result_dir}"))

            except Exception as e:
                # 在主线程中显示错误消息
                parent.after(0, lambda: tk.messagebox.showerror("错误", f"生成报告过程中出现错误：\n{str(e)}"))
                status_var.set("生成失败")
            finally:
                # 在主线程中恢复按钮
                parent.after(0, lambda: generate_btn.config(state='normal'))

        # 在新线程中执行
        import threading
        thread = threading.Thread(target=report_process)
        thread.daemon = True
        thread.start()

    generate_btn = ttk.Button(button_frame, text="生成报告", command=generate_report)
    generate_btn.pack(side=tk.LEFT, padx=10)

    def close_report_window():
        """关闭报告窗口时的清理函数"""
        try:
            # 关闭所有Matplotlib图表窗口
            plt.close('all')
        except Exception as e:
            print(f"关闭报告窗口时清理图表出错: {str(e)}")
        finally:
            report_window.destroy()

    ttk.Button(button_frame, text="关闭",
               command=close_report_window).pack(side=tk.LEFT, padx=10)


if __name__ == "__main__":
    main()
