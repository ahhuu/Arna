from src.core.context import AnalysisContext
from .process_gui import PreprocessingWindow
from .visual_gui import VisualizationWindow
from .report_gui import ReportWindow


def main():
    try:
        import tkinter as tk
        from .utils import center_window
    except Exception:
        # In headless contexts allow main to be called but do nothing
        def _fake_main():
            print('Tkinter unavailable or headless; main() is inactive in this environment')
        return _fake_main()

    root = tk.Tk()
    root.title("Android GNSS RINEX 分析工具")
    root.geometry('750x500')
    root.resizable(True, True)

    ctx = AnalysisContext()

    # create helper windows
    pp = PreprocessingWindow(ctx)
    vv = VisualizationWindow(ctx)
    rr = ReportWindow(ctx)

    # 主界面布局
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    header_font = ('Microsoft YaHei', 16, 'bold')
    desc_font = ('Microsoft YaHei', 10)
    
    welcome_label = tk.Label(main_frame, text="Android GNSS RINEX 分析工具", font=header_font)
    welcome_label.pack(pady=20)

    desc_frame = tk.LabelFrame(main_frame, text="功能说明", font=('Microsoft YaHei', 10, 'bold'), padx=15, pady=15)
    desc_frame.pack(fill=tk.X, pady=15)
    
    tk.Label(desc_frame, text="• 预处理：多普勒预测相位→码相不一致性建模校正→CMC变化阈值剔除→历元间双差剔除→BDS2/3 ISB分析校正", 
             font=desc_font, anchor='w').pack(fill=tk.X, pady=2)
    tk.Label(desc_frame, text="• 可视化：生成各类分析图表，支持单独保存和批量保存", 
             font=desc_font, anchor='w').pack(fill=tk.X, pady=2)
    tk.Label(desc_frame, text="• 报告：生成完整的分析报告，包含所有预处理分析结果", 
             font=desc_font, anchor='w').pack(fill=tk.X, pady=2)

    quick_frame = tk.LabelFrame(main_frame, text="快速操作", font=('Microsoft YaHei', 10, 'bold'), padx=15, pady=15)
    quick_frame.pack(fill=tk.X, pady=15)
    
    btn_frame = tk.Frame(quick_frame)
    btn_frame.pack()

    def on_preprocess():
        pp.show(root)

    def on_visual():
        vv.show(root)

    def on_report():
        rr.show(root)

    btn_config = {'font': desc_font, 'width': 15, 'pady': 5}
    tk.Button(btn_frame, text="开始预处理", command=on_preprocess, **btn_config).pack(side='left', padx=15)
    tk.Button(btn_frame, text="数据可视化", command=on_visual, **btn_config).pack(side='left', padx=15)
    tk.Button(btn_frame, text="生成报告", command=on_report, **btn_config).pack(side='left', padx=15)

    # 版权信息
    tk.Label(main_frame, text="© 2025 CZ", font=("Arial", 9), fg="#888888").pack(side=tk.BOTTOM, pady=10)

    # 菜单栏（保持与原始一致）
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    cleaning_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='预处理', menu=cleaning_menu)
    cleaning_menu.add_command(label='执行预处理', command=on_preprocess)

    charts_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='可视化', menu=charts_menu)
    charts_menu.add_command(label='选择图表类型', command=on_visual)

    report_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='报告', menu=report_menu)
    report_menu.add_command(label='生成分析报告', command=on_report)

    # 关闭时清理资源
    def on_closing():
        try:
            import matplotlib
            matplotlib.pyplot.close('all')
            root.destroy()
        except Exception:
            root.quit()

    root.protocol('WM_DELETE_WINDOW', on_closing)
    # 居中显示
    center_window(root, 750, 500)

    root.mainloop()


if __name__ == '__main__':
    main()
