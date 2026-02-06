from src.core.context import AnalysisContext
from .process_gui import PreprocessingWindow
from .visual_gui import VisualizationWindow
from .report_gui import ReportWindow


def main():
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        import matplotlib.pyplot as plt
        from .utils import center_window
    except Exception:
        # In headless contexts allow main to be called but do nothing
        def _fake_main():
            print('Tkinter unavailable or headless; main() is inactive in this environment')
        return _fake_main()

    root = tk.Tk()
    root.title("GNSS数据分析器")
    root.geometry('950x700')
    root.resizable(True, True)

    ctx = AnalysisContext()

    # Create windows with context
    pp = PreprocessingWindow(ctx)
    vv = VisualizationWindow(ctx)
    rr = ReportWindow(ctx)

    # 主菜单栏
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # 预处理菜单
    cleaning_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="预处理", menu=cleaning_menu)
    cleaning_menu.add_command(label="执行预处理", command=lambda: pp.show(root))

    # 可视化菜单
    charts_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="可视化", menu=charts_menu)
    charts_menu.add_command(label="选择图表类型", command=lambda: vv.show(root))

    # 报告菜单
    report_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="报告", menu=report_menu)
    report_menu.add_command(label="生成分析报告", command=lambda: rr.show(root))

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

    ttk.Label(desc_frame, text="• 预处理：多普勒预测相位→多普勒平滑伪距→码相不一致性建模校正→CMC变化阈值剔除→历元间双差剔除→BDS2/3 ISB分析校正",
              font=("Microsoft YaHei", 10), wraplength=900).pack(anchor=tk.W, pady=2)
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
               command=lambda: pp.show(root)).pack(side=tk.LEFT, padx=10)
    ttk.Button(quick_btn_frame, text="数据可视化",
               command=lambda: vv.show(root)).pack(side=tk.LEFT, padx=10)
    ttk.Button(quick_btn_frame, text="生成报告",
               command=lambda: rr.show(root)).pack(side=tk.LEFT, padx=10)

    # 版权信息
    copyright_frame = ttk.Frame(main_frame)
    copyright_frame.pack(fill=tk.X, pady=(20, 10))

    ttk.Label(copyright_frame, text="© 2025 CZ",
              font=("Microsoft YaHei", 9),
              foreground="gray").pack(anchor=tk.CENTER)

    # 关闭时清理资源
    def on_closing():
        try:
            plt.close('all')
            root.destroy()
        except Exception:
            root.quit()

    root.protocol('WM_DELETE_WINDOW', on_closing)
    
    # 居中显示
    center_window(root, 750, 500)

    root.mainloop()


if __name__ == '__main__':
    main()
