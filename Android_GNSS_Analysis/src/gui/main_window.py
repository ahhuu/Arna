from src.core.context import AnalysisContext
from .process_gui import PreprocessingWindow
from .visual_gui import VisualizationWindow
from .report_gui import ReportWindow
from .tools_launcher import ToolLauncher


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
    root.geometry('950x900')
    root.resizable(True, True)

    ctx = AnalysisContext()
    launcher = ToolLauncher()

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

    def _launch_tool(rel_parts):
        script = launcher.script_path(*rel_parts)
        ok, msg = launcher.launch(script)
        if not ok:
            messagebox.showerror("工具启动失败", msg)

    # 工具菜单
    tools_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="工具", menu=tools_menu)

    analysis_menu = tk.Menu(tools_menu, tearoff=0)
    tools_menu.add_cascade(label="分析工具", menu=analysis_menu)
    analysis_menu.add_command(
        label="API 不确定度分析",
        command=lambda: _launch_tool(("analysis_tools", "Api_Analysis_Tool.py")),
    )
    analysis_menu.add_command(
        label="SNR 权重建模",
        command=lambda: _launch_tool(("analysis_tools", "SNR_Weighting.py")),
    )

    conversion_menu = tk.Menu(tools_menu, tearoff=0)
    tools_menu.add_cascade(label="格式转换工具", menu=conversion_menu)
    conversion_menu.add_command(
        label="Android 原始数据转RINEX",
        command=lambda: _launch_tool(("conversion_tools", "Mod-Androidgnsslog_to_rinex.py")),
    )
    conversion_menu.add_command(
        label="随机模型表达式转换",
        command=lambda: _launch_tool(("conversion_tools", "ModelConverter.py")),
    )

    coord_menu = tk.Menu(tools_menu, tearoff=0)
    tools_menu.add_cascade(label="坐标工具", menu=coord_menu)
    coord_menu.add_command(
        label="批量 XYZ 写入 Coords",
        command=lambda: _launch_tool(("coordinate_tools", "batch_xyz_to_coords.py")),
    )
    coord_menu.add_command(
        label="静态坐标转换",
        command=lambda: _launch_tool(("coordinate_tools", "Static_Coordinate_Transformation.py")),
    )
    coord_menu.add_command(
        label="动态坐标转换",
        command=lambda: _launch_tool(("coordinate_tools", "Dynamic_Coordinate_Transformation.py")),
    )

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

    tools_quick_frame = ttk.LabelFrame(quick_frame, text="常用工具", padding="10")
    tools_quick_frame.pack(fill=tk.X, pady=(15, 0))

    row1 = ttk.Frame(tools_quick_frame)
    row1.pack(pady=4)
    ttk.Button(
        row1,
        text="API 不确定度分析",
        command=lambda: _launch_tool(("analysis_tools", "Api_Analysis_Tool.py")),
    ).pack(side=tk.LEFT, padx=6)
    ttk.Button(
        row1,
        text="SNR 权重建模",
        command=lambda: _launch_tool(("analysis_tools", "SNR_Weighting.py")),
    ).pack(side=tk.LEFT, padx=6)
    ttk.Button(
        row1,
        text="Android 原始数据转RINEX",
        command=lambda: _launch_tool(("conversion_tools", "Mod-Androidgnsslog_to_rinex.py")),
    ).pack(side=tk.LEFT, padx=6)

    row2 = ttk.Frame(tools_quick_frame)
    row2.pack(pady=4)
    ttk.Button(
        row2,
        text="模型表达式转换",
        command=lambda: _launch_tool(("conversion_tools", "ModelConverter.py")),
    ).pack(side=tk.LEFT, padx=6)
    ttk.Button(
        row2,
        text="批量 XYZ 写入 Coords",
        command=lambda: _launch_tool(("coordinate_tools", "batch_xyz_to_coords.py")),
    ).pack(side=tk.LEFT, padx=6)
    ttk.Button(
        row2,
        text="静态坐标转换",
        command=lambda: _launch_tool(("coordinate_tools", "Static_Coordinate_Transformation.py")),
    ).pack(side=tk.LEFT, padx=6)
    ttk.Button(
        row2,
        text="动态坐标转换",
        command=lambda: _launch_tool(("coordinate_tools", "Dynamic_Coordinate_Transformation.py")),
    ).pack(side=tk.LEFT, padx=6)

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
    center_window(root, 860, 600)

    root.mainloop()


if __name__ == '__main__':
    main()
