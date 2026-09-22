from src.core.context import AnalysisContext
from .process_gui import PreprocessingWindow
from .visual_gui import VisualizationWindow
from .report_gui import ReportWindow
from .tools_launcher import ToolLauncher
from .usage_tracker import UsageTracker

# ============================================================
# 所有工具的注册信息: (显示名称, 路径元组, 所属菜单类别)
# ============================================================
TOOLS_REGISTRY = [
    # 分析工具
    ("API 不确定度分析",       ("analysis_tools", "Api_Analysis_Tool.py"),               "analysis"),
    ("SNR 权重建模",           ("analysis_tools", "SNR_Weighting.py"),                   "analysis"),
    ("误差曲线分析",           ("analysis_tools", "Error_Curve_Analysis.py"),            "analysis"),
    # 格式转换工具
    ("Android 原始数据转RINEX", ("conversion_tools", "Mod-Androidgnsslog_to_rinex.py"),  "conversion"),
    ("随机模型表达式转换",     ("conversion_tools", "ModelConverter.py"),                "conversion"),
    ("GNSS 时间日期转换",      ("conversion_tools", "Time_Date_Converter.py"),           "conversion"),
    # 坐标工具
    ("批量 XYZ 写入 Coords",   ("coordinate_tools", "batch_xyz_to_coords.py"),           "coordinate"),
    ("静态坐标转换",           ("coordinate_tools", "Static_Coordinate_Transformation.py"), "coordinate"),
    ("动态坐标转换",           ("coordinate_tools", "Dynamic_Coordinate_Transformation.py"), "coordinate"),
]


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
    tracker = UsageTracker()

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

    def _launch_tool(label, rel_parts):
        """启动外部工具并记录使用次数"""
        tracker.record_usage(label)
        script = launcher.script_path(*rel_parts)
        ok, msg = launcher.launch(script)
        if not ok:
            messagebox.showerror("工具启动失败", msg)
        else:
            _refresh_quick_tools()

    # 工具菜单
    tools_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="工具", menu=tools_menu)

    # 按类别构建菜单
    analysis_menu = tk.Menu(tools_menu, tearoff=0)
    tools_menu.add_cascade(label="分析工具", menu=analysis_menu)
    conversion_menu = tk.Menu(tools_menu, tearoff=0)
    tools_menu.add_cascade(label="格式转换工具", menu=conversion_menu)
    coord_menu = tk.Menu(tools_menu, tearoff=0)
    tools_menu.add_cascade(label="坐标工具", menu=coord_menu)

    # 用 TOOLS_REGISTRY 填充各子菜单
    for label, path_parts, category in TOOLS_REGISTRY:
        target_menu = {
            "analysis": analysis_menu,
            "conversion": conversion_menu,
            "coordinate": coord_menu,
        }[category]
        target_menu.add_command(
            label=label,
            command=lambda lbl=label, pp=path_parts: _launch_tool(lbl, pp),
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

    ttk.Label(desc_frame, text="• 预处理：多普勒预测相位→伪距多路径改正→多普勒平滑伪距→码相不一致性建模校正→CMC变化阈值剔除→历元间双差剔除→BDS2/3 ISB分析校正",
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

    # ---------- 动态常用工具 (显示使用次数最多的前4个) ----------
    tools_quick_frame = ttk.LabelFrame(quick_frame, text="常用工具", padding="10")
    tools_quick_frame.pack(fill=tk.X, pady=(15, 0))

    # label -> path_parts 的快速查找表
    _tools_map = {lbl: pp for lbl, pp, _cat in TOOLS_REGISTRY}

    def _refresh_quick_tools():
        """刷新常用工具按钮区域"""
        # 清除旧按钮
        for w in tools_quick_frame.winfo_children():
            w.destroy()

        top_tools = tracker.get_top(4)
        if not top_tools:
            # 还没有使用记录 → 显示默认的前4个工具
            top_tools = [(TOOLS_REGISTRY[i][0], 0) for i in range(min(4, len(TOOLS_REGISTRY)))]

        # 按每行2个排列
        for i in range(0, len(top_tools), 2):
            row = ttk.Frame(tools_quick_frame)
            row.pack(pady=4)
            for j in range(2):
                if i + j < len(top_tools):
                    label_name, count = top_tools[i + j]
                    path = _tools_map.get(label_name)
                    if path is None:
                        continue
                    # 显示名称 + 使用次数角标
                    btn_text = f"{label_name}  [{count}]"
                    btn = ttk.Button(
                        row,
                        text=btn_text,
                        command=lambda lbl=label_name, pp=path: _launch_tool(lbl, pp),
                    )
                    btn.pack(side=tk.LEFT, padx=6)

    # 首次刷新
    _refresh_quick_tools()

    # 版权信息
    copyright_frame = ttk.Frame(main_frame)
    copyright_frame.pack(fill=tk.X, pady=(20, 10))

    ttk.Label(copyright_frame, text="© 2026 CZ",
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
