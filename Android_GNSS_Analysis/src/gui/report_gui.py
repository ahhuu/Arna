import datetime
import os
from typing import Optional
from src.core.context import AnalysisContext
from src.reporting.reporter import ReportGenerator


class ReportWindow:
    def __init__(self, context: Optional[AnalysisContext] = None):
        self.context = context or AnalysisContext()
        self.reporter = ReportGenerator()
        self.selected_source = None

    def _source(self):
        return (self.selected_source or
                self.context.results.get('processing_manifest_path') or
                self.context.output_dir or self.context.input_path)

    def generate_report(self) -> str:
        source = self._source()
        manifest_path = self.reporter.find_manifest(source)
        if manifest_path:
            return self.reporter.generate_text_report(self.reporter.load_manifest(manifest_path))
        return self.reporter.generate_text_report({
            'results': self.context.results,
            'input_path': self.context.input_path,
        })

    def save_report(self, output_dir: str, filename: Optional[str] = None):
        source = self._source()
        manifest_path = self.reporter.find_manifest(source)
        generated = {}
        task_root = None
        if manifest_path:
            task_root = os.path.dirname(os.path.dirname(manifest_path))
            preprocessing_dir = os.path.join(output_dir, 'preprocessing')
            generated['preprocessing'] = self.reporter.save_report_bundle(
                manifest_path, preprocessing_dir, 'gnss-preprocessing-report')
        source_path = os.path.abspath(source) if source else ''
        if task_root is None and source_path:
            if os.path.basename(source_path).lower() in ('preprocessing', 'visualization', 'report'):
                task_root = os.path.dirname(source_path)
            elif os.path.isdir(source_path):
                task_root = source_path
        visualization_dir = os.path.join(task_root, 'visualization') if task_root else None
        visualization_manifest = (os.path.join(visualization_dir, self.reporter.VISUALIZATION_MANIFEST_NAME)
                                  if visualization_dir else None)
        if visualization_manifest and os.path.isfile(visualization_manifest):
            with open(visualization_manifest, encoding='utf-8') as fh:
                import json
                visual_info = json.load(fh)
            generated['visualization'] = self.reporter.write_visualization_report(
                visualization_dir,
                visual_info.get('input_path', self.context.input_path or ''),
                selection=visual_info.get('selection', 'all'),
                metadata=visual_info.get('metadata', {}),
                generate_reports=True,
                report_output_dir=os.path.join(output_dir, 'visualization'),
            )
        if not generated:
            raise FileNotFoundError('没有找到预处理或可视化任务清单，请先执行相应处理。')
        return generated

    def show(self, parent):
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
        except Exception:
            return

        top = tk.Toplevel(parent)
        top.title('生成综合分析报告')
        top.geometry('900x700')
        top.transient(parent)
        top.grab_set()

        source_frame = ttk.LabelFrame(top, text='报告数据源', padding=10)
        source_frame.pack(fill='x', padx=8, pady=(8, 0))
        source_var = tk.StringVar(value=self._source() or '')
        ttk.Entry(source_frame, textvariable=source_var).pack(side='left', padx=(0, 8), fill='x', expand=True)

        status_var = tk.StringVar(value='可使用当前预处理会话，也可选择历史 preprocessing 目录或任务清单。')

        def choose_manifest():
            path = filedialog.askopenfilename(
                title='选择预处理任务清单',
                filetypes=[('Processing manifest', 'processing_manifest.json'), ('JSON', '*.json')])
            if path:
                self.selected_source = path
                source_var.set(path)
                refresh()

        def choose_directory():
            path = filedialog.askdirectory(title='选择 preprocessing 结果目录')
            if path:
                self.selected_source = path
                source_var.set(path)
                refresh()

        ttk.Button(source_frame, text='选择任务清单', command=choose_manifest).pack(side='left', padx=3)
        ttk.Button(source_frame, text='选择结果目录', command=choose_directory).pack(side='left', padx=3)

        ttk.Label(top, textvariable=status_var).pack(fill='x', padx=10, pady=6)
        text = tk.Text(top, wrap='word')
        text.pack(fill='both', expand=True, padx=8, pady=4)

        def refresh():
            typed = source_var.get().strip()
            if typed:
                self.selected_source = typed
            try:
                report = self.generate_report()
                text.delete('1.0', tk.END)
                text.insert('1.0', report)
                manifest = self.reporter.find_manifest(self._source())
                if manifest:
                    task_root = os.path.dirname(os.path.dirname(manifest))
                    visual_manifest = os.path.join(task_root, 'visualization', self.reporter.VISUALIZATION_MANIFEST_NAME)
                    visual_status = '；检测到可视化清单' if os.path.isfile(visual_manifest) else '；未检测到可视化清单'
                    status_var.set(f'已加载预处理清单: {manifest}{visual_status}')
                else:
                    status_var.set('尚无任务清单；当前仅显示会话提示。')
            except Exception as exc:
                status_var.set(f'读取失败: {exc}')
                messagebox.showerror('报告读取失败', str(exc))

        def default_report_dir():
            manifest_path = self.reporter.find_manifest(self._source())
            if manifest_path:
                preprocessing_dir = os.path.dirname(manifest_path)
                return os.path.join(os.path.dirname(preprocessing_dir), 'report')
            return os.path.join(os.getcwd(), 'report')

        def generated_message(paths):
            lines = []
            for report_type, report_paths in paths.items():
                if report_type == 'preprocessing':
                    lines.append(f"预处理HTML: {report_paths['html']}")
                    lines.append(f"预处理TXT: {report_paths['text']}")
                else:
                    lines.append(f"可视化HTML: {report_paths['html']}")
                    lines.append(f"可视化TXT: {report_paths['text']}")
            return '\n'.join(lines)

        def save_bundle():
            try:
                output_dir = default_report_dir()
                stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                paths = self.save_report(output_dir, f'gnss-preprocessing-report-{stamp}')
                status_var.set('报告生成完成')
                messagebox.showinfo('完成', '已生成报告：\n' + generated_message(paths))
            except Exception as exc:
                messagebox.showerror('报告生成失败', str(exc))

        def save_as():
            output_dir = filedialog.askdirectory(title='选择报告输出目录')
            if not output_dir:
                return
            try:
                stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                paths = self.save_report(output_dir, f'gnss-preprocessing-report-{stamp}')
                status_var.set('报告生成完成')
                messagebox.showinfo('完成', '已生成报告：\n' + generated_message(paths))
            except Exception as exc:
                messagebox.showerror('报告生成失败', str(exc))

        buttons = ttk.Frame(top)
        buttons.pack(fill='x', padx=8, pady=8)
        ttk.Button(buttons, text='刷新预览', command=refresh).pack(side='left', padx=4)
        ttk.Button(buttons, text='生成全部HTML+TXT', command=save_bundle).pack(side='left', padx=4)
        ttk.Button(buttons, text='另存为...', command=save_as).pack(side='left', padx=4)
        ttk.Button(buttons, text='关闭', command=top.destroy).pack(side='right', padx=4)
        refresh()
