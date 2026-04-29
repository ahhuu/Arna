import datetime
import os
from typing import Optional
from src.core.context import AnalysisContext
from src.reporting.reporter import ReportGenerator


class ReportWindow:
    def __init__(self, context: Optional[AnalysisContext] = None):
        self.context = context or AnalysisContext()
        self.reporter = ReportGenerator()

    def generate_report(self) -> str:
        report = self.reporter.generate_text_report({'results': self.context.results, 'input_path': self.context.input_path, 'output_dir': self.context.output_dir})
        self.context.results['last_report'] = report
        return report

    def save_report(self, output_dir: str, filename: Optional[str] = None) -> str:
        report = self.context.results.get('last_report') or self.generate_report()
        os.makedirs(output_dir, exist_ok=True)
        if not filename:
            obs_name = os.path.splitext(os.path.basename(self.context.input_path or 'report'))[0]
            filename = f"{obs_name}-report-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.txt"
        if not filename.lower().endswith('.txt'):
            filename += '.txt'
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        return path

    def show(self, parent):
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog
        except Exception:
            return

        top = tk.Toplevel(parent)
        top.title('报告')
        top.geometry('800x650')
        top.transient(parent)
        top.grab_set()

        file_frame = ttk.LabelFrame(top, text='选择手机RINEX文件', padding=10)
        file_frame.pack(fill='x', padx=6, pady=(6, 0))
        file_var = tk.StringVar(value=self.context.input_path or '')
        file_entry = ttk.Entry(file_frame, textvariable=file_var, width=60)
        file_entry.pack(side='left', padx=(0,10), fill='x', expand=True)

        def select_file():
            file_types = [
                ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
                ("All Files", "*.*")
            ]
            f = filedialog.askopenfilename(title='选择RINEX观测文件', filetypes=file_types)
            if f:
                file_var.set(f)
                self.context.set_input_path(f)
                base_dir = os.path.dirname(f)
                obs_name = os.path.splitext(os.path.basename(f))[0]
                self.context.set_output_dir(os.path.join(base_dir, 'Arna_results', obs_name, 'report'))
                status_var.set(f'已选择文件: {os.path.basename(f)}')

        ttk.Button(file_frame, text='浏览', command=select_file).pack(side='right')

        progress_frame = ttk.LabelFrame(top, text='状态', padding=10)
        progress_frame.pack(fill='x', padx=6, pady=(4, 0))
        status_var = tk.StringVar(value='请选择 RINEX 文件或使用当前已加载文件')
        ttk.Label(progress_frame, textvariable=status_var).pack(anchor='w')

        txt = tk.Text(top, wrap='word')
        txt.pack(fill='both', expand=True, padx=6, pady=6)

        def refresh():
            report = self.generate_report()
            txt.delete('1.0', tk.END)
            txt.insert('1.0', report)

        refresh()

        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, padx=6, pady=6)

        def get_report_dir():
            if self.context.output_dir:
                return self.context.output_dir
            current_file = file_var.get().strip() or self.context.input_path or ''
            if current_file:
                base_dir = os.path.dirname(current_file)
                obs_name = os.path.splitext(os.path.basename(current_file))[0]
                return os.path.join(base_dir, 'Arna_results', obs_name, 'report')
            return os.path.join(os.getcwd(), 'report')

        def on_save():
            out_dir = get_report_dir()
            filename = f"{os.path.splitext(os.path.basename(file_var.get() or self.context.input_path or 'report'))[0]}-report-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.txt"
            path = self.save_report(out_dir, filename=filename)
            status_var.set(f'报告已保存: {path}')
            tk.messagebox.showinfo('完成', f'报告已保存: {path}')

        ttk.Button(btn_frame, text='刷新', command=refresh).pack(side='left', padx=4)
        ttk.Button(btn_frame, text='保存', command=on_save).pack(side='left', padx=4)
