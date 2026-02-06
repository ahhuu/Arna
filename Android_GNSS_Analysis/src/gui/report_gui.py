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

    def save_report(self, output_dir: str, prefix: Optional[str] = None) -> str:
        report = self.context.results.get('last_report') or self.generate_report()
        path = self.reporter.save_logs('report', report, output_dir, prefix=prefix)
        return path

    def show(self, parent):
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog
        except Exception:
            return

        top = tk.Toplevel(parent)
        top.title('报告')
        top.geometry('800x600')
        top.transient(parent)
        top.grab_set()

        txt = tk.Text(top, wrap='word')
        txt.pack(fill='both', expand=True, padx=6, pady=6)

        def refresh():
            report = self.generate_report()
            txt.delete('1.0', tk.END)
            txt.insert('1.0', report)

        refresh()

        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, padx=6, pady=6)

        def on_save():
            file_types = [("Text Files", "*.txt"), ("All Files", "*.*")]
            out = filedialog.asksaveasfilename(title='保存报告', defaultextension='.txt', filetypes=file_types)
            if out:
                path = self.save_report(out.rsplit('/', 1)[0], prefix=out.rsplit('/', 1)[-1])
                tk.messagebox.showinfo('完成', f'报告已保存: {path}')

        ttk.Button(btn_frame, text='刷新', command=refresh).pack(side='left', padx=4)
        ttk.Button(btn_frame, text='保存', command=on_save).pack(side='left', padx=4)
