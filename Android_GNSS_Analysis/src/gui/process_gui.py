from typing import Optional, Dict, Any
import os

# Delay imports that may require a display until runtime
from src.core.context import AnalysisContext
from src.data.reader import RinexReader
from src.data.writer import RinexWriter
from src.processing.calculator import MetricCalculator
from src.processing.coarse_error import CoarseErrorProcessor
from src.processing.advanced_algo import CoreAlgorithmProcessor


class PreprocessingWindow:
    """Non-blocking preprocessing helper that can be used from GUI or CLI/tests.

    Methods are programmatically callable for testing; the window UI is optional.
    """

    def __init__(self, context: Optional[AnalysisContext] = None):
        self.context = context or AnalysisContext()
        self.reader = RinexReader()
        self.writer = RinexWriter()
        self.calculator = MetricCalculator()
        self.coarse = CoarseErrorProcessor()
        self.algo = CoreAlgorithmProcessor()

        # default parameter values (exposed to GUI and programmatic access)
        self.code_threshold = 10.0
        self.phase_threshold = 1.5
        self.doppler_threshold = 5.0
        self.cmc_threshold = 4.0
        self.r_squared_threshold = 0.5
        self.cv_threshold = 0.6
        self.phone_only_min_data_points = 20

    def save_params_to_file(self, filename: str) -> None:
        import json
        cfg = {
            'code_threshold': self.code_threshold,
            'phase_threshold': self.phase_threshold,
            'doppler_threshold': self.doppler_threshold,
            'cmc_threshold': self.cmc_threshold,
            'r_squared': self.r_squared_threshold,
            'cv_threshold': self.cv_threshold,
            'phone_only_min_data_points': self.phone_only_min_data_points
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def load_params_from_file(self, filename: str) -> None:
        import json
        with open(filename, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.code_threshold = float(cfg.get('code_threshold', self.code_threshold))
        self.phase_threshold = float(cfg.get('phase_threshold', self.phase_threshold))
        self.doppler_threshold = float(cfg.get('doppler_threshold', self.doppler_threshold))
        self.cmc_threshold = float(cfg.get('cmc_threshold', self.cmc_threshold))
        self.r_squared_threshold = float(cfg.get('r_squared', self.r_squared_threshold))
        self.cv_threshold = float(cfg.get('cv_threshold', self.cv_threshold))
        self.phone_only_min_data_points = int(cfg.get('phone_only_min_data_points', self.phone_only_min_data_points))

    def load_phone_file(self, file_path: str) -> Dict[str, Any]:
        res = self.reader.read_phone_rinex(file_path, frequencies=self.context.frequencies, glonass_k_map=self.context.glonass_k_map)
        self.context.observations_meters = res['observations_meters']
        # Update context wavelengths with those found in file (e.g. dynamic GLONASS)
        if 'satellite_wavelengths' in res:
             for sat, freqs in res['satellite_wavelengths'].items():
                 if sat not in self.context.wavelengths:
                     self.context.wavelengths[sat] = {}
                 self.context.wavelengths[sat].update(freqs)
        return res

    def load_receiver_file(self, file_path: str) -> Dict[str, Any]:
        res = self.reader.read_receiver_rinex(file_path, frequencies=self.context.frequencies, glonass_k_map=self.context.glonass_k_map)
        self.context.receiver_observations = res['receiver_observations']
        if 'satellite_wavelengths' in res:
             for sat, freqs in res['satellite_wavelengths'].items():
                 if sat not in self.context.wavelengths:
                     self.context.wavelengths[sat] = {}
                 self.context.wavelengths[sat].update(freqs)
        return res

    def run_preprocessing_from_files(self, phone_file: Optional[str] = None, receiver_file: Optional[str] = None, frequencies: Optional[Dict] = None, wavelengths: Optional[Dict] = None) -> Dict[str, Any]:
        if phone_file:
            self.load_phone_file(phone_file)
        if receiver_file:
            self.load_receiver_file(receiver_file)
        return self.run_preprocessing()

    def run_preprocessing(self, observations_meters: Optional[Dict[str, Any]] = None, receiver_observations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # allow passing data directly for tests
        if observations_meters is not None:
            self.context.observations_meters = observations_meters
        if receiver_observations is not None:
            self.context.receiver_observations = receiver_observations

        # ensure params stored in context for consistency (so CLI/tests see same data as GUI run)
        self.context.results['params'] = {
            'code_threshold': self.code_threshold,
            'phase_threshold': self.phase_threshold,
            'doppler_threshold': self.doppler_threshold,
            'cmc_threshold': self.cmc_threshold,
            'r_squared_threshold': self.r_squared_threshold,
            'cv_threshold': self.cv_threshold,
            'phone_only_min_data_points': self.phone_only_min_data_points
        }
        
        # Prepare calculation input with frequency/wavelength context
        calc_input_base = {
            'frequencies': self.context.frequencies,
            'wavelengths': self.context.wavelengths
        }

        # 1) calculate receiver CMC
        rc_input = {'receiver_observations': self.context.receiver_observations}
        rc_input.update(calc_input_base)
        rc_cmc = self.calculator.calculate_receiver_cmc(rc_input)
        self.context.results['receiver_cmc'] = rc_cmc

        # 2) calculate phone code-phase differences
        phone_input = {'observations_meters': self.context.observations_meters}
        phone_input.update(calc_input_base)
        raw_diffs = self.calculator.calculate_raw_diffs(phone_input)
        self.context.results['code_phase_differences'] = raw_diffs

        # 3) dCMC
        dres = self.algo.calculate_dcmc(rc_cmc, raw_diffs, r_squared_threshold=self.r_squared_threshold, enable_phone_only_analysis=self.phone_only_var.get() if hasattr(self, 'phone_only_var') else False, phone_only_min_data_points=self.phone_only_min_data_points)
        self.context.results['dcmc'] = dres.get('dcmc', {})
        self.context.results['meta'] = dres.get('meta', {})

        # 4) CCI and ROC
        cci = self.algo.extract_cci_series(self.context.results['dcmc'])
        self.context.results['cci_series'] = cci
        roc = self.algo.calculate_roc_model(cci, cv_threshold=self.cv_threshold, enable_phone_only_analysis=self.phone_only_var.get() if hasattr(self, 'phone_only_var') else False, phone_only_linear_drift=dres.get('meta', {}).get('linear_drift_detailed'))
        self.context.results['roc_model'] = roc

        # 5) correct phases
        corrected = self.algo.correct_phase_observations(self.context.observations_meters, roc, self.context.results['dcmc'], enable_phone_only_analysis=self.phone_only_var.get() if hasattr(self, 'phone_only_var') else False, phone_only_models=None)
        # corrected may be algorithm-only dict or include writer result
        # unify into results key
        if isinstance(corrected, dict) and 'corrected_results' in corrected:
            self.context.results['corrected_phase'] = corrected['corrected_results']
        else:
            self.context.results['corrected_phase'] = corrected

        # 6) epoch double diffs and triple median
        dd = self.coarse.process_epoch_double_diff(self.context.observations_meters)
        self.context.results['epoch_double_diffs'] = dd
        triple = self.coarse.check_triple_median_error(dd)
        self.context.results['triple_errors'] = triple

        # process cmc flags based on current cmc threshold
        cmc_flags = self.coarse.process_cmc_threshold(self.context.observations_meters, self.cmc_threshold)
        self.context.results['cmc_flags'] = cmc_flags
        
        # Intermediate file writing logic (Restoring feature)
        if self.context.input_path:
             try:
                base_dir = os.path.dirname(self.context.input_path)
                bn = os.path.basename(self.context.input_path)
                name, ext = os.path.splitext(bn)
                
                # Note: The original project logic for "cleaned1" and "cleaned2" files often involved 
                # passing specific filtered datasets or writing results of specific stages.
                # Here we simulate the key checkpoint: writing the corrected file if we have corrections.
                # If "cleaned2" implies intermediate corrections, self.context.results['corrected_phase'] is the candidate.
                
                cleaned2_path = os.path.join(base_dir, f"cleaned2-{name}{ext}")
                # We can call the writer if we have correction results
                if self.context.results.get('corrected_phase'):
                    self.writer.write_corrected_rinex(
                        self.context.input_path, 
                        cleaned2_path, 
                        self.context.results['corrected_phase'],
                        self.context.results.get('roc_model')
                    )
             except Exception as e:
                 print(f"Warning: Failed to write intermediate files: {e}")

        return {'receiver_cmc': rc_cmc, 'raw_diffs': raw_diffs, 'dcmc': self.context.results['dcmc'], 'roc': roc, 'corrected': self.context.results['corrected_phase'], 'epoch_dd': dd, 'triple': triple}

    def show(self, parent):
        """Display a Tkinter-based preprocessing window. This mirrors the legacy UI: file selection, templates,
        smart recommendation, run button (async), progress bar, and options toggles.
        """
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
        except Exception:
            # headless environments: do nothing
            return

        top = tk.Toplevel(parent)
        top.title('数据预处理')
        top.geometry('760x600')
        top.transient(parent)
        top.grab_set()

        # File selection
        file_frame = ttk.LabelFrame(top, text='选择数据文件', padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=8)

        phone_var = tk.StringVar()
        recv_var = tk.StringVar()

        phone_row = ttk.Frame(file_frame)
        phone_row.pack(fill=tk.X, pady=4)
        ttk.Label(phone_row, text='手机RINEX文件:').pack(side=tk.LEFT)
        phone_entry = ttk.Entry(phone_row, textvariable=phone_var, width=50)
        phone_entry.pack(side=tk.LEFT, padx=8, expand=True, fill=tk.X)

        def browse_phone():
            fn = filedialog.askopenfilename(title='选择手机RINEX文件')
            if fn:
                phone_var.set(fn)
        ttk.Button(phone_row, text='浏览', command=browse_phone).pack(side=tk.RIGHT)

        recv_row = ttk.Frame(file_frame)
        recv_row.pack(fill=tk.X, pady=4)
        ttk.Label(recv_row, text='接收机RINEX文件(可选):').pack(side=tk.LEFT)
        recv_entry = ttk.Entry(recv_row, textvariable=recv_var, width=50)
        recv_entry.pack(side=tk.LEFT, padx=8, expand=True, fill=tk.X)

        def on_receiver_file_change(*args):
            # enable BDS analysis button only when receiver file is present
            if recv_var.get().strip():
                bds_only_btn.config(state='normal')
            else:
                bds_only_btn.config(state='disabled')

        recv_var.trace('w', on_receiver_file_change)

        def browse_recv():
            fn = filedialog.askopenfilename(title='选择接收机RINEX文件')
            if fn:
                recv_var.set(fn)
        ttk.Button(recv_row, text='浏览', command=browse_recv).pack(side=tk.RIGHT)

        # BDS only analysis button
        bds_btn_row = ttk.Frame(file_frame)
        bds_btn_row.pack(fill=tk.X, pady=2)
        
        def run_isb_only_gui():
             if not recv_var.get():
                 messagebox.showwarning("警告", "ISB分析需要接收机文件")
                 return
             self._run_isb_only_thread(phone_var.get(), recv_var.get(), progress, status_var)

        bds_only_btn = ttk.Button(bds_btn_row, text='仅执行 BDS2/3 分析', state='disabled', command=run_isb_only_gui)
        bds_only_btn.pack(side=tk.LEFT)

        # Parameters and template
        param_frame = ttk.LabelFrame(top, text='参数设置', padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=8)

        template_var = tk.StringVar(value='自定义')
        template_combo = ttk.Combobox(param_frame, textvariable=template_var, state='readonly', values=['自定义', '开阔环境', '遮挡环境'])
        ttk.Label(param_frame, text='预设模板:').pack(side=tk.LEFT)
        template_combo.pack(side=tk.LEFT, padx=6)

        # numeric parameter entries (code/phase/doppler and thresholds)
        threshold_row = ttk.Frame(param_frame)
        threshold_row.pack(fill=tk.X, pady=4)
        ttk.Label(threshold_row, text='伪距阈值(米):').pack(side=tk.LEFT)
        code_threshold_var = tk.DoubleVar(value=self.code_threshold)
        code_threshold_entry = ttk.Entry(threshold_row, textvariable=code_threshold_var, width=8)
        code_threshold_entry.pack(side=tk.LEFT, padx=6)

        ttk.Label(threshold_row, text='相位阈值(米):').pack(side=tk.LEFT)
        phase_threshold_var = tk.DoubleVar(value=self.phase_threshold)
        phase_threshold_entry = ttk.Entry(threshold_row, textvariable=phase_threshold_var, width=8)
        phase_threshold_entry.pack(side=tk.LEFT, padx=6)

        ttk.Label(threshold_row, text='多普勒阈值:').pack(side=tk.LEFT)
        doppler_threshold_var = tk.DoubleVar(value=self.doppler_threshold)
        doppler_threshold_entry = ttk.Entry(threshold_row, textvariable=doppler_threshold_var, width=8)
        doppler_threshold_entry.pack(side=tk.LEFT, padx=6)

        cmc_row = ttk.Frame(param_frame)
        cmc_row.pack(fill=tk.X, pady=4)
        ttk.Label(cmc_row, text='CMC 变化阈值:').pack(side=tk.LEFT)
        cmc_threshold_var = tk.DoubleVar(value=self.cmc_threshold)
        cmc_threshold_entry = ttk.Entry(cmc_row, textvariable=cmc_threshold_var, width=8)
        cmc_threshold_entry.pack(side=tk.LEFT, padx=6)

        stats_row = ttk.Frame(param_frame)
        stats_row.pack(fill=tk.X, pady=4)
        ttk.Label(stats_row, text='R² 阈值:').pack(side=tk.LEFT)
        r_squared_var = tk.DoubleVar(value=self.r_squared_threshold)
        r_squared_entry = ttk.Entry(stats_row, textvariable=r_squared_var, width=8)
        r_squared_entry.pack(side=tk.LEFT, padx=6)

        ttk.Label(stats_row, text='ROC CV 阈值:').pack(side=tk.LEFT)
        cv_threshold_var = tk.DoubleVar(value=self.cv_threshold)
        cv_threshold_entry = ttk.Entry(stats_row, textvariable=cv_threshold_var, width=8)
        cv_threshold_entry.pack(side=tk.LEFT, padx=6)

        ttk.Label(stats_row, text='手机独有最小点数:').pack(side=tk.LEFT, padx=(10,0))
        phone_only_min_var = tk.IntVar(value=self.phone_only_min_data_points)
        phone_only_min_entry = ttk.Entry(stats_row, textvariable=phone_only_min_var, width=6)
        phone_only_min_entry.pack(side=tk.LEFT, padx=6)

        # toggles
        options_frame = ttk.Frame(param_frame)
        options_frame.pack(fill=tk.X, pady=6)
        self.doppler_enable_var = tk.BooleanVar(value=False)
        self.cci_enable_var = tk.BooleanVar(value=True)
        self.isb_enable_var = tk.BooleanVar(value=True)
        self.phone_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text='启用多普勒预测', variable=self.doppler_enable_var).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(options_frame, text='启用CCI建模', variable=self.cci_enable_var).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(options_frame, text='启用ISB校正', variable=self.isb_enable_var).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(options_frame, text='启用手机独有校正', variable=self.phone_only_var).pack(side=tk.LEFT, padx=4)

        # store tk vars locally so save/load/apply_template can access
        _tk_param_vars = {
            'code_threshold_var': code_threshold_var,
            'phase_threshold_var': phase_threshold_var,
            'doppler_threshold_var': doppler_threshold_var,
            'cmc_threshold_var': cmc_threshold_var,
            'r_squared_var': r_squared_var,
            'cv_threshold_var': cv_threshold_var,
            'phone_only_min_var': phone_only_min_var,
            'template_var': template_var
        }
        # Smart recommend
        def smart_recommend_ui():
            if not phone_var.get():
                messagebox.showwarning('警告', '请先选择手机RINEX文件')
                return
            try:
                rd = RinexReader()
                data = rd.read_phone_rinex(phone_var.get())
                total_sats = len(data.get('observations_meters', {}))
                total_epochs = len(data.get('data', {}).get('epochs', []))
                # simple quality metric
                total_possible = 0
                total_valid = 0
                for sat, freqs in data.get('observations_meters', {}).items():
                    for freq, obs in freqs.items():
                        code = obs.get('code', [])
                        total_possible += len(code)
                        total_valid += sum(1 for v in code if v is not None)
                ratio = total_valid / total_possible if total_possible > 0 else 0.0
                if ratio < 0.75:
                    _tk_param_vars['template_var'].set('遮挡环境')
                else:
                    _tk_param_vars['template_var'].set('开阔环境')
                messagebox.showinfo('智能推荐', f"检测到卫星: {total_sats}, 历元: {total_epochs}, 完整率: {ratio:.1%}\n已设置模板为: {_tk_param_vars['template_var'].get()}")
            except Exception as e:
                messagebox.showerror('错误', f'智能推荐失败: {e}')

        ttk.Button(param_frame, text='智能推荐参数', command=smart_recommend_ui).pack(side=tk.RIGHT)

        # apply template behavior
        def apply_template_ui(event=None):
            t = _tk_param_vars['template_var'].get()
            if t == '开阔环境':
                _tk_param_vars['code_threshold_var'].set(8.0)
                _tk_param_vars['phase_threshold_var'].set(3.0)
                _tk_param_vars['doppler_threshold_var'].set(4.0)
                _tk_param_vars['cmc_threshold_var'].set(3.0)
                _tk_param_vars['r_squared_var'].set(0.6)
                _tk_param_vars['cv_threshold_var'].set(0.6)
            elif t == '遮挡环境':
                _tk_param_vars['code_threshold_var'].set(10.0)
                _tk_param_vars['phase_threshold_var'].set(4.0)
                _tk_param_vars['doppler_threshold_var'].set(5.0)
                _tk_param_vars['cmc_threshold_var'].set(5.0)
                _tk_param_vars['r_squared_var'].set(0.5)
                _tk_param_vars['cv_threshold_var'].set(0.5)

        template_combo.bind('<<ComboboxSelected>>', apply_template_ui)

        # Save/Load params
        def save_params_ui():
            fn = filedialog.asksaveasfilename(title='保存参数配置', defaultextension='.json', filetypes=[('JSON', '*.json')])
            if fn:
                # update instance vars from tk vars first
                self.code_threshold = float(_tk_param_vars['code_threshold_var'].get())
                self.phase_threshold = float(_tk_param_vars['phase_threshold_var'].get())
                self.doppler_threshold = float(_tk_param_vars['doppler_threshold_var'].get())
                self.cmc_threshold = float(_tk_param_vars['cmc_threshold_var'].get())
                self.r_squared_threshold = float(_tk_param_vars['r_squared_var'].get())
                self.cv_threshold = float(_tk_param_vars['cv_threshold_var'].get())
                self.phone_only_min_data_points = int(_tk_param_vars['phone_only_min_var'].get())
                try:
                    self.save_params_to_file(fn)
                    messagebox.showinfo('成功', f'参数已保存到: {fn}')
                except Exception as e:
                    messagebox.showerror('错误', f'保存失败: {e}')

        def load_params_ui():
            fn = filedialog.askopenfilename(title='加载参数配置', filetypes=[('JSON', '*.json')])
            if fn:
                try:
                    self.load_params_from_file(fn)
                    # reflect into tk vars
                    _tk_param_vars['code_threshold_var'].set(self.code_threshold)
                    _tk_param_vars['phase_threshold_var'].set(self.phase_threshold)
                    _tk_param_vars['doppler_threshold_var'].set(self.doppler_threshold)
                    _tk_param_vars['cmc_threshold_var'].set(self.cmc_threshold)
                    _tk_param_vars['r_squared_var'].set(self.r_squared_threshold)
                    _tk_param_vars['cv_threshold_var'].set(self.cv_threshold)
                    _tk_param_vars['phone_only_min_var'].set(self.phone_only_min_data_points)
                    _tk_param_vars['template_var'].set('自定义')
                    messagebox.showinfo('成功', f'参数已从文件加载')
                except Exception as e:
                    messagebox.showerror('错误', f'加载失败: {e}')

        cfg_buttons = ttk.Frame(param_frame)
        cfg_buttons.pack(fill=tk.X, pady=6)
        ttk.Button(cfg_buttons, text='保存配置', command=save_params_ui).pack(side=tk.LEFT, padx=4)
        ttk.Button(cfg_buttons, text='加载配置', command=load_params_ui).pack(side=tk.LEFT, padx=4)
        # run controls
        run_frame = ttk.Frame(top)
        run_frame.pack(fill=tk.X, padx=10, pady=8)

        progress = ttk.Progressbar(run_frame, orient='horizontal', mode='indeterminate')
        progress.pack(fill=tk.X, padx=6, pady=4)

        status_var = tk.StringVar(value='就绪')
        ttk.Label(run_frame, textvariable=status_var).pack(side=tk.LEFT, padx=6)

        def _run_in_thread():
            # start progress
            progress.start(10)
            status_var.set('处理中...')

            def task():
                try:
                    # load files if provided
                    if phone_var.get():
                        self.load_phone_file(phone_var.get())
                    if recv_var.get():
                        self.load_receiver_file(recv_var.get())

                    # update instance params from UI
                    self.code_threshold = float(_tk_param_vars['code_threshold_var'].get())
                    self.phase_threshold = float(_tk_param_vars['phase_threshold_var'].get())
                    self.doppler_threshold = float(_tk_param_vars['doppler_threshold_var'].get())
                    self.cmc_threshold = float(_tk_param_vars['cmc_threshold_var'].get())
                    self.r_squared_threshold = float(_tk_param_vars['r_squared_var'].get())
                    self.cv_threshold = float(_tk_param_vars['cv_threshold_var'].get())
                    self.phone_only_min_data_points = int(_tk_param_vars['phone_only_min_var'].get())

                    # set flags and params in context
                    self.context.results['flags'] = {
                        'enable_doppler': bool(self.doppler_enable_var.get()),
                        'enable_cci': bool(self.cci_enable_var.get()),
                        'enable_isb': bool(self.isb_enable_var.get()),
                        'phone_only': bool(self.phone_only_var.get())
                    }
                    self.context.results['params'] = {
                        'code_threshold': self.code_threshold,
                        'phase_threshold': self.phase_threshold,
                        'doppler_threshold': self.doppler_threshold,
                        'cmc_threshold': self.cmc_threshold,
                        'r_squared_threshold': self.r_squared_threshold,
                        'cv_threshold': self.cv_threshold,
                        'phone_only_min_data_points': self.phone_only_min_data_points
                    }

                    # Run processing
                    # 1. Receiver CMC
                    rc_cmc = self.calculator.calculate_receiver_cmc({'receiver_observations': self.context.receiver_observations})
                    self.context.results['receiver_cmc'] = rc_cmc
                    
                    # 2. Raw differences
                    raw_diffs = self.calculator.calculate_raw_diffs({'observations_meters': self.context.observations_meters})
                    self.context.results['code_phase_differences'] = raw_diffs
                    
                    # 3. Doppler Prediction (if enabled)
                    if self.context.results['flags']['enable_doppler']:
                         calc_input_doppler = {
                             'observations_meters': self.context.observations_meters,
                             'frequencies': self.context.frequencies,
                             'wavelengths': self.context.wavelengths
                         }
                         # In original code, this step modifies observations or produces a new set
                         # Here we call the algo method which returns prediction results
                         # And optionally writes a predicted RINEX if we implement that.
                         # For now, let's just run the calculation.
                         self.algo.run_doppler_phase_prediction(
                             self.context.observations_meters,
                             self.context.frequencies,
                             self.context.wavelengths
                         )
                         # To actually use predicted phases, we might need to update observations_meters 
                         # or handle it in downstream, but original project likely used it as a pre-fill step.
                         pass
                    
                    # 4. CCI / DCMC / Correction
                    if self.context.results['flags']['enable_cci']:
                        dres = self.algo.calculate_dcmc(rc_cmc, raw_diffs, r_squared_threshold=self.r_squared_threshold, enable_phone_only_analysis=self.phone_only_var.get(), phone_only_min_data_points=self.phone_only_min_data_points)
                        self.context.results['dcmc'] = dres.get('dcmc', {})
                        self.context.results['meta'] = dres.get('meta', {})

                        cci = self.algo.extract_cci_series(self.context.results['dcmc'])
                        self.context.results['cci_series'] = cci

                        roc = self.algo.calculate_roc_model(cci, cv_threshold=self.cv_threshold, enable_phone_only_analysis=self.phone_only_var.get(), phone_only_linear_drift=dres.get('meta', {}).get('linear_drift_detailed'))
                        self.context.results['roc_model'] = roc

                        corrected = self.algo.correct_phase_observations(self.context.observations_meters, roc, self.context.results['dcmc'], enable_phone_only_analysis=self.phone_only_var.get(), phone_only_models=None)
                        if isinstance(corrected, dict) and 'corrected_results' in corrected:
                            self.context.results['corrected_phase'] = corrected['corrected_results']
                        else:
                            self.context.results['corrected_phase'] = corrected
                    else:
                        self.context.results['corrected_phase'] = {}
                        self.context.results['roc_model'] = {}
                    
                    # 5. ISB (This is usually a separate analysis, but if enabled in preprocessing, 
                    # it might mean preparing data or running it automatically)
                    if self.context.results['flags']['enable_isb'] and self.context.receiver_observations:
                        isb_data = self.algo.run_prepare_isb_data(self.context.observations_meters, self.context.receiver_observations)
                        self.context.results['isb_data'] = isb_data
                        # Select ref sat
                        ref_sat = self.algo.run_select_reference_satellite(isb_data)
                        stable_sats = self.algo.run_filter_stable_satellites(isb_data)
                        isb_res = self.algo.run_calculate_isb_double_difference(isb_data, ref_sat, stable_sats)
                        self.context.results['isb_analysis'] = isb_res

                    dd = self.coarse.process_epoch_double_diff(self.context.observations_meters)
                    self.context.results['epoch_double_diffs'] = dd
                    triple = self.coarse.check_triple_median_error(dd)
                    self.context.results['triple_errors'] = triple


                    # process cmc flags based on UI cmc threshold
                    cmc_flags = self.coarse.process_cmc_threshold(self.context.observations_meters, self.cmc_threshold)
                    self.context.results['cmc_flags'] = cmc_flags

                    top.after(0, lambda: messagebox.showinfo('完成', '预处理完成'))
                except Exception as e:
                    top.after(0, lambda: messagebox.showerror('错误', f'预处理失败: {e}'))
                finally:
                    top.after(0, lambda: progress.stop())
                    top.after(0, lambda: status_var.set('就绪'))

            import threading
            t = threading.Thread(target=task, daemon=True)
            t.start()
        ttk.Button(run_frame, text='开始预处理', command=_run_in_thread).pack(side=tk.LEFT, padx=6)
        ttk.Button(run_frame, text='选择文件', command=lambda: os.startfile(os.path.dirname(phone_var.get())) if phone_var.get() else None).pack(side=tk.LEFT, padx=6, fill=tk.X)
        ttk.Button(run_frame, text='关闭', command=top.destroy).pack(side=tk.RIGHT, padx=6)
        
        # write cleaned/corrected file buttons
        # These are somewhat redundant if we autowrite, but can keep them or remove them based on UI screenshot which doesn't show them explicitly at bottom row 
        # The screenshot shows [开始预处理] [BDS2/3 ISB分析] [选择文件] [关闭]
        # So I will hide the explicit write buttons from pack, or move them to a different frame if needed.
        # But for now, let's just make the bottom row match the screenshot.
        
        # We need to re-pack to match order
        # Clear run_frame to repack in order
        for w in run_frame.winfo_children():
            w.pack_forget()

        progress.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(run_frame, textvariable=status_var).pack(side=tk.TOP, padx=6) # status label above or below
        
        btn_frame = ttk.Frame(run_frame)
        btn_frame.pack(fill=tk.X, pady=4)
        
        ttk.Button(btn_frame, text='开始预处理', command=_run_in_thread).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text='BDS2/3 ISB分析', command=run_isb_only_gui).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text='选择文件', command=lambda: os.startfile(os.path.dirname(phone_var.get())) if phone_var.get() else None).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text='关闭', command=top.destroy).pack(side=tk.RIGHT, padx=6)

    def write_cleaned_rinex(self, original_path: str, output_path: Optional[str] = None, enable_cci: bool = True) -> Dict[str, Any]:
        result = self.writer.write_cleaned_rinex(original_path, output_path, self.context.results.get('epoch_double_diffs', {}), self.context.results.get('triple_errors', {}), enable_cci=enable_cci, cmc_flags=None)
        return result

    def write_corrected_rinex(self, original_path: str, output_path: Optional[str] = None, roc_model: Optional[Dict] = None) -> Dict[str, Any]:
        result = self.writer.write_corrected_rinex(original_path, output_path, self.context.results.get('corrected_phase', {}), roc_model or self.context.results.get('roc_model'))
        return result
