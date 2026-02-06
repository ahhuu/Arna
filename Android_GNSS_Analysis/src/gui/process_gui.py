from typing import Optional, Dict, Any
import os
import datetime
import traceback

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
        
        # Doppler smoothing parameters
        self.doppler_smoothing_enabled = False
        self.doppler_smoothing_window = 20
        self.doppler_smoothing_reset_threshold = 15.0
        
        # Triple sigma option
        self.use_triple_sigma = False
        
        # Threshold mode: 'fixed' or 'adaptive'
        self.threshold_mode = 'fixed'

    def save_params_to_file(self, filename: str) -> None:
        import json
        cfg = {
            'code_threshold': self.code_threshold,
            'phase_threshold': self.phase_threshold,
            'doppler_threshold': self.doppler_threshold,
            'cmc_threshold': self.cmc_threshold,
            'r_squared': self.r_squared_threshold,
            'cv_threshold': self.cv_threshold,
            'phone_only_min_data_points': self.phone_only_min_data_points,
            'doppler_smoothing_enabled': self.doppler_smoothing_enabled,
            'doppler_smoothing_window': self.doppler_smoothing_window,
            'doppler_smoothing_reset_threshold': self.doppler_smoothing_reset_threshold,
            'use_triple_sigma': self.use_triple_sigma,
            'threshold_mode': self.threshold_mode
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
        self.doppler_smoothing_enabled = bool(cfg.get('doppler_smoothing_enabled', self.doppler_smoothing_enabled))
        self.doppler_smoothing_window = int(cfg.get('doppler_smoothing_window', self.doppler_smoothing_window))
        self.doppler_smoothing_reset_threshold = float(cfg.get('doppler_smoothing_reset_threshold', self.doppler_smoothing_reset_threshold))
        self.use_triple_sigma = bool(cfg.get('use_triple_sigma', self.use_triple_sigma))
        self.threshold_mode = cfg.get('threshold_mode', 'fixed')

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
            'phone_only_min_data_points': self.phone_only_min_data_points,
            'doppler_smoothing_enabled': self.doppler_smoothing_enabled,
            'doppler_smoothing_window': self.doppler_smoothing_window,
            'doppler_smoothing_reset_threshold': self.doppler_smoothing_reset_threshold
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

        # 1.5) Apply Doppler Smoothing (if enabled)
        if self.doppler_smoothing_enabled:
            smoothing_result = self.algo.apply_doppler_smoothing(
                self.context.observations_meters,
                max_window=self.doppler_smoothing_window,
                reset_threshold_m=self.doppler_smoothing_reset_threshold
            )
            self.context.results['doppler_smoothing'] = smoothing_result
            # Replace pseudocode observations with smoothed values
            smoothed_obs = smoothing_result['smoothed_observations']
            for sat_id in smoothed_obs:
                for freq in smoothed_obs[sat_id]:
                    if sat_id in self.context.observations_meters and freq in self.context.observations_meters[sat_id]:
                        # Preserve original code, add smoothed version
                        self.context.observations_meters[sat_id][freq]['code_original'] = self.context.observations_meters[sat_id][freq].get('code', [])
                        self.context.observations_meters[sat_id][freq]['code'] = smoothed_obs[sat_id][freq]['code_smoothed']
        else:
            self.context.results['doppler_smoothing'] = None

        # 2) calculate phone code-phase differences
        phone_input = {'observations_meters': self.context.observations_meters}
        phone_input.update(calc_input_base)
        raw_diffs = self.calculator.calculate_code_phase_differences(phone_input)
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
        triple = self.coarse.check_triple_median_error(
            dd,
            max_threshold_limit={
                'code': self.code_threshold,
                'phase': self.phase_threshold,
                'doppler': self.doppler_threshold
            }
        )
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
        """Display a Tkinter-based preprocessing window matching Rinex_Analysis_Modules style."""
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
        except Exception:
            return

        top = tk.Toplevel(parent)
        top.title('数据预处理')
        top.geometry('800x770') # Slightly taller to fit new layout
        top.transient(parent)
        top.grab_set()

        # Styles can be added here if needed to strictly match, but ttk default is usually close enough.

        # --- 1. File Selection Section ---
        file_frame = ttk.LabelFrame(top, text='选择数据文件', padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        phone_var = tk.StringVar()
        recv_var = tk.StringVar()

        # Phone File
        phone_row = ttk.Frame(file_frame)
        phone_row.pack(fill=tk.X, pady=2)
        ttk.Label(phone_row, text='手机RINEX文件:', width=15).pack(side=tk.LEFT)
        ttk.Entry(phone_row, textvariable=phone_var).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(phone_row, text='浏览', command=lambda: self._browse_file(phone_var, '选择手机RINEX文件')).pack(side=tk.LEFT)

        # Receiver File
        recv_row = ttk.Frame(file_frame)
        recv_row.pack(fill=tk.X, pady=2)
        ttk.Label(recv_row, text='接收机RINEX文件(CCI建模和ISB分析必需):', width=40).pack(side=tk.LEFT) # Adjusted width
        ttk.Entry(recv_row, textvariable=recv_var).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(recv_row, text='浏览', command=lambda: self._browse_file(recv_var, '选择接收机RINEX文件')).pack(side=tk.LEFT)


        # --- 2. Parameter Settings Section ---
        param_frame = ttk.LabelFrame(top, text='参数设置', padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        # 2.1 Template
        template_row = ttk.Frame(param_frame)
        template_row.pack(fill=tk.X, pady=2)
        ttk.Label(template_row, text='参数模板').pack(side=tk.LEFT) # Section header logic inside frame? Or just row.
        
        # Sub-frame for template line
        tpl_sub = ttk.LabelFrame(param_frame, text='参数模板', padding=5) # Inner frame style
        tpl_sub.pack(fill=tk.X, pady=2)
        
        template_var = tk.StringVar(value='自定义')
        ttk.Label(tpl_sub, text='选择预设模板:').pack(side=tk.LEFT)
        template_combo = ttk.Combobox(tpl_sub, textvariable=template_var, state='readonly', values=['自定义', '开阔环境', '遮挡环境'])
        template_combo.pack(side=tk.LEFT, padx=5)
        
        # Smart recommend button
        def smart_recommend_ui():
             if not phone_var.get():
                 messagebox.showwarning('警告', '请先选择手机RINEX文件')
                 return
             # ... (reused logic)
             try:
                rd = RinexReader()
                data = rd.read_phone_rinex(phone_var.get())
                total_sats = len(data.get('observations_meters', {}))
                total_valid = 0; total_possible = 0
                for sat, freqs in data.get('observations_meters', {}).items():
                    for freq, obs in freqs.items():
                        total_possible += len(obs.get('code', []))
                        total_valid += sum(1 for v in obs.get('code', []) if v is not None)
                ratio = total_valid / total_possible if total_possible > 0 else 0
                rec = '遮挡环境' if ratio < 0.75 else '开阔环境'
                template_var.set(rec)
                self.apply_template(rec, _tk_param_vars)
                messagebox.showinfo('智能推荐', f"数据完整率: {ratio:.1%}\n已推荐并应用模板: {rec}")
             except Exception as e:
                 messagebox.showerror('错误', f"推荐失败: {e}")

        ttk.Button(tpl_sub, text='智能推荐参数', command=smart_recommend_ui).pack(side=tk.LEFT, padx=10)
        
        # Save/Load buttons aligned right
        ttk.Button(tpl_sub, text='加载配置', command=lambda: self.load_params_ui(_tk_param_vars)).pack(side=tk.RIGHT, padx=5)
        ttk.Button(tpl_sub, text='保存配置', command=lambda: self.save_params_ui(_tk_param_vars)).pack(side=tk.RIGHT, padx=5)


        # 2.2 Coarse Error Processing
        coarse_frame = ttk.LabelFrame(param_frame, text='粗差处理', padding=5)
        coarse_frame.pack(fill=tk.X, pady=5)
        
        # Threshold Mode Selection
        threshold_mode_var = tk.StringVar(value='固定')
        mode_row = ttk.Frame(coarse_frame)
        mode_row.pack(fill=tk.X, pady=2)
        ttk.Label(mode_row, text='阈值模式:').pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(mode_row, textvariable=threshold_mode_var, 
                                   state='readonly', width=15,
                                   values=['固定', '自适应'])
        mode_combo.pack(side=tk.LEFT, padx=5)
        
        # Dynamic hint label
        mode_hint_var = tk.StringVar(value='')
        ttk.Label(mode_row, textvariable=mode_hint_var, 
                  foreground='blue', font=('', 9)).pack(side=tk.LEFT, padx=5)
        
        # Layout: Code, Phase, Doppler thresholds in one row, CMC in another? Or matching screenshot.
        # Screenshot: "历元间双差最大阈值: 伪距(米): 10.0  相位(米): 1.5  多普勒(米/秒): 5.0"
        #             "CMC变化阈值(米): 4.0"
        
        c_row1 = ttk.Frame(coarse_frame)
        c_row1.pack(fill=tk.X, pady=2)
        ttk.Label(c_row1, text='历元间双差最大阈值:').pack(side=tk.LEFT)
        
        code_threshold_var = tk.DoubleVar(value=self.code_threshold)
        ttk.Label(c_row1, text='伪距(米):').pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(c_row1, textvariable=code_threshold_var, width=6).pack(side=tk.LEFT)

        phase_threshold_var = tk.DoubleVar(value=self.phase_threshold)
        ttk.Label(c_row1, text='相位(米):').pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(c_row1, textvariable=phase_threshold_var, width=6).pack(side=tk.LEFT)

        doppler_threshold_var = tk.DoubleVar(value=self.doppler_threshold)
        ttk.Label(c_row1, text='多普勒(米/秒):').pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(c_row1, textvariable=doppler_threshold_var, width=6).pack(side=tk.LEFT)
        
        # Triple Sigma Option - COMMENTED OUT (replaced by adaptive mode)
        # self.use_triple_sigma_var = tk.BooleanVar(value=False)
        # ttk.Checkbutton(c_row1, text='启用三倍中误差', variable=self.use_triple_sigma_var).pack(side=tk.LEFT, padx=10)

        c_row2 = ttk.Frame(coarse_frame)
        c_row2.pack(fill=tk.X, pady=2)
        cmc_threshold_var = tk.DoubleVar(value=self.cmc_threshold)
        ttk.Label(c_row2, text='CMC变化阈值(米):').pack(side=tk.LEFT)
        ttk.Entry(c_row2, textvariable=cmc_threshold_var, width=6).pack(side=tk.LEFT, padx=5)

        # 2.3 Optional Processing
        opt_frame = ttk.LabelFrame(param_frame, text='可选处理', padding=5)
        opt_frame.pack(fill=tk.X, pady=5)
        
        self.doppler_enable_var = tk.BooleanVar(value=False)
        self.cci_enable_var = tk.BooleanVar(value=False)
        self.isb_enable_var = tk.BooleanVar(value=False)
        
        # Layout: Vertical checks
        ttk.Checkbutton(opt_frame, text='启用多普勒预测相位 (基于多普勒观测值预测并填补缺失的载波相位观测值)', variable=self.doppler_enable_var).pack(anchor='w')
        ttk.Checkbutton(opt_frame, text='启用码相不一致性(CCI)处理 (需要接收机文件作为基准, 校正载波相位观测值)', variable=self.cci_enable_var).pack(anchor='w')
        ttk.Checkbutton(opt_frame, text='启用ISB处理 (需要接收机文件作为基准, 校正BDS系统间偏差)', variable=self.isb_enable_var).pack(anchor='w')
        
        # ADD DOPPLER SMOOTHING HERE
        doppler_smoothing_var = tk.BooleanVar(value=self.doppler_smoothing_enabled)
        ds_row = ttk.Frame(opt_frame)
        ds_row.pack(fill=tk.X, pady=2, anchor='w')
        ttk.Checkbutton(ds_row, text='启用多普勒平滑伪距', variable=doppler_smoothing_var).pack(side=tk.LEFT)
        
        doppler_window_var = tk.IntVar(value=self.doppler_smoothing_window)
        ttk.Label(ds_row, text='平滑窗口:').pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(ds_row, textvariable=doppler_window_var, width=5).pack(side=tk.LEFT)
        
        doppler_threshold_smooth_var = tk.DoubleVar(value=self.doppler_smoothing_reset_threshold)
        ttk.Label(ds_row, text='重置阈值(米):').pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(ds_row, textvariable=doppler_threshold_smooth_var, width=5).pack(side=tk.LEFT)

        # 2.4 Code-Phase Inconsistency Processing (Details)
        cci_frame = ttk.LabelFrame(param_frame, text='码相不一致性处理', padding=5)
        cci_frame.pack(fill=tk.X, pady=5)
        
        cci_row = ttk.Frame(cci_frame)
        cci_row.pack(fill=tk.X)
        
        r_squared_var = tk.DoubleVar(value=self.r_squared_threshold)
        ttk.Label(cci_row, text='R方阈值:').pack(side=tk.LEFT)
        ttk.Entry(cci_row, textvariable=r_squared_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(cci_row, text='(默认: 0.5, 线性漂移判断)').pack(side=tk.LEFT)
        
        cv_threshold_var = tk.DoubleVar(value=self.cv_threshold)
        ttk.Label(cci_row, text='CV阈值:').pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(cci_row, textvariable=cv_threshold_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(cci_row, text='(默认: 0.6, ROC模型选择)').pack(side=tk.LEFT)
        
        phone_only_var = tk.BooleanVar(value=False)
        phone_only_min_var = tk.IntVar(value=self.phone_only_min_data_points)
        # Checkbox for phone only
        cci_row2 = ttk.Frame(cci_frame)
        cci_row2.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(cci_row2, text='启用手机独有卫星分析 (检测手机独有卫星的码相不一致性)', variable=phone_only_var).pack(side=tk.LEFT)


        # --- 3. Processing Progress ---
        progress_frame = ttk.LabelFrame(top, text='处理进度', padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        progress = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate') # Changed to determinate for manual stepping or keep indeterminate
        # User screenshot shows empty bar. Indeterminate is safer if we don't calculate total steps precisely.
        # But step() usage suggests determinate-like behavior.
        progress.pack(fill=tk.X, padx=5, pady=5)
        
        status_var = tk.StringVar(value='等待开始...')
        ttk.Label(progress_frame, textvariable=status_var, anchor='center').pack(fill=tk.X)


        # --- 4. Verify & Store Vars ---
        # Define update_threshold_hints before creating _tk_param_vars
        def update_threshold_hints(event=None):
            if threshold_mode_var.get() == '自适应':
                mode_hint_var.set('(输入值将作为保底阈值)')
                # Auto-fill recommended floor thresholds
                code_threshold_var.set(5.0)
                phase_threshold_var.set(1.5)
                doppler_threshold_var.set(3.0)
                cmc_threshold_var.set(2.0)
            else:
                mode_hint_var.set('')
                # Restore default fixed thresholds
                code_threshold_var.set(10.0)
                phase_threshold_var.set(3.0)
                doppler_threshold_var.set(5.0)
                cmc_threshold_var.set(4.0)
        
        # Bind mode selection event
        mode_combo.bind('<<ComboboxSelected>>', update_threshold_hints)
        
        _tk_param_vars = {
            'code_threshold_var': code_threshold_var,
            'phase_threshold_var': phase_threshold_var,
            'doppler_threshold_var': doppler_threshold_var,
            'cmc_threshold_var': cmc_threshold_var,
            'r_squared_var': r_squared_var,
            'cv_threshold_var': cv_threshold_var,
            'phone_only_min_var': phone_only_min_var,
            'doppler_smoothing_var': doppler_smoothing_var,
            'doppler_window_var': doppler_window_var,
            'doppler_threshold_smooth_var': doppler_threshold_smooth_var,
            'threshold_mode_var': threshold_mode_var,
            'template_var': template_var
        }
        
        self.phone_only_var = phone_only_var # Bind to instance for easy access if needed, or pass explicitly
        
        # Bind template selection
        template_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_template(template_var.get(), _tk_param_vars))


        # --- 5. Bottom Buttons ---
        btn_frame = ttk.Frame(top, padding=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Spacer
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(5, weight=1)
        
        # Center buttons as shown in screenshot: [Start] [BDS] [Select File] [Close]
        # Actually they are somewhat centered. Pack side=LEFT with spacing works or grid.
        # Let's use pack in a centered inner frame.
        center_btn = ttk.Frame(btn_frame)
        center_btn.pack(side=tk.TOP, pady=5)
        
        def _run_in_thread():
             # Logic to start thread
             if not phone_var.get():
                 messagebox.showwarning("提示", "请选择手机RINEX文件")
                 return
             
             # 1. 提前在主线程中获取所有 UI 变量的值
             params_snapshot = {k: v.get() for k, v in _tk_param_vars.items()}
             phone_path = phone_var.get()
             recv_path = recv_var.get()
             phone_only_val = self.phone_only_var.get()
             doppler_enable_val = self.doppler_enable_var.get()
             cci_enable_val = self.cci_enable_var.get()
             isb_enable_val = self.isb_enable_var.get()

             # Disable buttons?
             progress['value'] = 0
             status_var.set("准备中...")
             
             import threading
             # 2. 将值（而不是对象）传递给后台线程
             t = threading.Thread(target=self._preprocessing_task, args=(
                 phone_path, recv_path, params_snapshot, progress, status_var, top, phone_only_val, 
                 doppler_enable_val, cci_enable_val, isb_enable_val
             ), daemon=True)
             t.start()

        def run_isb_only_wrapper():
            if not recv_var.get():
                 messagebox.showwarning("警告", "ISB分析需要接收机文件")
                 return
            if not phone_var.get():
                 messagebox.showwarning("警告", "ISB分析需要手机文件")
                 return
            # Call ISB only thread logic (assuming helper exists or create one)
            # reusing existing logic if possible or notify user
            messagebox.showinfo("提示", "功能开发中...")

        ttk.Button(center_btn, text='开始预处理', command=_run_in_thread).pack(side=tk.LEFT, padx=10)
        ttk.Button(center_btn, text='BDS2/3 ISB分析', command=run_isb_only_wrapper).pack(side=tk.LEFT, padx=10)
        # "Select File" button at bottom? Screenshot shows it. Maybe it opens folder?
        ttk.Button(center_btn, text='选择文件', command=lambda: os.startfile(os.path.dirname(phone_var.get())) if phone_var.get() else None).pack(side=tk.LEFT, padx=10)
        ttk.Button(center_btn, text='关闭', command=top.destroy).pack(side=tk.LEFT, padx=10)

    def _browse_file(self, var, title):
        import tkinter.filedialog as fd
        file_types = [
            ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
            ("All Files", "*.*")
        ]
        fn = fd.askopenfilename(title=title, filetypes=file_types)
        if fn: var.set(fn)
    
    def apply_template(self, t_name, vars_dict):
        if t_name == '开阔环境':
            vars_dict['code_threshold_var'].set(8.0)
            vars_dict['phase_threshold_var'].set(3.0)
            vars_dict['doppler_threshold_var'].set(4.0)
            vars_dict['cmc_threshold_var'].set(3.0)
            vars_dict['r_squared_var'].set(0.6)
            vars_dict['cv_threshold_var'].set(0.6)
        elif t_name == '遮挡环境':
            vars_dict['code_threshold_var'].set(10.0)
            vars_dict['phase_threshold_var'].set(4.0)
            vars_dict['doppler_threshold_var'].set(5.0)
            vars_dict['cmc_threshold_var'].set(5.0)
            vars_dict['r_squared_var'].set(0.5)
            vars_dict['cv_threshold_var'].set(0.5)

    def save_params_ui(self, vars_dict):
        # Update self vars from UI then save
        # Extract values from Tkinter variables
        params_dict = {
            'code_threshold_var': vars_dict['code_threshold_var'].get(),
            'phase_threshold_var': vars_dict['phase_threshold_var'].get(),
            'doppler_threshold_var': vars_dict['doppler_threshold_var'].get(),
            'cmc_threshold_var': vars_dict['cmc_threshold_var'].get(),
            'r_squared_var': vars_dict['r_squared_var'].get(),
            'cv_threshold_var': vars_dict['cv_threshold_var'].get(),
            'phone_only_min_var': vars_dict['phone_only_min_var'].get(),
            'doppler_smoothing_var': vars_dict['doppler_smoothing_var'].get(),
            'doppler_window_var': vars_dict['doppler_window_var'].get(),
            'doppler_threshold_smooth_var': vars_dict['doppler_threshold_smooth_var'].get(),
            'threshold_mode_var': vars_dict['threshold_mode_var'].get()
        }
        self._update_self_from_dict(params_dict)
        import tkinter.filedialog as fd
        file_types = [("JSON Files", "*.json"), ("All Files", "*.*")]
        fn = fd.asksaveasfilename(title='保存', defaultextension='.json', filetypes=file_types)
        if fn: self.save_params_to_file(fn)

    def load_params_ui(self, vars_dict):
        import tkinter.filedialog as fd
        file_types = [("JSON Files", "*.json"), ("All Files", "*.*")]
        fn = fd.askopenfilename(title='加载', filetypes=file_types)
        if fn:
            self.load_params_from_file(fn)
            # Update UI vars
            vars_dict['code_threshold_var'].set(self.code_threshold)
            vars_dict['phase_threshold_var'].set(self.phase_threshold)
            vars_dict['doppler_threshold_var'].set(self.doppler_threshold)
            vars_dict['cmc_threshold_var'].set(self.cmc_threshold)
            vars_dict['r_squared_var'].set(self.r_squared_threshold)
            vars_dict['cv_threshold_var'].set(self.cv_threshold)
            vars_dict['phone_only_min_var'].set(self.phone_only_min_data_points)
            vars_dict['doppler_smoothing_var'].set(self.doppler_smoothing_enabled)
            vars_dict['doppler_window_var'].set(self.doppler_smoothing_window)
            vars_dict['doppler_threshold_smooth_var'].set(self.doppler_smoothing_reset_threshold)
            if 'threshold_mode_var' in vars_dict:
                vars_dict['threshold_mode_var'].set(self.threshold_mode)

    def _update_self_from_dict(self, params_dict):
        """Thread-safe update from a dictionary of values."""
        self.code_threshold = float(params_dict.get('code_threshold_var', self.code_threshold))
        self.phase_threshold = float(params_dict.get('phase_threshold_var', self.phase_threshold))
        self.doppler_threshold = float(params_dict.get('doppler_threshold_var', self.doppler_threshold))
        self.cmc_threshold = float(params_dict.get('cmc_threshold_var', self.cmc_threshold))
        self.r_squared_threshold = float(params_dict.get('r_squared_var', self.r_squared_threshold))
        self.cv_threshold = float(params_dict.get('cv_threshold_var', self.cv_threshold))
        self.phone_only_min_data_points = int(params_dict.get('phone_only_min_var', self.phone_only_min_data_points))
        self.doppler_smoothing_enabled = bool(params_dict.get('doppler_smoothing_var', self.doppler_smoothing_enabled))
        self.doppler_smoothing_window = int(params_dict.get('doppler_window_var', self.doppler_smoothing_window))
        self.doppler_smoothing_reset_threshold = float(params_dict.get('doppler_threshold_smooth_var', self.doppler_smoothing_reset_threshold))
        # Map UI value to internal mode
        mode_ui_value = params_dict.get('threshold_mode_var', '固定')
        self.threshold_mode = 'adaptive' if mode_ui_value == '自适应' else 'fixed'

    def _preprocessing_task(self, phone_path, recv_path, params_snapshot, progress, status_var, top, phone_only, 
                            doppler_enable, cci_enable, isb_enable):
        from tkinter import messagebox
        try:
            # 3. 使用快照更新 self 属性
            self._update_self_from_dict(params_snapshot)
            
            self.context.isb_enable = isb_enable
            
            # Set Context Flags
            self.context.results['flags'] = {
                'enable_doppler': bool(doppler_enable),
                'enable_cci': bool(cci_enable),
                'enable_isb': bool(isb_enable)
            }
            
            # Load Data
            self.load_phone_file(phone_path)
            self.context.input_path = phone_path  # CRITICAL FIX: Set input path
            
            if recv_path:
                self.load_receiver_file(recv_path)
                
            # Setup Arna_results directory structure
            base_dir = os.path.dirname(phone_path)
            obs_name = os.path.splitext(os.path.basename(phone_path))[0]
            
            # Root results dir: input_dir/Arna_results/filename/preprocessing/
            project_dir = os.path.join(base_dir, "Arna_results", obs_name, "preprocessing")
            if not os.path.exists(project_dir):
                os.makedirs(project_dir)
            
            # Subfolders
            dirs = {
                'pred': os.path.join(project_dir, 'doppler prediction'),
                'smooth': os.path.join(project_dir, 'doppler smoothing'),
                'cci': os.path.join(project_dir, 'code-carrier inconsistency'),
                'coarse': os.path.join(project_dir, 'Coarse error'),
                'isb': os.path.join(project_dir, 'BDS23_ISB')
            }
            # 创建目录
            for d in dirs.values():
                os.makedirs(d, exist_ok=True)
            
            # Helper to print consistent messages
            
            # 4. 线程安全的 UI 更新函数
            def log_step(step_name, detail=""):
                msg = f"[{step_name}] {detail}" if detail else f"[{step_name}]"
                print(msg)
                # 使用 after 确保在主线程更新 UI
                top.after(0, lambda m=msg: status_var.set(m))
            
            def update_progress(delta=0, value=None):
                if value is not None:
                    top.after(0, lambda v=value: progress.configure(value=v))
                elif delta != 0:
                    top.after(0, lambda d=delta: progress.step(d))
            
            results_summary = []
            log_step("预处理开始", f"输出目录: {project_dir}")
            
            # --- Chain & Naming Variables ---
            current_chain_path = self.context.input_path
            
            # Extract strict original name and extension
            # Note: os.path.splitext handles .25o correctly as extension
            input_basename = os.path.basename(self.context.input_path)
            original_base_name, original_ext = os.path.splitext(input_basename)
            
            # --- Step 1: Doppler Prediction ---
            if self.context.results['flags']['enable_doppler']:
                 log_step("多普勒预测", "正在执行...")
                 update_progress(15)
                 
                 # Naming: Chain + Suffix
                 chain_base = os.path.splitext(os.path.basename(current_chain_path))[0]
                 pred_filename = f"{chain_base}-doppler predicted{original_ext}"
                 pred_path = os.path.join(dirs['pred'], pred_filename)
                 log_step("多普勒预测", f"输出路径: {pred_path}")

                 # Run Prediction using CHAIN input
                 doppler_results = self.algo.run_doppler_phase_prediction(
                     self.context.observations_meters,
                     self.context.frequencies,
                     self.context.wavelengths,
                     original_rinex_path=current_chain_path, # Use chain
                     output_path=pred_path,
                     writer=self.writer
                 )
                 self.context.results['doppler_prediction'] = doppler_results
                 
                 # Log
                 target_log_path = os.path.join(dirs['pred'], "doppler_prediction.log")
                 with open(target_log_path, 'w', encoding='utf-8') as f:
                     f.write("=" * 70 + "\n")
                     f.write("多普勒预测处理日志\n")
                     f.write("=" * 70 + "\n\n")
                     f.write(f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                     f.write(f"输入文件: {os.path.abspath(current_chain_path)}\n")
                     f.write(f"输出文件: {os.path.abspath(pred_path)}\n\n")
                     f.write("算法原理:\n")
                     f.write("  基于当前历元的多普勒观测值，结合载波频率，预测并修补缺失的载波相位观测值，确保护跳和周跳检测的连续性。\n\n")
                     f.write("一、总体统计信息\n" + "-" * 40 + "\n")
                     f.write(f"总缺失相位观测值: {doppler_results.get('total_missing', 0)} 个\n")
                     f.write(f"成功预测修补: {doppler_results.get('total_predicted', 0)} 个\n\n")
                     
                     f.write("二、卫星级统计信息\n" + "-" * 40 + "\n")
                     sv_stats = doppler_results.get('sv_missing_stats', {})
                     if sv_stats:
                         for sv in sorted(sv_stats.keys()):
                             stats = sv_stats[sv]
                             f.write(f"卫星 {sv:3}: 缺失 {stats.get('missing', 0):3} 个, 成功修补 {stats.get('predicted', 0):3} 个\n")
                     else:
                         f.write("无缺失或修补记录\n")
                     
                     if 'correction_log' in doppler_results and doppler_results['correction_log']:
                         f.write("\n三、详细修补记录\n" + "-" * 40 + "\n")
                         for log_item in doppler_results.get('correction_log', []):
                             f.write(f"Epoch {log_item['epoch_idx']:4} Time {log_item['time']} Sat {log_item['sat_id']} {log_item['freq']}: Predicted {log_item['predicted_phase_cycle']}\n")

                 # Update Chain
                 if os.path.exists(pred_path):
                      print(f"[系统] 重新加载中间文件: {pred_path}")
                      self.load_phone_file(pred_path)
                      current_chain_path = pred_path # Update chain
                      self.context.input_path = pred_path # Legacy compat
                 
                 results_summary.append("多普勒预测: 完成")


            # --- Step 2: Doppler Smoothing ---
            if self.doppler_smoothing_enabled:
                 log_step("多普勒平滑", "正在执行...")
                 update_progress(15)
                 
                 # Naming: Chain + Suffix
                 chain_base = os.path.splitext(os.path.basename(current_chain_path))[0]
                 smooth_filename = f"{chain_base}-doppler smoothed{original_ext}"
                 smooth_path = os.path.join(dirs['smooth'], smooth_filename)
                 
                 # Apply to CURRENT observations (loaded from chain)
                 smoothing_res = self.algo.apply_doppler_smoothing(
                    self.context.observations_meters,
                    max_window=self.doppler_smoothing_window,
                    reset_threshold_m=self.doppler_smoothing_reset_threshold,
                    input_file_name=original_base_name + original_ext
                 )
                 self.context.results['doppler_smoothing'] = smoothing_res
                 
                 # Update memory
                 smoothed_obs = smoothing_res['smoothed_observations']
                 for sat_id in smoothed_obs:
                     for freq in smoothed_obs[sat_id]:
                         if sat_id in self.context.observations_meters and freq in self.context.observations_meters[sat_id]:
                             self.context.observations_meters[sat_id][freq]['code_original'] = self.context.observations_meters[sat_id][freq].get('code', [])
                             self.context.observations_meters[sat_id][freq]['code'] = smoothed_obs[sat_id][freq]['code_smoothed']
                 
                 # Write File using CHAIN input as template
                 w_res = self.writer.write_doppler_smoothed_rinex(
                     current_chain_path, 
                     smooth_path, 
                     smoothed_obs
                 )
                 
                 # Log
                 log_path = os.path.join(dirs['smooth'], "doppler_smoothing.log")
                 with open(log_path, 'w', encoding='utf-8') as lf:
                     lf.write("=" * 70 + "\n")
                     lf.write("多普勒平滑处理日志\n")
                     lf.write("=" * 70 + "\n\n")
                     lf.write(f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                     lf.write(f"输入文件: {os.path.abspath(current_chain_path)}\n")
                     lf.write(f"输出文件: {os.path.abspath(smooth_path)}\n\n")
                     lf.write("算法原理:\n")
                     lf.write("  使用多普勒积分对伪距进行平滑处理。采用Hatch滤波公式，结合多普勒观测值得出的历元间距离变化量，降低伪距噪声，提高定位精度。\n\n")
                     lf.write(smoothing_res.get('log', ''))
                 
                 log_step("多普勒平滑", "生成日志: doppler_smoothing.log")
                     
                 # Update Chain
                 if os.path.exists(smooth_path):
                     log_step("多普勒平滑", f"生成文件: {smooth_filename}")
                     self.load_phone_file(smooth_path)
                     current_chain_path = smooth_path
                     self.context.input_path = smooth_path
                 
                 results_summary.append("多普勒平滑: 完成")

            # --- Step 3: CCI ---
            if self.context.results['flags']['enable_cci']:
                log_step("CCI建模与校正", "正在计算...")
                update_progress(20)
                
                # Calculations based on CURRENT observations
                rc_cmc = self.calculator.calculate_receiver_cmc({'receiver_observations': self.context.receiver_observations, 'frequencies': self.context.frequencies, 'wavelengths': self.context.wavelengths})
                self.context.results['receiver_cmc'] = rc_cmc
                
                raw_diffs = self.calculator.calculate_code_phase_differences({'observations_meters': self.context.observations_meters, 'frequencies': self.context.frequencies, 'wavelengths': self.context.wavelengths})
                dres = self.algo.calculate_dcmc(rc_cmc, raw_diffs, r_squared_threshold=self.r_squared_threshold, enable_phone_only_analysis=phone_only, phone_only_min_data_points=self.phone_only_min_data_points)
                self.context.results['dcmc'] = dres.get('dcmc', {})
                cci = self.algo.extract_cci_series(self.context.results['dcmc'])
                self.context.results['cci_series'] = cci
                roc = self.algo.calculate_roc_model(cci, cv_threshold=self.cv_threshold, enable_phone_only_analysis=phone_only, phone_only_linear_drift=dres.get('meta', {}).get('linear_drift_detailed'))
                self.context.results['roc_model'] = roc
                
                # Naming: Chain + Suffix
                chain_base = os.path.splitext(os.path.basename(current_chain_path))[0]
                cci_filename = f"{chain_base}-cc inconsistency{original_ext}"
                cci_path = os.path.join(dirs['cci'], cci_filename)
                
                log_step("CCI建模与校正", "应用校正...")
                corrected = self.algo.correct_phase_observations(
                    self.context.observations_meters, 
                    roc, 
                    self.context.results['dcmc'], 
                    enable_phone_only_analysis=phone_only, 
                    original_rinex_path=current_chain_path, 
                    writer=self.writer,
                    output_path=cci_path 
                )
                self.context.results['corrected_phase'] = corrected
                
                # Generate CCI Processing Log
                cci_log_path = os.path.join(dirs['cci'], "code_phase_inconsistency_processing.log")
                try:
                    with open(cci_log_path, 'w', encoding='utf-8') as lf:
                        lf.write("=" * 70 + "\n")
                        lf.write("码相不一致性建模和校正处理日志\n")
                        lf.write("=" * 70 + "\n\n")
                        lf.write(f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        lf.write(f"输入文件: {os.path.abspath(self.context.input_path)}\n")
                        lf.write(f"输出文件: {os.path.abspath(cci_path)}\n")
                        lf.write(f"接收机RINEX参考文件: {os.path.abspath(recv_path) if recv_path else 'N/A'}\n\n")
                        lf.write("算法原理:\n")
                        lf.write("  利用站间单差CMC（dCMC）序列，提取并建模接收机内部码相不一致性（ROC模型）。通过该模型对观测到的载波相位进行改正，消除系统性的码相漂移。\n\n")
                        
                        # dCMC统计
                        lf.write("=" * 50 + "\n")
                        lf.write("1. 站间单差CMC (dCMC) 计算\n")
                        lf.write("=" * 50 + "\n")
                        dcmc_data = self.context.results.get('dcmc', {})
                        lf.write(f"处理卫星-频率组合数: {sum(len(v) for v in dcmc_data.values())}\n")
                        lf.write(f"通过线性漂移检查的组合数: {len(dcmc_data)}\n\n")
                        
                        # ROC模型统计
                        lf.write("=" * 50 + "\n")
                        lf.write("2. ROC (Rate of Change) 模型\n")
                        lf.write("=" * 50 + "\n")
                        roc_model = self.context.results.get('roc_model', {})
                        lf.write(f"ROC模型总数: {len(roc_model)}\n")
                        system_level = sum(1 for v in roc_model.values() if v.get('model_type') == 'system_level')
                        individual_level = sum(1 for v in roc_model.values() if v.get('model_type') == 'individual_level')
                        lf.write(f"系统级模型: {system_level}\n")
                        lf.write(f"个体级模型: {individual_level}\n\n")
                        
                        # 详细ROC模型信息
                        lf.write("ROC模型详情:\n")
                        lf.write("-" * 50 + "\n")
                        for key, model in roc_model.items():
                            lf.write(f"{key}:\n")
                            lf.write(f"  ROC率: {model.get('roc_rate', 0):.6e} m/s\n")
                            lf.write(f"  模型类型: {model.get('model_type', 'N/A')}\n")
                            lf.write(f"  质量等级: {model.get('quality_level', 'N/A')}\n")
                            lf.write(f"  变异系数: {model.get('roc_cv', 0):.4f}\n")
                            lf.write(f"  参与卫星数: {model.get('num_satellites', 0)}\n\n")
                        
                        # 相位校正统计
                        if 'writer_result' in corrected:
                            wr = corrected['writer_result']
                            lf.write("=" * 50 + "\n")
                            lf.write("3. 载波相位校正\n")
                            lf.write("=" * 50 + "\n")
                            lf.write(f"总修改观测值数: {wr.get('total_modifications', 0)}\n")
                            lf.write(f"修改的卫星数: {len(wr.get('modification_details', {}).keys())}\n\n")
                        
                        lf.write("=" * 70 + "\n")
                        lf.write("处理完成\n")
                        lf.write("=" * 70 + "\n")
                    log_step("CCI建模与校正", f"生成日志: code_phase_inconsistency_processing.log")
                except Exception as e:
                    print(f"Warning: Failed to generate CCI log: {e}")
                
                # Update Chain
                if os.path.exists(cci_path):
                     log_step("CCI建模与校正", f"生成文件: {cci_filename}")
                     self.load_phone_file(cci_path)
                     current_chain_path = cci_path
                     self.context.input_path = cci_path
                
                results_summary.append("CCI建模: 完成")

            # --- Step 4: Cleaning 1 (CMC - cleaned1) ---
            log_step("CMC异常剔除", "正在执行...")
            update_progress(10)
            
            # 保存当前observations_meters的深拷贝，用于CMC日志生成
            import copy
            cmc_observations_snapshot = copy.deepcopy(self.context.observations_meters)
            
            # Call CMC threshold processing with mode support
            cmc_result = self.coarse.process_cmc_threshold(
                self.context.observations_meters, 
                self.cmc_threshold,
                mode=self.threshold_mode
            )
            
            # Handle both old and new return formats
            if isinstance(cmc_result, dict) and 'cmc_flags' in cmc_result:
                cmc_flags = cmc_result['cmc_flags']
                cmc_calculated_thresholds = cmc_result.get('calculated_thresholds', {})
            else:
                # Legacy format (dict directly)
                cmc_flags = cmc_result
                cmc_calculated_thresholds = {}
            
            self.context.results['cmc_flags'] = cmc_flags
            
            # Naming: cleaned1-Chain...
            chain_base = os.path.splitext(os.path.basename(current_chain_path))[0]
            clean1_filename = f"cleaned1-{chain_base}{original_ext}" 
            clean1_path = os.path.join(dirs['coarse'], clean1_filename)
            
            c1_res = self.writer.write_cleaned_rinex(
                 current_chain_path, 
                 output_path=clean1_path, 
                 double_diffs={}, 
                 triple_errors={}, 
                 enable_cci=True, 
                 cmc_flags=cmc_flags,
                 observations_meters=cmc_observations_snapshot,  # 使用快照而不是当前数据
                 cmc_threshold=self.cmc_threshold,
                 threshold_mode=self.threshold_mode,
                 calculated_thresholds=cmc_calculated_thresholds
            )
            
            # Update Chain
            if c1_res.get('output_path') and os.path.exists(c1_res['output_path']):
                 log_step("CMC异常剔除", f"生成文件: {clean1_filename}")
                 self.load_phone_file(c1_res['output_path'])
                 current_chain_path = c1_res['output_path']
                 self.context.input_path = c1_res['output_path']
            
            
            # Log generated by writer.py
            log_step("CMC异常剔除", "生成日志: code_phase_cleaning.log")
            
            results_summary.append("CMC剔除: 完成")

            # --- Step 5: Cleaning 2 (Double Diff - cleaned2) ---
            log_step("双差异常剔除", "正在执行...")
            update_progress(10)
            dd = self.coarse.process_epoch_double_diff(self.context.observations_meters)
            self.context.results['epoch_double_diffs'] = dd
            
            # Call check_triple_median_error with adaptive mode support
            triple = self.coarse.check_triple_median_error(
                dd, 
                use_triple_sigma=False,  # using mode='adaptive' instead
                mode=self.threshold_mode,
                adaptive_floor_thresholds={
                    'code': self.code_threshold,
                    'phase': self.phase_threshold,
                    'doppler': self.doppler_threshold
                },
                max_threshold_limit={
                    'code': self.code_threshold,
                    'phase': self.phase_threshold,
                    'doppler': self.doppler_threshold
                }
            )
            self.context.results['triple_errors'] = triple
            
            # Naming: cleaned2-Chain...
            chain_base = os.path.splitext(os.path.basename(current_chain_path))[0]
            clean2_filename = f"cleaned2-{chain_base}{original_ext}"
            clean2_path = os.path.join(dirs['coarse'], clean2_filename)
            
            c2_res = self.writer.write_cleaned_rinex(
                 current_chain_path, 
                 output_path=clean2_path, 
                 double_diffs=dd, 
                 triple_errors=triple, 
                 max_threshold_limit={
                     'code': self.code_threshold,
                     'phase': self.phase_threshold,
                     'doppler': self.doppler_threshold
                 },
                 enable_cci=False,
                 threshold_mode=self.threshold_mode
            )

            # Update Chain
            if c2_res.get('output_path') and os.path.exists(c2_res['output_path']):
                 log_step("双差异常剔除", f"生成文件: {clean2_filename}")
                 self.load_phone_file(c2_res['output_path'])
                 current_chain_path = c2_res['output_path']
                 self.context.input_path = c2_res['output_path']
            
            results_summary.append("双差剔除: 完成")
            
            
            # Log generated by writer.py
            log_step("双差异常剔除", "生成日志: double_diffs_cleaning.log")


            # --- Step 6: ISB ---
            if self.context.results['flags']['enable_isb'] and self.context.receiver_observations:
                log_step("ISB分析与校正", "正在执行...")
                update_progress(10)
                isb_data = self.algo.run_prepare_isb_data(self.context.observations_meters, self.context.receiver_observations)
                ref = self.algo.run_select_reference_satellite(isb_data)
                stable = self.algo.run_filter_stable_satellites(isb_data)
                isb_res = self.algo.run_calculate_isb_double_difference(isb_data, ref, stable)
                self.context.results['isb_analysis'] = isb_res
                
                # Naming: Chain + Suffix
                chain_base = os.path.splitext(os.path.basename(current_chain_path))[0]
                isb_filename = f"{chain_base}-isb{original_ext}"
                isb_path = os.path.join(dirs['isb'], isb_filename)
                
                self.algo.run_correct_isb_and_generate_rinex(
                     isb_res, 
                     input_rinex_path=current_chain_path, 
                     output_path=isb_path, 
                     writer=self.writer
                )
                
                # Update Chain
                if os.path.exists(isb_path):
                     log_step("ISB分析与校正", f"生成文件: {isb_filename}")
                     self.load_phone_file(isb_path)
                     current_chain_path = isb_path
                     self.context.input_path = isb_path
                     
                results_summary.append("ISB分析: 完成")
                
                # Generate ISB Analysis Log
                isb_log_path = os.path.join(dirs['isb'], "isb_analysis.log")
                try:
                    with open(isb_log_path, 'w', encoding='utf-8') as lf:
                        lf.write("=" * 70 + "\n")
                        lf.write("BDS-2/3系统间偏差(ISB)分析日志\n")
                        lf.write("=" * 70 + "\n\n")
                        lf.write(f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        lf.write(f"输入文件: {os.path.abspath(self.context.input_path)}\n")
                        lf.write(f"输出文件: {os.path.abspath(isb_path)}\n")
                        lf.write(f"接收机RINEX参考文件: {os.path.abspath(recv_path) if recv_path else 'N/A'}\n\n")
                        lf.write("算法原理:\n")
                        lf.write("  针对BDS-2和BDS-3系统间的偏差（ISB）进行估计和校正。通过选取稳定的参考卫星，计算双差观测值中的系统性偏差，并在观测值中进行改正。\n\n")
                        
                        # ISB统计
                        lf.write("=" * 50 + "\n")
                        lf.write("ISB估计结果\n")
                        lf.write("=" * 50 + "\n")
                        if isb_res:
                            lf.write(f"参考卫星: {ref}\n")
                            lf.write(f"ISB平均值: {isb_res.get('isb_mean', 0):.4f} 米\n")
                            lf.write(f"ISB标准差: {isb_res.get('isb_std', 0):.4f} 米\n")
                            lf.write(f"ISB中位数: {isb_res.get('isb_median', 0):.4f} 米\n")
                            lf.write(f"有效历元数: {len(isb_res.get('isb_epochs', []))}\n\n")
                            
                            # BDS-2稳定卫星
                            lf.write("BDS-2稳定卫星:\n")
                            bds2_stable = stable.get('bds2', [])
                            lf.write(f"  {', '.join(bds2_stable) if bds2_stable else '无'}\n\n")
                            
                            # BDS-3稳定卫星
                            lf.write("BDS-3稳定卫星:\n")
                            bds3_stable = stable.get('bds3', [])
                            lf.write(f"  {', '.join(bds3_stable) if bds3_stable else '无'}\n\n")
                            
                            # 双差详情
                            lf.write("=" * 50 + "\n")
                            lf.write("双差ISB估计详情\n")
                            lf.write("=" * 50 + "\n")
                            isb_estimates = isb_res.get('isb_estimates', [])
                            lf.write(f"历元数: {len(isb_estimates)}\n\n")
                            if isb_estimates:
                                lf.write("各历元ISB值:\n")
                                for i, isb_val in enumerate(isb_estimates[:10]):  # 只显示前10个
                                    lf.write(f"  历元 {i+1}: {isb_val:.4f} m\n")
                                if len(isb_estimates) > 10:
                                    lf.write(f"  ... 还有 {len(isb_estimates)-10} 个历元\n")
                        else:
                            lf.write("未能成功估计ISB\n")
                        
                        lf.write("\n" + "=" * 70 + "\n")
                        lf.write("处理完成\n")
                        lf.write("=" * 70 + "\n")
                    log_step("ISB分析与校正", "生成日志: isb_analysis.log")
                except Exception as e:
                    print(f"Warning: Failed to generate ISB log: {e}")
            
            log_step("完成")
            update_progress(value=100)
            
            summary_text = "\n".join(results_summary)
            top.after(0, lambda: messagebox.showinfo("处理结果", f"处理完成! 结果保存在:\n{project_dir}\n\n{summary_text}"))
            
        except Exception as e:
            traceback.print_exc()
            err = str(e)
            top.after(0, lambda: messagebox.showerror("错误", f"处理中发生错误: {err}"))
        finally:
            top.after(0, lambda: status_var.set("就绪"))
            top.after(0, lambda: progress.stop())

    def write_cleaned_rinex(self, original_path: str, output_path: Optional[str] = None, enable_cci: bool = True) -> Dict[str, Any]:
        result = self.writer.write_cleaned_rinex(original_path, output_path, self.context.results.get('epoch_double_diffs', {}), self.context.results.get('triple_errors', {}), enable_cci=enable_cci, cmc_flags=None)
        return result

    def write_corrected_rinex(self, original_path: str, output_path: Optional[str] = None, roc_model: Optional[Dict] = None) -> Dict[str, Any]:
        result = self.writer.write_corrected_rinex(original_path, output_path, self.context.results.get('corrected_phase', {}), roc_model or self.context.results.get('roc_model'))
        return result
