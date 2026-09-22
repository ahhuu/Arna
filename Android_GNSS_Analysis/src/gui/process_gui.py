from typing import Optional, Dict, Any
import os
import datetime
import traceback
import copy

# Delay imports that may require a display until runtime
from src.core.context import AnalysisContext
from src.data.reader import RinexReader
from src.data.writer import RinexWriter
from src.processing.calculator import MetricCalculator
from src.processing.coarse_error import CoarseErrorProcessor
from src.processing.advanced_algo import CoreAlgorithmProcessor
from src.visualization.prediction_artifacts import save_prediction_artifacts
from src.reporting.reporter import ReportGenerator


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

        # Doppler phase prediction.  Sigma values are exported for a future
        # PPP covariance model; RINEX observation flags are left untouched.
        self.doppler_prediction_max_length = 5
        self.doppler_prediction_max_interval_factor = 1.5
        self.doppler_prediction_real_sigma_m = 0.03
        self.doppler_prediction_first_sigma_m = 0.10
        self.doppler_prediction_sigma_growth_m = 0.05
        self.doppler_prediction_error_plot_max_length = 20

        # Global preprocessing scope.  Excluded observations remain unchanged
        # in output RINEX files but are absent from calculations/artifacts.
        self.processing_systems = ['G', 'R', 'E', 'C', 'J', 'I', 'S']
        self.processing_frequencies = sorted({
            freq for freq_map in self.context.frequencies.values() for freq in freq_map
        })

        # Pseudorange multipath correction (intentionally disabled by default)
        self.pseudorange_multipath_enabled = False
        self.pseudorange_multipath_pair = '自动选择'
        self.pseudorange_multipath_gain = 0.5
        self.pseudorange_multipath_min_arc_epochs = 30
        self.pseudorange_multipath_arc_gap_seconds = 3.0
        self.pseudorange_multipath_mad_scale = 6.0
        self.pseudorange_multipath_mad_floor_m = 0.5
        self.pseudorange_multipath_max_correction_m = 10.0
        self.pseudorange_multipath_reject_half_cycle = True
        
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
            'doppler_prediction_max_length': self.doppler_prediction_max_length,
            'doppler_prediction_max_interval_factor': self.doppler_prediction_max_interval_factor,
            'doppler_prediction_real_sigma_m': self.doppler_prediction_real_sigma_m,
            'doppler_prediction_first_sigma_m': self.doppler_prediction_first_sigma_m,
            'doppler_prediction_sigma_growth_m': self.doppler_prediction_sigma_growth_m,
            'doppler_prediction_error_plot_max_length': self.doppler_prediction_error_plot_max_length,
            'processing_systems': self.processing_systems,
            'processing_frequencies': self.processing_frequencies,
            'pseudorange_multipath_enabled': self.pseudorange_multipath_enabled,
            'pseudorange_multipath_pair': self.pseudorange_multipath_pair,
            'pseudorange_multipath_gain': self.pseudorange_multipath_gain,
            'pseudorange_multipath_min_arc_epochs': self.pseudorange_multipath_min_arc_epochs,
            'pseudorange_multipath_arc_gap_seconds': self.pseudorange_multipath_arc_gap_seconds,
            'pseudorange_multipath_mad_scale': self.pseudorange_multipath_mad_scale,
            'pseudorange_multipath_mad_floor_m': self.pseudorange_multipath_mad_floor_m,
            'pseudorange_multipath_max_correction_m': self.pseudorange_multipath_max_correction_m,
            'pseudorange_multipath_reject_half_cycle': self.pseudorange_multipath_reject_half_cycle,
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
        self.doppler_prediction_max_length = int(cfg.get('doppler_prediction_max_length', self.doppler_prediction_max_length))
        self.doppler_prediction_max_interval_factor = float(cfg.get('doppler_prediction_max_interval_factor', self.doppler_prediction_max_interval_factor))
        self.doppler_prediction_real_sigma_m = float(cfg.get('doppler_prediction_real_sigma_m', self.doppler_prediction_real_sigma_m))
        self.doppler_prediction_first_sigma_m = float(cfg.get('doppler_prediction_first_sigma_m', self.doppler_prediction_first_sigma_m))
        self.doppler_prediction_sigma_growth_m = float(cfg.get('doppler_prediction_sigma_growth_m', self.doppler_prediction_sigma_growth_m))
        self.doppler_prediction_error_plot_max_length = int(cfg.get('doppler_prediction_error_plot_max_length', self.doppler_prediction_error_plot_max_length))
        self.processing_systems = [str(value) for value in cfg.get('processing_systems', self.processing_systems)]
        self.processing_frequencies = [str(value) for value in cfg.get('processing_frequencies', self.processing_frequencies)]
        self.pseudorange_multipath_enabled = bool(cfg.get('pseudorange_multipath_enabled', self.pseudorange_multipath_enabled))
        self.pseudorange_multipath_pair = str(cfg.get('pseudorange_multipath_pair', self.pseudorange_multipath_pair))
        self.pseudorange_multipath_gain = float(cfg.get('pseudorange_multipath_gain', self.pseudorange_multipath_gain))
        self.pseudorange_multipath_min_arc_epochs = int(cfg.get('pseudorange_multipath_min_arc_epochs', self.pseudorange_multipath_min_arc_epochs))
        self.pseudorange_multipath_arc_gap_seconds = float(cfg.get('pseudorange_multipath_arc_gap_seconds', self.pseudorange_multipath_arc_gap_seconds))
        self.pseudorange_multipath_mad_scale = float(cfg.get('pseudorange_multipath_mad_scale', self.pseudorange_multipath_mad_scale))
        self.pseudorange_multipath_mad_floor_m = float(cfg.get('pseudorange_multipath_mad_floor_m', self.pseudorange_multipath_mad_floor_m))
        self.pseudorange_multipath_max_correction_m = float(cfg.get('pseudorange_multipath_max_correction_m', self.pseudorange_multipath_max_correction_m))
        self.pseudorange_multipath_reject_half_cycle = bool(cfg.get('pseudorange_multipath_reject_half_cycle', self.pseudorange_multipath_reject_half_cycle))
        self.use_triple_sigma = bool(cfg.get('use_triple_sigma', self.use_triple_sigma))
        self.threshold_mode = cfg.get('threshold_mode', 'fixed')

    def load_phone_file(self, file_path: str) -> Dict[str, Any]:
        res = self.reader.read_phone_rinex(file_path, frequencies=self.context.frequencies, glonass_k_map=self.context.glonass_k_map)
        self.context.observations_meters = self._filter_processing_scope(res['observations_meters'])
        # Update context wavelengths with those found in file (e.g. dynamic GLONASS)
        if 'satellite_wavelengths' in res:
             for sat, freqs in res['satellite_wavelengths'].items():
                 if sat not in self.context.wavelengths:
                     self.context.wavelengths[sat] = {}
                 self.context.wavelengths[sat].update(freqs)
        return res

    def load_receiver_file(self, file_path: str) -> Dict[str, Any]:
        res = self.reader.read_receiver_rinex(file_path, frequencies=self.context.frequencies, glonass_k_map=self.context.glonass_k_map)
        self.context.receiver_observations = self._filter_processing_scope(res['receiver_observations'])
        if 'satellite_wavelengths' in res:
             for sat, freqs in res['satellite_wavelengths'].items():
                 if sat not in self.context.wavelengths:
                     self.context.wavelengths[sat] = {}
                 self.context.wavelengths[sat].update(freqs)
        return res

    def _filter_processing_scope(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Select the global processing scope without mutating parsed data."""
        systems = set(self.processing_systems or [])
        frequencies = set(self.processing_frequencies or [])
        filtered = {}
        for sat_id, freq_map in (observations or {}).items():
            if systems and (not sat_id or sat_id[0] not in systems):
                continue
            selected_freqs = {
                freq: values for freq, values in (freq_map or {}).items()
                if not frequencies or freq in frequencies
            }
            if selected_freqs:
                filtered[sat_id] = selected_freqs
        return filtered

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
            'doppler_smoothing_reset_threshold': self.doppler_smoothing_reset_threshold,
            'pseudorange_multipath_enabled': self.pseudorange_multipath_enabled,
            'pseudorange_multipath_pair': self.pseudorange_multipath_pair,
            'pseudorange_multipath_gain': self.pseudorange_multipath_gain,
            'pseudorange_multipath_min_arc_epochs': self.pseudorange_multipath_min_arc_epochs,
            'pseudorange_multipath_arc_gap_seconds': self.pseudorange_multipath_arc_gap_seconds,
            'pseudorange_multipath_mad_scale': self.pseudorange_multipath_mad_scale,
            'pseudorange_multipath_mad_floor_m': self.pseudorange_multipath_mad_floor_m,
            'pseudorange_multipath_max_correction_m': self.pseudorange_multipath_max_correction_m,
            'pseudorange_multipath_reject_half_cycle': self.pseudorange_multipath_reject_half_cycle
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

        # Apply pseudorange multipath correction before Doppler/Hatch smoothing.
        if self.pseudorange_multipath_enabled:
            multipath_result = self.algo.apply_pseudorange_multipath_correction(
                self.context.observations_meters,
                freq_pair=self.pseudorange_multipath_pair,
                gain=self.pseudorange_multipath_gain,
                min_arc_epochs=self.pseudorange_multipath_min_arc_epochs,
                arc_gap_seconds=self.pseudorange_multipath_arc_gap_seconds,
                mad_scale=self.pseudorange_multipath_mad_scale,
                mad_floor_m=self.pseudorange_multipath_mad_floor_m,
                max_correction_m=self.pseudorange_multipath_max_correction_m,
                reject_half_cycle=self.pseudorange_multipath_reject_half_cycle,
            )
            self.context.results['multipath_correction'] = multipath_result
            self.context.observations_meters = multipath_result['corrected_observations']
        else:
            self.context.results['multipath_correction'] = None

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

        return {
            'receiver_cmc': rc_cmc,
            'raw_diffs': raw_diffs,
            'dcmc': self.context.results['dcmc'],
            'roc': roc,
            'corrected': self.context.results['corrected_phase'],
            'multipath': self.context.results.get('multipath_correction'),
            'epoch_dd': dd,
            'triple': triple,
        }

    def show(self, parent):
        """Display a Tkinter-based preprocessing window matching Rinex_Analysis_Modules style."""
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
        except Exception:
            return

        top = tk.Toplevel(parent)
        top.title('数据预处理')
        top.geometry('900x1000') # Extra space for optional multipath controls
        top.transient(parent)
        top.grab_set()

        class _HoverTip:
            """Small delayed tooltip that works for both ttk and tk widgets."""
            def __init__(self, widget, text, delay_ms=450):
                self.widget = widget
                self.text = text
                self.delay_ms = delay_ms
                self.after_id = None
                self.window = None
                widget.bind('<Enter>', self._schedule, add='+')
                widget.bind('<Leave>', self._hide, add='+')
                widget.bind('<ButtonPress>', self._hide, add='+')

            def _schedule(self, _event=None):
                self._cancel()
                self.after_id = self.widget.after(self.delay_ms, self._show)

            def _cancel(self):
                if self.after_id is not None:
                    try:
                        self.widget.after_cancel(self.after_id)
                    except Exception:
                        pass
                    self.after_id = None

            def _show(self):
                self.after_id = None
                if self.window is not None or not self.widget.winfo_exists():
                    return
                self.window = tk.Toplevel(self.widget)
                self.window.wm_overrideredirect(True)
                try:
                    x = self.widget.winfo_rootx() + 15
                    y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
                    self.window.wm_geometry(f'+{x}+{y}')
                except Exception:
                    pass
                label = tk.Label(
                    self.window, text=self.text, justify=tk.LEFT, wraplength=430,
                    background='#fffde7', foreground='#202020', relief=tk.SOLID,
                    borderwidth=1, padx=8, pady=6, font=('', 9),
                )
                label.pack()

            def _hide(self, _event=None):
                self._cancel()
                if self.window is not None:
                    try:
                        self.window.destroy()
                    except Exception:
                        pass
                    self.window = None

        tooltip_refs = []

        def _tip(widget, text):
            # Keep a reference for the lifetime of this window.
            tooltip_refs.append(_HoverTip(widget, text))
            return widget

        top._tooltip_refs = tooltip_refs

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
        _tip(ttk.Entry(phone_row, textvariable=phone_var),
             '待处理的手机端 RINEX 观测文件。多普勒预测、粗差处理、平滑和多路径改正都以此文件为输入。').pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        _tip(ttk.Button(phone_row, text='浏览', command=lambda: self._browse_file(phone_var, '选择手机RINEX文件')),
             '浏览并选择手机 RINEX 观测文件。').pack(side=tk.LEFT)

        # Receiver File
        recv_row = ttk.Frame(file_frame)
        recv_row.pack(fill=tk.X, pady=2)
        ttk.Label(recv_row, text='接收机RINEX文件(CCI建模和ISB分析必需):', width=40).pack(side=tk.LEFT) # Adjusted width
        _tip(ttk.Entry(recv_row, textvariable=recv_var),
             '可选的参考接收机 RINEX。启用 CCI 建模或 BDS2/3 ISB 分析时必须提供；仅做手机数据处理时可留空。').pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        _tip(ttk.Button(recv_row, text='浏览', command=lambda: self._browse_file(recv_var, '选择接收机RINEX文件')),
             '浏览并选择参考接收机 RINEX 文件。').pack(side=tk.LEFT)


        # --- 2. Parameter Settings Section ---
        param_frame = ttk.LabelFrame(top, text='参数设置', padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        # Global processing scope (applies to every preprocessing step).
        scope_frame = ttk.LabelFrame(param_frame, text='处理范围（全流程）', padding=5)
        scope_frame.pack(fill=tk.X, pady=2)
        processing_systems_var = tk.StringVar(value=','.join(self.processing_systems))
        processing_frequencies_var = tk.StringVar(value=','.join(self.processing_frequencies))
        scope_summary_var = tk.StringVar()

        system_names = {'G': 'GPS', 'R': 'GLONASS', 'E': 'Galileo', 'C': 'BDS',
                        'J': 'QZSS', 'I': 'NavIC', 'S': 'SBAS'}

        def update_scope_summary():
            systems = [item for item in processing_systems_var.get().split(',') if item]
            freqs = [item for item in processing_frequencies_var.get().split(',') if item]
            system_text = '、'.join(system_names.get(item, item) for item in systems) or '无'
            scope_summary_var.set(f'卫星系统: {system_text}；频率: {"、".join(freqs) or "无"}')
        processing_systems_var._scope_update = update_scope_summary

        def open_scope_selector():
            dialog = tk.Toplevel(top)
            dialog.title('选择预处理卫星系统和频率')
            dialog.transient(top)
            dialog.grab_set()
            dialog.resizable(False, False)

            selected_systems = set(item for item in processing_systems_var.get().split(',') if item)
            selected_freqs = set(item for item in processing_frequencies_var.get().split(',') if item)
            system_vars = {}
            freq_vars = {}

            system_box = ttk.LabelFrame(dialog, text='卫星系统', padding=8)
            system_box.pack(fill=tk.X, padx=10, pady=(10, 5))
            for column, code in enumerate(system_names):
                variable = tk.BooleanVar(value=code in selected_systems)
                system_vars[code] = variable
                ttk.Checkbutton(system_box, text=f'{system_names[code]} ({code})', variable=variable).grid(
                    row=column // 4, column=column % 4, sticky='w', padx=8, pady=3)

            freq_box = ttk.LabelFrame(dialog, text='观测频率', padding=8)
            freq_box.pack(fill=tk.X, padx=10, pady=5)
            all_freqs = sorted({freq for values in self.context.frequencies.values() for freq in values})
            for index, freq in enumerate(all_freqs):
                variable = tk.BooleanVar(value=freq in selected_freqs)
                freq_vars[freq] = variable
                ttk.Checkbutton(freq_box, text=freq, variable=variable).grid(
                    row=index // 6, column=index % 6, sticky='w', padx=8, pady=3)

            button_row = ttk.Frame(dialog, padding=8)
            button_row.pack(fill=tk.X)

            def select_all(value):
                for variable in list(system_vars.values()) + list(freq_vars.values()):
                    variable.set(value)

            def apply_scope():
                systems = [code for code, variable in system_vars.items() if variable.get()]
                freqs = [freq for freq, variable in freq_vars.items() if variable.get()]
                if not systems or not freqs:
                    messagebox.showwarning('处理范围', '至少选择一个卫星系统和一个频率。', parent=dialog)
                    return
                processing_systems_var.set(','.join(systems))
                processing_frequencies_var.set(','.join(freqs))
                update_scope_summary()
                dialog.destroy()

            ttk.Button(button_row, text='全选', command=lambda: select_all(True)).pack(side=tk.LEFT, padx=4)
            ttk.Button(button_row, text='全不选', command=lambda: select_all(False)).pack(side=tk.LEFT, padx=4)
            ttk.Button(button_row, text='确定', command=apply_scope).pack(side=tk.RIGHT, padx=4)
            ttk.Button(button_row, text='取消', command=dialog.destroy).pack(side=tk.RIGHT, padx=4)

        scope_button = ttk.Button(scope_frame, text='选择卫星系统/频率…', command=open_scope_selector)
        _tip(scope_button,
             '设置整个预处理链的处理范围。未选择的系统或频率不参与预测、粗差、CCI/ISB、多路径、平滑、统计、CSV或图表；输出RINEX中的原始观测保持不变。').pack(side=tk.LEFT)
        ttk.Label(scope_frame, textvariable=scope_summary_var, foreground='blue', wraplength=680).pack(
            side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        update_scope_summary()

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
        _tip(template_combo, '选择一组粗差阈值预设。“开阔环境”较严格，“遮挡环境”较宽松；选择后仍可逐项修改。')
        
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

        _tip(ttk.Button(tpl_sub, text='智能推荐参数', command=smart_recommend_ui),
             '读取手机 RINEX 的伪距完整率，低于 75% 推荐“遮挡环境”，否则推荐“开阔环境”，并立即应用对应预设。').pack(side=tk.LEFT, padx=10)
        
        # Save/Load buttons aligned right
        _tip(ttk.Button(tpl_sub, text='加载配置', command=lambda: self.load_params_ui(_tk_param_vars)),
             '从 JSON 文件加载本窗口的参数。').pack(side=tk.RIGHT, padx=5)
        _tip(ttk.Button(tpl_sub, text='保存配置', command=lambda: self.save_params_ui(_tk_param_vars)),
             '把本窗口当前参数保存为 JSON，便于复用。').pack(side=tk.RIGHT, padx=5)


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
        _tip(mode_combo, '固定：直接使用输入阈值。自适应：根据数据分布计算阈值，输入值作为保底阈值，防止阈值过小。')
        
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
        _tip(ttk.Entry(c_row1, textvariable=code_threshold_var, width=6),
             '伪距历元间双差允许的最大绝对值，单位米。超过阈值的伪距会被判为粗差；值越小越严格。自适应模式下作为保底阈值。').pack(side=tk.LEFT)

        phase_threshold_var = tk.DoubleVar(value=self.phase_threshold)
        ttk.Label(c_row1, text='相位(米):').pack(side=tk.LEFT, padx=(10, 2))
        _tip(ttk.Entry(c_row1, textvariable=phase_threshold_var, width=6),
             '载波相位历元间双差允许的最大绝对值，单位米。超过阈值的相位会被判为粗差；值越小越严格。').pack(side=tk.LEFT)

        doppler_threshold_var = tk.DoubleVar(value=self.doppler_threshold)
        ttk.Label(c_row1, text='多普勒(米/秒):').pack(side=tk.LEFT, padx=(10, 2))
        _tip(ttk.Entry(c_row1, textvariable=doppler_threshold_var, width=6),
             'Doppler 历元间双差允许的最大绝对值，单位米/秒。超过阈值的 Doppler 会被判为粗差；值越小越严格。').pack(side=tk.LEFT)
        
        # Triple Sigma Option - COMMENTED OUT (replaced by adaptive mode)
        # self.use_triple_sigma_var = tk.BooleanVar(value=False)
        # ttk.Checkbutton(c_row1, text='启用三倍中误差', variable=self.use_triple_sigma_var).pack(side=tk.LEFT, padx=10)

        c_row2 = ttk.Frame(coarse_frame)
        c_row2.pack(fill=tk.X, pady=2)
        cmc_threshold_var = tk.DoubleVar(value=self.cmc_threshold)
        ttk.Label(c_row2, text='CMC变化阈值(米):').pack(side=tk.LEFT)
        _tip(ttk.Entry(c_row2, textvariable=cmc_threshold_var, width=6),
             '码减相位（CMC）历元变化阈值，单位米。用于识别码相组合的突变；过小可能误删噪声较大的手机观测。').pack(side=tk.LEFT, padx=5)

        # 2.3 Optional Processing
        opt_frame = ttk.LabelFrame(param_frame, text='可选处理', padding=5)
        opt_frame.pack(fill=tk.X, pady=5)
        
        self.doppler_enable_var = tk.BooleanVar(value=False)
        self.cci_enable_var = tk.BooleanVar(value=False)
        self.isb_enable_var = tk.BooleanVar(value=False)
        
        # Layout: Vertical checks
        _tip(ttk.Checkbutton(opt_frame, text='启用多普勒预测相位 (基于多普勒观测值预测并填补缺失的载波相位观测值)', variable=self.doppler_enable_var),
             '使用相邻历元 Doppler 梯形积分填补短载波相位缺口。会输出预测后的 RINEX、完整率/误差图以及供未来 PPP 使用的 JSON/CSV 元数据。').pack(anchor='w')
        pred_row = ttk.Frame(opt_frame)
        pred_row.pack(fill=tk.X, pady=2, anchor='w')
        doppler_prediction_max_length_var = tk.IntVar(value=self.doppler_prediction_max_length)
        ttk.Label(pred_row, text='  最大连续预测(历元):').pack(side=tk.LEFT)
        _tip(ttk.Entry(pred_row, textvariable=doppler_prediction_max_length_var, width=5),
             '一个连续相位缺口最多填补多少个历元。超过该长度后保持缺失，直到出现新的真实相位。值越大，覆盖率提高但累计误差风险增大。').pack(side=tk.LEFT, padx=4)
        doppler_prediction_interval_factor_var = tk.DoubleVar(value=self.doppler_prediction_max_interval_factor)
        ttk.Label(pred_row, text='最大间隔倍数:').pack(side=tk.LEFT, padx=(8, 2))
        _tip(ttk.Entry(pred_row, textvariable=doppler_prediction_interval_factor_var, width=5),
             '允许的相邻采样间隔 / 名义采样间隔。名义间隔取该序列时间间隔中位数；例如 1 Hz 数据设为 1.5 时，间隔超过 1.5 秒不预测。').pack(side=tk.LEFT)
        doppler_prediction_first_sigma_var = tk.DoubleVar(value=self.doppler_prediction_first_sigma_m)
        ttk.Label(pred_row, text='首历元σ(m):').pack(side=tk.LEFT, padx=(8, 2))
        _tip(ttk.Entry(pred_row, textvariable=doppler_prediction_first_sigma_var, width=6),
             '第 1 个预测历元建议使用的标准差，单位米。当前只写入 prediction JSON/CSV，为以后 PPP 变权准备，不改变 RINEX 数值或当前处理结果。').pack(side=tk.LEFT)
        doppler_prediction_sigma_growth_var = tk.DoubleVar(value=self.doppler_prediction_sigma_growth_m)
        ttk.Label(pred_row, text='每历元σ增量(m):').pack(side=tk.LEFT, padx=(8, 2))
        _tip(ttk.Entry(pred_row, textvariable=doppler_prediction_sigma_growth_var, width=6),
             '连续预测每增加 1 个历元时，建议标准差增加的数值，单位米。例如首历元 0.10、增量 0.05，则第 5 历元为 0.30 m。仅写入 PPP 伴随元数据。').pack(side=tk.LEFT)
        _tip(ttk.Checkbutton(opt_frame, text='启用码相不一致性(CCI)处理 (需要接收机文件作为基准, 校正载波相位观测值)', variable=self.cci_enable_var),
             '用手机与参考接收机的站间单差 CMC 建模码相不一致性并改正手机载波相位。必须选择时间匹配的接收机 RINEX。').pack(anchor='w')
        _tip(ttk.Checkbutton(opt_frame, text='启用ISB处理 (需要接收机文件作为基准, 校正BDS系统间偏差)', variable=self.isb_enable_var),
             '分析并校正 BDS-2/BDS-3 系统间偏差。必须提供时间匹配的参考接收机 RINEX。').pack(anchor='w')
        
        # ADD DOPPLER SMOOTHING HERE
        doppler_smoothing_var = tk.BooleanVar(value=self.doppler_smoothing_enabled)
        ds_row = ttk.Frame(opt_frame)
        ds_row.pack(fill=tk.X, pady=2, anchor='w')
        _tip(ttk.Checkbutton(ds_row, text='启用多普勒平滑伪距', variable=doppler_smoothing_var),
             '使用 Doppler 推算的距离变化对伪距进行递推平滑。可降低随机噪声，但过长窗口会增加异常传播风险。').pack(side=tk.LEFT)
        
        doppler_window_var = tk.IntVar(value=self.doppler_smoothing_window)
        ttk.Label(ds_row, text='平滑窗口:').pack(side=tk.LEFT, padx=(10, 2))
        _tip(ttk.Entry(ds_row, textvariable=doppler_window_var, width=5),
             'Doppler 伪距平滑的最大窗口长度，单位历元。窗口越大通常越平滑，但对运动变化和异常的响应更慢。').pack(side=tk.LEFT)
        
        doppler_threshold_smooth_var = tk.DoubleVar(value=self.doppler_smoothing_reset_threshold)
        ttk.Label(ds_row, text='重置阈值(米):').pack(side=tk.LEFT, padx=(10, 2))
        _tip(ttk.Entry(ds_row, textvariable=doppler_threshold_smooth_var, width=5),
             '实测伪距与 Doppler 递推伪距偏差超过该值时重置平滑，单位米。值越小，窗口重置越频繁。').pack(side=tk.LEFT)

        # Pseudorange multipath correction.  This is intentionally opt-in and
        # is executed immediately before Doppler/Hatch smoothing.
        mp_frame = ttk.LabelFrame(opt_frame, text='伪距多路径改正（默认关闭，位于多普勒平滑之前）', padding=5)
        mp_frame.pack(fill=tk.X, pady=5, anchor='w')

        multipath_enable_var = tk.BooleanVar(value=self.pseudorange_multipath_enabled)
        multipath_enable_check = ttk.Checkbutton(
            mp_frame,
            text='启用双频伪距多路径改正',
            variable=multipath_enable_var,
        )
        _tip(multipath_enable_check,
             '使用双频 MP 组合估计并改正伪距多路径。仅修改伪距字段，不修改载波相位、Doppler、LLI 或 SNR；默认关闭。').pack(anchor='w')

        mp_row1 = ttk.Frame(mp_frame)
        mp_row1.pack(fill=tk.X, pady=2)
        multipath_pair_var = tk.StringVar(value=self.pseudorange_multipath_pair)
        ttk.Label(mp_row1, text='频率对:').pack(side=tk.LEFT)
        multipath_pair_combo = ttk.Combobox(
            mp_row1,
            textvariable=multipath_pair_var,
            state='readonly',
            width=14,
            values=['自动选择', 'L1C+L5Q', 'L1P+L5P', 'L2I+L5P', 'L2I+L1P', 'L1C+L7Q', 'L5Q+L7Q', 'L1C+L2C'],
        )
        _tip(multipath_pair_combo,
             '选择构造双频 MP 组合的频率对。“自动选择”会按星座优先级和实际有效连续弧段选择；指定频率对可能使不具备该频率的卫星被跳过。').pack(side=tk.LEFT, padx=5)

        multipath_gain_var = tk.DoubleVar(value=self.pseudorange_multipath_gain)
        ttk.Label(mp_row1, text='改正增益:').pack(side=tk.LEFT, padx=(12, 2))
        _tip(ttk.Entry(mp_row1, textvariable=multipath_gain_var, width=6),
             '将估计的多路径改正量应用到伪距的比例。0 表示不改，1 表示全量改正；默认 0.5 采用较保守的半量改正。').pack(side=tk.LEFT)

        multipath_min_arc_var = tk.IntVar(value=self.pseudorange_multipath_min_arc_epochs)
        ttk.Label(mp_row1, text='最小连续弧段(历元):').pack(side=tk.LEFT, padx=(12, 2))
        _tip(ttk.Entry(mp_row1, textvariable=multipath_min_arc_var, width=6),
             '参与多路径估计的连续弧段最少历元数。短于该长度的弧段不改正；增大可提高稳健性，但减少可处理数据。').pack(side=tk.LEFT)

        mp_row2 = ttk.Frame(mp_frame)
        mp_row2.pack(fill=tk.X, pady=2)
        multipath_gap_var = tk.DoubleVar(value=self.pseudorange_multipath_arc_gap_seconds)
        ttk.Label(mp_row2, text='最大弧段间隔(秒):').pack(side=tk.LEFT)
        _tip(ttk.Entry(mp_row2, textvariable=multipath_gap_var, width=6),
             '同一连续弧段允许的最大相邻时间间隔，单位秒。超过该间隔会切分弧段并重新估计弧段基准。').pack(side=tk.LEFT, padx=5)

        multipath_mad_scale_var = tk.DoubleVar(value=self.pseudorange_multipath_mad_scale)
        ttk.Label(mp_row2, text='MAD异常倍数:').pack(side=tk.LEFT, padx=(12, 2))
        _tip(ttk.Entry(mp_row2, textvariable=multipath_mad_scale_var, width=6),
             '基于 MAD 的异常判定倍数。值越小剔除越严格，值越大保留更多样本；异常点不会参与弧段基准或伪距改正。').pack(side=tk.LEFT)

        multipath_mad_floor_var = tk.DoubleVar(value=self.pseudorange_multipath_mad_floor_m)
        ttk.Label(mp_row2, text='MAD最小尺度(米):').pack(side=tk.LEFT, padx=(12, 2))
        _tip(ttk.Entry(mp_row2, textvariable=multipath_mad_floor_var, width=6),
             'MAD 鲁棒尺度的最小值，单位米。用于防止数据过于集中时异常阈值趋近于零；增大后判定更宽松。').pack(side=tk.LEFT)

        multipath_max_corr_var = tk.DoubleVar(value=self.pseudorange_multipath_max_correction_m)
        ttk.Label(mp_row2, text='最大单点改正(米):').pack(side=tk.LEFT, padx=(12, 2))
        _tip(ttk.Entry(mp_row2, textvariable=multipath_max_corr_var, width=6),
             '允许应用到单个伪距观测的最大绝对改正量，单位米。超过该值的样本保持原值并记录为拒绝。').pack(side=tk.LEFT)

        multipath_half_cycle_var = tk.BooleanVar(value=self.pseudorange_multipath_reject_half_cycle)
        multipath_half_cycle_check = ttk.Checkbutton(
            mp_row2,
            text='剔除未解决半周',
            variable=multipath_half_cycle_var,
        )
        _tip(multipath_half_cycle_check,
             '启用后，LLI 第 1 位指示“半周模糊度未解决”的载波相位不参与 MP 弧段和伪距改正。建议保持启用。').pack(side=tk.LEFT, padx=(12, 0))

        ttk.Label(
            mp_frame,
            text='说明：每个卫星/频率连续弧段使用MP原始组合减去弧段中位数；短弧段、周跳、失锁或异常点不改正。',
            foreground='blue',
        ).pack(anchor='w', pady=(3, 0))

        # 2.4 Code-Phase Inconsistency Processing (Details)
        cci_frame = ttk.LabelFrame(param_frame, text='码相不一致性处理', padding=5)
        cci_frame.pack(fill=tk.X, pady=5)
        
        cci_row = ttk.Frame(cci_frame)
        cci_row.pack(fill=tk.X)
        
        r_squared_var = tk.DoubleVar(value=self.r_squared_threshold)
        ttk.Label(cci_row, text='R方阈值:').pack(side=tk.LEFT)
        _tip(ttk.Entry(cci_row, textvariable=r_squared_var, width=6),
             'CCI 线性漂移拟合的决定系数 R² 门限，范围通常为 0–1。越高表示只接受线性特征更明显的序列。').pack(side=tk.LEFT, padx=5)
        ttk.Label(cci_row, text='(默认: 0.5, 线性漂移判断)').pack(side=tk.LEFT)
        
        cv_threshold_var = tk.DoubleVar(value=self.cv_threshold)
        ttk.Label(cci_row, text='CV阈值:').pack(side=tk.LEFT, padx=(20, 5))
        _tip(ttk.Entry(cci_row, textvariable=cv_threshold_var, width=6),
             '码相变化率变异系数（CV）门限，用于选择 ROC 模型层级。值越小，对变化率稳定性的要求越严格。').pack(side=tk.LEFT, padx=5)
        ttk.Label(cci_row, text='(默认: 0.6, ROC模型选择)').pack(side=tk.LEFT)
        
        phone_only_var = tk.BooleanVar(value=False)
        phone_only_min_var = tk.IntVar(value=self.phone_only_min_data_points)
        # Checkbox for phone only
        cci_row2 = ttk.Frame(cci_frame)
        cci_row2.pack(fill=tk.X, pady=2)
        _tip(ttk.Checkbutton(cci_row2, text='启用手机独有卫星分析 (检测手机独有卫星的码相不一致性)', variable=phone_only_var),
             '把参考接收机未共同观测、仅手机可见的卫星也纳入 CCI 分析。此类卫星缺少站间对照，结论应视为辅助结果。').pack(side=tk.LEFT)


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
            'processing_systems_var': processing_systems_var,
            'processing_frequencies_var': processing_frequencies_var,
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
            'doppler_prediction_max_length_var': doppler_prediction_max_length_var,
            'doppler_prediction_interval_factor_var': doppler_prediction_interval_factor_var,
            'doppler_prediction_first_sigma_var': doppler_prediction_first_sigma_var,
            'doppler_prediction_sigma_growth_var': doppler_prediction_sigma_growth_var,
            'pseudorange_multipath_enabled_var': multipath_enable_var,
            'pseudorange_multipath_pair_var': multipath_pair_var,
            'pseudorange_multipath_gain_var': multipath_gain_var,
            'pseudorange_multipath_min_arc_var': multipath_min_arc_var,
            'pseudorange_multipath_gap_var': multipath_gap_var,
            'pseudorange_multipath_mad_scale_var': multipath_mad_scale_var,
            'pseudorange_multipath_mad_floor_var': multipath_mad_floor_var,
            'pseudorange_multipath_max_corr_var': multipath_max_corr_var,
            'pseudorange_multipath_half_cycle_var': multipath_half_cycle_var,
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

        _tip(ttk.Button(center_btn, text='开始预处理', command=_run_in_thread),
             '按当前文件和参数启动预处理。处理在后台线程运行，输出保存到输入文件旁的 Arna_results 目录。').pack(side=tk.LEFT, padx=10)
        _tip(ttk.Button(center_btn, text='BDS2/3 ISB分析', command=run_isb_only_wrapper),
             '单独执行 BDS-2/BDS-3 系统间偏差分析；需要手机与参考接收机 RINEX。当前按钮仍处于功能开发状态。').pack(side=tk.LEFT, padx=10)
        # "Select File" button at bottom? Screenshot shows it. Maybe it opens folder?
        _tip(ttk.Button(center_btn, text='选择文件', command=lambda: os.startfile(os.path.dirname(phone_var.get())) if phone_var.get() else None),
             '在资源管理器中打开已选择手机 RINEX 所在的文件夹。').pack(side=tk.LEFT, padx=10)
        _tip(ttk.Button(center_btn, text='关闭', command=top.destroy),
             '关闭数据预处理窗口；不会删除已经生成的文件。').pack(side=tk.LEFT, padx=10)

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
            'processing_systems_var': vars_dict['processing_systems_var'].get(),
            'processing_frequencies_var': vars_dict['processing_frequencies_var'].get(),
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
            'doppler_prediction_max_length_var': vars_dict['doppler_prediction_max_length_var'].get(),
            'doppler_prediction_interval_factor_var': vars_dict['doppler_prediction_interval_factor_var'].get(),
            'doppler_prediction_first_sigma_var': vars_dict['doppler_prediction_first_sigma_var'].get(),
            'doppler_prediction_sigma_growth_var': vars_dict['doppler_prediction_sigma_growth_var'].get(),
            'pseudorange_multipath_enabled_var': vars_dict['pseudorange_multipath_enabled_var'].get(),
            'pseudorange_multipath_pair_var': vars_dict['pseudorange_multipath_pair_var'].get(),
            'pseudorange_multipath_gain_var': vars_dict['pseudorange_multipath_gain_var'].get(),
            'pseudorange_multipath_min_arc_var': vars_dict['pseudorange_multipath_min_arc_var'].get(),
            'pseudorange_multipath_gap_var': vars_dict['pseudorange_multipath_gap_var'].get(),
            'pseudorange_multipath_mad_scale_var': vars_dict['pseudorange_multipath_mad_scale_var'].get(),
            'pseudorange_multipath_mad_floor_var': vars_dict['pseudorange_multipath_mad_floor_var'].get(),
            'pseudorange_multipath_max_corr_var': vars_dict['pseudorange_multipath_max_corr_var'].get(),
            'pseudorange_multipath_half_cycle_var': vars_dict['pseudorange_multipath_half_cycle_var'].get(),
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
            vars_dict['processing_systems_var'].set(','.join(self.processing_systems))
            vars_dict['processing_frequencies_var'].set(','.join(self.processing_frequencies))
            update_scope = getattr(vars_dict.get('processing_systems_var'), '_scope_update', None)
            if callable(update_scope):
                update_scope()
            vars_dict['phase_threshold_var'].set(self.phase_threshold)
            vars_dict['doppler_threshold_var'].set(self.doppler_threshold)
            vars_dict['cmc_threshold_var'].set(self.cmc_threshold)
            vars_dict['r_squared_var'].set(self.r_squared_threshold)
            vars_dict['cv_threshold_var'].set(self.cv_threshold)
            vars_dict['phone_only_min_var'].set(self.phone_only_min_data_points)
            vars_dict['doppler_smoothing_var'].set(self.doppler_smoothing_enabled)
            vars_dict['doppler_window_var'].set(self.doppler_smoothing_window)
            vars_dict['doppler_threshold_smooth_var'].set(self.doppler_smoothing_reset_threshold)
            vars_dict['doppler_prediction_max_length_var'].set(self.doppler_prediction_max_length)
            vars_dict['doppler_prediction_interval_factor_var'].set(self.doppler_prediction_max_interval_factor)
            vars_dict['doppler_prediction_first_sigma_var'].set(self.doppler_prediction_first_sigma_m)
            vars_dict['doppler_prediction_sigma_growth_var'].set(self.doppler_prediction_sigma_growth_m)
            vars_dict['pseudorange_multipath_enabled_var'].set(self.pseudorange_multipath_enabled)
            vars_dict['pseudorange_multipath_pair_var'].set(self.pseudorange_multipath_pair)
            vars_dict['pseudorange_multipath_gain_var'].set(self.pseudorange_multipath_gain)
            vars_dict['pseudorange_multipath_min_arc_var'].set(self.pseudorange_multipath_min_arc_epochs)
            vars_dict['pseudorange_multipath_gap_var'].set(self.pseudorange_multipath_arc_gap_seconds)
            vars_dict['pseudorange_multipath_mad_scale_var'].set(self.pseudorange_multipath_mad_scale)
            vars_dict['pseudorange_multipath_mad_floor_var'].set(self.pseudorange_multipath_mad_floor_m)
            vars_dict['pseudorange_multipath_max_corr_var'].set(self.pseudorange_multipath_max_correction_m)
            vars_dict['pseudorange_multipath_half_cycle_var'].set(self.pseudorange_multipath_reject_half_cycle)
            if 'threshold_mode_var' in vars_dict:
                vars_dict['threshold_mode_var'].set(self.threshold_mode)

    def _update_self_from_dict(self, params_dict):
        """Thread-safe update from a dictionary of values."""
        systems_text = params_dict.get('processing_systems_var')
        frequencies_text = params_dict.get('processing_frequencies_var')
        if systems_text is not None:
            self.processing_systems = [item.strip() for item in str(systems_text).split(',') if item.strip()]
        if frequencies_text is not None:
            self.processing_frequencies = [item.strip() for item in str(frequencies_text).split(',') if item.strip()]
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
        self.doppler_prediction_max_length = max(1, int(params_dict.get('doppler_prediction_max_length_var', self.doppler_prediction_max_length)))
        self.doppler_prediction_max_interval_factor = max(1.0, float(params_dict.get('doppler_prediction_interval_factor_var', self.doppler_prediction_max_interval_factor)))
        self.doppler_prediction_first_sigma_m = max(0.0, float(params_dict.get('doppler_prediction_first_sigma_var', self.doppler_prediction_first_sigma_m)))
        self.doppler_prediction_sigma_growth_m = max(0.0, float(params_dict.get('doppler_prediction_sigma_growth_var', self.doppler_prediction_sigma_growth_m)))
        self.pseudorange_multipath_enabled = bool(params_dict.get('pseudorange_multipath_enabled_var', self.pseudorange_multipath_enabled))
        self.pseudorange_multipath_pair = str(params_dict.get('pseudorange_multipath_pair_var', self.pseudorange_multipath_pair))
        self.pseudorange_multipath_gain = float(params_dict.get('pseudorange_multipath_gain_var', self.pseudorange_multipath_gain))
        self.pseudorange_multipath_min_arc_epochs = int(params_dict.get('pseudorange_multipath_min_arc_var', self.pseudorange_multipath_min_arc_epochs))
        self.pseudorange_multipath_arc_gap_seconds = float(params_dict.get('pseudorange_multipath_gap_var', self.pseudorange_multipath_arc_gap_seconds))
        self.pseudorange_multipath_mad_scale = float(params_dict.get('pseudorange_multipath_mad_scale_var', self.pseudorange_multipath_mad_scale))
        self.pseudorange_multipath_mad_floor_m = float(params_dict.get('pseudorange_multipath_mad_floor_var', self.pseudorange_multipath_mad_floor_m))
        self.pseudorange_multipath_max_correction_m = float(params_dict.get('pseudorange_multipath_max_corr_var', self.pseudorange_multipath_max_correction_m))
        self.pseudorange_multipath_reject_half_cycle = bool(params_dict.get('pseudorange_multipath_half_cycle_var', self.pseudorange_multipath_reject_half_cycle))
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
                'enable_isb': bool(isb_enable),
                'enable_pseudorange_multipath': bool(self.pseudorange_multipath_enabled),
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
            self.context.set_output_dir(project_dir)
            
            # Subfolders
            dirs = {
                'pred': os.path.join(project_dir, 'doppler prediction'),
                'smooth': os.path.join(project_dir, 'doppler smoothing'),
                'multipath': os.path.join(project_dir, 'pseudorange multipath correction'),
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
                 prediction_source_observations = copy.deepcopy(self.context.observations_meters)
                 doppler_results = self.algo.run_doppler_phase_prediction(
                     self.context.observations_meters,
                     self.context.frequencies,
                     self.context.wavelengths,
                     original_rinex_path=current_chain_path, # Use chain
                     output_path=pred_path,
                     writer=self.writer,
                     max_prediction_length=self.doppler_prediction_max_length,
                     max_interval_factor=self.doppler_prediction_max_interval_factor,
                     real_phase_sigma_m=self.doppler_prediction_real_sigma_m,
                     first_prediction_sigma_m=self.doppler_prediction_first_sigma_m,
                     sigma_growth_m=self.doppler_prediction_sigma_growth_m,
                 )
                 self.context.results['doppler_prediction'] = doppler_results

                 artifact_paths = save_prediction_artifacts(
                     prediction_source_observations,
                     doppler_results,
                     dirs['pred'],
                     base_name=chain_base + '-doppler-predicted',
                     error_max_length=self.doppler_prediction_error_plot_max_length,
                 )
                 doppler_results['artifact_paths'] = artifact_paths
                 
                 # Log
                 target_log_path = os.path.join(dirs['pred'], "doppler_prediction.log")
                 with open(target_log_path, 'w', encoding='utf-8') as f:
                     f.write("=" * 70 + "\n")
                     f.write("多普勒预测处理日志\n")
                     f.write("=" * 70 + "\n\n")
                     f.write(f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                     f.write(f"输入文件: {os.path.abspath(current_chain_path)}\n")
                     f.write(f"输出文件: {os.path.abspath(pred_path)}\n\n")
                     f.write("全流程处理范围:\n")
                     f.write(f"  卫星系统: {', '.join(self.processing_systems)}\n")
                     f.write(f"  观测频率: {', '.join(self.processing_frequencies)}\n")
                     f.write("  未选择的系统/频率不参与计算、统计、CSV或图表；输出RINEX中的对应原始字段不会被删除或改写。\n\n")
                     f.write("预测门控及PPP元数据:\n")
                     f.write(f"  最大连续预测长度: {self.doppler_prediction_max_length} 历元\n")
                     f.write(f"  最大采样间隔倍数: {self.doppler_prediction_max_interval_factor}\n")
                     f.write(f"  预测1历元标准差: {self.doppler_prediction_first_sigma_m:.3f} m\n")
                     f.write(f"  每增加1历元标准差增量: {self.doppler_prediction_sigma_growth_m:.3f} m\n")
                     f.write("  权值仅写入JSON/CSV伴随文件，未编码到RINEX LLI/SSI。\n\n")
                     f.write("算法原理:\n")
                     f.write("  对已转换为相位距离变化率的相邻Doppler作梯形积分，预测并修补通过门控的短载波相位缺口，以提高相位序列连续性。\n\n")
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
                             f.write(
                                 f"Epoch {log_item['epoch_idx']:4} Time {log_item['time']} "
                                 f"Sat {log_item['sat_id']} {log_item['freq']}: "
                                 f"Predicted {log_item['predicted_phase_cycle']} "
                                 f"Length {log_item['prediction_length']} Sigma {log_item['sigma_m']:.3f}m\n"
                             )

                     f.write("\n四、伴随文件和图表\n" + "-" * 40 + "\n")
                     f.write("完整率口径: 分母为 Arna 解析出的对应卫星-频率记录历元数，不是基于星历/高度角计算的理论可见历元数。\n")
                     f.write("完整率柱状图读法: 每个频率两根柱（红=预测前，蓝=预测后）；整柱顶部为Code完整率，斜线区域顶部为Phase完整率。斜线不是与Code相加的堆叠值。\n")
                     f.write("预测误差口径: 在真实连续相位弧段上模拟多步预测，并与对应真实相位比较；跨LLI或Doppler-相位突变的边不参与统计。\n\n")
                     f.write(f"predicted_rinex: {os.path.abspath(pred_path)}\n")
                     f.write("  说明: 已将通过长度、时间间隔、LLI和Doppler可用性门控的预测相位写入原缺失字段；其他RINEX观测字段保持原语义。\n")
                     artifact_descriptions = {
                         'metadata_json': 'PPP接入伴随元数据(JSON)：逐条记录卫星、频率、时间、预测长度、建议sigma、接受状态和来源。',
                         'metadata_csv': 'PPP接入伴随元数据(CSV)：与JSON核心记录等价，便于表格检查和程序导入。',
                         'integrity_by_satellite_csv': '卫星-频率级完整率明细：Code、预测前Phase、预测后Phase的计数和比例。',
                         'integrity_summary_rms_csv': '系统-频率级完整率汇总：加权总体比例及卫星级完整率RMS。',
                         'prediction_error_samples_csv': '真实相位验证样本：锚点/目标时间、预测长度、预测值、真实值和误差。为控制文件大小，对长序列的锚点作确定性抽样。',
                         'prediction_error_rms_csv': '按真实卫星、频率和预测长度汇总的样本数、偏差、RMS、MAE和绝对误差P95。',
                         'phase_integrity_before_png': '预测前相位完整率时序图：每条短竖线表示该卫星/频率在对应真实时间存在载波相位。',
                         'phase_integrity_after_png': '预测后相位完整率时序图：同时显示真实相位和通过门控写入的预测相位。',
                         'observation_integrity_comparison_png': '预测前后观测完整率柱状图：整柱为Code完整率，斜线覆盖高度为Phase完整率。',
                         'prediction_error_rms_png': '真实相位验证得到的多步预测RMS图，按实际系统/频率分面、按实际卫星绘线。',
                     }
                     for artifact_name, artifact_path in artifact_paths.items():
                         if artifact_path:
                             f.write(f"{artifact_name}: {os.path.abspath(artifact_path)}\n")
                             f.write(f"  说明: {artifact_descriptions.get(artifact_name, '多普勒预测处理输出文件。')}\n")

                 # Update Chain
                 if os.path.exists(pred_path):
                      print(f"[系统] 重新加载中间文件: {pred_path}")
                      self.load_phone_file(pred_path)
                      current_chain_path = pred_path # Update chain
                      self.context.input_path = pred_path # Legacy compat
                 
                 results_summary.append("多普勒预测: 完成")


            # --- Step 2: Pseudorange Multipath Correction ---
            # Must be before Doppler/Hatch smoothing.  Doppler phase prediction,
            # when enabled above, is intentionally completed first so that the
            # MP calculation uses the current continuous phase observations.
            if self.pseudorange_multipath_enabled:
                 log_step("伪距多路径改正", "正在计算并应用...")
                 update_progress(15)

                 multipath_result = self.algo.apply_pseudorange_multipath_correction(
                     self.context.observations_meters,
                     freq_pair=self.pseudorange_multipath_pair,
                     gain=self.pseudorange_multipath_gain,
                     min_arc_epochs=self.pseudorange_multipath_min_arc_epochs,
                     arc_gap_seconds=self.pseudorange_multipath_arc_gap_seconds,
                     mad_scale=self.pseudorange_multipath_mad_scale,
                     mad_floor_m=self.pseudorange_multipath_mad_floor_m,
                     max_correction_m=self.pseudorange_multipath_max_correction_m,
                     reject_half_cycle=self.pseudorange_multipath_reject_half_cycle,
                 )
                 self.context.results['multipath_correction'] = multipath_result
                 self.context.observations_meters = multipath_result['corrected_observations']

                 chain_base = os.path.splitext(os.path.basename(current_chain_path))[0]
                 multipath_filename = f"{chain_base}-multipath corrected{original_ext}"
                 multipath_path = os.path.join(dirs['multipath'], multipath_filename)
                 writer_result = self.writer.write_pseudorange_multipath_corrected_rinex(
                     current_chain_path,
                     multipath_path,
                     multipath_result.get('corrections', {}),
                 )
                 multipath_result['writer_result'] = writer_result

                 log_path = os.path.join(dirs['multipath'], 'pseudorange_multipath_correction.log')
                 with open(log_path, 'w', encoding='utf-8') as lf:
                     lf.write('=' * 70 + '\n')
                     lf.write('伪距多路径改正日志\n')
                     lf.write('=' * 70 + '\n\n')
                     lf.write(f'处理时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                     lf.write(f'输入文件: {os.path.abspath(current_chain_path)}\n')
                     lf.write(f'输出文件: {os.path.abspath(multipath_path)}\n\n')
                     lf.write(multipath_result.get('log', ''))
                     lf.write('\n\n写回统计:\n')
                     lf.write(f'写回伪距观测数: {writer_result.get("total_modifications", 0)}\n')

                 if os.path.exists(multipath_path):
                     log_step("伪距多路径改正", f"生成文件: {multipath_filename}")
                     self.load_phone_file(multipath_path)
                     current_chain_path = multipath_path
                     self.context.input_path = multipath_path
                 results_summary.append("伪距多路径改正: 完成")
            else:
                 self.context.results['multipath_correction'] = None


            # --- Step 3: Doppler Smoothing ---
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

            # Persist a compact, restart-safe task description.  The report
            # generator reads this manifest instead of depending on GUI memory.
            manifest_path = ReportGenerator().write_processing_manifest(
                preprocessing_dir=project_dir,
                input_path=phone_path,
                receiver_path=recv_path,
                final_output_path=current_chain_path,
                parameters=params_snapshot,
                scope={
                    'systems': list(self.processing_systems),
                    'frequencies': list(self.processing_frequencies),
                    'excluded_observations_preserved_in_rinex': True,
                },
                flags=self.context.results.get('flags', {}),
                results=self.context.results,
                step_summary=results_summary,
            )
            self.context.results['processing_manifest_path'] = manifest_path
            log_step("报告清单", f"已生成: {manifest_path}")
            
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
