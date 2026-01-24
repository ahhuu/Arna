from typing import Optional, Dict, Any
import os
import matplotlib.pyplot as plt
from src.core.context import AnalysisContext
from src.visualization.plotter import GNSSPlotter
from src.data.reader import RinexReader
from src.processing.calculator import MetricCalculator


class VisualizationWindow:
    """Visualization helper that wraps GNSSPlotter.

    Provides programmatic API for producing/saving plots without launching UI mainloop (good for tests).
    """

    def __init__(self, context: Optional[AnalysisContext] = None):
        self.context = context or AnalysisContext()
        self.plotter = GNSSPlotter()

    def plot_raw(self, sat_id: str, save: bool = True, output_dir: Optional[str] = None):
        return self.plotter.plot_raw_observations({'observations_meters': self.context.observations_meters}, sat_id, save=save, output_dir=output_dir)

    def plot_derivative(self, sat_id: str, freq: str, save: bool = True, output_dir: Optional[str] = None):
        # expecting derivatives to be stored in results or computed on-the-fly
        derivatives = self.context.results.get('observable_derivatives') or {}
        return self.plotter.plot_derivatives(derivatives, sat_id, freq, save=save, output_dir=output_dir)

    def plot_code_phase_diff(self, sat_id: str, freq: str, save: bool = True, output_dir: Optional[str] = None):
        return self.plotter.plot_code_phase_raw_diff({'code_phase_differences': self.context.results.get('code_phase_differences', {})}, sat_id, freq, save=save, output_dir=output_dir)

    def plot_prediction_errors(self, sat_id: str, freq: str, save: bool = True, output_dir: Optional[str] = None):
        errors = self.context.results.get('phase_prediction_errors', {})
        return self.plotter.plot_prediction_errors(errors, sat_id, freq, save=save, output_dir=output_dir)

    def plot_epoch_dd(self, sat_id: str, freq: str, save: bool = True, output_dir: Optional[str] = None):
        return self.plotter.plot_epoch_double_diffs({'epoch_double_diffs': self.context.results.get('epoch_double_diffs', {})}, sat_id, freq, save=save, output_dir=output_dir)

    def plot_isb(self, save: bool = True, output_dir: Optional[str] = None):
        return self.plotter.plot_isb_analysis(self.context.results.get('isb_analysis', {}), save=save, output_dir=output_dir)

    def show(self, parent):
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
        except Exception:
            return

        top = tk.Toplevel(parent)
        top.title('图表生成')
        top.geometry('600x600')  # Reduced size since plots are popups
        top.transient(parent)
        top.grab_set()

        main_frame = ttk.Frame(top, padding=20)
        main_frame.pack(fill='both', expand=True)

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text='选择手机RINEX文件', padding=10)
        file_frame.pack(fill=tk.X, pady=8)
        file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=file_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=(0,10), fill=tk.X, expand=True)

        def select_file():
            f = filedialog.askopenfilename(title='选择RINEX观测文件')
            if f:
                file_var.set(f)
                load_satellite_info()
        ttk.Button(file_frame, text='浏览', command=select_file).pack(side=tk.RIGHT)

        # Chart types
        chart_frame = ttk.LabelFrame(main_frame, text='图表类型', padding=10)
        chart_frame.pack(fill=tk.X, pady=8)
        chart_types = [
            ('原始观测', 'raw_observations'),
            ('观测值一阶差分', 'derivatives'),
            ('伪距相位差值之差', 'code_phase_diffs'),
            ('伪距相位原始差值', 'code_phase_diff_raw'),
            ('相位预测误差', 'phase_pred_errors'),
            ('历元间双差', 'double_differences'),
            ('ISB分析', 'isb_analysis'),
            ('接收机CMC', 'receiver_cmc'),
            ('周跳探测分析 (MW & GF)', 'cycle_slip_detection')
        ]
        chart_var = tk.StringVar(value='raw_observations')
        
        # Grid layout for radio buttons
        grid_frame = ttk.Frame(chart_frame)
        grid_frame.pack(fill=tk.X)
        for i, (txt, val) in enumerate(chart_types):
            r = i // 2
            c = i % 2
            ttk.Radiobutton(grid_frame, text=txt, variable=chart_var, value=val).grid(row=r, column=c, sticky='w', padx=10, pady=5)
        
        # 周跳探测日志选项
        save_log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(chart_frame, text='保存周跳详细日志', variable=save_log_var).pack(anchor='w', pady=5)
        
        # 周跳探测频率组合选择（使用GNSS_FREQUENCIES动态生成）
        freq_pair_frame = ttk.Frame(chart_frame)
        freq_pair_frame.pack(fill=tk.X, pady=5)
        ttk.Label(freq_pair_frame, text='频率组合:').pack(side=tk.LEFT)
        freq_pair_var = tk.StringVar(value='自动选择')
        freq_pair_combo = ttk.Combobox(freq_pair_frame, textvariable=freq_pair_var, width=20, state="readonly")
        # 从config中生成频率对列表
        from src.core.config import GNSS_FREQUENCIES
        freq_pair_options = ['自动选择']
        for system, freqs in GNSS_FREQUENCIES.items():
            freq_list = list(freqs.keys())
            if len(freq_list) >= 2:
                freq_pair_options.append(f"{freq_list[0]}+{freq_list[1]}")
        freq_pair_combo['values'] = freq_pair_options
        # 确保默认值与下拉项一致
        freq_pair_var.set(freq_pair_options[0])
        freq_pair_combo.pack(side=tk.LEFT, padx=5)
        
        # 周跳探测阈值设置
        threshold_frame = ttk.LabelFrame(chart_frame, text='周跳阈值设置', padding=5)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        threshold_mode_var = tk.StringVar(value='dynamic')
        ttk.Label(threshold_frame, text='阈值模式:').pack(side=tk.LEFT)
        ttk.Radiobutton(threshold_frame, text='动态阈值', variable=threshold_mode_var, value='dynamic').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(threshold_frame, text='自定义阈值', variable=threshold_mode_var, value='custom').pack(side=tk.LEFT, padx=5)
        
        # 自定义阈值输入框
        custom_threshold_frame = ttk.Frame(threshold_frame)
        custom_threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(custom_threshold_frame, text='MW阈值(m):').pack(side=tk.LEFT, padx=5)
        mw_threshold_var = tk.DoubleVar(value=10.0)
        mw_threshold_entry = ttk.Entry(custom_threshold_frame, textvariable=mw_threshold_var, width=8)
        mw_threshold_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(custom_threshold_frame, text='GF阈值(m):').pack(side=tk.LEFT, padx=5)
        gf_threshold_var = tk.DoubleVar(value=0.05)
        gf_threshold_entry = ttk.Entry(custom_threshold_frame, textvariable=gf_threshold_var, width=8)
        gf_threshold_entry.pack(side=tk.LEFT, padx=2)

        # receiver file
        rx_frame = ttk.LabelFrame(main_frame, text='选择接收机RINEX文件(用于CMC/ISB)', padding=10)
        rx_frame.pack(fill=tk.X, pady=8)
        rx_var = tk.StringVar()
        rx_entry = ttk.Entry(rx_frame, textvariable=rx_var, width=50)
        rx_entry.pack(side=tk.LEFT, padx=(0,10), fill=tk.X, expand=True)

        def select_rx():
            f = filedialog.askopenfilename(title='选择接收机RINEX文件')
            if f:
                rx_var.set(f)
                load_receiver_satellite_info()
        ttk.Button(rx_frame, text='浏览', command=select_rx).pack(side=tk.RIGHT)

        # satellite selection
        sat_frame = ttk.LabelFrame(main_frame, text='卫星系统/PRN/频率选择', padding=10)
        sat_frame.pack(fill=tk.X, pady=8)
        sat_system_var = tk.StringVar()
        sat_prn_var = tk.StringVar()
        freq_var = tk.StringVar()
        
        # Use a cleaner layout for combos
        combo_frame = ttk.Frame(sat_frame)
        combo_frame.pack(fill=tk.X)
        
        ttk.Label(combo_frame, text='系统:').pack(side=tk.LEFT)
        sat_system_combo = ttk.Combobox(combo_frame, textvariable=sat_system_var, width=5, state="readonly")
        sat_system_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(combo_frame, text='PRN:').pack(side=tk.LEFT)
        sat_prn_combo = ttk.Combobox(combo_frame, textvariable=sat_prn_var, width=10, state="readonly")
        sat_prn_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(combo_frame, text='频率:').pack(side=tk.LEFT)
        freq_combo = ttk.Combobox(combo_frame, textvariable=freq_var, width=10, state="readonly")
        freq_combo.pack(side=tk.LEFT, padx=5)

        # progress
        progress_frame = ttk.LabelFrame(main_frame, text='处理进度', padding=10)
        progress_frame.pack(fill=tk.X, pady=8)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, variable=progress_var, mode='determinate')
        progress_bar.pack(fill=tk.X, pady=5)
        status_var = tk.StringVar(value='等待开始...')
        ttk.Label(progress_frame, textvariable=status_var).pack()

        # Action Buttons Area
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=15)

        def load_satellite_info():
            try:
                if not file_var.get():
                    return
                status_var.set('正在读取文件...')
                top.update_idletasks()
                
                rd = RinexReader()
                data = rd.read_phone_rinex(file_var.get())
                obs = data.get('observations_meters', {})
                if not obs:
                    messagebox.showwarning('警告', '文件中未找到观测数据')
                    status_var.set('未找到数据')
                    return
                # populate context for later plotting
                self.context.observations_meters = obs
                self.context.results['epochs'] = data.get('data', {}).get('epochs', [])

                # build frequencies/wavelengths maps consistent with config (system -> freq -> Hz/m)
                freq_map: Dict[str, Dict[str, float]] = {}
                wavelengths_map: Dict[str, Dict[str, float]] = {}

                def _guess_nominal_freq_hz(freq_name: str, system: str) -> float:
                    # common nominal base frequencies
                    if 'L1' in freq_name: return 1575.42e6
                    if 'L2' in freq_name: return 1227.60e6
                    if 'L5' in freq_name: return 1176.45e6
                    if 'L7' in freq_name or 'E5' in freq_name: return 1278.75e6
                    if system == 'R' and freq_name.startswith('L1'): return 1602e6  # GLONASS fallback
                    return 1575.42e6

                C = 299792458.0
                for sat_id, fmap in obs.items():
                    system = sat_id[0] if sat_id else ''
                    freq_map.setdefault(system, {})
                    wavelengths_map.setdefault(system, {})
                    for f, fv in fmap.items():
                        # prefer config value, else derive from wavelength, else nominal guess
                        cfg_freq = self.context.frequencies.get(system, {}).get(f)
                        wlist = fv.get('wavelength', [])
                        wl = next((x for x in wlist if x is not None), None)

                        freq_hz = cfg_freq
                        if freq_hz is None and wl is not None:
                            freq_hz = C / wl
                        if freq_hz is None:
                            try:
                                freq_hz = _guess_nominal_freq_hz(f, system)
                            except Exception:
                                freq_hz = None

                        if freq_hz is not None:
                            freq_map[system][f] = freq_hz
                            wavelengths_map[system][f] = C / freq_hz

                # merge back into context to keep downstream calculator expectations (dict of dicts)
                for sys, freqs in freq_map.items():
                    self.context.frequencies.setdefault(sys, {}).update(freqs)
                for sys, wl_map in wavelengths_map.items():
                    self.context.wavelengths.setdefault(sys, {}).update(wl_map)

                self.context.results['frequencies'] = self.context.frequencies
                self.context.results['wavelengths'] = self.context.wavelengths

                sats = sorted(list(obs.keys()))
                sat_prn_combo['values'] = sats
                if sats:
                    sat_prn_var.set(sats[0])
                    system_set = sorted({s[0] for s in sats if s})
                    sat_system_combo['values'] = system_set
                    sat_system_var.set(system_set[0])
                    freqs = sorted(list(obs[sats[0]].keys()))
                    freq_combo['values'] = freqs
                    if freqs:
                        freq_var.set(freqs[0])
                status_var.set('文件读取完成')
            except Exception as e:
                messagebox.showerror('错误', f'加载卫星信息失败: {e}')
                status_var.set('加载失败')

        def load_receiver_satellite_info():
            try:
                if not rx_var.get():
                    return
                status_var.set('正在读取接收机文件...')
                top.update_idletasks()
                
                rd = RinexReader()
                data = rd.read_receiver_rinex(rx_var.get())
                obs = data.get('receiver_observations', {})
                if not obs:
                    messagebox.showwarning('警告', '接收机文件中未找到观测数据')
                    status_var.set('未找到数据')
                    return
                # populate context
                self.context.receiver_observations = obs
                self.context.results['receiver_epochs'] = data.get('data', {}).get('epochs', [])
                # build freqs/wavelengths with fallback
                freq_map = {}
                wavelengths_map = {}
                C = 299792458.0
                for sat_id, fmap in obs.items():
                    system = sat_id[0] if sat_id else ''
                    freq_map.setdefault(system, {})
                    wavelengths_map.setdefault(system, {})
                    for f, fv in fmap.items():
                        cfg_freq = self.context.frequencies.get(system, {}).get(f)
                        wl_list = fv.get('wavelength', [])
                        wl = next((x for x in wl_list if x is not None), None)
                        freq_hz = cfg_freq if cfg_freq is not None else (C / wl if wl else 1575.42e6)
                        freq_map[system][f] = freq_hz
                        wavelengths_map[system][f] = C / freq_hz

                self.context.results['receiver_frequencies'] = freq_map
                self.context.results['receiver_wavelengths'] = wavelengths_map

                # If phone not loaded, use receiver to populate combos
                if not self.context.observations_meters:
                    sats = sorted(list(obs.keys()))
                    sat_prn_combo['values'] = sats
                    if sats:
                        sat_prn_var.set(sats[0])
                        satsys = sorted({s[0] for s in sats if s})
                        sat_system_combo['values'] = satsys
                        sat_system_var.set(satsys[0])
                        freqs = sorted(list(obs[sats[0]].keys()))
                        freq_combo['values'] = freqs
                        if freqs:
                            freq_var.set(freqs[0])
                status_var.set('接收机文件加载完成')
            except Exception as e:
                messagebox.showerror('错误', f'加载接收机信息失败: {e}')
                status_var.set('加载失败')

        def get_freq_pair_options_for_system(system: str):
            """根据卫星系统返回推荐的频率组合列表（以用户期望的顺序）。"""
            from src.core.config import GNSS_FREQUENCIES
            freqs = GNSS_FREQUENCIES.get(system, {})
            keys = list(freqs.keys())
            opts = []
            if system in ('G', 'R', 'J'):
                if 'L1C' in keys and 'L5Q' in keys:
                    opts.append('L1C+L5Q')
            elif system == 'E':
                for a, b in [('L1C', 'L5Q'), ('L1C', 'L7Q'), ('L5Q', 'L7Q')]:
                    if a in keys and b in keys:
                        opts.append(f'{a}+{b}')
            elif system == 'C':
                for a, b in [('L2I', 'L1P'), ('L2I', 'L5P'), ('L1P', 'L5P')]:
                    if a in keys and b in keys:
                        opts.append(f'{a}+{b}')
            else:
                if len(keys) >= 2:
                    opts.append(f'{keys[0]}+{keys[1]}')
            return ['自动选择'] + opts if opts else ['自动选择']

        def on_system_change(*args):
            sys = sat_system_var.get()
            if not sys: return
            
            # Prefer phone data, else receiver
            target = self.context.observations_meters if self.context.observations_meters else self.context.receiver_observations
            if not target: return
            
            sats = [s for s in target.keys() if s.startswith(sys)]
            sats.sort()
            sat_prn_combo['values'] = sats
            if sats:
                sat_prn_var.set(sats[0])
                freqs = sorted(list(target[sats[0]].keys()))
                freq_combo['values'] = freqs
                if freqs:
                    freq_var.set(freqs[0])

            # 更新频率组合下拉，按系统提供推荐组合
            try:
                pair_opts = get_freq_pair_options_for_system(sys)
                freq_pair_combo['values'] = pair_opts
                freq_pair_var.set(pair_opts[0])
            except Exception:
                # 保持原有选项不变
                pass

        sat_system_var.trace('w', on_system_change)
        
        def on_prn_change(*args):
             prn = sat_prn_var.get()
             if not prn: return
             target = self.context.observations_meters if self.context.observations_meters else self.context.receiver_observations
             if target and prn in target:
                 # 当 PRN 改变时，同步更新频率组合选项（确保与所选系统一致）
                 sys_code = prn[0] if prn else ''
                 try:
                     pair_opts = get_freq_pair_options_for_system(sys_code)
                     freq_pair_combo['values'] = pair_opts
                     freq_pair_var.set(pair_opts[0])
                 except Exception:
                     pass
                 freqs = sorted(list(target[prn].keys()))
                 freq_combo['values'] = freqs
                 # if current freq not in list, select first
                 if freq_var.get() not in freqs and freqs:
                     freq_var.set(freqs[0])
        
        sat_prn_var.trace('w', on_prn_change)

        def pre_calculate_metrics():
            """Ensure metrics are calculated before plotting."""
            if 'epochs' not in self.context.results: return
            
            mc = MetricCalculator()
            inputs = {
                'observations_meters': self.context.observations_meters,
                'epochs': self.context.results.get('epochs', []),
                'frequencies': self.context.frequencies,
                'wavelengths': self.context.wavelengths
            }
            
            # Helper to check and calc
            def check_calc(key, method_name, **kwargs):
                if key not in self.context.results:
                    method = getattr(mc, method_name)
                    self.context.results[key] = method(inputs, **kwargs)

            # Lazy calc based on need, or just Calc All? 
            # Calculating all is safer for 'pre_calculate'
            if 'observable_derivatives' not in self.context.results:
                self.context.results['observable_derivatives'] = mc.calculate_derivatives(inputs)
            if 'code_phase_differences' not in self.context.results:
                self.context.results['code_phase_differences'] = mc.calculate_code_phase_differences(inputs)
            if 'phase_prediction_errors' not in self.context.results:
                self.context.results['phase_prediction_errors'] = mc.calculate_phase_prediction_errors(inputs)
            if 'epoch_double_diffs' not in self.context.results:
                self.context.results['epoch_double_diffs'] = mc.calculate_epoch_double_differences(inputs)

        def gen_chart():
            chart_type = chart_var.get()
            sat = sat_prn_var.get()
            freqv = freq_var.get()
            
            status_var.set(f'正在生成 {chart_type} ...')
            top.update_idletasks()
            try:
                out = None
                
                # Receiver CMC Special Case
                if chart_type == 'receiver_cmc':
                    if not self.context.receiver_observations:
                        messagebox.showerror('错误', '请先加载接收机文件')
                        return
                    if 'receiver_cmc' not in self.context.results:
                        mc = MetricCalculator()
                        self.context.results['receiver_cmc'] = mc.calculate_receiver_cmc({
                            'receiver_observations': self.context.receiver_observations,
                            'receiver_frequencies': self.context.results.get('receiver_frequencies'),
                            'receiver_wavelengths': self.context.results.get('receiver_wavelengths')
                        })
                    out = self.plotter.plot_receiver_cmc(self.context.results['receiver_cmc'], save=False)

                # ISB Special Case
                elif chart_type == 'isb_analysis':
                    if not self.context.observations_meters or not self.context.receiver_observations:
                        messagebox.showerror('错误', 'ISB分析需要同时加载手机和接收机文件')
                        return
                    # We assume ISB analysis might be complex. 
                    # For simplicity, if not in results, we try to calculating it
                    if 'isb_analysis' not in self.context.results:
                         messagebox.showinfo("提示", "ISB分析可能需要较长时间计算，请稍候...")
                         top.update_idletasks()
                         mc = MetricCalculator()
                         # NOTE: update with actual args needed for calculate_isb
                         self.context.results['isb_analysis'] = mc.calculate_isb({
                             'observations_meters': self.context.observations_meters,
                             'receiver_observations': self.context.receiver_observations,
                             'epochs': self.context.results.get('epochs', [])
                         })
                    out = self.plotter.plot_isb_analysis(self.context.results.get('isb_analysis', {}), save=False)

                # Cycle Slip Detection Special Case
                elif chart_type == 'cycle_slip_detection':
                    if not self.context.observations_meters:
                        messagebox.showerror('错误', '请先加载手机RINEX文件')
                        return
                    
                    # 执行周跳探测
                    status_var.set('正在执行周跳探测...')
                    top.update_idletasks()
                    
                    from src.processing.cycle_slip_detector import CycleSlipDetector
                    from src.reporting.cycle_slip_logger import CycleSlipLogger
                    
                    # 解析频率组合选择
                    freq_pair_selection = freq_pair_var.get()
                    freq_pair = None
                    if freq_pair_selection and freq_pair_selection != '自动选择':
                        parts = [p.strip() for p in freq_pair_selection.split('+') if p.strip()]
                        if len(parts) == 2:
                            freq_pair = tuple(parts)
                        else:
                            messagebox.showwarning('警告', f'频率组合解析失败: {freq_pair_selection}，将使用自动选择')
                            freq_pair = None
                    
                    # 读取阈值设置并校验
                    use_custom = threshold_mode_var.get() == 'custom'
                    mw_thresh = None
                    gf_thresh = None
                    if use_custom:
                        try:
                            mw_val = float(mw_threshold_var.get())
                            gf_val = float(gf_threshold_var.get())
                        except Exception:
                            messagebox.showerror('输入错误', '请为 MW 和 GF 输入有效的数字阈值')
                            status_var.set('准备就绪')
                            return
                        if mw_val <= 0 or gf_val <= 0:
                            messagebox.showerror('输入错误', '阈值必须为正数')
                            status_var.set('准备就绪')
                            return
                        mw_thresh = mw_val
                        gf_thresh = gf_val
                    
                    detector = CycleSlipDetector(
                        use_custom_threshold=use_custom,
                        custom_mw_threshold=mw_thresh,
                        custom_gf_threshold=gf_thresh
                    )
                    detection_results = detector.detect_cycle_slips(
                        self.context.observations_meters,
                        self.context.frequencies,
                        self.context.wavelengths,
                        freq_pair=freq_pair
                    )
                    
                    # 保存到context以便后续使用
                    self.context.results['cycle_slip_detection'] = detection_results
                    
                    # 构建保存路径：results/文件名/cycleslips/
                    phone_file = file_var.get()
                    if phone_file:
                        filename = os.path.splitext(os.path.basename(phone_file))[0]
                        cycleslip_dir = os.path.join('results', filename, 'cycleslips')
                    else:
                        cycleslip_dir = os.path.join('results', 'cycleslips')
                    os.makedirs(cycleslip_dir, exist_ok=True)
                    
                    # 如果选中保存日志
                    if save_log_var.get():
                        logger = CycleSlipLogger(output_dir=cycleslip_dir)
                        log_path = logger.save_cycle_slip_log(detection_results)
                        csv_path = logger.save_cycle_slip_csv(detection_results)
                        status_var.set(f'日志已保存: {log_path}')
                        messagebox.showinfo('日志保存', f'周跳探测日志已保存:\n{log_path}\n{csv_path}')
                    
                    # 检查选中的卫星是否有结果
                    if sat not in detection_results:
                        available_sats = list(detection_results.keys())
                        if available_sats:
                            messagebox.showwarning('警告', f'卫星 {sat} 无周跳探测结果，将显示 {available_sats[0]} 的结果')
                            sat = available_sats[0]
                            sat_prn_var.set(sat)
                        else:
                            messagebox.showerror('错误', '没有可用的周跳探测结果')
                            return
                    
                    # 绘制图表（使用相同的保存路径）
                    out = self.plotter.plot_cycle_slip_analysis(detection_results[sat], sat, save=False, output_dir=cycleslip_dir)

                # Standard Charts
                else:
                    if not self.context.observations_meters:
                        messagebox.showerror('错误', '请先加载手机RINEX文件')
                        return
                        
                    if chart_type == 'raw_observations':
                        out = self.plotter.plot_raw_observations({'observations_meters': self.context.observations_meters}, sat, save=False)
                    else:
                        pre_calculate_metrics()
                        if chart_type == 'derivatives':
                            out = self.plotter.plot_derivatives(self.context.results['observable_derivatives'], sat, freqv, save=False)
                        elif chart_type == 'code_phase_diff_raw':
                            out = self.plotter.plot_code_phase_raw_diff({'code_phase_differences': self.context.results['code_phase_differences']}, sat, freqv, save=False)
                        elif chart_type == 'code_phase_diffs':
                            out = self.plotter.plot_code_phase_diff_variation({'code_phase_differences': self.context.results['code_phase_differences']}, sat, freqv, save=False)
                        elif chart_type == 'phase_pred_errors':
                            out = self.plotter.plot_prediction_errors(self.context.results['phase_prediction_errors'], sat, freqv, save=False)
                        elif chart_type == 'double_differences':
                            out = self.plotter.plot_epoch_double_diffs({'epoch_double_diffs': self.context.results['epoch_double_diffs']}, sat, freqv, save=False)

                
                if out and out.get('figure'):
                    status_var.set('图表生成完成')
                    plt.show()
                else:
                    status_var.set('未生成图表')
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror('错误', f'生成图表失败: {e}')
                status_var.set('失败')

        
        # Batch Save buttons
        def batch_save_all():
             if not self.context.observations_meters:
                 messagebox.showwarning('警告', '无数据')
                 return
             out_dir = filedialog.askdirectory(title='选择保存目录')
             if not out_dir: return
             
             status_var.set('正在批量保存所有图表...')
             top.update_idletasks()
             
             try:
                 pre_calculate_metrics()
                 sats = sorted(self.context.observations_meters.keys())
                 count = 0
                 
                 # Save Raw for all sats
                 for sat in sats:
                     self.plotter.plot_raw_observations({'observations_meters': self.context.observations_meters}, sat, save=True, output_dir=out_dir)
                     count += 1
                     
                 # Add other batch saves here if needed, but start with raw
                 status_var.set(f'批量保存完成: {count} 张')
                 messagebox.showinfo('完成', '批量保存完成')
             except Exception as e:
                 messagebox.showerror('错误', f'保存失败: {e}')

        def batch_save_selected():
             if not self.context.observations_meters:
                 messagebox.showwarning('警告', '无数据')
                 return
             chart_type = chart_var.get()
             out_dir = filedialog.askdirectory(title='选择保存目录')
             if not out_dir: return
             
             status_var.set(f'正在批量保存 {chart_type}...')
             top.update_idletasks()
             
             try:
                 # 特殊处理周跳探测
                 if chart_type == 'cycle_slip_detection':
                     from src.processing.cycle_slip_detector import CycleSlipDetector
                     from src.reporting.cycle_slip_logger import CycleSlipLogger
                     
                     # 构建保存路径
                     phone_file = file_var.get()
                     if phone_file:
                         filename = os.path.splitext(os.path.basename(phone_file))[0]
                         cycleslip_dir = os.path.join(out_dir, filename, 'cycleslips')
                     else:
                         cycleslip_dir = os.path.join(out_dir, 'cycleslips')
                     os.makedirs(cycleslip_dir, exist_ok=True)
                     
                     # 解析频率组合选择
                     freq_pair_selection = freq_pair_var.get()
                     freq_pair = None
                     if freq_pair_selection and freq_pair_selection != '自动选择':
                         parts = [p.strip() for p in freq_pair_selection.split('+') if p.strip()]
                         if len(parts) == 2:
                             freq_pair = tuple(parts)
                         else:
                             messagebox.showwarning('警告', f'频率组合解析失败: {freq_pair_selection}，将使用自动选择')
                             freq_pair = None
                     
                     # 读取并校验阈值设置
                     use_custom = threshold_mode_var.get() == 'custom'
                     mw_thresh = None
                     gf_thresh = None
                     if use_custom:
                         try:
                             mw_val = float(mw_threshold_var.get())
                             gf_val = float(gf_threshold_var.get())
                         except Exception:
                             messagebox.showerror('输入错误', '请为 MW 和 GF 输入有效的数字阈值，批量保存已取消')
                             return
                         if mw_val <= 0 or gf_val <= 0:
                             messagebox.showerror('输入错误', '阈值必须为正数，批量保存已取消')
                             return
                         mw_thresh = mw_val
                         gf_thresh = gf_val
                     
                     detector = CycleSlipDetector(
                         use_custom_threshold=use_custom,
                         custom_mw_threshold=mw_thresh,
                         custom_gf_threshold=gf_thresh
                     )
                     detection_results = detector.detect_cycle_slips(
                         self.context.observations_meters,
                         self.context.frequencies,
                         self.context.wavelengths,
                         freq_pair=freq_pair
                     )
                     
                     # 保存日志
                     if save_log_var.get():
                         logger = CycleSlipLogger(output_dir=cycleslip_dir)
                         log_path = logger.save_cycle_slip_log(detection_results)
                         csv_path = logger.save_cycle_slip_csv(detection_results)
                     
                     # 为每个卫星保存图表
                     count = 0
                     for sat in detection_results.keys():
                         try:
                             self.plotter.plot_cycle_slip_analysis(detection_results[sat], sat, save=True, output_dir=cycleslip_dir)
                             count += 1
                         except:
                             continue
                     
                     status_var.set(f'批量保存完成: {count} 张周跳探测图')
                     messagebox.showinfo('完成', f'已保存 {count} 张周跳探测图表')
                     return
                 
                 pre_calculate_metrics()
                 sats = sorted(self.context.observations_meters.keys())
                 count = 0
                 for sat in sats:
                     freqs = list(self.context.observations_meters[sat].keys())
                     for freq in freqs:
                        try:
                            if chart_type == 'raw_observations':
                                self.plotter.plot_raw_observations({'observations_meters': self.context.observations_meters}, sat, save=True, output_dir=out_dir)
                                count += 1; break 
                            elif chart_type == 'derivatives':
                                self.plotter.plot_derivatives(self.context.results['observable_derivatives'], sat, freq, save=True, output_dir=out_dir)
                                count += 1
                            elif chart_type == 'phase_pred_errors':
                                self.plotter.plot_prediction_errors(self.context.results.get('phase_prediction_errors', {}), sat, freq, save=True, output_dir=out_dir)
                                count += 1
                            # Add others...
                        except: continue
                 
                 status_var.set(f'批量保存完成: {count} 张')
                 messagebox.showinfo('完成', f'已保存 {count} 张图表')
             except Exception as e:
                 messagebox.showerror('错误', f'保存失败: {e}')

        ttk.Button(btn_frame, text='生成图表', command=gen_chart).pack(side=tk.LEFT, padx=15)
        ttk.Button(btn_frame, text='批量保存所有图表', command=batch_save_all).pack(side=tk.LEFT, padx=15)
        ttk.Button(btn_frame, text='批量保存选中类型', command=batch_save_selected).pack(side=tk.LEFT, padx=15)
        ttk.Button(btn_frame, text='关闭', command=top.destroy).pack(side=tk.RIGHT, padx=15)



