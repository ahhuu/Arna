from typing import Optional, Dict, Any
import os
import matplotlib.pyplot as plt
from src.core.context import AnalysisContext
from src.visualization.plotter import GNSSPlotter
from src.data.reader import RinexReader
from src.processing.calculator import MetricCalculator
from src.reporting.reporter import ReportGenerator
from src.visualization.batch_exporter import VisualizationBatchExporter


class VisualizationWindow:
    """Visualization helper that wraps GNSSPlotter.

    Provides programmatic API for producing/saving plots without launching UI mainloop (good for tests).
    """

    def __init__(self, context: Optional[AnalysisContext] = None):
        self.context = context or AnalysisContext()
        self.plotter = GNSSPlotter()

    def plot_raw(self, sat_id: str, save: bool = True, output_dir: Optional[str] = None):
        return self.plotter.plot_raw_observations({'observations_meters': self.context.observations_meters}, sat_id, save=save, output_dir=output_dir)

    def plot_satellite_frequency_sequence(self, save: bool = True, output_dir: Optional[str] = None):
        return self.plotter.plot_satellite_frequency_sequence({'observations_meters': self.context.observations_meters}, save=save, output_dir=output_dir)

    def plot_pseudorange_multipath(self, sat_id: str, freq_pair: Optional[tuple] = None,
                                   save: bool = True, output_dir: Optional[str] = None):
        mc = MetricCalculator()
        cache_key = 'pseudorange_multipath_auto'
        if freq_pair:
            cache_key = f'pseudorange_multipath_{freq_pair[0]}+{freq_pair[1]}'
        if not self.context.is_result_current(cache_key):
            self.context.cache_result(cache_key, mc.calculate_pseudorange_multipath(
                {'observations_meters': self.context.observations_meters},
                freq_pair=freq_pair,
            ))
        return self.plotter.plot_pseudorange_multipath(
            self.context.results[cache_key],
            sat_id,
            freq_pair=freq_pair,
            save=save,
            output_dir=output_dir,
        )

    def plot_pseudorange_multipath_overview(self, save: bool = True, output_dir: Optional[str] = None,
                                            smoothing_window: int = 5,
                                            system_filters=None,
                                            sat_filters=None,
                                            freq_filters=None):
        return self.plotter.plot_pseudorange_multipath_overview(
            {'observations_meters': self.context.observations_meters},
            save=save,
            output_dir=output_dir,
            smoothing_window=smoothing_window,
            system_filters=system_filters,
            sat_filters=sat_filters,
            freq_filters=freq_filters,
        )

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

    def plot_ionofree_cmc(self, sat_id: str = None, save: bool = True, output_dir: Optional[str] = None):
        """Plot ionosphere-free combination CMC for one or all satellites."""
        ionofree = self.context.results.get('ionofree_cmc', {})
        if not ionofree:
            mc = MetricCalculator()
            if not self.context.is_result_current('code_phase_differences'):
                inputs = {
                    'observations_meters': self.context.observations_meters,
                    'frequencies': self.context.current_frequencies or self.context.frequencies,
                    'wavelengths': self.context.current_wavelengths or self.context.wavelengths
                }
                self.context.cache_result('code_phase_differences', mc.calculate_code_phase_differences(inputs))
            ionofree = mc.calculate_ionofree_cmc({'code_phase_differences': self.context.results['code_phase_differences']})
            self.context.cache_result('ionofree_cmc', ionofree)
        return self.plotter.plot_ionofree_cmc(ionofree, sat_id=sat_id, save=save, output_dir=output_dir)

    def plot_inter_freq_bias(self, freq1: str, freq2: str, constellation: Optional[str] = None, 
                            save: bool = True, output_dir: Optional[str] = None):
        """Plot inter-frequency bias analysis with ISD correction."""
        from src.processing.inter_freq_bias import InterFrequencyBiasAnalyzer
        analyzer = InterFrequencyBiasAnalyzer()
        analysis_result = analyzer.analyze_inter_freq_bias(
            self.context.observations_meters,
            freq1, freq2, constellation
        )
        self.context.results['inter_freq_bias'] = analysis_result
        return self.plotter.plot_inter_freq_bias(analysis_result, save=save, output_dir=output_dir)

    def show(self, parent):
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
        except Exception:
            return

        top = tk.Toplevel(parent)
        top.title('图表生成')
        top.geometry('800x900')  # Increased size
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
            file_types = [
                ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
                ("All Files", "*.*")
            ]
            f = filedialog.askopenfilename(title='选择RINEX观测文件', filetypes=file_types)
            if f:
                file_var.set(f)
                load_satellite_info()
        ttk.Button(file_frame, text='浏览', command=select_file).pack(side=tk.RIGHT)

        # receiver file
        rx_frame = ttk.LabelFrame(main_frame, text='选择接收机RINEX文件(用于CMC/ISB)', padding=10)
        rx_frame.pack(fill=tk.X, pady=8)
        rx_var = tk.StringVar()
        rx_entry = ttk.Entry(rx_frame, textvariable=rx_var, width=50)
        rx_entry.pack(side=tk.LEFT, padx=(0,10), fill=tk.X, expand=True)

        def select_rx():
            file_types = [
                ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
                ("All Files", "*.*")
            ]
            f = filedialog.askopenfilename(title='选择接收机RINEX文件', filetypes=file_types)
            if f:
                rx_var.set(f)
                load_receiver_satellite_info()
        ttk.Button(rx_frame, text='浏览', command=select_rx).pack(side=tk.RIGHT)

        # Chart types
        chart_frame = ttk.LabelFrame(main_frame, text='图表类型', padding=10)
        chart_frame.pack(fill=tk.X, pady=8)
        chart_types = [
            # 基础数据概览
            ('原始观测', 'raw_observations'),
            ('卫星-频点序列', 'sat_freq_sequence'),
            ('卫星数量', 'satellite_count'),
            ('载噪比分析', 'cnr_analysis'),
            ('数据完整率', 'data_integrity'),
            # 观测数据质量
            ('观测噪声分析', 'observation_noise'),
            ('多普勒质量分析', 'doppler_quality'),
            ('伪距多路径-星座', 'pseudorange_multipath_overview'),
            ('伪距多路径-卫星', 'pseudorange_multipath'),
            # 码-相位一致性
            ('伪距相位原始差值', 'code_phase_diff_raw'),
            ('伪距相位差值之差', 'code_phase_diffs'),
            ('观测值一阶差分', 'derivatives'),
            ('相位预测误差', 'phase_pred_errors'),
            # 精密分析
            ('历元间双差', 'double_differences'),
            ('周跳探测分析 (MW & GF & LLI)', 'cycle_slip_detection'),
            ('接收机CMC', 'receiver_cmc'),
            ('无电离层组合CMC', 'ionofree_cmc'),
            ('ISB分析', 'isb_analysis'),
            ('伪距频间偏差与ISD验证', 'inter_freq_bias'),
        ]
        chart_var = tk.StringVar(value='raw_observations')
        
        # Grid layout for radio buttons
        grid_frame = ttk.Frame(chart_frame)
        grid_frame.pack(fill=tk.X)
        for i, (txt, val) in enumerate(chart_types):
            r = i // 2
            c = i % 2
            ttk.Radiobutton(grid_frame, text=txt, variable=chart_var, value=val).grid(row=r, column=c, sticky='w', padx=10, pady=5)
        
        # 周跳探测频率组合选择（使用GNSS_FREQUENCIES动态生成）
        freq_pair_frame = ttk.Frame(chart_frame)
        freq_pair_frame.pack(fill=tk.X, pady=5)
        ttk.Label(freq_pair_frame, text='频率组合:').pack(side=tk.LEFT)
        freq_pair_var = tk.StringVar(value='自动选择')
        freq_pair_combo = ttk.Combobox(freq_pair_frame, textvariable=freq_pair_var, width=20, state="readonly")
        # 从config中生成频率对列表（用于周跳探测等）
        from src.core.config import GNSS_FREQUENCIES
        cycleslip_freq_options = ['自动选择']
        for system, freqs in GNSS_FREQUENCIES.items():
            freq_list = list(freqs.keys())
            if len(freq_list) >= 2:
                cycleslip_freq_options.append(f"{freq_list[0]}+{freq_list[1]}")
        # 从IF_FREQ_PAIRS生成无电离层组合专用频率对选项
        from src.processing.calculator import MetricCalculator as _MC

        def _get_multipath_pair_options_for_system(sys_code: str = ''):
            """根据星座系统返回伪距多路径可用的频率对。"""
            opts = ['自动选择']
            seen = set()
            if sys_code and sys_code in GNSS_FREQUENCIES:
                freq_keys = list(GNSS_FREQUENCIES[sys_code].keys())
                for idx, f1 in enumerate(freq_keys):
                    for f2 in freq_keys[idx + 1:]:
                        key = tuple(sorted((f1, f2)))
                        if key in seen:
                            continue
                        seen.add(key)
                        opts.append(f"{key[0]}+{key[1]}")
            else:
                for _, freqs in GNSS_FREQUENCIES.items():
                    freq_keys = list(freqs.keys())
                    for idx, f1 in enumerate(freq_keys):
                        for f2 in freq_keys[idx + 1:]:
                            key = tuple(sorted((f1, f2)))
                            if key in seen:
                                continue
                            seen.add(key)
                            opts.append(f"{key[0]}+{key[1]}")
            return opts

        def _get_ionofree_options_for_system(sys_code: str = ''):
            """根据卫星系统返回IF组合频率对选项，无系统时返回所有系统的去重合集"""
            opts = ['自动选择']
            if sys_code and sys_code in _MC.IF_FREQ_PAIRS:
                for f1, f2 in _MC.IF_FREQ_PAIRS[sys_code]:
                    opts.append(f"{f1}+{f2}")
            else:
                seen = set()
                for pairs in _MC.IF_FREQ_PAIRS.values():
                    for f1, f2 in pairs:
                        key = f"{f1}+{f2}"
                        if key not in seen:
                            seen.add(key)
                            opts.append(key)
            return opts

        freq_pair_combo['values'] = cycleslip_freq_options
        freq_pair_var.set(cycleslip_freq_options[0])
        freq_pair_combo.pack(side=tk.LEFT, padx=5)

        multipath_window_frame = ttk.Frame(chart_frame)
        ttk.Label(multipath_window_frame, text='平滑窗口:').pack(side=tk.LEFT)
        multipath_window_var = tk.StringVar(value='5')
        multipath_window_combo = ttk.Combobox(
            multipath_window_frame,
            textvariable=multipath_window_var,
            values=['3', '5', '7', '11'],
            width=4,
            state='readonly',
        )
        multipath_window_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(multipath_window_frame, text='历元').pack(side=tk.LEFT)

        # 切换图表类型时动态更新参数区显示状态
        def _on_chart_type_changed(*_args):
            current_type = chart_var.get()

            if current_type == 'ionofree_cmc':
                sys_code = sat_system_var.get() or ''
                opts = _get_ionofree_options_for_system(sys_code)
                freq_pair_combo['values'] = opts
            elif current_type == 'pseudorange_multipath':
                sys_code = sat_system_var.get() or ''
                freq_pair_combo['values'] = _get_multipath_pair_options_for_system(sys_code)
            elif current_type == 'inter_freq_bias':
                sys_code = sat_system_var.get() or ''
                freq_pair_combo['values'] = _get_ionofree_options_for_system(sys_code)
            else:
                freq_pair_combo['values'] = cycleslip_freq_options
            freq_pair_var.set('自动选择')

            if current_type in ('ionofree_cmc', 'cycle_slip_detection', 'inter_freq_bias', 'pseudorange_multipath'):
                if not freq_pair_frame.winfo_ismapped():
                    freq_pair_frame.pack(fill=tk.X, pady=5)
            else:
                freq_pair_frame.pack_forget()

            if current_type == 'pseudorange_multipath':
                if not multipath_window_frame.winfo_ismapped():
                    multipath_window_frame.pack(fill=tk.X, pady=5)
            else:
                multipath_window_frame.pack_forget()

            if current_type == 'cycle_slip_detection':
                if not threshold_frame.winfo_ismapped():
                    threshold_frame.pack(fill=tk.X, pady=5)
            else:
                threshold_frame.pack_forget()

            if current_type in ('sat_freq_sequence', 'satellite_count', 'pseudorange_multipath_overview', 'cnr_analysis', 'data_integrity', 'observation_noise', 'doppler_quality'):
                if not sequence_filter_frame.winfo_ismapped():
                    sequence_filter_frame.pack(fill=tk.X, pady=(8, 0))
            else:
                sequence_filter_frame.pack_forget()

            if current_type in ('receiver_cmc', 'isb_analysis'):
                if not rx_frame.winfo_ismapped():
                    rx_frame.pack(fill=tk.X, pady=8, before=chart_frame)
            else:
                rx_frame.pack_forget()
        chart_var.trace_add('write', _on_chart_type_changed)
        
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

        sequence_filter_frame = ttk.Frame(sat_frame)
        ttk.Label(sequence_filter_frame, text='数据过滤:').pack(side=tk.LEFT)
        sequence_buttons_frame = ttk.Frame(sequence_filter_frame)
        sequence_buttons_frame.pack(side=tk.LEFT, padx=6)

        def _build_dropdown_group(parent, title):
            button_label = tk.StringVar(value=title)
            button = tk.Menubutton(parent, textvariable=button_label, relief='raised', direction='below')
            menu = tk.Menu(button, tearoff=0)
            button.configure(menu=menu)
            button.pack(side=tk.LEFT, padx=4)

            state_map = {'vars': {}, 'menu': menu, 'label': button_label, 'title': title}

            def _update_button_label():
                values = list(state_map['vars'].values())
                total = len(values)
                selected = sum(1 for var in values if var.get())
                if total == 0:
                    button_label.set(title)
                    return
                elif selected == total:
                    button_label.set(f'{title}(全部)')
                elif selected == 0:
                    button_label.set(f'{title}(无)')
                else:
                    button_label.set(f'{title}({selected}/{total})')

                menu.entryconfig(0, label='清空' if selected == total else '全选')

            def _toggle_all():
                values = list(state_map['vars'].values())
                if not values:
                    return
                all_checked = all(var.get() for var in values)
                new_state = not all_checked
                for var in values:
                    var.set(new_state)
                _update_button_label()

            state_map['toggle_all'] = _toggle_all
            state_map['update_button_label'] = _update_button_label
            return state_map

        sequence_group_widgets = {
            'systems': _build_dropdown_group(sequence_buttons_frame, '系统'),
            'sats': _build_dropdown_group(sequence_buttons_frame, '卫星'),
            'freqs': _build_dropdown_group(sequence_buttons_frame, '频点'),
        }

        def _selected_values(kind: str):
            group = sequence_group_widgets[kind]
            return [key for key, var in group['vars'].items() if var.get()]

        def _populate_group(kind: str, values):
            group = sequence_group_widgets[kind]
            menu = group['menu']
            menu.delete(0, tk.END)
            group['vars'].clear()

            values = sorted(values)
            if not values:
                menu.add_command(label='当前无可用观测数据', state='disabled')
            else:
                menu.add_command(label='全选', command=group['toggle_all'])
                menu.add_separator()
                for value in values:
                    var = tk.BooleanVar(value=True)
                    group['vars'][value] = var
                    menu.add_checkbutton(label=value, variable=var, command=group['update_button_label'])

            group['update_button_label']()

        def _refresh_sequence_filter_values():
            obs = self.context.observations_meters or {}
            systems = sorted({sat_id[0] for sat_id in obs.keys() if sat_id})
            sats = sorted(obs.keys())
            freqs = sorted({freq for sat_freqs in obs.values() for freq in sat_freqs.keys()})
            _populate_group('systems', systems)
            _populate_group('sats', sats)
            _populate_group('freqs', freqs)

        title_font_var = tk.IntVar(value=16)
        label_font_var = tk.IntVar(value=14)
        tick_font_var = tk.IntVar(value=14)
        legend_font_var = tk.IntVar(value=14)
        annotation_font_var = tk.IntVar(value=10)

        def _apply_font_settings():
            try:
                plt.rcParams.update({
                    'axes.titlesize': int(title_font_var.get()),
                    'axes.labelsize': int(label_font_var.get()),
                    'xtick.labelsize': int(tick_font_var.get()),
                    'ytick.labelsize': int(tick_font_var.get()),
                    'legend.fontsize': int(legend_font_var.get()),
                    'figure.titlesize': int(title_font_var.get()),
                    'font.size': int(annotation_font_var.get())
                })
            except Exception:
                pass

        _on_chart_type_changed()

        font_frame = ttk.LabelFrame(main_frame, text='字体设置', padding=10)
        font_frame.pack(fill=tk.X, pady=8)
        ttk.Label(font_frame, text='标题').grid(row=0, column=0, sticky='w', padx=4, pady=2)
        ttk.Spinbox(font_frame, from_=8, to=24, textvariable=title_font_var, width=5).grid(row=0, column=1, sticky='w', padx=4, pady=2)
        ttk.Label(font_frame, text='坐标轴').grid(row=0, column=2, sticky='w', padx=4, pady=2)
        ttk.Spinbox(font_frame, from_=8, to=20, textvariable=label_font_var, width=5).grid(row=0, column=3, sticky='w', padx=4, pady=2)
        ttk.Label(font_frame, text='刻度').grid(row=0, column=4, sticky='w', padx=4, pady=2)
        ttk.Spinbox(font_frame, from_=8, to=18, textvariable=tick_font_var, width=5).grid(row=0, column=5, sticky='w', padx=4, pady=2)
        ttk.Label(font_frame, text='图例').grid(row=0, column=6, sticky='w', padx=4, pady=2)
        ttk.Spinbox(font_frame, from_=8, to=18, textvariable=legend_font_var, width=5).grid(row=0, column=7, sticky='w', padx=4, pady=2)
        ttk.Label(font_frame, text='注释').grid(row=0, column=8, sticky='w', padx=4, pady=2)
        ttk.Spinbox(font_frame, from_=6, to=18, textvariable=annotation_font_var, width=5).grid(row=0, column=9, sticky='w', padx=4, pady=2)
        ttk.Button(font_frame, text='重置', command=lambda: _reset_font_settings()).grid(row=0, column=10, sticky='w', padx=10, pady=2)

        def _reset_font_settings():
            title_font_var.set(16)
            label_font_var.set(14)
            tick_font_var.set(14)
            legend_font_var.set(14)
            annotation_font_var.set(10)
            _apply_font_settings()

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
                        cfg_freq = self.context.nominal_frequencies.get(system, {}).get(f)
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

                self.context.replace_phone_data(
                    obs,
                    data.get('data', {}).get('epochs', []),
                    source_path=file_var.get(),
                    frequencies=freq_map,
                    wavelengths=wavelengths_map,
                )

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
                _refresh_sequence_filter_values()
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
                # build freqs/wavelengths with fallback
                freq_map = {}
                wavelengths_map = {}
                C = 299792458.0
                for sat_id, fmap in obs.items():
                    system = sat_id[0] if sat_id else ''
                    freq_map.setdefault(system, {})
                    wavelengths_map.setdefault(system, {})
                    for f, fv in fmap.items():
                        cfg_freq = self.context.nominal_frequencies.get(system, {}).get(f)
                        wl_list = fv.get('wavelength', [])
                        wl = next((x for x in wl_list if x is not None), None)
                        freq_hz = cfg_freq if cfg_freq is not None else (C / wl if wl else 1575.42e6)
                        freq_map[system][f] = freq_hz
                        wavelengths_map[system][f] = C / freq_hz

                self.context.replace_receiver_data(
                    obs,
                    data.get('data', {}).get('epochs', []),
                    source_path=rx_var.get(),
                    frequencies=freq_map,
                    wavelengths=wavelengths_map,
                )

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
            if chart_var.get() == 'ionofree_cmc':
                opts = _get_ionofree_options_for_system(sys)
                freq_pair_combo['values'] = opts
                freq_pair_var.set(opts[0])
            else:
                try:
                    pair_opts = get_freq_pair_options_for_system(sys)
                    freq_pair_combo['values'] = pair_opts
                    freq_pair_var.set(pair_opts[0])
                except Exception:
                    pass

        sat_system_var.trace('w', on_system_change)
        
        def on_prn_change(*args):
             prn = sat_prn_var.get()
             if not prn: return
             target = self.context.observations_meters if self.context.observations_meters else self.context.receiver_observations
             if target and prn in target:
                 # 当 PRN 改变时，同步更新频率组合选项
                 sys_code = prn[0] if prn else ''
                 if chart_var.get() == 'ionofree_cmc':
                     opts = _get_ionofree_options_for_system(sys_code)
                     freq_pair_combo['values'] = opts
                     freq_pair_var.set(opts[0])
                 else:
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
                'frequencies': self.context.current_frequencies or self.context.frequencies,
                'wavelengths': self.context.current_wavelengths or self.context.wavelengths
            }
            
            # Helper to check and calc
            def check_calc(key, method_name, **kwargs):
                if key not in self.context.results:
                    method = getattr(mc, method_name)
                    self.context.results[key] = method(inputs, **kwargs)

            # Lazy calc based on need, or just Calc All? 
            # Calculating all is safer for 'pre_calculate'
            if not self.context.is_result_current('observable_derivatives'):
                self.context.cache_result('observable_derivatives', mc.calculate_derivatives(inputs))
            if not self.context.is_result_current('code_phase_differences'):
                self.context.cache_result('code_phase_differences', mc.calculate_code_phase_differences(inputs))
            if not self.context.is_result_current('phase_prediction_errors'):
                self.context.cache_result('phase_prediction_errors', mc.calculate_phase_prediction_errors(inputs))
            if not self.context.is_result_current('epoch_double_diffs'):
                self.context.cache_result('epoch_double_diffs', mc.calculate_epoch_double_differences(inputs))

        def _get_pseudorange_multipath_results(freq_pair_selection: str = '自动选择', smoothing_window: int = 5):
            mc = MetricCalculator()
            selected_freq_pair = None
            if freq_pair_selection and freq_pair_selection != '自动选择':
                parts = [p.strip() for p in freq_pair_selection.split('+') if p.strip()]
                if len(parts) == 2:
                    selected_freq_pair = tuple(parts)
            cache_key = (
                'pseudorange_multipath_auto'
                if selected_freq_pair is None
                else f'pseudorange_multipath_{selected_freq_pair[0]}+{selected_freq_pair[1]}'
            )
            cache_key = f'{cache_key}_w{smoothing_window}'
            if not self.context.is_result_current(cache_key):
                self.context.cache_result(cache_key, mc.calculate_pseudorange_multipath(
                    {'observations_meters': self.context.observations_meters},
                    freq_pair=selected_freq_pair,
                    smoothing_window=smoothing_window,
                ))
            return self.context.results.get(cache_key, {}), selected_freq_pair

        def _save_pseudorange_multipath_overview(output_dir: str, save: bool = True):
            smoothing_window = int(multipath_window_var.get() or '5')
            return self.plotter.plot_pseudorange_multipath_overview(
                {'observations_meters': self.context.observations_meters},
                save=save,
                output_dir=output_dir,
                smoothing_window=smoothing_window,
                system_filters=_selected_values('systems'),
                sat_filters=_selected_values('sats'),
                freq_filters=_selected_values('freqs'),
            )

        def _sequence_plot_filters():
            return {
                'system_filters': _selected_values('systems'),
                'sat_filters': _selected_values('sats'),
                'freq_filters': _selected_values('freqs'),
            }

        def gen_chart():
            _apply_font_settings()
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
                    if not self.context.is_result_current('receiver_cmc', depends_on='receiver'):
                        mc = MetricCalculator()
                        receiver_cmc = mc.calculate_receiver_cmc({
                            'receiver_observations': self.context.receiver_observations,
                            'receiver_frequencies': self.context.results.get('receiver_frequencies'),
                            'receiver_wavelengths': self.context.results.get('receiver_wavelengths')
                        })
                        self.context.cache_result('receiver_cmc', receiver_cmc, depends_on='receiver')
                    if not self.context.results['receiver_cmc']:
                        messagebox.showwarning('警告', '未生成接收机CMC，请检查伪距和载波相位观测')
                        return
                    out = self.plotter.plot_receiver_cmc(self.context.results['receiver_cmc'], save=False)

                # Pseudorange Multipath Special Case
                elif chart_type == 'pseudorange_multipath':
                    if not self.context.observations_meters:
                        messagebox.showerror('错误', '请先加载手机RINEX文件')
                        return
                    multipath_results, selected_freq_pair = _get_pseudorange_multipath_results(
                        freq_pair_var.get(),
                        smoothing_window=int(multipath_window_var.get() or '5'),
                    )
                    if not multipath_results:
                        messagebox.showerror('错误', '无法生成伪距多路径-卫星结果，请检查双频数据或频率组合')
                        return
                    if not sat:
                        messagebox.showerror('错误', '请先选择卫星')
                        return
                    out = self.plotter.plot_pseudorange_multipath(
                        multipath_results,
                        sat,
                        freq_pair=selected_freq_pair,
                        save=False,
                    )

                elif chart_type == 'pseudorange_multipath_overview':
                    if not self.context.observations_meters:
                        messagebox.showerror('错误', '请先加载手机RINEX文件')
                        return
                    out = _save_pseudorange_multipath_overview(output_dir=None, save=False)

                # Ionosphere-Free CMC Special Case
                elif chart_type == 'ionofree_cmc':
                    if not self.context.observations_meters:
                        messagebox.showerror('错误', '请先加载手机RINEX文件')
                        return
                    # 确保CMC已计算
                    mc = MetricCalculator()
                    if not self.context.is_result_current('code_phase_differences'):
                        inputs = {
                            'observations_meters': self.context.observations_meters,
                            'epochs': self.context.results.get('epochs', []),
                            'frequencies': self.context.current_frequencies or self.context.frequencies,
                            'wavelengths': self.context.current_wavelengths or self.context.wavelengths
                        }
                        self.context.cache_result('code_phase_differences', mc.calculate_code_phase_differences(inputs))
                    # 解析GUI频率组合选择
                    freq_pair_selection = freq_pair_var.get()
                    selected_freq_pair = None
                    if freq_pair_selection and freq_pair_selection != '自动选择':
                        parts = [p.strip() for p in freq_pair_selection.split('+') if p.strip()]
                        if len(parts) == 2:
                            selected_freq_pair = tuple(parts)
                    # 用频率对作为缓存key，不同频率对不共享缓存
                    cache_key = f'ionofree_cmc_{freq_pair_selection}'
                    if not self.context.is_result_current(cache_key):
                        self.context.cache_result(cache_key, mc.calculate_ionofree_cmc(
                            {'code_phase_differences': self.context.results['code_phase_differences']},
                            freq_pair=selected_freq_pair))
                    ionofree = self.context.results[cache_key]
                    if not ionofree:
                        messagebox.showwarning('警告', '无法计算无电离层组合CMC（需要双频数据，请检查频率组合选择）')
                        return
                    sat = sat_prn_var.get()
                    if sat and sat in ionofree:
                        out = self.plotter.plot_ionofree_cmc(ionofree, sat_id=sat, save=False)
                    else:
                        # 支持多频率对key格式 (如 "E05:L1C+L5Q")，按sat_id前缀匹配
                        matched_keys = [k for k in ionofree if k == sat or k.startswith(sat + ':')]
                        if matched_keys:
                            out = self.plotter.plot_ionofree_cmc(ionofree, sat_id=matched_keys[0], save=False)
                        else:
                            first_sat = sorted(ionofree.keys())[0]
                            out = self.plotter.plot_ionofree_cmc(ionofree, sat_id=first_sat, save=False)

                # ISB Special Case
                elif chart_type == 'isb_analysis':
                    if not self.context.observations_meters or not self.context.receiver_observations:
                        messagebox.showerror('错误', 'ISB分析需要同时加载手机和接收机文件')
                        return
                    # We assume ISB analysis might be complex. 
                    # For simplicity, if not in results, we try to calculating it
                    if not self.context.is_result_current('isb_analysis', depends_on='both'):
                         messagebox.showinfo("提示", "ISB分析可能需要较长时间计算，请稍候...")
                         top.update_idletasks()
                         mc = MetricCalculator()
                         # NOTE: update with actual args needed for calculate_isb
                         self.context.cache_result('isb_analysis', mc.calculate_isb({
                             'observations_meters': self.context.observations_meters,
                             'receiver_observations': self.context.receiver_observations,
                             'epochs': self.context.results.get('epochs', [])
                         }), depends_on='both')
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
                        self.context.current_frequencies or self.context.frequencies,
                        self.context.current_wavelengths or self.context.wavelengths,
                        freq_pair=freq_pair
                    )
                    
                    # 保存到context以便后续使用
                    self.context.cache_result('cycle_slip_detection', detection_results)
                    
                    # 构建保存路径：results/文件名/cycleslips/
                    phone_file = file_var.get()
                    if phone_file:
                        filename = os.path.splitext(os.path.basename(phone_file))[0]
                        cycleslip_dir = os.path.join('results', filename, 'cycleslips')
                    else:
                        cycleslip_dir = os.path.join('results', 'cycleslips')
                    os.makedirs(cycleslip_dir, exist_ok=True)
                    
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
                    
                    # 伪距频间偏差分析
                    if chart_type == 'inter_freq_bias':
                        status_var.set('正在计算频间偏差...')
                        top.update_idletasks()
                        
                        # 解析频率组合
                        freq_pair_selection = freq_pair_var.get()
                        if not freq_pair_selection or freq_pair_selection == '自动选择':
                            # 自动选择第一个可用的频率对
                            system = sat[0] if sat else None
                            from src.core.config import GNSS_FREQUENCIES
                            if system and system in GNSS_FREQUENCIES:
                                freq_keys = list(GNSS_FREQUENCIES[system].keys())
                                if len(freq_keys) >= 2:
                                    freq1_name, freq2_name = freq_keys[0], freq_keys[1]
                                else:
                                    messagebox.showerror('错误', f'系统 {system} 的频率数量不足')
                                    return
                            else:
                                messagebox.showerror('错误', '无法确定频率组合，请手动选择')
                                return
                        else:
                            parts = [p.strip() for p in freq_pair_selection.split('+') if p.strip()]
                            if len(parts) == 2:
                                freq1_name, freq2_name = parts[0], parts[1]
                            else:
                                messagebox.showerror('错误', '频率组合格式错误')
                                return
                        
                        # 执行分析
                        try:
                            from src.processing.inter_freq_bias import InterFrequencyBiasAnalyzer
                            analyzer = InterFrequencyBiasAnalyzer()
                            
                            # 获取星座系统（可选过滤）
                            system = sat_system_var.get() if sat_system_var.get() else None
                            
                            analysis_result = analyzer.analyze_inter_freq_bias(
                                self.context.observations_meters,
                                freq1_name,
                                freq2_name,
                                constellation=system
                            )
                            
                            if 'error' in analysis_result:
                                messagebox.showerror('错误', analysis_result['error'])
                                return
                            
                            # 保存到context
                            self.context.cache_result('inter_freq_bias', analysis_result)
                            
                            # 绘图
                            out = self.plotter.plot_inter_freq_bias(analysis_result, save=False)
                            status_var.set('频间偏差分析完成')
                            
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            messagebox.showerror('错误', f'频间偏差分析失败: {e}')
                            status_var.set('分析失败')
                            return
                    
                    elif chart_type == 'raw_observations':
                        out = self.plotter.plot_raw_observations({'observations_meters': self.context.observations_meters}, sat, save=False)
                    elif chart_type == 'sat_freq_sequence':
                        out = self.plotter.plot_satellite_frequency_sequence({'observations_meters': self.context.observations_meters}, save=False, **_sequence_plot_filters())
                    elif chart_type == 'satellite_count':
                        out = self.plotter.plot_satellite_count({'observations_meters': self.context.observations_meters}, save=False, **_sequence_plot_filters())
                    elif chart_type == 'cnr_analysis':
                        out = self.plotter.plot_cnr_analysis({'observations_meters': self.context.observations_meters}, save=False, **_sequence_plot_filters())
                    elif chart_type == 'data_integrity':
                        out = self.plotter.plot_data_integrity({'observations_meters': self.context.observations_meters}, save=False, **_sequence_plot_filters())
                    elif chart_type == 'observation_noise':
                        out = self.plotter.plot_observation_noise({'observations_meters': self.context.observations_meters}, save=False, **_sequence_plot_filters())
                    elif chart_type == 'doppler_quality':
                        out = self.plotter.plot_doppler_quality({'observations_meters': self.context.observations_meters}, save=False, **_sequence_plot_filters())
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

        
        def save_visualization_manifest(project_dir, selection, chart_count=None):
            """Refresh the restart-safe inventory; reports are made on demand."""
            return ReportGenerator().write_visualization_report(
                project_dir,
                file_var.get() or self.context.input_path or '',
                selection=selection,
                metadata={'chart_count_this_run': chart_count},
                generate_reports=False,
            )

        # Batch Save buttons
        def _batch_parameters():
            selected_pair = None
            selection = freq_pair_var.get()
            if selection and selection != '自动选择':
                parts = tuple(part.strip() for part in selection.split('+') if part.strip())
                if len(parts) == 2:
                    selected_pair = parts
            use_custom = threshold_mode_var.get() == 'custom'
            return {
                **_sequence_plot_filters(),
                'smoothing_window': int(multipath_window_var.get() or '5'),
                'freq_pair': selected_pair,
                'use_custom_threshold': use_custom,
                'mw_threshold': float(mw_threshold_var.get()) if use_custom else None,
                'gf_threshold': float(gf_threshold_var.get()) if use_custom else None,
            }

        def batch_save_all():
             if not self.context.observations_meters:
                 messagebox.showwarning('警告', '无数据')
                 return
             
             # Auto-detect output dir
             phone_file = file_var.get()
             if not phone_file:
                 messagebox.showwarning('警告', '未选择文件')
                 return
             
             base_dir = os.path.dirname(phone_file)
             obs_name = os.path.splitext(os.path.basename(phone_file))[0]
             
             # Root results dir: input_dir/Arna_results/filename/visualization/
             project_dir = os.path.join(base_dir, "Arna_results", obs_name, "visualization")
             if not os.path.exists(project_dir):
                 os.makedirs(project_dir)

             exporter = VisualizationBatchExporter(self.context, self.plotter)
             status_var.set(f'正在保存全部图表至: {project_dir} ...')
             top.update_idletasks()
             summary = exporter.export_all(project_dir, _batch_parameters())
             save_visualization_manifest(project_dir, 'all', summary['success_count'])
             status_var.set(
                 f'全部导出完成: 成功 {summary["success_count"]}，'
                 f'跳过 {summary["skipped_count"]}，失败 {summary["failure_count"]}'
             )
             messagebox.showinfo(
                 '批量导出完成',
                 f'成功: {summary["success_count"]}\n'
                 f'跳过: {summary["skipped_count"]}\n'
                 f'失败: {summary["failure_count"]}\n'
                 f'错误清单: {summary["error_log"]}'
             )
             return
             
             status_var.set(f'正在保存至: {project_dir} ...')
             top.update_idletasks()
             
             try:
                 _apply_font_settings()
                 pre_calculate_metrics()
                 sats = sorted(self.context.observations_meters.keys())
                 count = 0
                 # We'll compute per-satellite per-pair when saving to ensure all available pairs are exported
                 smoothing_w = int(multipath_window_var.get() or '5')
                 
                 # Raw Folder
                 raw_dir = os.path.join(project_dir, 'Raw_observations')
                 if not os.path.exists(raw_dir): os.makedirs(raw_dir)
                 
                 # Save Raw for all sats
                 for sat in sats:
                     self.plotter.plot_raw_observations({'observations_meters': self.context.observations_meters}, sat, save=True, output_dir=raw_dir)
                     count += 1

                 # Pseudorange multipath: save all valid freq-pairs per satellite
                 from src.processing.calculator import MetricCalculator
                 multi_dir = os.path.join(project_dir, 'Pseudorange_Multipath_Satellite')
                 if not os.path.exists(multi_dir): os.makedirs(multi_dir)
                 mc = MetricCalculator()
                 for sat in sats:
                     sat_freqs = list(self.context.observations_meters.get(sat, {}).keys())
                     if len(sat_freqs) < 2:
                         continue
                     # iterate combinations of frequency names
                     from itertools import combinations
                     for f1, f2 in combinations(sat_freqs, 2):
                         try:
                             res = mc.calculate_pseudorange_multipath(
                                 {'observations_meters': {sat: self.context.observations_meters[sat]}},
                                 freq_pair=(f1, f2),
                                 smoothing_window=smoothing_w,
                             )
                             if sat in res:
                                 self.plotter.plot_pseudorange_multipath(
                                     res,
                                     sat,
                                     freq_pair=(f1, f2),
                                     save=True,
                                     output_dir=multi_dir,
                                 )
                                 count += 1
                         except Exception:
                             continue
                 
                 # Satellite frequency sequence
                 seq_dir = os.path.join(project_dir, 'Satellite_frequency_sequence')
                 if not os.path.exists(seq_dir): os.makedirs(seq_dir)
                 try:
                     self.plotter.plot_satellite_frequency_sequence(
                         {'observations_meters': self.context.observations_meters},
                         save=True,
                         output_dir=seq_dir,
                         **_sequence_plot_filters(),
                     )
                     count += 1
                 except Exception:
                     pass

                 # Satellite count
                 count_dir = os.path.join(project_dir, 'Satellite_count')
                 if not os.path.exists(count_dir): os.makedirs(count_dir)
                 try:
                     self.plotter.plot_satellite_count({'observations_meters': self.context.observations_meters}, save=True, output_dir=count_dir)
                     count += 1
                 except Exception:
                     pass

                 # CNR Analysis
                 cnr_dir = os.path.join(project_dir, 'CNR_Analysis')
                 if not os.path.exists(cnr_dir): os.makedirs(cnr_dir)
                 try:
                     self.plotter.plot_cnr_analysis({'observations_meters': self.context.observations_meters}, save=True, output_dir=cnr_dir)
                     count += 1
                 except Exception:
                     pass

                 # Observation Noise Analysis
                 noise_dir = os.path.join(project_dir, 'Observation_Noise')
                 if not os.path.exists(noise_dir): os.makedirs(noise_dir)
                 try:
                     self.plotter.plot_observation_noise({'observations_meters': self.context.observations_meters}, save=True, output_dir=noise_dir)
                     count += 1
                 except Exception:
                     pass

                 # Doppler Quality Analysis
                 doppler_dir = os.path.join(project_dir, 'Doppler_Quality')
                 if not os.path.exists(doppler_dir): os.makedirs(doppler_dir)
                 try:
                     self.plotter.plot_doppler_quality({'observations_meters': self.context.observations_meters}, save=True, output_dir=doppler_dir)
                     count += 1
                 except Exception:
                     pass
                 
                 save_visualization_manifest(project_dir, 'all', count)
                 status_var.set(f'批量保存完成: {count} 张')
                 messagebox.showinfo('完成', f'批量保存完成\n保存路径: {project_dir}')
             except Exception as e:
                 messagebox.showerror('错误', f'保存失败: {e}')

        def batch_save_selected():
            if not self.context.observations_meters:
                messagebox.showwarning('警告', '无数据')
                return

            phone_file = file_var.get()
            if not phone_file:
                messagebox.showwarning('警告', '未选择文件')
                return

            base_dir = os.path.dirname(phone_file)
            obs_name = os.path.splitext(os.path.basename(phone_file))[0]

            # Root results dir: input_dir/Arna_results/filename/visualization/
            project_dir = os.path.join(base_dir, "Arna_results", obs_name, "visualization")
            if not os.path.exists(project_dir):
                os.makedirs(project_dir)

            chart_type = chart_var.get()

            exporter = VisualizationBatchExporter(self.context, self.plotter)
            status_var.set(f'正在批量保存 {chart_type} ...')
            top.update_idletasks()
            result = exporter.export_chart(chart_type, project_dir, _batch_parameters())
            save_visualization_manifest(project_dir, chart_type, result['count'])
            status_var.set(
                f'导出完成: 成功 {result["count"]}，'
                f'跳过 {len(result["skipped"])}，失败 {len(result["errors"])}'
            )
            if result['errors']:
                error_path = os.path.join(project_dir, 'visualization_export_errors.json')
                import json
                with open(error_path, 'w', encoding='utf-8') as fh:
                    json.dump({'results': [result]}, fh, ensure_ascii=False, indent=2, default=str)
            messagebox.showinfo(
                '导出完成',
                f'成功: {result["count"]}\n跳过: {len(result["skipped"])}\n'
                f'失败: {len(result["errors"])}\n目录: '
                f'{os.path.join(project_dir, exporter.FOLDERS[chart_type])}'
            )
            return

            # Map chart_type to Folder Name
            folder_map = {
                'raw_observations': 'Raw_observations',
                'sat_freq_sequence': 'Satellite_frequency_sequence',
                'satellite_count': 'Satellite_count',
                 'pseudorange_multipath_overview': 'Pseudorange_Multipath_Constellation',
                'pseudorange_multipath': 'Pseudorange_Multipath_Satellite',
                'derivatives': 'Derivatives',
                'code_phase_diffs': 'Code_phase_diffs',
                'code_phase_diff_raw': 'Code_phase_diffs_raw',
                'phase_pred_errors': 'Prediction_errors',
                'double_differences': 'Double_differences',
                'isb_analysis': 'ISB_analysis',
                'receiver_cmc': 'Receiver_CMC',
                'ionofree_cmc': 'Ionofree_CMC',
                'cycle_slip_detection': 'Cycle_slips',
                'inter_freq_bias': 'Inter_freq_bias',
                'cnr_analysis': 'CNR_Analysis'
                ,'data_integrity': 'Data_Integrity'
                ,'observation_noise': 'Observation_Noise'
                ,'doppler_quality': 'Doppler_Quality'
            }
            folder_name = folder_map.get(chart_type, chart_type)
            target_dir = os.path.join(project_dir, folder_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            status_var.set(f'正在批量保存 {chart_type} 至 {target_dir}...')
            top.update_idletasks()

            try:
                _apply_font_settings()
                smoothing_w = int(multipath_window_var.get() or '5')

                if chart_type == 'sat_freq_sequence':
                    self.plotter.plot_satellite_frequency_sequence(
                        {'observations_meters': self.context.observations_meters},
                        save=True,
                        output_dir=target_dir,
                        **_sequence_plot_filters(),
                    )
                    save_visualization_manifest(project_dir, chart_type, 1)
                    status_var.set(f'已保存卫星-频点序列图至 {target_dir}')
                    messagebox.showinfo('完成', f'卫星-频点序列图已保存\n保存路径: {target_dir}')
                    return

                if chart_type == 'satellite_count':
                    self.plotter.plot_satellite_count(
                        {'observations_meters': self.context.observations_meters},
                        save=True,
                        output_dir=target_dir,
                        **_sequence_plot_filters()
                    )
                    save_visualization_manifest(project_dir, chart_type, 1)
                    status_var.set(f'已保存卫星数量图至 {target_dir}')
                    messagebox.showinfo('完成', f'卫星数量图已保存\n保存路径: {target_dir}')
                    return

                if chart_type == 'data_integrity':
                    self.plotter.plot_data_integrity(
                        {'observations_meters': self.context.observations_meters},
                        save=True,
                        output_dir=target_dir,
                        **_sequence_plot_filters()
                    )
                    save_visualization_manifest(project_dir, chart_type, 1)
                    status_var.set(f'已保存数据完整率至 {target_dir}')
                    messagebox.showinfo('完成', f'数据完整率图已保存\n保存路径: {target_dir}')
                    return

                if chart_type == 'cnr_analysis':
                    self.plotter.plot_cnr_analysis(
                        {'observations_meters': self.context.observations_meters},
                        save=True,
                        output_dir=target_dir,
                        **_sequence_plot_filters()
                    )
                    save_visualization_manifest(project_dir, chart_type, 1)
                    status_var.set(f'已保存载噪比分析图至 {target_dir}')
                    messagebox.showinfo('完成', f'载噪比分析图已保存\n保存路径: {target_dir}')
                    return

                if chart_type == 'observation_noise':
                    self.plotter.plot_observation_noise(
                        {'observations_meters': self.context.observations_meters},
                        save=True,
                        output_dir=target_dir,
                        **_sequence_plot_filters()
                    )
                    save_visualization_manifest(project_dir, chart_type, 1)
                    status_var.set(f'已保存观测噪声分析图至 {target_dir}')
                    messagebox.showinfo('完成', f'观测噪声分析图已保存\n保存路径: {target_dir}')
                    return

                if chart_type == 'doppler_quality':
                    result = self.plotter.plot_doppler_quality(
                        {'observations_meters': self.context.observations_meters},
                        save=True,
                        output_dir=target_dir,
                        **_sequence_plot_filters()
                    )
                    save_visualization_manifest(project_dir, chart_type, 1)
                    status_var.set(f'已保存多普勒质量分析图至 {target_dir}')
                    messagebox.showinfo('完成', f'多普勒质量分析图和日志已保存\n保存路径: {target_dir}\n日志: {result.get("log")}')
                    return

                if chart_type == 'pseudorange_multipath_overview':
                    out = _save_pseudorange_multipath_overview(output_dir=target_dir, save=True)
                    save_visualization_manifest(project_dir, chart_type, 1)
                    status_var.set(f'已保存伪距多路径-星座图至 {target_dir}')
                    if out and out.get('log_path'):
                        messagebox.showinfo('完成', f'伪距多路径-星座图已保存\n保存路径: {target_dir}\n日志: {out["log_path"]}')
                    else:
                        messagebox.showinfo('完成', f'伪距多路径-星座图已保存\n保存路径: {target_dir}')
                    return

                if chart_type == 'pseudorange_multipath':
                    # Save all available frequency pairs per satellite
                    from itertools import combinations
                    mc = MetricCalculator()
                    count = 0
                    for sat in sorted(self.context.observations_meters.keys()):
                        sat_freqs = list(self.context.observations_meters.get(sat, {}).keys())
                        if len(sat_freqs) < 2:
                            continue
                        for f1, f2 in combinations(sat_freqs, 2):
                            try:
                                res = mc.calculate_pseudorange_multipath(
                                    {'observations_meters': {sat: self.context.observations_meters[sat]}},
                                    freq_pair=(f1, f2),
                                    smoothing_window=smoothing_w,
                                )
                            except Exception:
                                continue

                            if sat in res:
                                try:
                                    self.plotter.plot_pseudorange_multipath(
                                        res,
                                        sat,
                                        freq_pair=(f1, f2),
                                        save=True,
                                        output_dir=target_dir,
                                    )
                                    count += 1
                                except Exception:
                                    continue

                    save_visualization_manifest(project_dir, chart_type, count)
                    status_var.set(f'已保存伪距多路径-卫星图至 {target_dir}')
                    messagebox.showinfo('完成', f'伪距多路径-卫星图已保存\n保存路径: {target_dir}')
                    return

                if chart_type == 'cycle_slip_detection':
                    from src.processing.cycle_slip_detector import CycleSlipDetector
                    from src.reporting.cycle_slip_logger import CycleSlipLogger

                    cycleslip_dir = target_dir
                    freq_pair_selection = freq_pair_var.get()
                    freq_pair = None
                    if freq_pair_selection and freq_pair_selection != '自动选择':
                        parts = [p.strip() for p in freq_pair_selection.split('+') if p.strip()]
                        if len(parts) == 2:
                            freq_pair = tuple(parts)

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
                        self.context.current_frequencies or self.context.frequencies,
                        self.context.current_wavelengths or self.context.wavelengths,
                        freq_pair=freq_pair
                    )

                    logger = CycleSlipLogger(output_dir=cycleslip_dir)
                    logger.save_cycle_slip_log(detection_results)
                    logger.save_cycle_slip_csv(detection_results)

                    count = 0
                    for sat in detection_results.keys():
                        try:
                            self.plotter.plot_cycle_slip_analysis(detection_results[sat], sat, save=True, output_dir=cycleslip_dir)
                            count += 1
                        except Exception:
                            continue

                    save_visualization_manifest(project_dir, chart_type, count)
                    status_var.set(f'批量保存完成: {count} 张周跳探测图')
                    messagebox.showinfo('完成', f'已保存 {count} 张周跳探测图表\n目录: {cycleslip_dir}')
                    return

                count = 0

                if chart_type == 'ionofree_cmc':
                    mc = MetricCalculator()
                    if not self.context.is_result_current('code_phase_differences'):
                        inputs = {
                            'observations_meters': self.context.observations_meters,
                            'epochs': self.context.results.get('epochs', []),
                            'frequencies': self.context.current_frequencies or self.context.frequencies,
                            'wavelengths': self.context.current_wavelengths or self.context.wavelengths
                        }
                        self.context.cache_result('code_phase_differences', mc.calculate_code_phase_differences(inputs))

                    freq_pair_selection = freq_pair_var.get()
                    selected_freq_pair = None
                    if freq_pair_selection and freq_pair_selection != '自动选择':
                        parts = [p.strip() for p in freq_pair_selection.split('+') if p.strip()]
                        if len(parts) == 2:
                            selected_freq_pair = tuple(parts)

                    cache_key = f'ionofree_cmc_{freq_pair_selection}'
                    if not self.context.is_result_current(cache_key):
                        self.context.cache_result(cache_key, mc.calculate_ionofree_cmc(
                            {'code_phase_differences': self.context.results['code_phase_differences']},
                            freq_pair=selected_freq_pair))
                    ionofree = self.context.results.get(cache_key, {})
                    for sid in sorted(ionofree.keys()):
                        try:
                            self.plotter.plot_ionofree_cmc(ionofree, sat_id=sid, save=True, output_dir=target_dir)
                            count += 1
                        except Exception:
                            continue

                elif chart_type == 'isb_analysis':
                    if not self.context.is_result_current('isb_analysis', depends_on='both'):
                        messagebox.showerror('错误', '请先进行ISB分析或加载数据')
                        return
                    self.plotter.plot_isb_analysis(self.context.results['isb_analysis'], save=True, output_dir=target_dir)
                    count = 1

                elif chart_type == 'inter_freq_bias':
                    try:
                        from src.processing.inter_freq_bias import InterFrequencyBiasAnalyzer
                        from src.core.config import GNSS_FREQUENCIES
                        from itertools import combinations

                        analyzer = InterFrequencyBiasAnalyzer()

                        available_systems = set()
                        for sat_id in self.context.observations_meters.keys():
                            available_systems.add(sat_id[0])

                        status_var.set(f'正在分析频间偏差，发现 {len(available_systems)} 个星座系统...')
                        top.update_idletasks()

                        for system in sorted(available_systems):
                            if system not in GNSS_FREQUENCIES:
                                continue
                            freq_keys = list(GNSS_FREQUENCIES[system].keys())
                            if len(freq_keys) < 2:
                                continue
                            freq_pairs = list(combinations(freq_keys, 2))
                            status_var.set(f'正在分析 {system} 系统，共 {len(freq_pairs)} 个频率组合...')
                            top.update_idletasks()
                            for freq1_name, freq2_name in freq_pairs:
                                try:
                                    analysis_result = analyzer.analyze_inter_freq_bias(
                                        self.context.observations_meters,
                                        freq1_name,
                                        freq2_name,
                                        constellation=system
                                    )
                                    if 'error' in analysis_result:
                                        continue
                                    raw_diffs = analysis_result.get('raw_diffs', {})
                                    if not raw_diffs:
                                        continue
                                    self.plotter.plot_inter_freq_bias(
                                        analysis_result,
                                        save=True,
                                        output_dir=target_dir
                                    )
                                    count += 1
                                    status_var.set(f'已保存 {count} 张: {system} {freq1_name}-{freq2_name}')
                                    top.update_idletasks()
                                except Exception as e:
                                    print(f"警告: {system} {freq1_name}-{freq2_name} 分析失败: {e}")
                                    continue

                        if count == 0:
                            messagebox.showwarning('警告', '未找到有效的频率组合数据')
                            return

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        messagebox.showerror('错误', f'批量频间偏差分析失败: {e}')
                        return

                else:
                    pre_calculate_metrics()
                    sats = sorted(self.context.observations_meters.keys())
                    for sat in sats:
                        freqs = list(self.context.observations_meters[sat].keys())
                        if chart_type == 'raw_observations':
                            try:
                                self.plotter.plot_raw_observations({'observations_meters': self.context.observations_meters}, sat, save=True, output_dir=target_dir)
                                count += 1
                            except Exception:
                                pass
                        else:
                            for freq in freqs:
                                try:
                                    if chart_type == 'derivatives':
                                        self.plotter.plot_derivatives(self.context.results['observable_derivatives'], sat, freq, save=True, output_dir=target_dir)
                                        count += 1
                                    elif chart_type == 'phase_pred_errors':
                                        self.plotter.plot_prediction_errors(self.context.results.get('phase_prediction_errors', {}), sat, freq, save=True, output_dir=target_dir)
                                        count += 1
                                    elif chart_type == 'code_phase_diffs':
                                        self.plotter.plot_code_phase_diff_variation({'code_phase_differences': self.context.results['code_phase_differences']}, sat, freq, save=True, output_dir=target_dir)
                                        count += 1
                                    elif chart_type == 'code_phase_diff_raw':
                                        self.plotter.plot_code_phase_raw_diff({'code_phase_differences': self.context.results['code_phase_differences']}, sat, freq, save=True, output_dir=target_dir)
                                        count += 1
                                    elif chart_type == 'double_differences':
                                        self.plotter.plot_epoch_double_diffs({'epoch_double_diffs': self.context.results['epoch_double_diffs']}, sat, freq, save=True, output_dir=target_dir)
                                        count += 1
                                except Exception:
                                    continue

                save_visualization_manifest(project_dir, chart_type, count)
                status_var.set(f'批量保存完成: {count} 张')
                messagebox.showinfo('完成', f'已保存 {count} 张图表\n目录: {target_dir}')
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror('错误', f'保存失败: {e}')

        ttk.Button(btn_frame, text='生成图表', command=gen_chart).pack(side=tk.LEFT, padx=15)
        ttk.Button(btn_frame, text='批量保存所有图表', command=batch_save_all).pack(side=tk.LEFT, padx=15)
        ttk.Button(btn_frame, text='批量保存选中类型', command=batch_save_selected).pack(side=tk.LEFT, padx=15)
        ttk.Button(btn_frame, text='关闭', command=top.destroy).pack(side=tk.RIGHT, padx=15)



