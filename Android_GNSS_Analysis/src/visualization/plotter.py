from typing import Dict, Any, Optional, Sequence
import os
import statistics
import matplotlib
# Use Agg backend for headless environments (tests/CLI). GUI can set a different backend on startup.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


class GNSSPlotter:
    """Matplotlib-based plotting utilities. Returns Figure objects or saves files.

    Methods are lightweight and accept the data structures produced by the processing layer.
    All `save` flags default to True for GUI convenience; tests should call with save=False.
    """

    def _ensure_output_dir(self, output_dir: Optional[str]) -> str:
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _save_fig(self, fig: Figure, name: str, output_dir: Optional[str] = None) -> str:
        out_dir = self._ensure_output_dir(output_dir)
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return path

    def plot_raw_observations(self, data: Dict[str, Any], sat_id: str, save: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Plot raw code/phase with phase aligned to code (mirrors original tool output)."""
        obs = data.get('observations_meters', data) if isinstance(data, dict) else data
        sat = obs.get(sat_id)
        if not sat:
            raise ValueError(f"No data for satellite {sat_id}")

        fig, ax = plt.subplots(figsize=(12, 6))
        style_dict = {
            'L1C': {'color': 'blue', 'linestyle': '-', 'marker': 's', 'phase_marker': 'o', 'label_code': 'L1C code', 'label_phase': 'L1C phase'},
            'L1D': {'color': 'cyan', 'linestyle': '-', 'marker': '^', 'phase_marker': 'x', 'label_code': 'L1D code', 'label_phase': 'L1D phase'},
            'L1P': {'color': 'red', 'linestyle': '-', 'marker': 'D', 'phase_marker': '*', 'label_code': 'L1P code', 'label_phase': 'L1P phase'},
            'L2I': {'color': 'blue', 'linestyle': '-', 'marker': 'o', 'phase_marker': '^', 'label_code': 'L2I code', 'label_phase': 'L2I phase'},
            'L5Q': {'color': 'cyan', 'linestyle': '-', 'marker': '^', 'phase_marker': 's', 'label_code': 'L5Q code', 'label_phase': 'L5Q phase'},
            'L7Q': {'color': 'magenta', 'linestyle': '-', 'marker': '^', 'phase_marker': 'D', 'label_code': 'L7Q code', 'label_phase': 'L7Q phase'},
            'L5P': {'color': 'magenta', 'linestyle': '-', 'marker': 'D', 'phase_marker': '^', 'label_code': 'L5P code', 'label_phase': 'L5P phase'},
        }

        plotted = False
        # prepare a palette for distinct series colors (one color per plotted line)
        base_colors = plt.get_cmap('tab20').colors
        ncolors = len(base_colors)
        series_idx = 0

        # collect plotted series for overlap detection
        series_records = []  # each item: {'name': str, 'y': list_or_array, 'line': Line2D}

        for freq, vals in sat.items():
            times = vals.get('times', []) or []
            code_vals = vals.get('code', []) or []
            phase_cycles = vals.get('phase_cycle') or vals.get('phase') or []
            wl_list = vals.get('wavelength', []) or []
            wl = next((w for w in wl_list if w is not None), None)
            if wl is None:
                continue

            epochs = list(range(1, len(times) + 1))
            valid_idx = [i for i in range(len(epochs))
                         if i < len(code_vals) and i < len(phase_cycles)
                         and code_vals[i] is not None and phase_cycles[i] is not None]
            if len(valid_idx) < 2:
                continue

            phase_m_valid = [phase_cycles[i] * wl for i in valid_idx]
            code_valid = [code_vals[i] for i in valid_idx]
            if not code_valid:
                continue

            adjustment_constant = statistics.mean([c - p for c, p in zip(code_valid, phase_m_valid)])
            adjusted_phase = []
            for i in range(len(epochs)):
                if i < len(phase_cycles) and phase_cycles[i] is not None:
                    adjusted_phase.append(phase_cycles[i] * wl + adjustment_constant)
                else:
                    adjusted_phase.append(None)

            style = style_dict.get(freq, {'linestyle': '-', 'label_code': f'{freq} code', 'label_phase': f'{freq} phase'})

            # assign distinct colors: one for code series, one for phase series
            code_color = base_colors[series_idx % ncolors]
            series_idx += 1
            phase_color = base_colors[series_idx % ncolors]
            series_idx += 1

            # plot without markers; use solid for code and dashed for phase
            code_line, = ax.plot(epochs, code_vals, linestyle='-', color=code_color,
                                 label=style.get('label_code', f'{freq} code'), linewidth=1.2)
            phase_line, = ax.plot(epochs, adjusted_phase, linestyle='--', color=phase_color, alpha=0.9,
                                 label=style.get('label_phase', f'{freq} phase'), linewidth=1.0)

            # store for overlap detection (convert None -> np.nan)
            import numpy as _np
            code_arr = _np.array([_np.nan if v is None else v for v in code_vals], dtype=float)
            phase_arr = _np.array([_np.nan if v is None else v for v in adjusted_phase], dtype=float)

            series_records.append({'name': style.get('label_code', f'{freq} code'), 'y': code_arr, 'line': code_line})
            series_records.append({'name': style.get('label_phase', f'{freq} phase'), 'y': phase_arr, 'line': phase_line})

            plotted = True

        # detect overlapping (nearly identical) series and apply small display-space offsets
        if series_records:
            from matplotlib.transforms import ScaledTranslation
            import numpy as _np

            n = len(series_records)
            visited = [False] * n
            clusters = []

            # threshold: either absolute small (1e-3 m) or relative small (1e-9)
            abs_thresh = 1e-3
            rel_thresh = 1e-9

            for i in range(n):
                if visited[i]:
                    continue
                visited[i] = True
                group = [i]
                yi = series_records[i]['y']
                mean_abs_yi = _np.nanmean(_np.abs(yi)) if _np.any(~_np.isnan(yi)) else 0.0
                for j in range(i + 1, n):
                    if visited[j]:
                        continue
                    yj = series_records[j]['y']
                    # indices where both are valid
                    mask = ~_np.isnan(yi) & ~_np.isnan(yj)
                    if not _np.any(mask):
                        continue
                    mad = _np.nanmean(_np.abs(yi[mask] - yj[mask]))
                    mean_abs = (_np.nanmean(_np.abs(yi[mask])) + _np.nanmean(_np.abs(yj[mask]))) / 2.0
                    rel = mad / mean_abs if mean_abs != 0 else mad
                    if mad <= abs_thresh or rel <= rel_thresh:
                        visited[j] = True
                        group.append(j)
                if len(group) > 1:
                    clusters.append(group)

            # Merge legend labels for nearly-identical series (instead of visual offsets)
            for group in clusters:
                labels = [series_records[idx]['name'] for idx in group]
                merged_label = ' / '.join(labels) + ' (identical)'
                rep_idx = group[0]
                # assign merged label to representative line and de-emphasize others
                for idx in group:
                    line = series_records[idx]['line']
                    if idx == rep_idx:
                        # representative shows merged label and slightly thicker line
                        line.set_label(merged_label)
                        try:
                            lw = line.get_linewidth()
                            line.set_linewidth(lw + 0.6)
                        except Exception:
                            pass
                    else:
                        # hide from legend and make slightly transparent
                        line.set_label('')
                        try:
                            line.set_alpha(0.5)
                        except Exception:
                            pass

        if plotted:
            ax.legend(loc='upper right', fontsize='small')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Observation (m)')
        ax.set_title(f'{sat_id} Raw Observations')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save:
            path = self._save_fig(fig, f"raw_observations_{sat_id}", output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_derivatives(self, derivatives: Dict[str, Any], sat_id: str, freq: str, save: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        sat = derivatives.get(sat_id, {})
        if freq not in sat:
            raise ValueError(f"No derivative data for {sat_id} {freq}")
        d = sat[freq]
        times = d.get('times', []) or []
        pr = d.get('pr_derivative', []) or []
        ph = d.get('ph_derivative', []) or []
        dop = d.get('doppler', []) or []

        valid_idx = [i for i in range(len(times))
                     if i < len(pr) and i < len(ph) and i < len(dop)
                     and pr[i] is not None and ph[i] is not None and dop[i] is not None]
        epochs = list(range(1, len(valid_idx) + 1))
        valid_pr = [pr[i] for i in valid_idx]
        valid_ph = [ph[i] for i in valid_idx]
        valid_dop = [dop[i] for i in valid_idx]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        if valid_idx:
            ax1.plot(epochs, valid_pr, 'b-', label='PR derivative (m/s)')
            ax1.plot(epochs, valid_ph, 'g-', label='PH derivative (m/s)')
            ax1.plot(epochs, valid_dop, 'r-', label='Doppler (m/s)')
        ax1.set_title(f"Derivatives {sat_id} {freq}")
        ax1.set_ylabel('Rate (m/s)')
        _h,_l = ax1.get_legend_handles_labels()
        if _h:
            ax1.legend()
        ax1.grid(True)

        if valid_idx:
            dop_minus_pr = [valid_dop[i] - valid_pr[i] for i in range(len(valid_idx))]
            dop_minus_ph = [valid_dop[i] - valid_ph[i] for i in range(len(valid_idx))]
            ax2.plot(epochs, dop_minus_pr, 'm-', label='-λ·D - dP/dt')
            ax2.plot(epochs, dop_minus_ph, 'c-', label='-λ·D - dΦ/dt')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Difference (m/s)')
        _h,_l = ax2.get_legend_handles_labels()
        if _h:
            ax2.legend()
        ax2.grid(True)

        fig.tight_layout()
        if save:
            path = self._save_fig(fig, f"derivatives_{sat_id}_{freq}", output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_code_phase_diff_variation(self, data: Dict[str, Any], sat_id: str, freq: str, save: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        diffs = data.get('code_phase_differences', data)
        sat = diffs.get(sat_id, {})
        if freq not in sat:
            raise ValueError(f"No diff data for {sat_id} {freq}")
        d = sat[freq]
        changes_raw = d.get('diff_changes', []) or []
        changes = [c for c in changes_raw if c is not None]
        epochs = list(range(1, len(changes) + 1))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, changes, marker='o')
        ax.set_title(f"Code-Phase Diff Changes {sat_id} {freq}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Change (m)')
        ax.grid(True)
        if save:
            path = self._save_fig(fig, f"diff_variation_{sat_id}_{freq}", output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_code_phase_raw_diff(self, data: Dict[str, Any], sat_id: str, freq: str, save: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        diffs = data.get('code_phase_differences', data)
        sat = diffs.get(sat_id, {})
        if freq not in sat:
            raise ValueError(f"No diff data for {sat_id} {freq}")
        d = sat[freq]
        values_raw = d.get('code_phase_diff', []) or []
        values = [v for v in values_raw if v is not None]
        epochs = list(range(1, len(values) + 1))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, values, marker='.')
        ax.set_title(f"Code-Phase Raw Diff {sat_id} {freq}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Code-Phase (m)')
        ax.grid(True)
        if save:
            path = self._save_fig(fig, f"raw_diff_{sat_id}_{freq}", output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_prediction_errors(self, errors: Dict[str, Any], sat_id: str, freq: str, save: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        sat = errors.get(sat_id, {})
        if freq not in sat:
            raise ValueError(f"No prediction error data for {sat_id} {freq}")
        d = sat[freq]
        errs_raw = d.get('prediction_error', []) or []
        errs = [e for e in errs_raw if e is not None]
        epochs = list(range(1, len(errs) + 1))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, errs, marker='o')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_title(f"Prediction Errors {sat_id} {freq}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error (m)')
        ax.grid(True)
        if save:
            path = self._save_fig(fig, f"prediction_errors_{sat_id}_{freq}", output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_epoch_double_diffs(self, data: Dict[str, Any], sat_id: str, freq: str, save: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        dd = data.get('epoch_double_diffs', data)
        sat = dd.get(sat_id, {})
        if freq not in sat:
            raise ValueError(f"No epoch double diff data for {sat_id} {freq}")
        d = sat[freq]
        code = d.get('dd_code', []) or []
        phase = d.get('dd_phase', []) or []
        dop = d.get('dd_doppler', []) or []
        epochs = list(range(1, len(code) + 1))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        if epochs:
            ax1.plot(epochs, code, 'b-', label='DD code')
            ax2.plot(epochs, phase, 'g-', label='DD phase')
            ax3.plot(epochs, dop, 'm-', label='DD doppler')
        ax1.set_title(f"Epoch Double Differences {sat_id} {freq}")
        ax1.set_ylabel('DD code (m)')
        ax2.set_ylabel('DD phase (m)')
        ax3.set_ylabel('DD doppler (m/s)')
        ax3.set_xlabel('Epoch')
        for a in (ax1, ax2, ax3):
            a.grid(True)
            _h,_l = a.get_legend_handles_labels()
            if _h:
                a.legend(loc='upper right')

        fig.tight_layout()
        if save:
            path = self._save_fig(fig, f"dd_{sat_id}_{freq}", output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_isb_analysis(self, isb_results: Dict[str, Any], save: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        # Simple ISB summary plot (histogram of ISB estimates)
        estimates = isb_results.get('isb_estimates', [])
        fig, ax = plt.subplots(figsize=(8, 4))
        if estimates:
            ax.hist(estimates, bins=20)
        ax.set_title('ISB Estimates')
        ax.set_xlabel('ISB (m)')
        ax.grid(True)
        if save:
            path = self._save_fig(fig, 'isb_estimates', output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_receiver_cmc(self, cmc_results: Dict[str, Any], save: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        # plot CMC time series for each sat/freq as small multiples
        fig, axes = plt.subplots(len(cmc_results) or 1, 1, figsize=(10, 3 * max(1, len(cmc_results))))
        if not hasattr(axes, '__iter__'):
            axes = [axes]
        for ax, (sat, freqs) in zip(axes, cmc_results.items()):
            for freq, info in freqs.items():
                ax.plot(info.get('times', []), info.get('cmc_m', []), label=freq)
            ax.set_title(f"CMC {sat}")
            ax.legend()
            ax.grid(True)
        fig.autofmt_xdate()
        if save:
            path = self._save_fig(fig, 'receiver_cmc', output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def save_all_plots(self, figures: Sequence[Figure], output_dir: str) -> Dict[str, str]:
        results = {}
        out_dir = self._ensure_output_dir(output_dir)
        for i, fig in enumerate(figures):
            name = f"figure_{i}"
            path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(path, bbox_inches='tight')
            results[name] = path
            plt.close(fig)
        return results
    
    def plot_cycle_slip_analysis(self, detection_result: Dict[str, Any], sat_id: str, 
                                 save: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        绘制周跳探测分析图（MW + GF）
        
        Args:
            detection_result: 单颗卫星的周跳探测结果
            sat_id: 卫星ID
            save: 是否保存图片
            output_dir: 保存目录
            
        Returns:
            包含figure和path的字典
        """
        mw_data = detection_result.get('mw', {})
        gf_data = detection_result.get('gf', {})
        freq_pair = detection_result.get('freq_pair', ('?', '?'))
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'周跳探测 - {sat_id} ({freq_pair[0]}/{freq_pair[1]})', fontsize=14, fontweight='bold')
        
        # 子图(a): MW检验
        if 'delta_mw' in mw_data and 'epochs' in mw_data:
            epochs = mw_data['epochs']
            delta_mw = mw_data['delta_mw']
            threshold = mw_data.get('threshold_history', [])
            
            # 绘制MW差异
            ax1.plot(epochs, delta_mw, 'b-', linewidth=1.5, label='|Nw(i) - mean(Nw)|')
            
            # 绘制阈值线
            mw_mode = mw_data.get('threshold_mode', 'dynamic')
            # 自定义阈值：绘制一条水平线并显示具体数值
            if mw_mode == 'custom':
                thr_val = mw_data.get('threshold_value')
                if thr_val is None and threshold:
                    # 兜底取第二项或第一项
                    try:
                        thr_val = float(threshold[1]) if len(threshold) > 1 else float(threshold[0])
                    except Exception:
                        thr_val = None
                if thr_val is not None:
                    ax1.axhline(y=thr_val, color='r', linestyle='--', linewidth=1.2, label=f'阈值 ({thr_val:.2f} m)')
            else:
                # 动态阈值：绘制随历元变化的阈值曲线
                if isinstance(threshold, (list, tuple)) and len(threshold) == len(epochs):
                    # 防止初始阈值为0导致图像左侧垂直连线，若首元素为0则用第二元素替代显示
                    thr_plot = list(threshold)
                    if len(thr_plot) > 1 and thr_plot[0] == 0:
                        thr_plot[0] = thr_plot[1]
                    ax1.plot(epochs, thr_plot, 'r--', linewidth=1.2, label=f'{mw_data.get("threshold_sigma", 4)}σ阈值')
                else:
                    try:
                        thr_val = float(threshold)
                        ax1.axhline(y=thr_val, color='r', linestyle='--', linewidth=1.2, label=f'{mw_data.get("threshold_sigma", 4)}σ阈值')
                    except Exception:
                        pass
            
            # 标记周跳点
            cycle_slips = mw_data.get('cycle_slips', [])
            if cycle_slips:
                slip_epochs = [cs['epoch'] for cs in cycle_slips]
                slip_deltas = [cs['delta'] for cs in cycle_slips]
                ax1.scatter(slip_epochs, slip_deltas, c='red', marker='x', s=100, 
                           linewidths=2, label='周跳', zorder=5)
            
            # 标记粗差点
            outliers = mw_data.get('outliers', [])
            if outliers:
                outlier_epochs = [o['epoch'] for o in outliers]
                outlier_deltas = [o['delta'] for o in outliers]
                ax1.scatter(outlier_epochs, outlier_deltas, c='orange', marker='s', s=80,
                           linewidths=2, label='粗差', zorder=5)
            
            ax1.set_xlabel('历元 (Epoch)', fontsize=11)
            ax1.set_ylabel('ΔMW / m', fontsize=11)
            ax1.set_title('(a) MW检验', fontsize=12, loc='left')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best')
        else:
            ax1.text(0.5, 0.5, 'MW数据不可用', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('(a) MW检验', fontsize=12, loc='left')
        
        # 子图(b): GF检验
        if 'delta_gf' in gf_data and 'epochs' in gf_data:
            epochs = gf_data['epochs']
            delta_gf = gf_data['delta_gf']
            threshold = gf_data.get('threshold', 0.4)
            
            # 绘制GF差异
            ax2.plot(epochs, delta_gf, 'b-', linewidth=1.5, label='|GF(i) - GF(i-1)|')
            
            # 绘制阈值线
            ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=1.2, 
                       label=f'阈值 ({threshold:.2f}m)')
            
            # 标记周跳点
            cycle_slips = gf_data.get('cycle_slips', [])
            if cycle_slips:
                slip_epochs = [cs['epoch'] for cs in cycle_slips]
                slip_deltas = [cs['delta'] for cs in cycle_slips]
                ax2.scatter(slip_epochs, slip_deltas, c='red', marker='x', s=100,
                           linewidths=2, label='周跳', zorder=5)
            
            ax2.set_xlabel('历元 (Epoch)', fontsize=11)
            ax2.set_ylabel('ΔGF / m', fontsize=11)
            ax2.set_title('(b) GF检验', fontsize=12, loc='left')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best')
        else:
            ax2.text(0.5, 0.5, 'GF数据不可用', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('(b) GF检验', fontsize=12, loc='left')
        
        plt.tight_layout()
        
        if save:
            path = self._save_fig(fig, f'cycle_slip_{sat_id}', output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

