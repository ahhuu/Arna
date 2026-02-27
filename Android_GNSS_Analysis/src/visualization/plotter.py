from typing import Dict, Any, Optional, Sequence
import os
import statistics
import matplotlib
# Use Agg backend for headless environments (tests/CLI). GUI can set a different backend on startup.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# 尝试导入mplcursors用于交互式数据点显示
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False

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
        # 同一频率内伪距和相位使用区别明显的marker（实心 vs 空心，填充 vs 线框）
        style_dict = {
            'L1C': {'color': 'blue', 'linestyle': '-', 'marker': 's', 'phase_marker': 'o', 'label_code': 'L1C code', 'label_phase': 'L1C phase'},
            'L1D': {'color': 'cyan', 'linestyle': '-', 'marker': 'D', 'phase_marker': '^', 'label_code': 'L1D code', 'label_phase': 'L1D phase'},
            'L1P': {'color': 'red', 'linestyle': '-', 'marker': 's', 'phase_marker': 'o', 'label_code': 'L1P code', 'label_phase': 'L1P phase'},
            'L2I': {'color': 'blue', 'linestyle': '-', 'marker': 'D', 'phase_marker': '^', 'label_code': 'L2I code', 'label_phase': 'L2I phase'},
            'L5Q': {'color': 'green', 'linestyle': '-', 'marker': 's', 'phase_marker': 'o', 'label_code': 'L5Q code', 'label_phase': 'L5Q phase'},
            'L7Q': {'color': 'magenta', 'linestyle': '-', 'marker': 'D', 'phase_marker': '^', 'label_code': 'L7Q code', 'label_phase': 'L7Q phase'},
            'L5P': {'color': 'orange', 'linestyle': '-', 'marker': 's', 'phase_marker': 'o', 'label_code': 'L5P code', 'label_phase': 'L5P phase'},
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

            style = style_dict.get(freq, {'linestyle': '-', 'marker': 'o', 'phase_marker': 's', 'label_code': f'{freq} code', 'label_phase': f'{freq} phase'})

            # assign distinct colors: one for code series, one for phase series
            code_color = base_colors[series_idx % ncolors]
            series_idx += 1
            phase_color = base_colors[series_idx % ncolors]
            series_idx += 1

            # plot with different markers for each frequency to distinguish them clearly
            # 使用markevery间隔显示标记，markersize增大，同频率内code和phase使用区别明显的marker
            code_marker = style.get('marker', 's')
            phase_marker = style.get('phase_marker', 'o')
            
            # 计算标记间隔：数据点少于30个时每5个标记一次，否则每10个标记一次
            marker_interval = 5 if len(epochs) < 30 else max(10, len(epochs) // 30)
            
            code_line, = ax.plot(epochs, code_vals, linestyle='-', marker=code_marker, markersize=8, 
                                 markevery=marker_interval, markerfacecolor=code_color, markeredgecolor=code_color,
                                 color=code_color, label=style.get('label_code', f'{freq} code'), linewidth=1.5)
            phase_line, = ax.plot(epochs, adjusted_phase, linestyle='--', marker=phase_marker, markersize=8,
                                  markevery=marker_interval, markerfacecolor='white', markeredgecolor=phase_color, markeredgewidth=1.5,
                                  color=phase_color, alpha=0.9, label=style.get('label_phase', f'{freq} phase'), linewidth=1.2)

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

        # 添加交互式数据点显示（点击数据点后用方向键切换）
        if MPLCURSORS_AVAILABLE and not save:
            cursor = mplcursors.cursor(ax)
            cursor.connect("add", lambda sel: sel.annotation.set(
                text=f'Epoch: {sel.target[0]:.0f}\nValue: {sel.target[1]:.3f} m\n\n点击后按←/→键\n(图表窗口需有焦点)',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)))

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
        # derivatives的计算是从第1个历元开始（i=1），所以valid_idx[j]对应的是原始数据的第valid_idx[j]+1个历元
        # 但derivatives本身的索引就是从1开始的（range(1, len(times))），所以直接使用valid_idx+1即可
        epochs = [idx + 1 for idx in valid_idx]  # +1转换为1-based历元号
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

        # 添加交互式数据点显示（点击数据点后用方向键切换）
        if MPLCURSORS_AVAILABLE and not save:
            cursor = mplcursors.cursor([ax1, ax2])
            cursor.connect("add", lambda sel: sel.annotation.set(
                text=f'Epoch: {sel.target[0]:.0f}\nValue: {sel.target[1]:.6f} m/s\n\n点击后按←/→键\n(图表窗口需有焦点)',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)))

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
        epoch_indices_raw = d.get('epoch_indices', []) or []
        
        # 过滤掉 None 值，并保持索引对应关系
        changes = []
        epoch_indices = []
        for i, c in enumerate(changes_raw):
            if c is not None:
                changes.append(c)
                # 使用真实的历元索引（1-based，与RINEX文件一致）
                if i < len(epoch_indices_raw):
                    epoch_indices.append(epoch_indices_raw[i] + 1)  # +1 转换为1-based
                else:
                    epoch_indices.append(i + 1)  # fallback到相对索引
        
        # 如果没有epoch_indices数据（旧数据格式），使用相对编号
        if not epoch_indices:
            epoch_indices = list(range(1, len(changes) + 1))
        
        epochs = epoch_indices

        fig, ax = plt.subplots(figsize=(10, 4))
        line, = ax.plot(epochs, changes, marker='o', markersize=4, linestyle='-', linewidth=1)
        ax.set_title(f"Code-Phase Diff Changes {sat_id} {freq}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Change (m)')
        ax.grid(True)

        # 添加交互式数据点显示（只在数据点上点击，然后用方向键切换）
        if MPLCURSORS_AVAILABLE and not save:
            # 创建一个不可见的scatter来确保只响应数据点，但有足够的点击区域
            scatter = ax.scatter(epochs, changes, s=50, alpha=0, picker=True)
            cursor = mplcursors.cursor(scatter)
            cursor.connect("add", lambda sel: sel.annotation.set(
                text=f'Epoch: {sel.target[0]:.0f}\nChange: {sel.target[1]:.3f} m\n\n点击后按←/→键\n(图表窗口需有焦点)',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)))

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
        epoch_indices_raw = d.get('epoch_indices', []) or []
        
        # 过滤掉None值，并保持索引对应关系
        values = []
        epoch_indices = []
        for i, v in enumerate(values_raw):
            if v is not None:
                values.append(v)
                if i < len(epoch_indices_raw):
                    epoch_indices.append(epoch_indices_raw[i] + 1)  # +1转换为1-based
                else:
                    epoch_indices.append(i + 1)  # fallback
        
        if not epoch_indices:
            epoch_indices = list(range(1, len(values) + 1))
        
        epochs = epoch_indices

        fig, ax = plt.subplots(figsize=(10, 4))
        line, = ax.plot(epochs, values, marker='o', markersize=3, linestyle='-', linewidth=1)
        ax.set_title(f"Code-Phase Raw Diff {sat_id} {freq}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Code-Phase (m)')
        ax.grid(True)

        # 添加交互式数据点显示（只在数据点上点击，然后用方向键切换）
        if MPLCURSORS_AVAILABLE and not save:
            scatter = ax.scatter(epochs, values, s=50, alpha=0, picker=True)
            cursor = mplcursors.cursor(scatter)
            cursor.connect("add", lambda sel: sel.annotation.set(
                text=f'Epoch: {sel.target[0]:.0f}\nCode-Phase: {sel.target[1]:.3f} m\n\n点击后按←/→键\n(图表窗口需有焦点)',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)))

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
        epoch_indices_raw = d.get('epoch_indices', []) or []
        
        # 过滤掉None值，并保持索引对应关系
        errs = []
        epoch_indices = []
        for i, e in enumerate(errs_raw):
            if e is not None:
                errs.append(e)
                if i < len(epoch_indices_raw):
                    epoch_indices.append(epoch_indices_raw[i] + 1)  # +1转换为1-based
                else:
                    epoch_indices.append(i + 1)  # fallback
        
        if not epoch_indices:
            epoch_indices = list(range(1, len(errs) + 1))
        
        epochs = epoch_indices

        fig, ax = plt.subplots(figsize=(10, 4))
        line, = ax.plot(epochs, errs, linestyle='-', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_title(f"Prediction Errors {sat_id} {freq}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error (m)')
        ax.grid(True)

        # 添加交互式数据点显示（只在数据点上点击，然后用方向键切换）
        if MPLCURSORS_AVAILABLE and not save:
            scatter = ax.scatter(epochs, errs, s=50, alpha=0, picker=True)
            cursor = mplcursors.cursor(scatter)
            cursor.connect("add", lambda sel: sel.annotation.set(
                text=f'Epoch: {sel.target[0]:.0f}\nError: {sel.target[1]:.6f} m\n\n点击后按←/→键\n(图表窗口需有焦点)',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)))

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
        epoch_indices_raw = d.get('epoch_indices', []) or []
        
        # 使用真实历元索引（epoch_indices记录的是双差对应的历元）
        if epoch_indices_raw:
            epochs = [idx + 1 for idx in epoch_indices_raw]  # +1转换为1-based
        else:
            epochs = list(range(1, len(code) + 1))  # fallback到相对编号

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

        # 添加交互式数据点显示（点击数据点后用方向键切换）
        if MPLCURSORS_AVAILABLE and not save:
            cursor = mplcursors.cursor([ax1, ax2, ax3])
            def on_add(sel):
                x, y = sel.target
                # 根据子图类型显示不同的单位
                if sel.artist.axes == ax3:
                    unit = 'm/s'
                else:
                    unit = 'm'
                sel.annotation.set(text=f'Epoch: {x:.0f}\nValue: {y:.6f} {unit}\n\n点击后按←/→键\n(图表窗口需有焦点)',
                                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8))
            cursor.connect("add", on_add)

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

        # 添加交互式数据点显示（点击数据点后用方向键切换）
        if MPLCURSORS_AVAILABLE and not save:
            cursor = mplcursors.cursor(axes)
            cursor.connect("add", lambda sel: sel.annotation.set(
                text=f'Value: {sel.target[1]:.3f} m\n\n点击后按←/→键\n(图表窗口需有焦点)',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)))

        if save:
            path = self._save_fig(fig, 'receiver_cmc', output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_ionofree_cmc(self, ionofree_results: Dict[str, Any], sat_id: str = None,
                          save: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """绘制无电离层组合CMC时间序列图.

        若指定 sat_id，则只绘制该颗卫星；否则绘制所有卫星（每卫星一子图）。
        返回 {'figure': fig|None, 'path': path|None, 'paths': [...]}
        """
        if sat_id is not None:
            # 单颗卫星绘制
            sat_data = ionofree_results.get(sat_id)
            if not sat_data:
                raise ValueError(f"无电离层组合CMC中无卫星 {sat_id} 的数据")
            return self._plot_single_ionofree_cmc(sat_id, sat_data, save, output_dir)
        else:
            # 所有卫星分别绘制
            paths = []
            last_fig = None
            for sid in sorted(ionofree_results.keys()):
                out = self._plot_single_ionofree_cmc(sid, ionofree_results[sid], save, output_dir)
                if out.get('path'):
                    paths.append(out['path'])
                if out.get('figure'):
                    last_fig = out['figure']
            return {'figure': last_fig, 'path': None, 'paths': paths}

    def _plot_single_ionofree_cmc(self, result_key: str, sat_data: Dict[str, Any],
                                   save: bool, output_dir: Optional[str]) -> Dict[str, Any]:
        """绘制单颗卫星的无电离层组合CMC图."""
        times = sat_data.get('times', [])
        cmc_if = sat_data.get('cmc_if', [])
        freq_pair = sat_data.get('freq_pair', ('?', '?'))
        real_sat_id = sat_data.get('sat_id', result_key)
        alpha = sat_data.get('alpha', 0)
        beta = sat_data.get('beta', 0)
        noise_factor = sat_data.get('noise_factor', 0)

        if not cmc_if:
            raise ValueError(f"卫星 {real_sat_id} 的无电离层组合CMC数据为空")

        # 生成历元索引 (1-based)
        epochs = list(range(1, len(cmc_if) + 1))

        fig, ax = plt.subplots(figsize=(12, 5))

        line, = ax.plot(epochs, cmc_if, marker='o', markersize=2, linestyle='-',
                        linewidth=0.8, color='#2196F3', alpha=0.8)

        # 标题和标签
        f1_name, f2_name = freq_pair
        ax.set_title(f"Ionosphere-Free CMC  {real_sat_id}  ({f1_name} + {f2_name})", fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('CMC_IF (m)', fontsize=11)
        ax.grid(True, alpha=0.3)

        # 统计信息
        import statistics as _stats
        mean_val = _stats.mean(cmc_if)
        std_val = _stats.stdev(cmc_if) if len(cmc_if) > 1 else 0.0
        min_val = min(cmc_if)
        max_val = max(cmc_if)

        stats_text = (
            f"α = {alpha:.4f},  β = {beta:.4f}\n"
            f"噪声放大: ×{noise_factor:.2f}\n"
            f"均值: {mean_val:.4f} m\n"
            f"标准差: {std_val:.4f} m\n"
            f"范围: [{min_val:.4f}, {max_val:.4f}] m\n"
            f"历元数: {len(cmc_if)}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # 均值参考线
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label=f'Mean={mean_val:.4f} m')
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()

        # 交互式数据点显示
        if MPLCURSORS_AVAILABLE and not save:
            scatter = ax.scatter(epochs, cmc_if, s=50, alpha=0, picker=True)
            cursor = mplcursors.cursor(scatter)
            cursor.connect("add", lambda sel: sel.annotation.set(
                text=f'Epoch: {sel.target[0]:.0f}\nCMC_IF: {sel.target[1]:.4f} m\n\n点击后按←/→键',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)))

        if save:
            safe_key = result_key.replace(':', '_')
            path = self._save_fig(fig, f"ionofree_cmc_{safe_key}", output_dir)
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
            
            # 绘制MW差异（带标记点）
            ax1.plot(epochs, delta_mw, 'b-', marker='o', markersize=3,
                    linewidth=1.5, label='|Nw(i) - mean(Nw)|')
            
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
            
            # 绘制GF差异（带标记点）
            ax2.plot(epochs, delta_gf, 'b-', marker='o', markersize=3,
                    linewidth=1.5, label='|GF(i) - GF(i-1)|')
            
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

        # 添加交互式数据点显示（点击数据点后用方向键切换）
        if MPLCURSORS_AVAILABLE and not save:
            cursor = mplcursors.cursor([ax1, ax2])
            def on_add(sel):
                x, y = sel.target
                if sel.artist.axes == ax1:
                    label_text = 'ΔMW'
                else:
                    label_text = 'ΔGF'
                sel.annotation.set(text=f'历元: {x:.0f}\n{label_text}: {y:.3f} m\n\n点击后按←/→键\n(图表窗口需有焦点)',
                                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8))
            cursor.connect("add", on_add)
        
        if save:
            path = self._save_fig(fig, f'cycle_slip_{sat_id}', output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_inter_freq_bias(self, analysis_result: Dict[str, Any], 
                            save: bool = True, 
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        绘制伪距频间偏差分析图：上下两个子图
        - 上图：原始频间差值（Raw Inter-Frequency Difference）
        - 下图：ISD处理后的频间差值（After Inter-Satellite Single Difference）
        
        参数:
            analysis_result: InterFrequencyBiasAnalyzer.analyze_inter_freq_bias 的返回结果
            save: 是否保存图片
            output_dir: 输出目录
        """
        import numpy as np
        
        raw_diffs = analysis_result.get('raw_diffs', {})
        isd_diffs = analysis_result.get('isd_diffs', {})
        freq_pair = analysis_result.get('freq_pair', ('', ''))
        constellation = analysis_result.get('constellation', 'All')
        
        if not raw_diffs:
            raise ValueError("未找到频间差数据")
        
        # 创建图形：两个子图，共享X轴
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 颜色映射（为每颗卫星分配不同颜色）
        satellites = sorted(raw_diffs.keys())
        colors = plt.cm.tab20(np.linspace(0, 1, len(satellites)))
        color_map = dict(zip(satellites, colors))
        
        # ========== 上图：原始频间差 ==========
        for sat_id, sat_data in raw_diffs.items():
            times = sat_data['times']
            diffs = sat_data['diff']
            
            # 转换为相对秒数（从第一个历元开始）
            if times:
                start_time = times[0]
                relative_seconds = [(t - start_time).total_seconds() for t in times]
                
                # 绘制散点
                ax1.scatter(relative_seconds, diffs, 
                           c=[color_map[sat_id]], 
                           s=15, alpha=0.6, label=sat_id)
        
        ax1.set_ylabel('频间差 (m)', fontsize=12)
        ax1.set_title(f'(a) 原始伪距频间差 {freq_pair[0]}-{freq_pair[1]} (星座: {constellation or "All"})', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # 图例（放在右侧，避免遮挡数据）
        ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), 
                  fontsize=8, ncol=1, framealpha=0.9)
        
        # ========== 下图：ISD处理后的频间差 ==========
        for sat_id, sat_data in isd_diffs.items():
            times = sat_data['times']
            isd_diff_values = sat_data['isd_diff']
            
            # 过滤有效值
            valid_data = [(t, d) for t, d in zip(times, isd_diff_values) if d is not None]
            if not valid_data:
                continue
            
            valid_times, valid_diffs = zip(*valid_data)
            
            # 转换为相对秒数
            if valid_times:
                start_time = valid_times[0]
                relative_seconds = [(t - start_time).total_seconds() for t in valid_times]
                
                # 绘制散点
                ax2.scatter(relative_seconds, valid_diffs,
                           c=[color_map[sat_id]],
                           s=15, alpha=0.6, label=sat_id)
        
        ax2.set_xlabel('相对时间 (秒)', fontsize=12)
        ax2.set_ylabel('频间差 (m)', fontsize=12)
        ax2.set_title(f'(b) 星间单差(ISD)后频间差 {freq_pair[0]}-{freq_pair[1]}', 
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # 添加 ±10m 参考线（期望值范围）
        ax2.axhline(y=10, color='red', linestyle=':', linewidth=0.8, alpha=0.5, label='±10m')
        ax2.axhline(y=-10, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
        
        ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
                  fontsize=8, ncol=1, framealpha=0.9)
        
        # 添加统计信息文本框
        try:
            from src.processing.inter_freq_bias import InterFrequencyBiasAnalyzer
            analyzer = InterFrequencyBiasAnalyzer()
            stats = analyzer.get_statistics(analysis_result)
            
            raw_stats = stats.get('raw_stats')
            isd_stats = stats.get('isd_stats')
            improvement = stats.get('improvement')
            
            stats_text = ""
            if raw_stats:
                stats_text += f"原始差值 RMS: {raw_stats['rms']:.3f} m\n"
                stats_text += f"原始差值 STD: {raw_stats['std']:.3f} m\n"
            if isd_stats:
                stats_text += f"ISD后 RMS: {isd_stats['rms']:.3f} m\n"
                stats_text += f"ISD后 STD: {isd_stats['std']:.3f} m\n"
            if improvement is not None:
                stats_text += f"改善率: {improvement:.1f}%"
            
            if stats_text:
                ax2.text(0.02, 0.98, stats_text,
                        transform=ax2.transAxes,
                        fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        except Exception:
            pass  # 如果统计计算失败，不影响绘图
        
        plt.tight_layout()

        # 添加交互式数据点显示（点击数据点后用方向键切换）
        if MPLCURSORS_AVAILABLE and not save:
            cursor = mplcursors.cursor([ax1, ax2])
            cursor.connect("add", lambda sel: sel.annotation.set(
                text=f'时间: {sel.target[0]:.0f}s\n频间差: {sel.target[1]:.3f} m\n\n点击后按←/→键\n(图表窗口需有焦点)',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)))
        
        if save:
            suffix = f"_{constellation}" if constellation else ""
            path = self._save_fig(fig, f'inter_freq_bias_{freq_pair[0]}_{freq_pair[1]}{suffix}', output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

