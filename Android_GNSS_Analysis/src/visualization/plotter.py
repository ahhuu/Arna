from typing import Dict, Any, Optional, Sequence
import os
import math
import statistics
import datetime
import matplotlib
# Use Agg backend for headless environments (tests/CLI). GUI can set a different backend on startup.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

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

    def _satellite_sort_key(self, sat_id: str):
        system_order = {'G': 0, 'R': 1, 'E': 2, 'C': 3, 'J': 4, 'I': 5}
        if not sat_id:
            return (99, '', 0, sat_id)
        system = sat_id[0]
        prn_text = sat_id[1:]
        try:
            prn_value = int(prn_text)
        except Exception:
            prn_value = 0
        return (system_order.get(system, 99), system, prn_value, sat_id)

    def _has_valid_observation(self, freq_values: Dict[str, Any], index: int) -> bool:
        fields = ('code', 'phase_cycle', 'phase')
        for field in fields:
            values = freq_values.get(field, []) or []
            if index < len(values) and values[index] is not None:
                return True
        return False

    def _frequency_sequence_color(self, sat_id: str, freq: str, color_map: Dict[str, Any]):
        system = sat_id[0] if sat_id else ''
        freq_key = freq.upper()

        if system == 'C':
            if freq_key in {'L1P', 'L1D'}:
                return color_map['L1C']
            if freq_key == 'L2I':
                return color_map['L2I']
            if freq_key == 'L5P':
                return color_map['L5Q']
            if freq_key == 'L7Q':
                return color_map['L7Q']

        if freq_key in {'L1P', 'L1D'}:
            return color_map['L1C']
        if freq_key == 'L5P':
            return color_map['L5Q']
        if freq_key not in color_map:
            return color_map['default']
        return color_map[freq_key]

    def _frequency_sequence_legend_label(self, sat_id: str, freq: str) -> str:
        system_prefix_map = {
            'C': 'B',
        }
        system = sat_id[0] if sat_id else ''
        display_system = system_prefix_map.get(system, system or '?')
        return f'{display_system}: {freq}'

    def _normalize_frequency_filters(self, values):
        if not values:
            return set()
        if isinstance(values, str):
            return {values} if values else set()
        return {str(value) for value in values if value}

    def _normalize_sat_freq_filters(self, values):
        if not values:
            return set()
        normalized = set()
        for value in values:
            if not value:
                continue
            if isinstance(value, tuple) and len(value) == 2:
                normalized.add((str(value[0]), str(value[1])))
            elif isinstance(value, list) and len(value) == 2:
                normalized.add((str(value[0]), str(value[1])))
            elif isinstance(value, str) and '|' in value:
                sat_id, freq = value.split('|', 1)
                normalized.add((sat_id, freq))
        return normalized

    def _selection_allows(self, selection, value: str) -> bool:
        if selection is None:
            return True
        normalized = self._normalize_frequency_filters(selection)
        return value in normalized

    def _frequency_sequence_accepts(self, sat_id: str, freq: str,
                                    system_filters=None,
                                    sat_filters=None,
                                    freq_filters=None) -> bool:
        if not self._selection_allows(system_filters, sat_id[0] if sat_id else ''):
            return False
        if not self._selection_allows(sat_filters, sat_id):
            return False
        if not self._selection_allows(freq_filters, freq):
            return False
        return True

    def _satellite_count_color(self, system: str) -> str:
        colors = {
            'AVE': 'tab:red',
            'G': 'tab:orange',
            'C': 'tab:green',
            'R': 'tab:blue',
            'E': 'tab:purple',
            'J': 'tab:brown',
            'I': 'tab:cyan',
            'B': 'tab:red'
        }
        return colors.get(system.upper() if system else '', 'tab:gray')

    def _stat_label(self, prefix: str, values: Sequence[int]) -> str:
        if not values:
            return prefix
        avg = sum(values) / len(values)
        return f'{prefix} AVE: {avg:.1f} MIN: {min(values)} MAX: {max(values)}'

    def plot_satellite_count(self, observations: Dict[str, Any], save: bool = True,
                             output_dir: Optional[str] = None,
                             system_filters=None,
                             sat_filters=None,
                             freq_filters=None) -> Dict[str, Any]:
        """Plot the number of visible satellites over GPST time, grouped by GNSS system."""
        obs = observations.get('observations_meters', observations) if isinstance(observations, dict) else observations
        if not obs:
            raise ValueError('No observations available for satellite count plot')

        time_sat_map = {}
        systems = set()
        for sat_id, freq_map in obs.items():
            if not sat_id:
                continue
            if not self._selection_allows(system_filters, sat_id[0]) or not self._selection_allows(sat_filters, sat_id):
                continue
            for freq, values in (freq_map or {}).items():
                if not self._frequency_sequence_accepts(sat_id, freq, system_filters, sat_filters, freq_filters):
                    continue
                times = values.get('times', []) or []
                for idx, t in enumerate(times):
                    if self._has_valid_observation(values, idx):
                        time_sat_map.setdefault(t, set()).add(sat_id)
                        systems.add(sat_id[0])

        if not time_sat_map:
            raise ValueError('No valid satellite observations found for count plot')

        sorted_times = sorted(time_sat_map.keys())
        total_counts = [len(time_sat_map[t]) for t in sorted_times]
        system_counts = {
            system: [sum(1 for sat in time_sat_map[t] if sat.startswith(system)) for t in sorted_times]
            for system in sorted(systems)
        }

        max_count = max(total_counts + [max(vals) for vals in system_counts.values()] if system_counts else total_counts)
        tick_max = max(10, int(math.ceil(max_count / 10.0) * 10))
        y_ticks = list(range(0, tick_max + 1, 10))

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(sorted_times, total_counts, color=self._satellite_count_color('AVE'), linewidth=2.0, alpha=0.9, label=self._stat_label('AVE', total_counts))

        legend_items = [Line2D([0], [0], color=self._satellite_count_color('AVE'), linewidth=2.0, label=self._stat_label('AVE', total_counts))]
        for system, counts in system_counts.items():
            if not any(counts):
                continue
            label = self._stat_label(system, counts)
            color = self._satellite_count_color(system)
            ax.plot(sorted_times, counts, color=color, linewidth=1.6, alpha=0.8, label=label)
            legend_items.append(Line2D([0], [0], color=color, linewidth=1.6, label=label))

        ax.set_xlabel('')
        ax.set_ylabel('Satellite Count')
        ax.set_title('Number Of Sat', pad=8)
        ax.grid(True, axis='x', alpha=0.25)
        ax.grid(True, axis='y', alpha=0.15)

        if len(sorted_times) == 1:
            x_min = sorted_times[0] - datetime.timedelta(minutes=1)
            x_max = sorted_times[0] + datetime.timedelta(minutes=1)
            ax.set_xlim(x_min, x_max)
        else:
            ax.set_xlim(sorted_times[0], sorted_times[-1])
        ax.set_ylim(0, tick_max)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(y) for y in y_ticks])
        ax.margins(x=0.02, y=0.04)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=0, ha='center')

        legend = ax.legend(handles=legend_items, loc='center right', bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=plt.rcParams.get('legend.fontsize', 'small'), borderpad=0.4, fancybox=True, framealpha=0.9)
        legend.set_draggable(True)
        fig.subplots_adjust(left=0.08, right=0.93, top=0.96, bottom=0.08)

        if save:
            path = self._save_fig(fig, 'satellite_count', output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_satellite_frequency_sequence(self, observations: Dict[str, Any], save: bool = True,
                                          output_dir: Optional[str] = None,
                                          system_filters=None,
                                          sat_filters=None,
                                          freq_filters=None) -> Dict[str, Any]:
        """Plot satellite PRN on the y-axis and GPST on the x-axis for all observed frequencies.

        Only frequencies that contain at least one valid observation in the current RINEX data are drawn.
        """
        obs = observations.get('observations_meters', observations) if isinstance(observations, dict) else observations
        if not obs:
            raise ValueError('No observations available for satellite frequency sequence plot')

        satellite_ids = sorted(obs.keys(), key=self._satellite_sort_key)
        visible_satellite_ids = []
        freq_names = []
        for sat_id in satellite_ids:
            sat_has_visible_data = False
            for freq, values in (obs.get(sat_id) or {}).items():
                times = values.get('times', []) or []
                if self._frequency_sequence_accepts(sat_id, freq, system_filters, sat_filters, freq_filters) and any(self._has_valid_observation(values, idx) for idx in range(len(times))):
                    freq_names.append(freq)
                    sat_has_visible_data = True
            if sat_has_visible_data:
                visible_satellite_ids.append(sat_id)

        if not freq_names:
            raise ValueError('No valid frequency observations found in current RINEX data')

        unique_freqs = sorted(set(freq_names))
        cmap = plt.get_cmap('tab20')
        color_map = {
            'L1C': cmap(0),
            'L2I': cmap(2),
            'L5Q': cmap(4),
            'L7Q': cmap(6),
            'default': cmap(8),
        }

        fig_height = max(4.8, 0.24 * len(satellite_ids) + 1.4)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        y_positions = {sat_id: idx for idx, sat_id in enumerate(visible_satellite_ids)}
        all_valid_times = []

        for sat_id in visible_satellite_ids:
            y = y_positions[sat_id]
            sat_data = obs.get(sat_id) or {}
            for freq in unique_freqs:
                if not self._frequency_sequence_accepts(sat_id, freq, system_filters, sat_filters, freq_filters):
                    continue
                freq_values = sat_data.get(freq)
                if not freq_values:
                    continue
                times = freq_values.get('times', []) or []
                valid_times = [times[idx] for idx in range(len(times)) if self._has_valid_observation(freq_values, idx)]
                if not valid_times:
                    continue
                all_valid_times.extend(valid_times)

                color = self._frequency_sequence_color(sat_id, freq, color_map)
                label = self._frequency_sequence_legend_label(sat_id, freq)
                y_vals = [y] * len(valid_times)
                ax.plot(valid_times, y_vals, color=color, linewidth=0.8, alpha=0.35, label=label)
                ax.scatter(valid_times, y_vals, color=color, s=10, alpha=0.9, edgecolors='none')

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(visible_satellite_ids)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('Satellite Visibility by Frequency', pad=8)
        ax.grid(True, axis='x', alpha=0.25)
        ax.grid(True, axis='y', alpha=0.15)

        if all_valid_times:
            x_min = min(all_valid_times)
            x_max = max(all_valid_times)
            ax.set_xlim(x_min, x_max)
            ax.margins(x=0, y=0)

        if len(visible_satellite_ids) > 1:
            ax.set_ylim(-0.35, len(visible_satellite_ids) - 0.65)
        else:
            ax.set_ylim(-0.5, 0.5)

        ax.margins(y=0)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=0, ha='center')

        legend_items = {}
        for sat_id in visible_satellite_ids:
            sat_data = obs.get(sat_id) or {}
            for freq in unique_freqs:
                if not self._frequency_sequence_accepts(sat_id, freq, system_filters, sat_filters, freq_filters):
                    continue
                freq_values = sat_data.get(freq)
                if not freq_values:
                    continue
                times = freq_values.get('times', []) or []
                if not any(self._has_valid_observation(freq_values, idx) for idx in range(len(times))):
                    continue
                label = self._frequency_sequence_legend_label(sat_id, freq)
                if label not in legend_items:
                    legend_items[label] = self._frequency_sequence_color(sat_id, freq, color_map)

        legend_handles = [
            Line2D([0], [0], color=color, marker='o', linestyle='-', linewidth=2.2,
                   markersize=7, markerfacecolor=color, markeredgecolor=color, label=label)
            for label, color in legend_items.items()
        ]
        legend = ax.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            fontsize=plt.rcParams.get('legend.fontsize', 'small'),
        )
        legend.set_draggable(True)

        fig.subplots_adjust(left=0.08, right=0.85, top=0.92, bottom=0.08)

        if MPLCURSORS_AVAILABLE and not save:
            cursor = mplcursors.cursor(ax, hover=False)
            cursor.connect(
                'add',
                lambda sel: sel.annotation.set(
                    text=f'GPST: {sel.target[0]}\nPRN: {satellite_ids[int(round(sel.target[1]))] if 0 <= int(round(sel.target[1])) < len(satellite_ids) else "N/A"}',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                ),
            )

        if save:
            path = self._save_fig(fig, 'satellite_frequency_sequence', output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

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
            ax.legend(loc='upper right', fontsize=plt.rcParams.get('legend.fontsize', 'small'))
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

    def plot_pseudorange_multipath(self, multipath_results: Dict[str, Any], sat_id: str,
                                   freq_pair: Optional[tuple] = None,
                                   save: bool = True,
                                   output_dir: Optional[str] = None) -> Dict[str, Any]:
        """绘制预计算的单颗卫星伪距多路径结果。"""
        sat_data = multipath_results.get(sat_id)
        if not sat_data:
            raise ValueError(f"No pseudorange multipath data for satellite {sat_id}")

        freq1, freq2 = sat_data.get('freq_pair', ('?', '?'))
        segments = sat_data.get('segments', {}) or {}
        stats = sat_data.get('stats', {}) or {}
        coeffs = sat_data.get('coefficients', {}) or {}

        if freq_pair and len(freq_pair) == 2:
            freq1, freq2 = freq_pair
            if freq1 not in segments or freq2 not in segments:
                raise ValueError(f"No pseudorange multipath data for satellite {sat_id} with pair {freq1}/{freq2}")

        if freq1 not in segments or freq2 not in segments:
            raise ValueError(f"No pseudorange multipath data for satellite {sat_id}")

        flat_freq1 = [value for segment in segments.get(freq1, []) for value in segment.get('values', [])]
        flat_freq2 = [value for segment in segments.get(freq2, []) for value in segment.get('values', [])]
        if not flat_freq1 and not flat_freq2:
            raise ValueError(f"No valid multipath samples for {sat_id}")

        colors = {freq1: 'tab:blue', freq2: 'tab:orange'}
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        if not hasattr(axes, '__iter__'):
            axes = [axes]

        for ax, freq_key, title_tag in (
            (axes[0], freq1, '(a)'),
            (axes[1], freq2, '(b)'),
        ):
            segs = segments.get(freq_key, [])
            for seg_idx, seg in enumerate(segs):
                label = freq_key if seg_idx == 0 else None
                ax.plot(seg.get('times', []), seg.get('values', []), color=colors[freq_key], marker='o', markersize=3,
                        linewidth=1.0, alpha=0.85, label=label)
            ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.35)
            ax.set_ylabel('PMP (m)')
            stat = stats.get(freq_key, {})
            if 'rms' not in stat:
                values = flat_freq1 if freq_key == freq1 else flat_freq2
                stat = {
                    **stat,
                    'rms': math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0,
                }
            ax.set_title(
                f'{title_tag} {freq_key}  mean={stat.get("mean", 0.0):.3f} m  '
                f'rms={stat.get("rms", 0.0):.3f} m  std={stat.get("std", 0.0):.3f} m',
                loc='left'
            )
            ax.grid(True, alpha=0.3)
            if segs:
                ax.legend(loc='upper right')

        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=0, ha='center')

        fig.suptitle(f'Pseudorange Multipath Satellite  {sat_id}  ({freq1} + {freq2})', fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if MPLCURSORS_AVAILABLE and not save:
            cursor = mplcursors.cursor(axes)

            def _on_add(sel):
                x_val, y_val = sel.target
                try:
                    x_text = mdates.num2date(x_val).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    x_text = str(x_val)
                freq_key = freq1 if sel.artist.axes == axes[0] else freq2
                sel.annotation.set(
                    text=f'频率: {freq_key}\n时间: {x_text}\nPMP: {y_val:.4f} m\n\n点击后按←/→键\n(图表窗口需有焦点)',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)
                )

            cursor.connect('add', _on_add)

        if save:
            path = self._save_fig(fig, f'pseudorange_multipath_satellite_{sat_id}_{freq1}_{freq2}', output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_pseudorange_multipath_overview(self, observations: Dict[str, Any],
                                            save: bool = True,
                                            output_dir: Optional[str] = None,
                                            smoothing_window: int = 5,
                                            system_filters=None,
                                            sat_filters=None,
                                            freq_filters=None) -> Dict[str, Any]:
        """Plot constellation-level pseudorange multipath overview.

        Each constellation gets one subplot. Within each subplot, satellites are drawn in distinct colors.
        The selected PMP pair for each satellite follows the requested default/fallback rules.
        """
        obs = observations.get('observations_meters', observations) if isinstance(observations, dict) else observations
        if not obs:
            raise ValueError('No observations available for pseudorange multipath overview plot')

        from src.processing.calculator import MetricCalculator
        from src.reporting.reporter import ReportGenerator
        import numpy as np

        system_name_map = {
            'G': 'GPS',
            'C': 'BDS',
            'E': 'Galileo',
            'J': 'QZSS',
            'R': 'GLONASS',
            'I': 'IRNSS',
        }
        system_order = {'G': 0, 'C': 1, 'E': 2, 'J': 3, 'R': 4, 'I': 5}

        mc = MetricCalculator()
        grouped_results: Dict[str, list] = {}
        log_lines = []
        total_satellites = 0
        total_plotted = 0

        preferred_pairs_by_system = {
            'G': [('L1C', 'L5Q')],
            'C': [('L1P', 'L5P'), ('L2I', 'L5P'), ('L2I', 'L1P')],
            'E': [('L1C', 'L5Q'), ('L1C', 'L7Q'), ('L5Q', 'L7Q')],
            'J': [('L1C', 'L5Q')],
        }

        def _has_valid_code(sat_data: Dict[str, Any], freq_name: str) -> bool:
            d = sat_data.get(freq_name, {})
            code_vals = d.get('code', []) if isinstance(d, dict) else []
            return any(v is not None for v in code_vals)

        def _ordered_pairs_for_sat(system: str, sat_data: Dict[str, Any]):
            from itertools import combinations

            valid_freqs = [f for f in sat_data.keys() if _has_valid_code(sat_data, f)]
            if len(valid_freqs) < 2:
                return []

            pair_list = []
            seen = set()
            for f1, f2 in preferred_pairs_by_system.get(system, []):
                if f1 in valid_freqs and f2 in valid_freqs:
                    key = tuple(sorted((f1, f2)))
                    if key not in seen:
                        seen.add(key)
                        pair_list.append((f1, f2))

            for f1, f2 in combinations(valid_freqs, 2):
                key = tuple(sorted((f1, f2)))
                if key not in seen:
                    seen.add(key)
                    pair_list.append((f1, f2))
            return pair_list

        for sat_id in sorted(obs.keys(), key=self._satellite_sort_key):
            sat_data = obs.get(sat_id) or {}
            system = sat_id[0] if sat_id else ''

            if not self._selection_allows(system_filters, system):
                continue
            if not self._selection_allows(sat_filters, sat_id):
                continue

            ordered_pairs = _ordered_pairs_for_sat(system, sat_data)
            if not ordered_pairs:
                log_lines.append(f'{sat_id}: skipped, no valid frequency pair')
                continue

            target_freqs = {f for f in sat_data.keys() if _has_valid_code(sat_data, f)}
            if freq_filters:
                target_freqs = {f for f in target_freqs if f in set(freq_filters)}

            covered_freqs = set()
            sat_has_valid = False
            for pair_idx, pair in enumerate(ordered_pairs):
                if freq_filters and (pair[0] not in set(freq_filters) or pair[1] not in set(freq_filters)):
                    continue

                # avoid redundant pair if it doesn't add new frequency coverage
                if sat_has_valid and not ((set(pair) - covered_freqs) & target_freqs):
                    continue

                try:
                    sat_result = mc.calculate_pseudorange_multipath(
                        {'observations_meters': {sat_id: sat_data}},
                        freq_pair=pair,
                        smoothing_window=smoothing_window,
                    ).get(sat_id)
                except Exception as exc:
                    log_lines.append(f'{sat_id}: skipped, calculation failed for {pair[0]}+{pair[1]} ({exc})')
                    continue

                if not sat_result:
                    continue

                sat_has_valid = True
                grouped_results.setdefault(system, []).append({
                    'sat_id': sat_id,
                    'pair': pair,
                    'rule': 'preferred' if pair_idx == 0 else f'fallback_{pair_idx}',
                    'result': sat_result,
                })
                covered_freqs.update(pair)
                total_plotted += 1

                stats = sat_result.get('stats', {}) or {}
                s1 = stats.get(pair[0], {})
                s2 = stats.get(pair[1], {})
                log_lines.append(
                    f"{sat_id}: pair={pair[0]}+{pair[1]} | "
                    f"{pair[0]}: count={s1.get('count', 0)} mean={s1.get('mean', 0.0):.3f} "
                    f"rms={s1.get('rms', 0.0):.3f} std={s1.get('std', 0.0):.3f} | "
                    f"{pair[1]}: count={s2.get('count', 0)} mean={s2.get('mean', 0.0):.3f} "
                    f"rms={s2.get('rms', 0.0):.3f} std={s2.get('std', 0.0):.3f}"
                )

                if target_freqs and target_freqs.issubset(covered_freqs):
                    break

            if sat_has_valid:
                total_satellites += 1
            else:
                log_lines.append(f'{sat_id}: skipped, no valid PMP data under current filters')

        if not grouped_results:
            raise ValueError('No valid pseudorange multipath overview data found')

        system_keys = [system for system in sorted(grouped_results.keys(), key=lambda s: (system_order.get(s, 99), s)) if grouped_results.get(system)]
        fig_height = max(4.8, 3.8 * len(system_keys))
        fig, axes = plt.subplots(len(system_keys), 1, figsize=(15, fig_height), sharex=True)
        if len(system_keys) == 1:
            axes = [axes]

        line_meta = {}
        for ax, system in zip(axes, system_keys):
            sat_items = grouped_results.get(system, [])
            if not sat_items:
                continue

            sat_ids = sorted({item['sat_id'] for item in sat_items}, key=self._satellite_sort_key)
            colors = plt.get_cmap('tab20')(np.linspace(0, 1, max(len(sat_ids), 2), endpoint=False))
            sat_color_map = {sid: colors[idx % len(colors)] for idx, sid in enumerate(sat_ids)}
            all_times = []
            freq_summary = {}
            sat_legend_handles = []
            sat_legend_seen = set()

            for idx, item in enumerate(sat_items):
                sat_id = item['sat_id']
                pair = item['pair']
                sat_result = item['result']
                freq1, freq2 = pair
                color = sat_color_map.get(sat_id, colors[idx % len(colors)])

                for freq_key, line_style in ((freq1, '-'), (freq2, '--')):
                    for seg_idx, seg in enumerate((sat_result.get('segments', {}) or {}).get(freq_key, [])):
                        times = seg.get('times', []) or []
                        values = seg.get('values', []) or []
                        if not times or not values:
                            continue
                        all_times.extend(times)
                        line, = ax.plot(
                            times,
                            values,
                            color=color,
                            linestyle=line_style,
                            marker='o',
                            markersize=2.2,
                            linewidth=1.0,
                            alpha=0.85,
                        )
                        line_meta[line] = {
                            'sat_id': sat_id,
                            'freq_key': freq_key,
                            'pair': pair,
                        }

                if sat_id not in sat_legend_seen:
                    sat_legend_seen.add(sat_id)
                    sat_legend_handles.append(
                        Line2D(
                            [0], [0],
                            color=color,
                            linestyle='-',
                            linewidth=2.0,
                            marker='o',
                            markersize=5,
                            label=sat_id,
                        )
                    )

                freq_summary.setdefault(freq1, []).extend([
                    v
                    for seg in (sat_result.get('segments', {}) or {}).get(freq1, [])
                    for v in (seg.get('values', []) or [])
                    if v is not None
                ])
                freq_summary.setdefault(freq2, []).extend([
                    v
                    for seg in (sat_result.get('segments', {}) or {}).get(freq2, [])
                    for v in (seg.get('values', []) or [])
                    if v is not None
                ])

            ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.35)
            ax.set_ylabel('PMP (m)')
            ax.set_title(f'{system_name_map.get(system, system or "?")} ({system})', loc='left', fontweight='bold')
            ax.grid(True, alpha=0.3)
            if all_times:
                ax.set_xlim(min(all_times), max(all_times))

            if freq_summary:
                rms_parts = []
                if system == 'C':
                    ordered_keys = [k for k in ('L2I', 'L1P', 'L5P') if k in freq_summary]
                    ordered_keys += [k for k in sorted(freq_summary.keys()) if k not in ordered_keys]
                else:
                    ordered_keys = sorted(freq_summary.keys())

                for freq_key in ordered_keys:
                    vals = freq_summary.get(freq_key, [])
                    rms_val = math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else 0.0
                    rms_parts.append(f'{freq_key}:{rms_val:.2f}')
                ax.text(
                    0.01,
                    0.98,
                    'RMS ' + '  '.join(rms_parts),
                    transform=ax.transAxes,
                    va='top',
                    ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
                )

            if sat_legend_handles:
                legend = ax.legend(
                    handles=sat_legend_handles,
                    loc='upper left',
                    bbox_to_anchor=(1.01, 1.0),
                    ncol=1,
                    frameon=True,
                )
                legend.set_draggable(True)

        if MPLCURSORS_AVAILABLE and not save and line_meta:
            cursor = mplcursors.cursor(list(line_meta.keys()), hover=False)

            def _on_overview_add(sel):
                artist = sel.artist
                meta = line_meta.get(artist, {})
                sat_id = meta.get('sat_id', '?')
                freq_key = meta.get('freq_key', '?')
                pair = meta.get('pair', ('?', '?'))
                x_val, y_val = sel.target
                try:
                    t_text = mdates.num2date(x_val).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    t_text = str(x_val)
                sel.annotation.set(
                    text=(
                        f'卫星: {sat_id}\n'
                        f'频率: {freq_key}\n'
                        f'组合: {pair[0]}+{pair[1]}\n'
                        f'时间: {t_text}\n'
                        f'PMP: {y_val:.4f} m\n\n'
                        '点击后按←/→键\n(图表窗口需有焦点)'
                    ),
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)
                )

            cursor.connect('add', _on_overview_add)

        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=0, ha='center')
        fig.suptitle('Pseudorange Multipath Constellation', fontweight='bold', fontsize=plt.rcParams['figure.titlesize'])
        fig.tight_layout(rect=[0, 0, 0.95, 0.96])

        log_text = '\n'.join([
            'Pseudorange Multipath Constellation Analysis Summary',
            f'Smoothing window: {smoothing_window}',
            f'Satellites plotted: {total_plotted}',
            f'System filters: {system_filters if system_filters else "ALL"}',
            f'Satellite filters: {sat_filters if sat_filters else "ALL"}',
            f'Frequency filters: {freq_filters if freq_filters else "ALL"}',
            'Selection rules:',
            '  GPS/G -> L1C+L5Q',
            '  BDS/C -> L1P+L5P -> L2I+L5P -> L2I+L1P',
            '  Galileo/E -> L1C+L5Q -> L1C+L7Q -> L5Q+L7Q',
            '  J -> L1C+L5Q',
            '',
            *log_lines,
        ])

        log_path = None
        if save:
            path = self._save_fig(fig, 'pseudorange_multipath_constellation', output_dir)
            try:
                reporter = ReportGenerator()
                log_path = reporter.save_logs('pseudorange_multipath_constellation', log_text, self._ensure_output_dir(output_dir), prefix='pmp')
            except Exception:
                log_path = None
            return {'figure': None, 'path': path, 'log_path': log_path, 'log_text': log_text}
        return {'figure': fig, 'path': None, 'log_path': None, 'log_text': log_text}

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
        绘制周跳探测分析图（MW + GF + LLI）
        
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
        lli_data = detection_result.get('lli', {})
        freq_pair = detection_result.get('freq_pair', ('?', '?'))
        
        # 创建三个子图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 13))
        fig.suptitle(f'周跳探测 - {sat_id} ({freq_pair[0]}/{freq_pair[1]}) [MW/GF/LLI]', fontsize=14, fontweight='bold')
        
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

        # 子图(c): LLI标识检验
        if 'epochs' in lli_data and 'lock_loss_union' in lli_data:
            epochs = lli_data.get('epochs', [])
            lock_loss_union = lli_data.get('lock_loss_union', [])
            half_cycle_union = lli_data.get('half_cycle_union', [])
            lli_freqs = lli_data.get('lli_freqs', [])
            bit0_by_freq = lli_data.get('bit0_by_freq', {})
            events = lli_data.get('cycle_slips', [])
            half_events = lli_data.get('half_cycle_events', [])

            ax3.step(epochs, lock_loss_union, where='mid', color='purple', linewidth=1.8, label='LLI bit0 (Loss of lock)')
            if len(epochs) == len(half_cycle_union):
                ax3.step(epochs, half_cycle_union, where='mid', color='orange', linewidth=1.3, alpha=0.8, label='LLI bit1 (Half-cycle)')

            # 绘制各频率 bit0 轨迹（多频并列）
            if lli_freqs:
                cmap = plt.get_cmap('tab10')
                for idx, fn in enumerate(lli_freqs):
                    bit0_vals = bit0_by_freq.get(fn, [])
                    if len(bit0_vals) == len(epochs):
                        ax3.plot(
                            epochs,
                            bit0_vals,
                            color=cmap(idx % 10),
                            alpha=0.35,
                            linewidth=1.0,
                            label=f'{fn} bit0'
                        )

            if events:
                slip_epochs = [e['epoch'] for e in events]
                slip_vals = [1] * len(slip_epochs)
                ax3.scatter(slip_epochs, slip_vals, c='red', marker='x', s=90,
                            linewidths=2, label='LLI周跳(bit0)', zorder=6)
            if half_events:
                he_epochs = [e['epoch'] for e in half_events]
                he_vals = [1] * len(he_epochs)
                ax3.scatter(he_epochs, he_vals, c='goldenrod', marker='o', s=38,
                            linewidths=1.0, label='半周模糊(bit1)', zorder=5)

            ax3.set_ylim(-0.1, 1.2)
            ax3.set_yticks([0, 1])
            ax3.set_xlabel('历元 (Epoch)', fontsize=11)
            ax3.set_ylabel('LLI Flag', fontsize=11)
            ax3.set_title('(c) LLI标识检验（bit0=周跳/失锁，bit1=半周模糊）', fontsize=12, loc='left')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='best')
        else:
            ax3.text(0.5, 0.5, 'LLI数据不可用', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('(c) LLI标识检验', fontsize=12, loc='left')
        
        plt.tight_layout()

        # 添加交互式数据点显示（点击数据点后用方向键切换）
        if MPLCURSORS_AVAILABLE and not save:
            cursor = mplcursors.cursor([ax1, ax2, ax3])
            def on_add(sel):
                x, y = sel.target
                if sel.artist.axes == ax1:
                    label_text = 'ΔMW'
                    unit = 'm'
                elif sel.artist.axes == ax2:
                    label_text = 'ΔGF'
                    unit = 'm'
                else:
                    label_text = 'LLI'
                    unit = ''
                unit_suffix = f' {unit}' if unit else ''
                sel.annotation.set(text=f'历元: {x:.0f}\n{label_text}: {y:.3f}{unit_suffix}\n\n点击后按←/→键\n(图表窗口需有焦点)',
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

    def plot_cnr_analysis(self, observations: Dict[str, Any], save: bool = True,
                         output_dir: Optional[str] = None,
                         system_filters=None,
                         sat_filters=None,
                         freq_filters=None) -> Dict[str, Any]:
        """Plot Carrier-to-Noise Ratio (CNR/C/N0) analysis with histogram and box plot.
        
        Grouped by satellite system and frequency (e.g., GPS-L1C, GPS-L5Q, GLO-L1C)
        Left subplot: histogram of CNR values by frequency band
        Right subplot: box plot of CNR distribution by frequency band
        """
        obs = observations.get('observations_meters', observations) if isinstance(observations, dict) else observations
        if not obs:
            raise ValueError('No observations available for CNR analysis')

        import numpy as np
        
        # Extract CNR data grouped by system-frequency (not by individual satellite)
        cnr_by_freq = {}  # (system, freq) -> [cnr_values]
        freq_labels = []  # display labels like 'GPS-L1C', 'GPS-L5Q', etc.
        freq_data = []    # corresponding CNR data arrays
        
        # Map system code to full name for display
        system_names = {'G': 'GPS', 'R': 'GLO', 'E': 'Galileo', 'C': 'BDS', 'J': 'J-GPS', 'I': 'IRNSS'}
        
        for sat_id in sorted(obs.keys(), key=self._satellite_sort_key):
            if not self._selection_allows(system_filters, sat_id[0]) or not self._selection_allows(sat_filters, sat_id):
                continue
            
            system_code = sat_id[0]
            system_name = system_names.get(system_code, system_code)
            
            sat_data = obs.get(sat_id, {})
            for freq in sorted(sat_data.keys()):
                if not self._frequency_sequence_accepts(sat_id, freq, system_filters, sat_filters, freq_filters):
                    continue
                
                freq_values = sat_data.get(freq, {})
                snr_list = freq_values.get('snr', []) or []
                
                # Filter out None values
                valid_snr = [s for s in snr_list if s is not None and isinstance(s, (int, float))]
                
                if valid_snr:
                    # Create key as (system_name, freq) for grouping across all satellites
                    key = (system_name, freq)
                    if key not in cnr_by_freq:
                        cnr_by_freq[key] = []
                    cnr_by_freq[key].extend(valid_snr)
        
        if not cnr_by_freq:
            raise ValueError('No valid CNR data found')

        # Prepare data for plotting - sort by system then frequency
        system_order = {'GPS': 0, 'GLO': 1, 'Galileo': 2, 'BDS': 3, 'J-GPS': 4, 'IRNSS': 5}
        sorted_keys = sorted(cnr_by_freq.keys(), key=lambda x: (system_order.get(x[0], 99), x[1]))
        
        for system_name, freq in sorted_keys:
            label = f'{system_name[:3]} {freq}'
            freq_labels.append(label)
            freq_data.append(cnr_by_freq[(system_name, freq)])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: KDE curves showing C/N0 density by frequency band
        colors = plt.cm.tab20(np.linspace(0, 1, len(freq_labels)))
        all_min = min(min(data) for data in freq_data)
        all_max = max(max(data) for data in freq_data)
        span = all_max - all_min
        pad = max(span * 0.1, 1.0)
        x_grid = np.linspace(all_min - pad, all_max + pad, 512)

        for i, (label, data) in enumerate(zip(freq_labels, freq_data)):
            sample = np.asarray(data, dtype=float)
            if sample.size < 2:
                continue

            sample_std = np.std(sample, ddof=1)
            if not np.isfinite(sample_std) or sample_std == 0:
                sample_std = 1.0
            bandwidth = 1.06 * sample_std * (sample.size ** (-1 / 5))
            bandwidth = max(bandwidth, 0.5)

            diff = (x_grid[:, None] - sample[None, :]) / bandwidth
            density = np.exp(-0.5 * diff ** 2).sum(axis=1)
            density /= (sample.size * bandwidth * np.sqrt(2 * np.pi))

            ax1.plot(x_grid, density, color=colors[i], linewidth=1.8, label=label)
            ax1.fill_between(x_grid, density, color=colors[i], alpha=0.12)

        ax1.set_xlabel('C/N0 (dB-Hz)', fontsize=plt.rcParams['axes.labelsize'])
        ax1.set_ylabel('Probability Density', fontsize=plt.rcParams['axes.labelsize'])
        ax1.set_title('C/N0 Distribution by Frequency Band', fontsize=plt.rcParams['font.size'])
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(loc='upper right', fontsize=10)

        # Right: Box plot showing CNR statistics by frequency band
        bp = ax2.boxplot(
            freq_data,
            labels=freq_labels,
            patch_artist=True,
            widths=0.6,
            flierprops=dict(
                marker='o',
                markerfacecolor='none',
                markeredgecolor='red',
                markersize=5,
                linestyle='none'
            ),
            medianprops=dict(color='green', linewidth=1.5)
        )
        
        for patch in bp['boxes']:
            patch.set_facecolor('none')
            patch.set_alpha(1.0)
        
        ax2.set_ylabel('C/N0 (dB-Hz)', fontsize=plt.rcParams['axes.labelsize'])
        ax2.set_title('C/N0 Statistics by Frequency Band', fontsize=plt.rcParams['font.size'])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for readability
        ax2.set_xticklabels(freq_labels, rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])
        
        fig.tight_layout()

        if MPLCURSORS_AVAILABLE and not save:
            kde_cursor = mplcursors.cursor(ax1.lines, hover=False)

            def _on_kde_click(sel):
                x_val, y_val = sel.target
                sel.annotation.set(
                    text=f'C/N0: {x_val:.1f} dB-Hz\nDensity: {y_val:.4f}',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)
                )

            kde_cursor.connect('add', _on_kde_click)

            box_cursor = mplcursors.cursor(ax2, hover=True)
            
            def _on_boxplot_click(sel):
                sel.annotation.set(
                    text=f'C/N0: {sel.target[1]:.1f} dB-Hz',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)
                )
            
            box_cursor.connect('add', _on_boxplot_click)

        if save:
            path = self._save_fig(fig, 'cnr_analysis', output_dir)
            return {'figure': None, 'path': path}
        return {'figure': fig, 'path': None}

    def plot_data_integrity(self, observations: Dict[str, Any], save: bool = True,
                            output_dir: Optional[str] = None,
                            system_filters=None,
                            sat_filters=None,
                            freq_filters=None) -> Dict[str, Any]:
        """Compute and plot data integrity rate.

        Left: bar chart of integrity rate per system-frequency (e.g., GPS L1C)
        Right: bar chart of integrity rate per satellite (averaged across frequencies)
        """
        obs = observations.get('observations_meters', observations) if isinstance(observations, dict) else observations
        if not obs:
            raise ValueError('No observations available for data integrity analysis')

        # collect per-satellite visible epochs (union of all timestamps per satellite)
        import numpy as np
        sat_visible_epochs = {}
        for sat_id, freqs in obs.items():
            if not self._selection_allows(system_filters, sat_id[0]) or not self._selection_allows(sat_filters, sat_id):
                continue
            sat_times = set()
            for freq, data in freqs.items():
                times = data.get('times', []) or []
                for t in times:
                    sat_times.add(t)
            if sat_times:
                sat_visible_epochs[sat_id] = len(sat_times)

        if not sat_visible_epochs:
            raise ValueError('No epoch times found')

        # compute actual counts per sat-freq (require both code and phase present)
        actual_counts = {}  # (sat_id, freq) -> count
        for sat_id in sorted(obs.keys(), key=self._satellite_sort_key):
            if not self._selection_allows(system_filters, sat_id[0]) or not self._selection_allows(sat_filters, sat_id):
                continue
            for freq, data in obs[sat_id].items():
                if not self._frequency_sequence_accepts(sat_id, freq, system_filters, sat_filters, freq_filters):
                    continue
                codes = data.get('code', []) or []
                phases = data.get('phase', []) or []
                cnt = 0
                for c, p in zip(codes, phases):
                    if c is not None and p is not None:
                        cnt += 1
                actual_counts[(sat_id, freq)] = cnt

        if not actual_counts:
            raise ValueError('No valid data points for integrity calculation')

        # group by system-frequency — only include sats that actually have data for this freq
        system_names = {'G': 'GPS', 'R': 'GLO', 'E': 'Galileo', 'C': 'BDS', 'J': 'J-GPS', 'I': 'IRNSS'}
        sf_groups = {}  # (system_name, freq) -> list of sat keys
        sat_list = set()
        for (sat_id, freq), cnt in actual_counts.items():
            if cnt <= 0:
                continue
            system = system_names.get(sat_id[0], sat_id[0])
            key = (system, freq)
            sf_groups.setdefault(key, []).append((sat_id, cnt))
            sat_list.add(sat_id)

        # determine selected systems (map filters to single-letter codes if needed)
        selected_system_codes = set()
        if system_filters:
            # accept list of single-letter codes or full names
            reverse_map = {v: k for k, v in system_names.items()}
            for item in system_filters:
                if not item:
                    continue
                s = str(item)
                if len(s) == 1:
                    selected_system_codes.add(s)
                else:
                    # try match full name or first 3 letters
                    uc = s.upper()
                    if uc in reverse_map:
                        selected_system_codes.add(reverse_map[uc])
                    else:
                        # try matching by prefix (GPS->G)
                        for full, code in reverse_map.items():
                            if full.upper().startswith(uc[:3]):
                                selected_system_codes.add(code)
                                break
        else:
            # no explicit filter -> infer from data
            for sat_id in obs.keys():
                if sat_id:
                    selected_system_codes.add(sat_id[0])

        # decide plotting mode
        multi_system_mode = len(selected_system_codes) >= 2

        # compute ratios per system-frequency
        sf_labels = []
        sf_ratios = []
        for key in sorted(sf_groups.keys(), key=lambda x: (x[0], x[1])):
            system_name = key[0]
            # consider only selected systems
            code = next((k for k, v in system_names.items() if v == system_name), system_name[:1])
            if code not in selected_system_codes and not multi_system_mode:
                # if single-system mode and this key not in selected, skip
                continue
            sats = sf_groups[key]
            actual_total = sum(cnt for _, cnt in sats)
            expected_total = sum(sat_visible_epochs.get(sat_id, 0) for sat_id, _ in sats)
            ratio = actual_total / expected_total if expected_total > 0 else 0.0
            sf_labels.append(f"{system_name[:3]} {key[1]}")
            sf_ratios.append(ratio)

        # compute per-satellite overall ratio — only freqs with cnt>0
        sat_ratios = {}
        for sat in sorted(sat_list, key=self._satellite_sort_key):
            if sat and sat[0] not in selected_system_codes and not multi_system_mode:
                continue
            keys = [(s, f) for (s, f) in actual_counts.keys() if s == sat]
            if not keys:
                continue
            # only include frequencies where the satellite actually has data
            active_keys = [(s, f) for (s, f) in keys if actual_counts.get((s, f), 0) > 0]
            if not active_keys:
                continue
            actual_total = sum(actual_counts[(sat, f)] for (_, f) in active_keys)
            expected_total = sat_visible_epochs.get(sat, 0) * len(active_keys)
            sat_ratios[sat] = (actual_total / expected_total) if expected_total > 0 else 0.0

        # plotting: single-axis only
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        # reuse CNR system order for consistent ordering
        system_order = {'GPS': 0, 'GLO': 1, 'Galileo': 2, 'BDS': 3, 'J-GPS': 4, 'IRNSS': 5}

        if multi_system_mode:
            # prepare and filter out zero-data SF entries, sort by system_order then freq
            items = []
            for key in sf_groups.keys():
                system_name, freq = key
                sats = sf_groups[key]
                actual_total = sum(cnt for _, cnt in sats)
                if actual_total <= 0:
                    continue
                expected_total = sum(sat_visible_epochs.get(sat_id, 0) for sat_id, _ in sats)
                ratio = actual_total / expected_total if expected_total > 0 else 0.0
                items.append((system_name, freq, ratio))

            items_sorted = sorted(items, key=lambda x: (system_order.get(x[0], 99), x[1]))
            labels = [f"{it[0][:3]} {it[1]}" for it in items_sorted]
            ratios = [it[2] for it in items_sorted]

            x = np.arange(len(labels))
            bars = ax.bar(x, np.array(ratios) * 100.0, width=0.5, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])
            ax.set_ylabel('Data Integrity Rate (%)')
            ax.set_title('Data Integrity by Frequency Band')
            ax.grid(True, axis='y', alpha=0.3)

            for bar, val in zip(bars, ratios):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{val*100:.1f}", ha='center', va='bottom', fontsize=plt.rcParams.get('font.size', 10))

        else:
            # Single system selected: grouped bar chart per-satellite with frequencies colored
            single_code = next(iter(selected_system_codes)) if selected_system_codes else None
            sat_freqs = {}
            freq_counts = {}
            for (sat_id, freq), cnt in actual_counts.items():
                if single_code and sat_id[0] != single_code:
                    continue
                sat_freqs.setdefault(sat_id, {})[freq] = cnt
                freq_counts[freq] = freq_counts.get(freq, 0) + cnt

            # remove frequencies with zero total across sats
            freq_set = sorted([f for f, c in freq_counts.items() if c > 0])
            sats = sorted(sat_freqs.keys(), key=self._satellite_sort_key)

            if not sats:
                ax.text(0.5, 0.5, 'No data for selected system', ha='center')
            else:
                n_freq = len(freq_set)
                x = np.arange(len(sats))
                total_width = 0.8
                bar_width = total_width / max(n_freq, 1)
                colors = plt.cm.tab10(np.linspace(0, 1, n_freq))
                for i, freq in enumerate(freq_set):
                    vals = []
                    for sat in sats:
                        cnt = sat_freqs.get(sat, {}).get(freq, 0)
                        sat_expected = sat_visible_epochs.get(sat, 0)
                        vals.append((cnt / sat_expected) * 100.0 if sat_expected > 0 else 0.0)
                    offsets = x - total_width/2 + i*bar_width + bar_width/2
                    bars = ax.bar(offsets, vals, width=bar_width*0.9, color=colors[i], label=freq)
                    for bar, sat, v in zip(bars, sats, vals):
                        bar._sat = sat
                        bar._freq = freq
                        cnt_val = sat_freqs.get(sat, {}).get(freq, 0)
                        bar._cnt = cnt_val
                        bar._visible = sat_visible_epochs.get(sat, 0)

                ax.set_xticks(x)
                ax.set_xticklabels(sats, rotation=45, ha='right')
                ax.set_ylabel('Data Integrity Rate (%)')
                ax.set_title(f'Data Integrity per Satellite ({single_code})')
                ax.legend(title='Frequency', fontsize=8)
                ax.grid(True, axis='y', alpha=0.3)

        fig.tight_layout()

        # interactive tooltip for single-system mode
        if MPLCURSORS_AVAILABLE and not save and not multi_system_mode:
            cursor = mplcursors.cursor(ax.patches, hover=False)

            def _on_di_add(sel):
                patch = sel.artist
                sat = getattr(patch, '_sat', '?')
                freq = getattr(patch, '_freq', '?')
                cnt = getattr(patch, '_cnt', None)
                visible = getattr(patch, '_visible', 0)
                rate_val = patch.get_height()
                if cnt is not None:
                    sel.annotation.set(
                        text=f'{sat} {freq}\nRate: {rate_val:.1f}%\nActual: {cnt}/{visible}',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)
                    )
                else:
                    sel.annotation.set(
                        text=f'{sat} {freq}\nRate: {rate_val:.1f}%',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)
                    )

            cursor.connect('add', _on_di_add)

        if save:
            path = self._save_fig(fig, 'data_integrity', output_dir)
            # write a simple text log summarizing the statistics
            out_dir = self._ensure_output_dir(output_dir)
            log_path = os.path.join(out_dir, 'data_integrity_log.txt')
            with open(log_path, 'w', encoding='utf-8') as fh:
                fh.write(f'Data Integrity Report\n')
                fh.write(f'Generated: {datetime.datetime.utcnow().isoformat()}Z\n')
                fh.write('Per-satellite visible epoch counts:\n')
                fh.write(f'{"Satellite":<12}{"VisibleEpochs":>14}\n')
                for sat in sorted(sat_visible_epochs.keys(), key=self._satellite_sort_key):
                    fh.write(f'{sat:<12}{sat_visible_epochs[sat]:>14}\n')
                fh.write('\nSystem-Frequency Statistics:\n')
                fh.write(f'{"System-Freq":<12}{"ActualEpochs":>14}{"ExpectedEpochs":>16}{"IntegrityRate(%)":>18}\n')
                for key in sorted(sf_groups.keys(), key=lambda x: (system_order.get(x[0], 99), x[1])):
                    system_name, freq = key
                    sats = sf_groups[key]
                    actual_total = sum(cnt for _, cnt in sats)
                    expected_total = sum(sat_visible_epochs.get(sat_id, 0) for sat_id, _ in sats)
                    if expected_total > 0:
                        ratio = actual_total / expected_total
                    else:
                        ratio = 0.0
                    fh.write(f'{system_name[:3] + " " + freq:<12}{actual_total:>14}{expected_total:>16}{ratio*100:>18.2f}\n')
                fh.write('\nPer-Satellite Statistics:\n')
                fh.write(f'{"Satellite":<12}{"VisibleEpochs":>14}{"ActiveFreqs":>12}{"ActualEpochs":>14}{"IntegrityRate(%)":>18}\n')
                for sat in sorted(sat_ratios.keys(), key=self._satellite_sort_key):
                    freq_keys = [(s, f) for (s, f) in actual_counts.keys() if s == sat]
                    active_keys = [(s, f) for (s, f) in freq_keys if actual_counts.get((s, f), 0) > 0]
                    actual_total = sum(actual_counts[(sat, f)] for (_, f) in active_keys)
                    visible = sat_visible_epochs.get(sat, 0)
                    active_count = len(active_keys)
                    expected_total = visible * active_count
                    rate = actual_total / expected_total if expected_total > 0 else 0.0
                    fh.write(f'{sat:<12}{visible:>14}{active_count:>12}{actual_total:>14}{rate*100:>18.2f}\n')
                fh.write('\nNote: IntegrityRate = ActualEpochs / ExpectedEpochs, where ExpectedEpochs = SatelliteVisibleEpochs * ActiveFreqs\n')
                fh.write('Only frequencies with at least one valid code+phase observation are counted (ActiveFreqs).\n')
            return {'figure': None, 'path': path, 'log': log_path}
        return {'figure': fig, 'path': None}

    def plot_doppler_quality(self, observations: Dict[str, Any], save: bool = True,
                             output_dir: Optional[str] = None,
                             system_filters=None, sat_filters=None,
                             freq_filters=None) -> Dict[str, Any]:
        """Plot Doppler completeness, phase consistency, outliers and smoothness."""
        import numpy as np
        from src.processing.calculator import MetricCalculator

        obs = observations.get('observations_meters', observations) if isinstance(observations, dict) else observations
        filtered = {}
        for sat_id in sorted(obs, key=self._satellite_sort_key):
            if not self._selection_allows(system_filters, sat_id[0]):
                continue
            if not self._selection_allows(sat_filters, sat_id):
                continue
            selected = {}
            for freq, values in obs[sat_id].items():
                if self._frequency_sequence_accepts(sat_id, freq, system_filters, sat_filters, freq_filters):
                    selected[freq] = values
            if selected:
                filtered[sat_id] = selected
        if not filtered:
            raise ValueError('No observations after filtering for Doppler quality analysis')

        quality = MetricCalculator().calculate_doppler_quality({'observations_meters': filtered})
        rows = quality.get('summary', [])
        if not rows:
            raise ValueError('No Doppler quality data available')

        selected_systems = sorted({sat_id[0] for sat_id in filtered})
        all_satellites_selected = not sat_filters or set(sat_filters) >= set(obs.keys())
        aggregate_mode = len(selected_systems) > 1 and all_satellites_selected

        def percentile(values, percent):
            values = sorted(values)
            if not values:
                return None
            position = (len(values) - 1) * percent / 100.0
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            if lower == upper:
                return values[lower]
            weight = position - lower
            return values[lower] * (1.0 - weight) + values[upper] * weight

        if aggregate_mode:
            system_names = {'G': 'GPS', 'R': 'GLO', 'E': 'Galileo', 'C': 'BDS',
                            'J': 'QZSS', 'I': 'IRNSS', 'S': 'SBAS'}
            grouped = {}
            for row in rows:
                grouped.setdefault((row['system'], row['freq']), []).append(row)
            plot_rows = []
            for (system, freq), members in sorted(grouped.items()):
                system_label = system_names.get(system, system)
                residuals = [value for member in members
                             for value in member.get('phase_consistency_residuals', [])]
                accelerations = [value for member in members
                                 for value in member.get('accelerations_mps2', [])]
                total_epochs = sum(member.get('total_epochs', 0) for member in members)
                valid_doppler = sum(member.get('valid_doppler_epochs', 0) for member in members)
                valid_residuals = sum(member.get('valid_residual_count', 0) for member in members)
                outliers = sum(member.get('outlier_count', 0) for member in members)
                median_residual = statistics.median(residuals) if residuals else None
                plot_rows.append({
                    'sat_id': system_label,
                    'system': system,
                    'freq': freq,
                    'group_label': f'{system_label}\n{freq}',
                    'member_count': len(members),
                    'total_epochs': total_epochs,
                    'valid_doppler_epochs': valid_doppler,
                    'completeness_percent': 100.0 * valid_doppler / total_epochs if total_epochs else 0.0,
                    'residual_rms_m': (math.sqrt(sum(v * v for v in residuals) / len(residuals))
                                       if residuals else None),
                    'residual_mad_m': (statistics.median([abs(v - median_residual) for v in residuals])
                                       if residuals else None),
                    'residual_p95_m': percentile([abs(v) for v in residuals], 95),
                    'outlier_count': outliers,
                    'valid_residual_count': valid_residuals,
                    'outlier_rate_percent': 100.0 * outliers / valid_residuals if valid_residuals else 0.0,
                    'acceleration_p95_mps2': percentile([abs(v) for v in accelerations], 95),
                })
            grouping_label = 'System-Frequency Summary'
        else:
            plot_rows = rows
            for row in plot_rows:
                row['group_label'] = f"{row['sat_id']}\n{row['freq']}"
                row['member_count'] = 1
            grouping_label = 'Satellite-Frequency Detail'

        labels = [row['group_label'] for row in plot_rows]
        x = np.arange(len(plot_rows))
        fig, axes = plt.subplots(2, 2, figsize=(20, 11))
        # Showing every satellite/frequency label makes a full multi-GNSS data set
        # unreadable. Keep every bar, but display an evenly sampled set of ticks.
        max_visible_ticks = 22
        tick_step = max(1, int(math.ceil(len(plot_rows) / max_visible_ticks)))
        tick_positions = list(range(0, len(plot_rows), tick_step))
        if plot_rows and tick_positions[-1] != len(plot_rows) - 1:
            tick_positions.append(len(plot_rows) - 1)
        plots = [
            ('completeness_percent', 'Completeness (%)', 'Doppler Data Completeness', 'steelblue'),
            ('residual_rms_m', 'RMS (m)', 'Doppler-Phase Consistency Residual', 'tomato'),
            ('outlier_rate_percent', 'Outlier Rate (%)', 'Consistency Outlier Rate', 'darkorange'),
            ('acceleration_p95_mps2', 'P95 (m/s^2)', 'Doppler Smoothness (Acceleration P95)', 'seagreen'),
        ]
        for ax, (key, ylabel, title, color) in zip(axes.flat, plots):
            values = [row.get(key) if row.get(key) is not None else 0.0 for row in plot_rows]
            bars = ax.bar(x, values, color=color, alpha=0.82)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([labels[index] for index in tick_positions],
                               rotation=45, ha='right', fontsize=8,
                               rotation_mode='anchor')
            ax.set_xlim(-0.75, len(plot_rows) - 0.25)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, axis='y', alpha=0.3)
            for bar, row, value in zip(bars, plot_rows, values):
                bar._doppler_quality_row = row
                if len(plot_rows) <= 20:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{value:.2f}', ha='center', va='bottom', fontsize=7)

        fig.suptitle(f'Doppler Quality Analysis - {grouping_label}', fontsize=15, fontweight='bold')
        fig.tight_layout(rect=(0, 0.02, 1, 0.97), h_pad=2.2, w_pad=2.0)

        if MPLCURSORS_AVAILABLE and not save:
            cursor = mplcursors.cursor([patch for ax in axes.flat for patch in ax.patches], hover=False)
            def _on_add(selection):
                row = getattr(selection.artist, '_doppler_quality_row', {})
                selection.annotation.set_text(
                    f"{row.get('sat_id')} {row.get('freq')} ({row.get('member_count', 1)} satellites)\n"
                    f"Completeness: {row.get('completeness_percent', 0):.2f}%\n"
                    f"Residual RMS: {row.get('residual_rms_m') if row.get('residual_rms_m') is not None else 'NA'}\n"
                    f"Residual P95: {row.get('residual_p95_m') if row.get('residual_p95_m') is not None else 'NA'}\n"
                    f"Outliers: {row.get('outlier_count', 0)}/{row.get('valid_residual_count', 0)}"
                )
            cursor.connect('add', _on_add)

        if save:
            path = self._save_fig(fig, 'doppler_quality', output_dir)
            out_dir = self._ensure_output_dir(output_dir)
            log_path = os.path.join(out_dir, 'doppler_quality_log.txt')
            with open(log_path, 'w', encoding='utf-8') as fh:
                fh.write('Doppler Quality Analysis Report\n')
                fh.write(f'Generated: {datetime.datetime.utcnow().isoformat()}Z\n\n')
                fh.write(f'Plot grouping: {grouping_label}\n')
                fh.write('Multi-system overview uses system-frequency aggregation; a single system or explicit satellite selection uses satellite-frequency detail.\n\n')
                fh.write('Doppler-phase residual = phase range change - integrated Doppler range-rate\n')
                fh.write('Outlier threshold = max(0.5 m, 4 * 1.4826 * MAD)\n\n')
                fh.write('Plotted Groups:\n')
                fh.write(f'{"Group":<18}{"Members":>8}{"Complete%":>11}{"RMS(m)":>11}{"MAD(m)":>11}'
                         f'{"P95(m)":>11}{"Outlier%":>11}{"AccP95":>11}{"N":>8}\n')
                for row in plot_rows:
                    def number(name, width=11, precision=4):
                        value = row.get(name)
                        return f'{value:>{width}.{precision}f}' if value is not None else f'{"NA":>{width}}'
                    name = f"{row['sat_id']} {row['freq']}"
                    fh.write(f'{name:<18}{row.get("member_count", 1):>8}{row["completeness_percent"]:>11.2f}'
                             f'{number("residual_rms_m")}{number("residual_mad_m")}'
                             f'{number("residual_p95_m")}{row["outlier_rate_percent"]:>11.2f}'
                             f'{number("acceleration_p95_mps2")}{row["valid_residual_count"]:>8}\n')
                fh.write('\nSatellite-Frequency Detail:\n')
                fh.write(f'{"Satellite/Freq":<18}{"Complete%":>11}{"RMS(m)":>11}{"P95(m)":>11}{"Outlier%":>11}{"AccP95":>11}{"N":>8}\n')
                for row in rows:
                    name = f"{row['sat_id']} {row['freq']}"
                    def detail_number(key):
                        value = row.get(key)
                        return f'{value:>11.4f}' if value is not None else f'{"NA":>11}'
                    fh.write(f'{name:<18}{row["completeness_percent"]:>11.2f}'
                             f'{detail_number("residual_rms_m")}{detail_number("residual_p95_m")}'
                             f'{row["outlier_rate_percent"]:>11.2f}{detail_number("acceleration_p95_mps2")}'
                             f'{row["valid_residual_count"]:>8}\n')
                fh.write('\nCross-frequency Doppler consistency (m/s):\n')
                fh.write(f'{"Satellite/Pair":<24}{"N":>8}{"Bias":>12}{"RMS":>12}{"P95":>12}\n')
                for item in quality.get('cross_frequency', []):
                    name = f"{item['sat_id']} {item['freq1']}-{item['freq2']}"
                    fh.write(f'{name:<24}{item["count"]:>8}{item["bias_mps"]:>12.4f}'
                             f'{item["rms_mps"]:>12.4f}{item["p95_mps"]:>12.4f}\n')
            return {'figure': None, 'path': path, 'log': log_path, 'quality': quality}
        return {'figure': fig, 'path': None, 'log': None, 'quality': quality}

    def plot_observation_noise(self, observations: Dict[str, Any], save: bool = True,
                                output_dir: Optional[str] = None,
                                system_filters=None,
                                sat_filters=None,
                                freq_filters=None) -> Dict[str, Any]:
        """Plot pseudorange and carrier phase observation noise (third-order difference).

        Dual-mode layout (matching data_integrity):
          - >=2 systems: left=code noise bars, right=phase noise bars (system-freq x-axis)
          -  1 system:   grouped bars by satellite with colored frequencies

        Displays σ (sigma, STD/sqrt(20)) and RMS at bar tops and in title annotations.
        """
        from src.processing.calculator import MetricCalculator

        obs = observations.get('observations_meters', observations) if isinstance(observations, dict) else observations
        if not obs:
            raise ValueError('No observations available for noise analysis')

        # === pre-filter observations to respect all three filter types ===
        filtered_obs = {}
        for sat_id in sorted(obs.keys(), key=self._satellite_sort_key):
            if not self._selection_allows(system_filters, sat_id[0]) or not self._selection_allows(sat_filters, sat_id):
                continue
            filtered_freqs = {}
            for freq, data in obs[sat_id].items():
                if not self._frequency_sequence_accepts(sat_id, freq, system_filters, sat_filters, freq_filters):
                    continue
                filtered_freqs[freq] = data
            if filtered_freqs:
                filtered_obs[sat_id] = filtered_freqs

        if not filtered_obs:
            raise ValueError('No observations after filtering')

        # compute results using only filtered data
        mc = MetricCalculator()
        noise_results = mc.calculate_observation_noise({'observations_meters': filtered_obs})
        summary = noise_results.pop('summary', {})

        if not noise_results:
            raise ValueError('No valid noise data computed')

        import numpy as np
        system_names = {'G': 'GPS', 'R': 'GLO', 'E': 'Galileo', 'C': 'BDS', 'J': 'J-GPS', 'I': 'IRNSS'}
        system_order = {'GPS': 0, 'GLO': 1, 'Galileo': 2, 'BDS': 3, 'J-GPS': 4, 'IRNSS': 5}

        # === determine selected systems ===
        selected_system_codes = set()
        if system_filters:
            reverse_map = {v: k for k, v in system_names.items()}
            for item in system_filters:
                s = str(item)
                if len(s) == 1:
                    selected_system_codes.add(s)
                else:
                    uc = s.upper()
                    if uc in reverse_map:
                        selected_system_codes.add(reverse_map[uc])
                    else:
                        for full, code in reverse_map.items():
                            if full.upper().startswith(uc[:3]):
                                selected_system_codes.add(code)
                                break
        else:
            for sat_id in obs.keys():
                if sat_id:
                    selected_system_codes.add(sat_id[0])
        multi_system_mode = len(selected_system_codes) >= 2

        # === build summary items with filtering ===
        summary_items = []  # (system, freq, sigma_p, sigma_l, rms_p, rms_l, n_sats)
        for (system, freq), stats in summary.items():
            code = next((k for k, v in system_names.items() if v == system), system[:1])
            if code not in selected_system_codes and not multi_system_mode:
                continue
            summary_items.append((system, freq, stats['sigma_p'], stats['sigma_l'],
                                  stats.get('rms_p', stats['sigma_p'] * math.sqrt(20)),
                                  stats.get('rms_l', stats['sigma_l'] * math.sqrt(20)),
                                  stats['n_sats']))

        if not summary_items:
            raise ValueError('No noise summary data after filtering')

        system_order_map = {s: system_order.get(s, 99) for s in set(it[0] for it in summary_items)}

        # === plotting ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        if multi_system_mode:
            # sort by system_order then freq
            items_sorted = sorted(summary_items, key=lambda x: (system_order.get(x[0], 99), x[1]))
            labels = [f"{it[0][:3]} {it[1]}" for it in items_sorted]
            sigma_p_vals = [it[2] for it in items_sorted]
            sigma_l_vals = [it[3] for it in items_sorted]
            # use sigma_p / sigma_l from summary as unified estimate
            code_vals = [it[2] for it in items_sorted]
            phase_vals = [it[3] for it in items_sorted]

            x = np.arange(len(labels))
            width = 0.5
            rms_p_vals = [it[4] for it in items_sorted]
            rms_l_vals = [it[5] for it in items_sorted]

            bars_p = ax1.bar(x, sigma_p_vals, width=width, color='tomato', alpha=0.8)
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])
            ax1.set_ylabel('Code Noise σ (m)')
            ax1.set_title('Pseudorange Noise by Frequency Band')
            ax1.grid(True, axis='y', alpha=0.3)
            for bar, val in zip(bars_p, sigma_p_vals):
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, h * 1.008,
                        f"{val:.3f}",
                        ha='center', va='bottom', fontsize=plt.rcParams.get('font.size', 8))

            bars_l = ax2.bar(x, sigma_l_vals, width=width, color='steelblue', alpha=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])
            ax2.set_ylabel('Phase Noise σ (m)')
            ax2.set_title('Carrier Phase Noise by Frequency Band')
            ax2.grid(True, axis='y', alpha=0.3)
            for bar, val in zip(bars_l, sigma_l_vals):
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, h * 1.008,
                        f"{val:.4f}",
                        ha='center', va='bottom', fontsize=plt.rcParams.get('font.size', 8))

        else:
            # Single system — grouped bars per satellite with colored frequencies
            single_code = next(iter(selected_system_codes)) if selected_system_codes else None
            system_name = system_names.get(single_code, single_code)

            # collect per-sat-per-freq noise values
            sat_freqs = {}  # sat -> {freq: {sigma_p, sigma_l, ...}}
            freq_set = []
            for sat_id, freq_map in noise_results.items():
                if single_code and sat_id[0] != single_code:
                    continue
                sat_freqs[sat_id] = {}
                for freq, stats in freq_map.items():
                    sat_freqs[sat_id][freq] = stats
                    if freq not in freq_set:
                        freq_set.append(freq)

            freq_set = sorted(freq_set)
            sats = sorted(sat_freqs.keys(), key=self._satellite_sort_key)

            if not sats:
                ax1.text(0.5, 0.5, 'No data for selected system', ha='center', transform=ax1.transAxes)
                ax2.text(0.5, 0.5, 'No data for selected system', ha='center', transform=ax2.transAxes)
            else:
                n_freq = len(freq_set)
                x = np.arange(len(sats))
                total_width = 0.8
                bar_width = total_width / max(n_freq, 1)
                colors = plt.cm.tab10(np.linspace(0, 1, n_freq))

                for i, freq in enumerate(freq_set):
                    vals_p = []
                    vals_l = []
                    for sat in sats:
                        info = sat_freqs.get(sat, {}).get(freq, None)
                        vals_p.append(info['sigma_p'] if info else 0.0)
                        vals_l.append(info['sigma_l'] if info else 0.0)
                    offsets = x - total_width / 2 + i * bar_width + bar_width / 2

                    ax1.bar(offsets, vals_p, width=bar_width * 0.9, color=colors[i], label=freq)
                    ax2.bar(offsets, vals_l, width=bar_width * 0.9, color=colors[i], label=freq)

                # attach metadata to each bar for interactivity
                n_sats = len(sats)
                for bar_idx in range(min(len(ax1.patches), len(ax2.patches))):
                    fi = bar_idx // n_sats  # frequency index
                    si = bar_idx % n_sats   # satellite index
                    if si < n_sats and fi < len(freq_set):
                        sat = sats[si]
                        freq_name = freq_set[fi]
                        info = sat_freqs.get(sat, {}).get(freq_name, None)
                        if info:
                            ax1.patches[bar_idx]._sat = sat
                            ax1.patches[bar_idx]._freq = freq_name
                            ax1.patches[bar_idx]._info = info
                            ax2.patches[bar_idx]._sat = sat
                            ax2.patches[bar_idx]._freq = freq_name
                            ax2.patches[bar_idx]._info = info

                ax1.set_xticks(x)
                ax1.set_xticklabels(sats, rotation=45, ha='right')
                ax1.set_ylabel('Code Noise σ (m)')
                ax1.set_title(f'Pseudorange Noise ({single_code})')
                ax1.legend(title='Freq', fontsize=8)
                ax1.grid(True, axis='y', alpha=0.3)

                ax2.set_xticks(x)
                ax2.set_xticklabels(sats, rotation=45, ha='right')
                ax2.set_ylabel('Phase Noise σ (m)')
                ax2.set_title(f'Carrier Phase Noise ({single_code})')
                ax2.legend(title='Freq', fontsize=8)
                ax2.grid(True, axis='y', alpha=0.3)

        fig.tight_layout()

        # interactive tooltip for single-system mode (too many bars for static text)
        if MPLCURSORS_AVAILABLE and not save and not multi_system_mode:
            for ax, key, label in [(ax1, 'sigma_p', 'Code σ'), (ax2, 'sigma_l', 'Phase σ')]:
                cursor = mplcursors.cursor(ax.patches, hover=False)

                def _make_on_add(k, lbl):
                    def _on_add(sel):
                        patch = sel.artist
                        sat = getattr(patch, '_sat', '?')
                        freq = getattr(patch, '_freq', '?')
                        info = getattr(patch, '_info', None)
                        if info:
                            val = info.get(k, 0)
                            sel.annotation.set(
                                text=f'{sat} {freq}\n{lbl}: {val:.6f} m',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)
                            )
                        else:
                            sel.annotation.set(
                                text=f'{sat} {freq}\n{lbl}: {patch.get_height():.4f} m',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8)
                            )
                    return _on_add

                cursor.connect('add', _make_on_add(key, label))

        # === log ===
        log_path = None
        if save:
            path = self._save_fig(fig, 'observation_noise', output_dir)
            out_dir = self._ensure_output_dir(output_dir)
            log_path = os.path.join(out_dir, 'observation_noise_log.txt')
            with open(log_path, 'w', encoding='utf-8') as fh:
                fh.write('Observation Noise Report (Third-Order Difference Method)\n')
                fh.write(f'Generated: {datetime.datetime.utcnow().isoformat()}Z\n\n')
                fh.write('System-Frequency Summary (σ = RMS(Δ)/√20):\n')
                fh.write(f'{"System-Freq":<14}{"Sats":>6}{"Code σ (m)":>14}{"Phase σ (m)":>16}\n')
                items_sorted = sorted(summary_items, key=lambda x: (system_order.get(x[0], 99), x[1]))
                for system, freq, sp, sl, rp, rl, ns in items_sorted:
                    fh.write(f'{system[:3] + " " + freq:<14}{ns:>6}{sp:>14.4f}{sl:>16.6f}\n')
                fh.write('\nNote: σ = RMS(Δ)/√20, where Δ is the third-order difference\n')
            return {'figure': None, 'path': path, 'log': log_path}
        return {'figure': fig, 'path': None, 'log': None}


