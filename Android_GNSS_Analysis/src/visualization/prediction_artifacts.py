"""Auditable artifacts for Doppler-filled carrier-phase observations."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np


def _json_default(value):
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _sat_key(sat_id: str):
    order = {'G': 0, 'R': 1, 'E': 2, 'C': 3, 'J': 4, 'I': 5}
    try:
        prn = int(sat_id[1:])
    except (TypeError, ValueError):
        prn = 0
    return order.get(sat_id[:1], 99), prn, sat_id


def _active_lli(value) -> bool:
    try:
        return value is not None and (int(value) & 1) != 0
    except (TypeError, ValueError):
        return False


def _prediction_series(results: Dict[str, Any], sat_id: str, freq: str):
    return results.get('predicted_phases', {}).get(sat_id, {}).get(freq, {})


def calculate_integrity_rows(observations: Dict[str, Any], results: Dict[str, Any]):
    """Return separate code/phase integrity using each recorded sat/freq series as denominator."""
    rows = []
    for sat_id in sorted(observations, key=_sat_key):
        for freq in sorted(observations[sat_id]):
            data = observations[sat_id][freq]
            times = data.get('times', []) or []
            if not times:
                continue
            code = data.get('code', []) or []
            original_phase = data.get('phase', []) or []
            pred = _prediction_series(results, sat_id, freq)
            filled_phase = pred.get('filled_phase_m', original_phase) or []
            expected = len(times)
            code_count = sum(i < len(code) and code[i] is not None for i in range(expected))
            before_count = sum(i < len(original_phase) and original_phase[i] is not None for i in range(expected))
            after_count = sum(i < len(filled_phase) and filled_phase[i] is not None for i in range(expected))
            if code_count == 0 and before_count == 0 and after_count == 0:
                continue
            rows.append({
                'system': sat_id[:1], 'satellite': sat_id, 'frequency': freq,
                'expected_epochs': expected,
                'code_count': code_count, 'code_rate': code_count / expected,
                'phase_before_count': before_count, 'phase_before_rate': before_count / expected,
                'phase_after_count': after_count, 'phase_after_rate': after_count / expected,
            })
    return rows


def calculate_multistep_prediction_errors(observations: Dict[str, Any], max_length: int = 20,
                                           max_interval_factor: float = 1.5,
                                           max_anchor_samples_per_series: int = 50):
    """Mask real continuous phase samples and compare Doppler predictions with truth."""
    samples = []
    for sat_id in sorted(observations, key=_sat_key):
        for freq, data in sorted(observations[sat_id].items()):
            times = data.get('times', []) or []
            phase = data.get('phase', []) or []
            doppler = data.get('doppler', []) or []
            lli = data.get('phase_lli', []) or []
            if min(len(times), len(phase), len(doppler)) < 2:
                continue
            intervals = []
            for idx in range(1, len(times)):
                try:
                    dt = (times[idx] - times[idx - 1]).total_seconds()
                except Exception:
                    continue
                if dt > 0:
                    intervals.append(dt)
            if not intervals:
                continue
            nominal = float(np.median(intervals))
            # Build phase-continuous arcs from the real measurements.  Smartphone
            # RINEX LLI is often incomplete, so also reject edges whose measured
            # phase change strongly disagrees with integrated Doppler.
            edge_residuals = []
            raw_edges = {}
            for idx in range(1, len(times)):
                if phase[idx - 1] is None or phase[idx] is None or doppler[idx - 1] is None or doppler[idx] is None:
                    continue
                try:
                    dt = (times[idx] - times[idx - 1]).total_seconds()
                except Exception:
                    continue
                if dt <= 0 or dt > nominal * max_interval_factor:
                    continue
                residual = (phase[idx] - phase[idx - 1]) - 0.5 * (doppler[idx - 1] + doppler[idx]) * dt
                raw_edges[idx] = residual
                edge_residuals.append(residual)
            if not edge_residuals:
                continue
            residual_array = np.asarray(edge_residuals, dtype=float)
            residual_median = float(np.median(residual_array))
            residual_mad = float(np.median(np.abs(residual_array - residual_median)))
            edge_threshold = max(0.5, 4.0 * 1.4826 * residual_mad)
            valid_edges = {
                idx for idx, residual in raw_edges.items()
                if abs(residual - residual_median) <= edge_threshold
            }
            anchor_step = max(1, int(math.ceil((len(times) - 1) / max_anchor_samples_per_series)))
            for start in range(0, len(times) - 1, anchor_step):
                if phase[start] is None or doppler[start] is None or _active_lli(lli[start] if start < len(lli) else None):
                    continue
                predicted = float(phase[start])
                for length in range(1, min(max_length, len(times) - start - 1) + 1):
                    idx = start + length
                    if doppler[idx] is None or phase[idx] is None:
                        break
                    if idx not in valid_edges:
                        break
                    if _active_lli(lli[idx - 1] if idx - 1 < len(lli) else None):
                        break
                    try:
                        dt = (times[idx] - times[idx - 1]).total_seconds()
                    except Exception:
                        break
                    if dt <= 0 or dt > nominal * max_interval_factor:
                        break
                    predicted += 0.5 * (doppler[idx - 1] + doppler[idx]) * dt
                    samples.append({
                        'satellite': sat_id, 'frequency': freq, 'prediction_length': length,
                        'anchor_time': times[start], 'target_time': times[idx],
                        'predicted_phase_m': predicted, 'actual_phase_m': phase[idx],
                        'error_m': float(phase[idx] - predicted),
                        'absolute_error_m': abs(float(phase[idx] - predicted)),
                    })
    return samples


def _write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: List[str]):
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_default(value) if hasattr(value, 'isoformat') else value for key, value in row.items()})


def _plot_phase_timeline(observations, results, after: bool, path: str):
    groups = defaultdict(list)
    for sat_id, freqs in observations.items():
        for freq, data in freqs.items():
            phase = (_prediction_series(results, sat_id, freq).get('filled_phase_m')
                     if after else data.get('phase'))
            if phase and any(value is not None for value in phase):
                groups[(sat_id[:1], freq)].append((sat_id, data.get('times', []), phase))
    keys = sorted(groups, key=lambda item: ({'G': 0, 'R': 1, 'E': 2, 'C': 3}.get(item[0], 9), item[1]))
    if not keys:
        return None
    fig = Figure(figsize=(max(8, 2.4 * len(keys)), 6))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(keys), squeeze=False, sharex=True)
    for ax, key in zip(axes[0], keys):
        sats = sorted(groups[key], key=lambda item: _sat_key(item[0]))
        for y, (sat_id, times, phase) in enumerate(sats):
            valid_times = [times[i] for i in range(min(len(times), len(phase))) if phase[i] is not None]
            if valid_times:
                display_step = max(1, int(math.ceil(len(valid_times) / 2000)))
                valid_times = valid_times[::display_step]
                ax.scatter(valid_times, [y] * len(valid_times), marker='|', s=20, color='#3558a8')
        ax.set_title(f'{key[0]} {key[1]}')
        ax.set_yticks(range(len(sats)))
        ax.set_yticklabels([item[0] for item in sats], fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='x', alpha=0.15)
    fig.suptitle('Carrier phase availability after Doppler prediction' if after else 'Carrier phase availability before Doppler prediction')
    fig.supxlabel('Observation time (HH:MM)')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    fig.clear()
    return path


def _aggregate_integrity(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['system'], row['frequency'])].append(row)
    summary = []
    for (system, freq), items in sorted(grouped.items()):
        total_expected = sum(item['expected_epochs'] for item in items)
        entry = {'system': system, 'frequency': freq, 'satellite_frequency_count': len(items)}
        for count_key, rate_key in (
                ('code_count', 'code_rate'), ('phase_before_count', 'phase_before_rate'),
                ('phase_after_count', 'phase_after_rate')):
            entry[rate_key] = sum(item[count_key] for item in items) / total_expected if total_expected else 0.0
            entry[f'{rate_key}_rms'] = math.sqrt(sum(item[rate_key] ** 2 for item in items) / len(items))
        summary.append(entry)
    return summary


def _plot_integrity_summary(summary, path):
    labels = [f"{row['system']} {row['frequency']}" for row in summary]
    x = np.arange(len(labels))
    width = 0.34
    before_color = '#ef2929'
    after_color = '#3f5aa9'
    fig = Figure(figsize=(max(9, len(labels) * 1.1), 5.5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    code_rates = [100 * row['code_rate'] for row in summary]
    phase_before_rates = [100 * row['phase_before_rate'] for row in summary]
    phase_after_rates = [100 * row['phase_after_rate'] for row in summary]

    # Match the paper's encoding: the full solid bar is code integrity, while
    # the hatched overlay (not an added stack) reaches the phase integrity.
    ax.bar(x - width / 2, code_rates, width, color=before_color,
           edgecolor='#222222', linewidth=0.6)
    ax.bar(x + width / 2, code_rates, width, color=after_color,
           edgecolor='#222222', linewidth=0.6)
    ax.bar(x - width / 2, phase_before_rates, width, color=before_color,
           edgecolor='white', linewidth=0.8, hatch='///')
    ax.bar(x + width / 2, phase_after_rates, width, color=after_color,
           edgecolor='white', linewidth=0.8, hatch='///')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Integrity rate (%)')
    ax.set_title('Code and phase integrity before and after Doppler phase prediction')
    ax.grid(True, axis='y', alpha=0.25)
    fig.legend(handles=[
        Patch(facecolor=before_color, edgecolor='#222222', label='Before prediction'),
        Patch(facecolor=after_color, edgecolor='#222222', label='After prediction'),
        Patch(facecolor='#888888', edgecolor='#222222', label='Code: full bar height'),
        Patch(facecolor='#888888', edgecolor='white', hatch='///', label='Phase: hatched height'),
    ], loc='upper center', bbox_to_anchor=(0.5, 0.935), ncol=4,
       frameon=True, title='Scenario and bar encoding')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    fig.savefig(path, dpi=180, bbox_inches='tight')
    fig.clear()
    return path


def _plot_prediction_error_rms(samples, path):
    grouped = defaultdict(list)
    for row in samples:
        grouped[(row['satellite'], row['frequency'], row['prediction_length'])].append(row['error_m'])
    series = defaultdict(list)
    for (sat, freq, length), values in grouped.items():
        rms = math.sqrt(sum(value * value for value in values) / len(values))
        series[(sat, freq)].append((length, rms, len(values)))
    panels = sorted({(sat[:1], freq) for sat, freq in series},
                    key=lambda item: ({'G': 0, 'R': 1, 'E': 2, 'C': 3, 'J': 4}.get(item[0], 9), item[1]))
    ncols = min(4, max(1, len(panels)))
    nrows = int(math.ceil(len(panels) / ncols))
    fig = Figure(figsize=(4.2 * ncols, 3.2 * nrows))
    FigureCanvasAgg(fig)
    # Prediction accuracy can differ greatly between signal frequencies.  A
    # shared y-axis lets one degraded frequency flatten every healthy panel,
    # so keep only the prediction-length x-axis shared.
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=False)
    for ax, panel in zip(axes.flat, panels):
        panel_series = [item for item in series.items() if item[0][0][:1] == panel[0] and item[0][1] == panel[1]]
        panel_max = 0.0
        for (sat, freq), values in sorted(panel_series, key=lambda item: _sat_key(item[0][0])):
            values.sort()
            panel_max = max(panel_max, max((v[1] for v in values), default=0.0))
            ax.plot([v[0] for v in values], [v[1] for v in values], marker='o', markersize=2.2,
                    linewidth=0.8, label=sat)
        # RMS is non-negative.  Give each frequency its own readable range and
        # retain a small headroom for line markers and legends.
        ax.set_ylim(0.0, panel_max * 1.08 if panel_max > 0.0 else 1.0)
        ax.set_title(f'{panel[0]} {panel[1]}')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=5 if len(panel_series) > 12 else 6,
                  ncol=3 if len(panel_series) > 12 else 2,
                  loc='upper left')
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    # With sharex=True Matplotlib normally labels only the physical last row.
    # A partially filled grid needs the last *visible* axis in every column to
    # carry x tick labels instead.
    for column in range(ncols):
        visible_in_column = [
            axes[row, column] for row in range(nrows)
            if row * ncols + column < len(panels)
        ]
        if visible_in_column:
            visible_in_column[-1].tick_params(axis='x', which='both', labelbottom=True)

    fig.suptitle('Doppler phase prediction RMS evaluated on measured carrier phase')
    fig.supxlabel('Prediction length (epochs)', y=0.012)
    # Keep the global y label to the left of all tick labels and reserve a
    # matching margin so it cannot overlap values such as 0.50/0.25.
    fig.supylabel('RMS prediction error (m)', x=0.006)
    fig.tight_layout(rect=(0.035, 0.035, 1.0, 0.97))
    fig.savefig(path, dpi=180, bbox_inches='tight')
    fig.clear()
    return path


def _prediction_error_summary(samples):
    grouped = defaultdict(list)
    for row in samples:
        grouped[(row['satellite'], row['frequency'], row['prediction_length'])].append(row['error_m'])
    rows = []
    for (satellite, frequency, length), values in sorted(grouped.items(), key=lambda item: (_sat_key(item[0][0]), item[0][1], item[0][2])):
        array = np.asarray(values, dtype=float)
        rows.append({
            'satellite': satellite,
            'frequency': frequency,
            'prediction_length': length,
            'sample_count': len(values),
            'bias_m': float(np.mean(array)),
            'rms_m': float(np.sqrt(np.mean(array ** 2))),
            'mae_m': float(np.mean(np.abs(array))),
            'p95_abs_m': float(np.percentile(np.abs(array), 95)),
        })
    return rows


def save_prediction_artifacts(observations: Dict[str, Any], results: Dict[str, Any], output_dir: str,
                              base_name: str = 'doppler_prediction', error_max_length: int = 20):
    os.makedirs(output_dir, exist_ok=True)
    metadata_json = os.path.join(output_dir, f'{base_name}.prediction.json')
    metadata_csv = os.path.join(output_dir, f'{base_name}.prediction.csv')
    with open(metadata_json, 'w', encoding='utf-8') as stream:
        json.dump({'schema_version': 1, 'parameters': results.get('parameters', {}),
                   'records': results.get('metadata_records', [])}, stream,
                  ensure_ascii=False, indent=2, default=_json_default)
    metadata_fields = ['time', 'sat_id', 'freq', 'epoch_idx', 'prediction_length', 'sigma_m',
                       'source', 'accepted', 'status', 'predicted_phase_cycle', 'predicted_phase_m']
    _write_csv(metadata_csv, results.get('metadata_records', []), metadata_fields)

    integrity_rows = calculate_integrity_rows(observations, results)
    integrity_summary = _aggregate_integrity(integrity_rows)
    integrity_csv = os.path.join(output_dir, f'{base_name}.integrity_by_satellite.csv')
    summary_csv = os.path.join(output_dir, f'{base_name}.integrity_summary_rms.csv')
    _write_csv(integrity_csv, integrity_rows, list(integrity_rows[0].keys()) if integrity_rows else [])
    _write_csv(summary_csv, integrity_summary, list(integrity_summary[0].keys()) if integrity_summary else [])

    errors = calculate_multistep_prediction_errors(
        observations, max_length=error_max_length,
        max_interval_factor=results.get('parameters', {}).get('max_interval_factor', 1.5),
    )
    errors_csv = os.path.join(output_dir, f'{base_name}.prediction_error_samples.csv')
    error_fields = ['satellite', 'frequency', 'prediction_length', 'anchor_time', 'target_time',
                    'predicted_phase_m', 'actual_phase_m', 'error_m', 'absolute_error_m']
    _write_csv(errors_csv, errors, error_fields)
    error_summary = _prediction_error_summary(errors)
    error_summary_csv = os.path.join(output_dir, f'{base_name}.prediction_error_rms.csv')
    error_summary_fields = ['satellite', 'frequency', 'prediction_length', 'sample_count',
                            'bias_m', 'rms_m', 'mae_m', 'p95_abs_m']
    _write_csv(error_summary_csv, error_summary, error_summary_fields)

    paths = {
        'metadata_json': metadata_json, 'metadata_csv': metadata_csv,
        'integrity_by_satellite_csv': integrity_csv, 'integrity_summary_rms_csv': summary_csv,
        'prediction_error_samples_csv': errors_csv,
        'prediction_error_rms_csv': error_summary_csv,
        'phase_integrity_before_png': _plot_phase_timeline(
            observations, results, False, os.path.join(output_dir, 'phase_integrity_before.png')),
        'phase_integrity_after_png': _plot_phase_timeline(
            observations, results, True, os.path.join(output_dir, 'phase_integrity_after.png')),
        'observation_integrity_comparison_png': _plot_integrity_summary(
            integrity_summary, os.path.join(output_dir, 'observation_integrity_before_after.png')) if integrity_summary else None,
        'prediction_error_rms_png': _plot_prediction_error_rms(
            errors, os.path.join(output_dir, 'prediction_error_rms_by_length.png')) if errors else None,
    }
    return paths
