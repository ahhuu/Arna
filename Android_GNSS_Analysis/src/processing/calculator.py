from typing import Dict, Any, List, Tuple
import statistics
import math
from src.core.config import GNSS_FREQUENCIES, SPEED_OF_LIGHT
from src.processing.advanced_algo import CoreAlgorithmProcessor


class MetricCalculator:
    """Compute intermediate metrics without modifying original data."""
    
    def __init__(self):
        self.processor = CoreAlgorithmProcessor()

    def calculate_isb(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate ISB analysis using CoreAlgorithmProcessor.
        
        Expects data to contain:
            - observations_meters: phone data
            - receiver_observations: receiver data
            - epochs: list of epochs (optional, for time range info)
        """
        obs = data.get('observations_meters', {})
        rx_obs = data.get('receiver_observations', {})
        
        # 1. Prepare data
        isb_data = self.processor.run_prepare_isb_data(obs, rx_obs)
        if not isb_data.get('common_times'):
             return {'error': 'No common time points found'}

        # 2. Select reference satellite
        ref_sat = self.processor.run_select_reference_satellite(isb_data)
        if not ref_sat:
            return {'error': 'No suitable reference satellite found'}
            
        # 3. Filter stable satellites
        stable_sats = self.processor.run_filter_stable_satellites(isb_data)
        
        # 4. Calculate Double Differences & ISB
        results = self.processor.run_calculate_isb_double_difference(isb_data, ref_sat, stable_sats)
        
        # Add metadata to results
        results['reference_satellite'] = ref_sat
        results['stable_satellites'] = stable_sats
        return results

    def calculate_derivatives(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Port of analyzer.calculate_observable_derivatives.

        Expects:
            data['epochs']: list of epochs where each epoch has {'time': ..., 'satellites': {sat_id: {obs_type: value}}}
            data['frequencies']: {system: [freqs...]}
            data['wavelengths']: {system: {freq: wavelength_m}}
        Returns:
            {sat_id: {freq: {'times': [...], 'pr_derivative': [...], 'ph_derivative': [...], 'doppler': [...]}}}
        """
        derivatives: Dict[str, Dict[str, Any]] = {}
        epochs = data.get('epochs', [])
        frequencies = data.get('frequencies', {})
        wavelengths = data.get('wavelengths', {})

        if not epochs:
            return derivatives

        # satellites present in first epoch
        first_epoch = epochs[0]
        sat_ids = list(first_epoch.get('satellites', {}).keys())

        for sat_id in sat_ids:
            system = sat_id[0] if sat_id else ''
            available_freqs = frequencies.get(system, {})
            freq_derivatives = {}

            # initialize
            for freq in available_freqs:
                freq_derivatives[freq] = {'times': [], 'pr_derivative': [], 'ph_derivative': [], 'doppler': []}

            # collect time series
            for epoch in epochs:
                time = epoch.get('time')
                sat_obs = epoch.get('satellites', {}).get(sat_id, {})
                for freq in available_freqs:
                    code_key = f'C{freq[1:]}'
                    phase_key = f'L{freq[1:]}'
                    doppler_key = f'D{freq[1:]}'
                    code_val = sat_obs.get(code_key)
                    phase_val = sat_obs.get(phase_key)
                    doppler_val = sat_obs.get(doppler_key)
                    if code_val is not None or phase_val is not None or doppler_val is not None:
                        freq_derivatives[freq]['times'].append(time)
                        freq_derivatives[freq]['pr_derivative'].append(code_val)
                        freq_derivatives[freq]['ph_derivative'].append(phase_val)
                        freq_derivatives[freq]['doppler'].append(doppler_val)

            # compute derivatives per freq
            for freq in list(freq_derivatives.keys()):
                times = freq_derivatives[freq]['times']
                pr_vals = freq_derivatives[freq]['pr_derivative']
                ph_vals = freq_derivatives[freq]['ph_derivative']
                dop_vals = freq_derivatives[freq]['doppler']
                wavelength = wavelengths.get(system, {}).get(freq)

                if wavelength is None or not times:
                    # nothing to compute
                    freq_derivatives[freq]['times'] = []
                    freq_derivatives[freq]['pr_derivative'] = []
                    freq_derivatives[freq]['ph_derivative'] = []
                    freq_derivatives[freq]['doppler'] = []
                    continue

                pr_d = []
                ph_d = []
                # compute pr and ph derivatives (length len(times)-1)
                for i in range(1, len(times)):
                    dt = (times[i] - times[i - 1]).total_seconds()
                    if dt > 0 and pr_vals[i] is not None and pr_vals[i - 1] is not None:
                        pr_d.append((pr_vals[i] - pr_vals[i - 1]) / dt)
                    else:
                        pr_d.append(None)

                    if dt > 0 and ph_vals[i] is not None and ph_vals[i - 1] is not None:
                        cycle_rate = (ph_vals[i] - ph_vals[i - 1]) / dt
                        ph_d.append(cycle_rate * wavelength)
                    else:
                        ph_d.append(None)

                dop_m = []
                for d in dop_vals:
                    if d is not None:
                        dop_m.append(-d * wavelength)
                    else:
                        dop_m.append(None)

                # align times to derivative (remove first timestamp)
                freq_derivatives[freq]['times'] = times[1:]
                freq_derivatives[freq]['pr_derivative'] = pr_d
                freq_derivatives[freq]['ph_derivative'] = ph_d
                freq_derivatives[freq]['doppler'] = dop_m[1:] if len(dop_m) > 1 else dop_m

            derivatives[sat_id] = freq_derivatives

        return derivatives

    def calculate_code_phase_differences(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Port of analyzer.calculate_code_phase_differences.

        Expects data to contain 'observations_meters' and optionally 'phase_stagnation'.
        Returns structure: {sat_id: {freq: {times, code_phase_diff, diff_changes, ...}}}
        """
        source = data.get('observations_meters', {})
        phase_stagnation = data.get('phase_stagnation', {})

        differences: Dict[str, Any] = {}

        for sat_id, freq_data in source.items():
            freq_differences: Dict[str, Any] = {}
            sat_stag = phase_stagnation.get(sat_id, {}) if phase_stagnation else {}

            for freq, obs in freq_data.items():
                times = obs.get('times', [])
                code_values = obs.get('code', [])
                phase_values = obs.get('phase', [])

                stagnant_epochs = sat_stag.get(freq, {}).get('stagnant_epochs', []) if sat_stag else []

                freq_differences[freq] = {
                    'times': [],
                    'code_phase_diff': [],
                    'diff_changes': [],
                    'epoch_indices': [],  # 新增：记录原始历元索引
                    'original_epochs': len(times),
                    'filtered_epochs': 0,
                    'stagnant_epochs_removed': len(stagnant_epochs),
                    'missing_epochs': 0
                }

                prev_diff = None
                missing_obs = 0
                for i in range(len(times)):
                    if i in stagnant_epochs or code_values[i] is None or phase_values[i] is None:
                        if code_values[i] is None or phase_values[i] is None:
                            missing_obs += 1
                        continue
                    diff = code_values[i] - phase_values[i]
                    freq_differences[freq]['times'].append(times[i])
                    freq_differences[freq]['code_phase_diff'].append(diff)
                    freq_differences[freq]['epoch_indices'].append(i)  # 新增：记录原始索引
                    if prev_diff is not None:
                        freq_differences[freq]['diff_changes'].append(abs(diff - prev_diff))
                    else:
                        freq_differences[freq]['diff_changes'].append(None)
                    prev_diff = diff

                freq_differences[freq]['filtered_epochs'] = len(freq_differences[freq]['code_phase_diff'])
                freq_differences[freq]['missing_epochs'] = missing_obs

            differences[sat_id] = freq_differences

        return differences

    def calculate_diff_variations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics for diff_changes produced by calculate_raw_diffs."""
        raw = data.get('code_phase_differences', {})
        summary: Dict[str, Any] = {}

        for sat_id, freq_data in raw.items():
            summary[sat_id] = {}
            for freq, d in freq_data.items():
                changes = [c for c in d.get('diff_changes', []) if c is not None]
                if changes:
                    mean = statistics.mean(changes)
                    stdev = statistics.stdev(changes) if len(changes) > 1 else 0.0
                    mn = min(changes)
                    mx = max(changes)
                else:
                    mean = stdev = mn = mx = 0.0
                summary[sat_id][freq] = {
                    'count': len(changes),
                    'mean_change': mean,
                    'std_change': stdev,
                    'min_change': mn,
                    'max_change': mx,
                    'raw': d
                }
        return summary

    def calculate_phase_prediction_errors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Port of analyzer.calculate_phase_prediction_errors.

        Expects data to contain 'observations_meters', 'frequencies', 'wavelengths'.
        Returns dict of prediction errors per sat/freq.
        """
        observations = data.get('observations_meters', {})
        frequencies = data.get('frequencies', {})
        wavelengths = data.get('wavelengths', {})

        errors: Dict[str, Any] = {}

        for sat_id, freq_data in observations.items():
            freq_errors: Dict[str, Any] = {}
            for freq, obs in freq_data.items():
                times = obs.get('times', [])
                phase_cycles = obs.get('phase_cycle', [])
                doppler_mps = obs.get('doppler', [])
                system = sat_id[0] if sat_id else ''

                # tolerate list-based frequency collections from UI
                freq_by_system = frequencies.get(system, {}) if isinstance(frequencies, dict) else {}
                if isinstance(freq_by_system, dict):
                    frequency = freq_by_system.get(freq)
                else:
                    frequency = None

                wavelength = wavelengths.get(system, {}).get(freq) if isinstance(wavelengths, dict) else None
                if frequency is None:
                    # fallback to config or derive from wavelength
                    frequency = GNSS_FREQUENCIES.get(system, {}).get(freq)
                if frequency is None and wavelength:
                    frequency = SPEED_OF_LIGHT / wavelength

                freq_errors[freq] = {
                    'times': [],
                    'actual_phase': [],
                    'predicted_phase': [],
                    'prediction_error': [],
                    'doppler_mps': [],
                    'doppler_hz': [],
                    'epoch_indices': []  # 新增：记录原始历元索引
                }

                for i in range(1, len(times)):
                    if (phase_cycles[i - 1] is not None and doppler_mps[i - 1] is not None and i < len(doppler_mps)
                            and doppler_mps[i] is not None and phase_cycles[i] is not None
                            and wavelength is not None):
                        dt = (times[i] - times[i - 1]).total_seconds()
                        doppler_now_hz = doppler_mps[i] / wavelength
                        doppler_old_hz = doppler_mps[i - 1] / wavelength
                        doppler_arith = (doppler_now_hz + doppler_old_hz) / 2
                        # RINEX D is converted by the reader to phase range-rate
                        # (-D * wavelength, m/s).  Integrating it gives phase change.
                        phase_change = dt * doppler_arith
                        predicted_phase = phase_cycles[i - 1] + phase_change
                        error = (phase_cycles[i] - predicted_phase) * wavelength
                        freq_errors[freq]['times'].append(times[i])
                        freq_errors[freq]['actual_phase'].append(phase_cycles[i])
                        freq_errors[freq]['predicted_phase'].append(predicted_phase)
                        freq_errors[freq]['prediction_error'].append(error)
                        freq_errors[freq]['doppler_mps'].append((doppler_mps[i - 1] + doppler_mps[i]) / 2)
                        freq_errors[freq]['doppler_hz'].append(doppler_arith)
                        freq_errors[freq]['epoch_indices'].append(i)  # 新增：记录原始历元索引

                if freq_errors[freq]['times']:
                    freq_errors[freq]['total_errors'] = len(freq_errors[freq]['prediction_error'])
                else:
                    freq_errors[freq]['total_errors'] = 0

            errors[sat_id] = freq_errors

        return errors

    def calculate_receiver_cmc(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Based on analyzer.calculate_receiver_cmc

        Expects data['receiver_observations'] in the same structure as the original analyzer.
        Returns: {sat_id: {freq: {'times': [...], 'cmc_m': [...]}}}
        """
        source = data.get('receiver_observations', {})
        target_freqs = {
            'G': ['L1C', 'L5Q'],
            'R': ['L1C'],
            'E': ['L1C', 'L5Q', 'L7Q'],
            'C': ['L2I', 'L1P', 'L5P']
        }

        cmc_results: Dict = {}
        for sat_id, freq_data in source.items():
            system = sat_id[0] if sat_id else ''
            allowed = set(target_freqs.get(system, []))
            freq_out = {}
            for freq, obs in freq_data.items():
                if freq not in allowed:
                    continue
                times = obs.get('times', [])
                code_m = obs.get('code', [])
                phase_m = obs.get('phase', [])
                cmc_vals = []
                out_times = []
                for t, c, p in zip(times, code_m, phase_m):
                    if c is None or p is None:
                        continue
                    cmc_vals.append(c - p)
                    out_times.append(t)
                if cmc_vals:
                    freq_out[freq] = {'times': out_times, 'cmc_m': cmc_vals}
            if freq_out:
                cmc_results[sat_id] = freq_out

        return cmc_results

    @staticmethod
    def _is_active_lli(value) -> bool:
        """检查 LLI 的位0是否为1（真正的锁定丢失/周跳）"""
        try:
            raw = int(value)
            return (raw & 1) != 0  # 只检查位0：锁定丢失
        except Exception:
            return False

    def _select_pseudorange_multipath_pair(self, system: str, sat_data: Dict[str, Any],
                                           freq_pair: Tuple[str, str] = None):
        available_freqs = list(sat_data.keys())
        if len(available_freqs) < 2:
            return None

        # 辅助函数：检查频率是否有有效的 code 数据
        def has_valid_code(freq_name):
            freq_data = sat_data.get(freq_name, {})
            code_vals = freq_data.get('code', []) if 'code' in freq_data else []
            return any(v is not None for v in code_vals)

        if freq_pair is not None:
            req_f1, req_f2 = freq_pair
            if req_f1 in sat_data and req_f2 in sat_data:
                return req_f1, req_f2

        if system in self.IF_FREQ_PAIRS:
            for f1_name, f2_name in self.IF_FREQ_PAIRS[system]:
                if f1_name in sat_data and f2_name in sat_data:
                    if has_valid_code(f1_name) and has_valid_code(f2_name):
                        return f1_name, f2_name

        sys_freqs = GNSS_FREQUENCIES.get(system, {}) if system else {}
        best_pair = None
        best_sep = -1.0
        for i in range(len(available_freqs)):
            for j in range(i + 1, len(available_freqs)):
                a = available_freqs[i]
                b = available_freqs[j]
                fa = sys_freqs.get(a)
                fb = sys_freqs.get(b)
                if not fa or not fb:
                    continue
                # 必须都有有效的 code
                if not (has_valid_code(a) and has_valid_code(b)):
                    continue
                sep = abs(fa - fb)
                if sep > best_sep:
                    best_sep = sep
                    best_pair = (a, b)
        if best_pair and best_sep >= 1e6:
            return best_pair
        return None

    def select_default_pseudorange_multipath_pair(self, system: str, sat_data: Dict[str, Any]):
        """Select a preferred PMP frequency pair for overview plots.

        Requested fallback order:
        - GPS (G): L1C+L5Q
        - BDS (C): L1P+L5P -> L2I+L5P -> L2I+L1P
        - Galileo (E): L1C+L5Q -> L1C+L7Q -> L5Q+L7Q
        - J: L1C+L5Q

        Returns:
            (freq_pair, rule_text) where freq_pair is a tuple or None.
        """
        system = (system or '').upper()
        available_freqs = set(sat_data.keys())

        preferred_pairs = {
            'G': [('L1C', 'L5Q')],
            'C': [('L1P', 'L5P'), ('L2I', 'L5P'), ('L2I', 'L1P')],
            'E': [('L1C', 'L5Q'), ('L1C', 'L7Q'), ('L5Q', 'L7Q')],
            'J': [('L1C', 'L5Q')],
        }

        def has_valid_code(freq_name):
            freq_data = sat_data.get(freq_name, {})
            code_vals = freq_data.get('code', []) if 'code' in freq_data else []
            return any(v is not None for v in code_vals)

        for idx, pair in enumerate(preferred_pairs.get(system, [])):
            f1_name, f2_name = pair
            if f1_name in available_freqs and f2_name in available_freqs and has_valid_code(f1_name) and has_valid_code(f2_name):
                if idx == 0:
                    return pair, f'{system}:preferred'
                return pair, f'{system}:fallback_{idx}'

        # For systems not explicitly configured, keep the legacy generic selection.
        fallback_pair = self._select_pseudorange_multipath_pair(system, sat_data, freq_pair=None)
        if fallback_pair:
            return fallback_pair, f'{system}:generic'
        return None, f'{system}:no_valid_pair'

    @staticmethod
    def _smooth_time_series(values: List[float], window_size: int = 5) -> List[float]:
        """Apply a centered moving-average smoother to a single continuous segment."""
        if not values:
            return []
        if window_size <= 1 or len(values) <= 2:
            return values[:]

        if window_size % 2 == 0:
            window_size += 1
        half_window = window_size // 2
        smoothed = []
        for idx in range(len(values)):
            start = max(0, idx - half_window)
            end = min(len(values), idx + half_window + 1)
            smoothed.append(sum(values[start:end]) / (end - start))
        return smoothed

    def calculate_pseudorange_multipath(self, data: Dict[str, Any], freq_pair: Tuple[str, str] = None,
                                        smoothing_window: int = 5) -> Dict[str, Any]:
        """Compute pseudorange multipath series from dual-frequency code/phase observations.

        Returns a plot-ready structure keyed by satellite id. Each item contains the selected
        frequency pair, coefficients, per-frequency continuous segments, and summary statistics.
        """
        observations = data.get('observations_meters', {})
        if not observations:
            return {}

        results: Dict[str, Any] = {}

        for sat_id, sat_data in observations.items():
            system = sat_id[0] if sat_id else ''

            pair = self._select_pseudorange_multipath_pair(system, sat_data, freq_pair=freq_pair)
            if not pair:
                continue

            freq1, freq2 = pair
            d1 = sat_data.get(freq1, {})
            d2 = sat_data.get(freq2, {})
            times1 = d1.get('times', []) or []
            times2 = d2.get('times', []) or []
            code1 = d1.get('code', []) or []
            code2 = d2.get('code', []) or []
            phase1_m = d1.get('phase', []) or []
            phase2_m = d2.get('phase', []) or []
            phase1_cycle = d1.get('phase_cycle', []) or []
            phase2_cycle = d2.get('phase_cycle', []) or []
            lli1 = d1.get('phase_lli', []) or []
            lli2 = d2.get('phase_lli', []) or []
            wl1_list = d1.get('wavelength', []) or []
            wl2_list = d2.get('wavelength', []) or []

            if len(times1) < 2 or len(times2) < 2:
                continue

            sys_freqs = GNSS_FREQUENCIES.get(system, {}) if system else {}
            f1 = sys_freqs.get(freq1)
            f2 = sys_freqs.get(freq2)
            if not f1 or not f2 or abs(f1 - f2) < 1e6:
                continue

            denom = f1 ** 2 - f2 ** 2
            if abs(denom) < 1e-6:
                continue
            coeff_i = (f1 ** 2 + f2 ** 2) / denom
            coeff_j = (2 * f2 ** 2) / denom
            coeff_k = (2 * f1 ** 2) / denom

            time2_idx = {t: j for j, t in enumerate(times2)}

            segments = {
                freq1: [],
                freq2: [],
            }
            current = {
                freq1: {'times': [], 'values': []},
                freq2: {'times': [], 'values': []},
            }

            def _flush_segment():
                for freq_key in (freq1, freq2):
                    times = current[freq_key]['times']
                    values = current[freq_key]['values']
                    if not times:
                        continue
                    window_size = max(1, int(smoothing_window or 1))
                    window_size = min(window_size, len(values))
                    if window_size % 2 == 0 and window_size > 1:
                        window_size -= 1
                    if window_size < 3:
                        window_size = 1
                    smoothed = self._smooth_time_series(values, window_size=window_size)
                    centered = [value - smooth for value, smooth in zip(values, smoothed)]
                    segments[freq_key].append({
                        'times': times[:],
                        'values': centered,
                        'raw_values': values[:],
                        'smoothed_values': smoothed,
                        'smoothing_window': window_size,
                        'count': len(centered),
                    })
                    current[freq_key]['times'].clear()
                    current[freq_key]['values'].clear()

            def _phase_m(idx: int, phase_m_values, phase_cycle_values, wl_values):
                value = phase_m_values[idx] if idx < len(phase_m_values) else None
                if value is not None:
                    return value
                if idx < len(phase_cycle_values):
                    cycle_value = phase_cycle_values[idx]
                    wavelength = wl_values[idx] if idx < len(wl_values) else None
                    if cycle_value is not None and wavelength is not None:
                        return cycle_value * wavelength
                return None

            for i, t1 in enumerate(times1):
                j = time2_idx.get(t1)
                if j is None and hasattr(t1, 'total_seconds'):
                    for t2_candidate, idx in time2_idx.items():
                        try:
                            if abs((t1 - t2_candidate).total_seconds()) < 0.1:
                                j = idx
                                break
                        except Exception:
                            continue

                if j is None:
                    _flush_segment()
                    continue

                code_1 = code1[i] if i < len(code1) else None
                code_2 = code2[j] if j < len(code2) else None
                p1 = _phase_m(i, phase1_m, phase1_cycle, wl1_list)
                p2 = _phase_m(j, phase2_m, phase2_cycle, wl2_list)

                if code_1 is None or code_2 is None:
                    _flush_segment()
                    continue
                if p1 is None or p2 is None:
                    _flush_segment()
                    continue
                if (self._is_active_lli(lli1[i] if i < len(lli1) else None)
                        or self._is_active_lli(lli2[j] if j < len(lli2) else None)):
                    _flush_segment()
                    continue

                epoch_time = t1
                mp1_raw = code_1 - coeff_i * p1 + coeff_j * p2
                mp2_raw = code_2 - coeff_k * p1 + coeff_i * p2
                current[freq1]['times'].append(epoch_time)
                current[freq1]['values'].append(mp1_raw)
                current[freq2]['times'].append(epoch_time)
                current[freq2]['values'].append(mp2_raw)

            _flush_segment()

            flat_freq1 = [value for segment in segments[freq1] for value in segment['values']]
            flat_freq2 = [value for segment in segments[freq2] for value in segment['values']]
            if not flat_freq1 and not flat_freq2:
                continue

            def _summary(values: List[float]) -> Dict[str, float]:
                if not values:
                    return {'count': 0, 'mean': 0.0, 'rms': 0.0, 'std': 0.0}
                return {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'rms': math.sqrt(sum(v * v for v in values) / len(values)),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                }

            results[sat_id] = {
                'freq_pair': (freq1, freq2),
                'frequencies': (f1, f2),
                'coefficients': {
                    'coeff_i': coeff_i,
                    'coeff_j': coeff_j,
                    'coeff_k': coeff_k,
                },
                'segments': segments,
                'stats': {
                    freq1: _summary(flat_freq1),
                    freq2: _summary(flat_freq2),
                },
            }

        return results

    # ---- IF组合频率对配置 ----
    # 各系统用于无电离层组合的频率对列表: [(高频, 低频), ...]
    # 一个系统可有多个频率对，如Galileo同时支持L1C+L5Q和L1C+L7Q
    IF_FREQ_PAIRS = {
        'G': [('L1C', 'L5Q')],
        'E': [('L1C', 'L5Q'), ('L1C', 'L7Q')],
        'C': [('L1P', 'L5P')],
        'J': [('L1C', 'L5Q')],
    }

    @staticmethod
    def _ionofree_coefficients(f1_hz: float, f2_hz: float) -> Tuple[float, float]:
        """计算无电离层组合系数 alpha, beta.

        alpha = f1^2 / (f1^2 - f2^2),  beta = -f2^2 / (f1^2 - f2^2)
        满足 alpha + beta = 1, 且消除一阶电离层延迟.
        """
        f1sq = f1_hz ** 2
        f2sq = f2_hz ** 2
        denom = f1sq - f2sq
        if abs(denom) < 1e-6:
            raise ValueError(f"两个频率太接近, 无法组合: f1={f1_hz}, f2={f2_hz}")
        alpha = f1sq / denom
        beta = -f2sq / denom
        return alpha, beta

    def calculate_ionofree_cmc(self, data: Dict[str, Any], source_key: str = 'code_phase_differences',
                                freq_pair: Tuple[str, str] = None) -> Dict[str, Any]:
        """计算无电离层组合CMC (Ionosphere-Free CMC).

        对每颗卫星，将两个频率的CMC按无电离层组合系数线性组合，消除一阶电离层延迟。
        支持两种输入源:
          - 'code_phase_differences': 手机CMC数据 (来自 calculate_code_phase_differences)
          - 'receiver_cmc': 接收机CMC数据 (来自 calculate_receiver_cmc)

        当一个系统配置了多个频率对（如Galileo的L1C+L5Q和L1C+L7Q）时,
        结果key使用 "sat_id:f1+f2" 格式以区分不同组合。
        若该系统仅有一个频率对，key仍为 "sat_id"。

        参数:
            data: 包含 CMC 数据的字典
            source_key: 数据来源键名
            freq_pair: 可选，指定使用的频率对 (f1_name, f2_name)，
                       如 ('L1C', 'L5Q')。指定后所有系统统一使用此频率对，
                       忽略 IF_FREQ_PAIRS 配置。为 None 时使用默认配置。

        返回:
            {key: {'times': [...], 'cmc_if': [...], 'freq_pair': (f1, f2),
                   'sat_id': str, 'alpha': float, 'beta': float, 'noise_factor': float}}
        """
        source = data.get(source_key, {})
        if not source:
            return {}

        ionofree_results: Dict[str, Any] = {}

        for sat_id, freq_data in source.items():
            system = sat_id[0] if sat_id else ''

            # 确定使用的频率对列表
            if freq_pair is not None:
                # 用户指定了频率对，直接使用（仅一个）
                freq_pairs_to_use = [freq_pair]
            else:
                freq_pairs_to_use = self.IF_FREQ_PAIRS.get(system)
                if freq_pairs_to_use is None:
                    continue  # 该系统不支持IF组合 (如GLONASS)

            multiple = len(freq_pairs_to_use) > 1

            for f1_name, f2_name in freq_pairs_to_use:
                # 获取两个频率的数据
                d1 = freq_data.get(f1_name)
                d2 = freq_data.get(f2_name)
                if d1 is None or d2 is None:
                    continue  # 缺少某个频率

                # 根据source_key确定值字段名
                if source_key == 'code_phase_differences':
                    val_key = 'code_phase_diff'
                elif source_key == 'receiver_cmc':
                    val_key = 'cmc_m'
                else:
                    val_key = 'code_phase_diff'

                times1 = d1.get('times', [])
                vals1 = d1.get(val_key, [])
                times2 = d2.get('times', [])
                vals2 = d2.get(val_key, [])

                if not times1 or not times2:
                    continue

                # 获取频率值并计算IF系数
                sys_freqs = GNSS_FREQUENCIES.get(system, {})
                f1_hz = sys_freqs.get(f1_name)
                f2_hz = sys_freqs.get(f2_name)
                if f1_hz is None or f2_hz is None:
                    continue

                alpha, beta = self._ionofree_coefficients(f1_hz, f2_hz)
                noise_factor = math.sqrt(alpha ** 2 + beta ** 2)

                # 数据对齐: 按时间匹配两个频率的历元
                # 构建f2的时间索引 (支持datetime和数值型时间)
                time2_idx: Dict[Any, int] = {}
                for j, t in enumerate(times2):
                    time2_idx[t] = j

                out_times = []
                out_cmc_if = []

                for i, t1 in enumerate(times1):
                    v1 = vals1[i]
                    if v1 is None:
                        continue

                    # 精确时间匹配
                    j = time2_idx.get(t1)

                    # 如果精确匹配失败且时间是datetime类型，尝试容差匹配
                    if j is None and hasattr(t1, 'total_seconds'):
                        for t2_candidate, idx in time2_idx.items():
                            try:
                                if abs((t1 - t2_candidate).total_seconds()) < 0.1:
                                    j = idx
                                    break
                            except Exception:
                                continue

                    if j is None:
                        continue
                    v2 = vals2[j]
                    if v2 is None:
                        continue

                    cmc_if = alpha * v1 + beta * v2
                    out_times.append(t1)
                    out_cmc_if.append(cmc_if)

                if out_cmc_if:
                    # 多频率对时用 "sat_id:f1+f2" 作key，单频率对直接用 sat_id
                    result_key = f"{sat_id}:{f1_name}+{f2_name}" if multiple else sat_id
                    ionofree_results[result_key] = {
                        'times': out_times,
                        'cmc_if': out_cmc_if,
                        'freq_pair': (f1_name, f2_name),
                        'sat_id': sat_id,
                        'alpha': alpha,
                        'beta': beta,
                        'noise_factor': noise_factor,
                    }

        return ionofree_results

    def calculate_epoch_double_diffs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute epoch-based double differences (port of analyzer.calculate_epoch_double_differences)."""
        result: Dict[str, Any] = {}
        obs = data.get('observations_meters', {})
        for sat_id, freq_data in obs.items():
            result[sat_id] = {}
            for freq, d in freq_data.items():
                code = d.get('code', [])
                phase = d.get('phase', [])
                dop = d.get('doppler', [])
                times = d.get('times', [])
                n = len(code)
                if n < 3:
                    continue
                dd_code = []
                dd_phase = []
                dd_dop = []
                dd_times = []  # 新增：记录时间
                dd_epoch_indices = []  # 新增：记录原始历元索引
                for i in range(n - 2):
                    # 双差使用i, i+1, i+2三个历元，结果对应第i+2个历元
                    if i + 2 < len(times):
                        dd_times.append(times[i + 2])
                    dd_epoch_indices.append(i + 2)  # 记录双差对应的历元索引
                    
                    # code
                    a = code[i]
                    b = code[i + 1]
                    c = code[i + 2]
                    if a is None or b is None or c is None:
                        dd_code.append(None)
                    else:
                        dd_code.append(c - 2 * b + a)

                    # phase (check length)
                    if len(phase) >= n:
                        a = phase[i]
                        b = phase[i + 1]
                        c = phase[i + 2]
                        if a is None or b is None or c is None:
                            dd_phase.append(None)
                        else:
                            dd_phase.append(c - 2 * b + a)
                    else:
                        dd_phase.append(None)

                    # doppler (check length)
                    if len(dop) >= n:
                        a = dop[i]
                        b = dop[i + 1]
                        c = dop[i + 2]
                        if a is None or b is None or c is None:
                            dd_dop.append(None)
                        else:
                            dd_dop.append(c - 2 * b + a)
                    else:
                        dd_dop.append(None)

                result[sat_id][freq] = {
                    'times': dd_times if dd_times else d.get('times', [])[2:],
                    'dd_code': dd_code,
                    'dd_phase': dd_phase,
                    'dd_doppler': dd_dop,
                    'epoch_indices': dd_epoch_indices  # 新增：双差对应的历元索引
                }
        return result

    def calculate_epoch_double_differences(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility wrapper for older callers expecting calculate_epoch_double_differences.

        Forwards to calculate_epoch_double_diffs (canonical implementation).
        """
        return self.calculate_epoch_double_diffs(data)

    def calculate_doppler_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Doppler completeness, phase consistency, smoothness and frequency consistency."""
        observations = data.get('observations_meters', data) if isinstance(data, dict) else data
        if not observations:
            return {}

        def percentile(values, percent):
            ordered = sorted(values)
            if not ordered:
                return None
            position = (len(ordered) - 1) * percent / 100.0
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            if lower == upper:
                return ordered[lower]
            weight = position - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        results = {}
        summary_rows = []
        cross_frequency = []
        for sat_id, frequency_data in observations.items():
            sat_result = {}
            doppler_by_frequency = {}
            for freq, obs in frequency_data.items():
                times = obs.get('times', []) or []
                phases = obs.get('phase', []) or []
                dopplers = obs.get('doppler', []) or []
                count = len(times)
                valid_doppler = sum(1 for i in range(count)
                                    if i < len(dopplers) and dopplers[i] is not None)
                residuals = []
                accelerations = []
                residual_times = []
                doppler_by_frequency[freq] = {
                    times[i]: dopplers[i] for i in range(min(count, len(dopplers)))
                    if dopplers[i] is not None
                }
                positive_steps = []
                for i in range(1, count):
                    try:
                        dt = (times[i] - times[i - 1]).total_seconds()
                    except AttributeError:
                        dt = float(times[i] - times[i - 1])
                    if dt > 0:
                        positive_steps.append(dt)
                nominal_dt = statistics.median(positive_steps) if positive_steps else None

                for i in range(1, count):
                    try:
                        dt = (times[i] - times[i - 1]).total_seconds()
                    except AttributeError:
                        dt = float(times[i] - times[i - 1])
                    continuous = dt > 0 and (nominal_dt is None or dt <= 1.5 * nominal_dt)
                    if not continuous or i >= len(dopplers):
                        continue
                    d0, d1 = dopplers[i - 1], dopplers[i]
                    if d0 is not None and d1 is not None:
                        accelerations.append((d1 - d0) / dt)
                        if i < len(phases) and phases[i - 1] is not None and phases[i] is not None:
                            residuals.append((phases[i] - phases[i - 1]) - 0.5 * (d0 + d1) * dt)
                            residual_times.append(times[i])

                abs_residuals = [abs(value) for value in residuals]
                median_residual = statistics.median(residuals) if residuals else None
                mad = (statistics.median([abs(value - median_residual) for value in residuals])
                       if residuals else None)
                robust_sigma = 1.4826 * mad if mad is not None else None
                threshold = max(0.5, 4.0 * robust_sigma) if robust_sigma is not None else 0.5
                outlier_count = sum(1 for value in abs_residuals if value > threshold)
                stats = {
                    'total_epochs': count,
                    'valid_doppler_epochs': valid_doppler,
                    'completeness_percent': 100.0 * valid_doppler / count if count else 0.0,
                    'residual_times': residual_times,
                    'phase_consistency_residuals': residuals,
                    'residual_bias_m': statistics.mean(residuals) if residuals else None,
                    'residual_rms_m': math.sqrt(sum(v * v for v in residuals) / len(residuals)) if residuals else None,
                    'residual_mad_m': mad,
                    'residual_p95_m': percentile(abs_residuals, 95),
                    'outlier_threshold_m': threshold,
                    'outlier_count': outlier_count,
                    'outlier_rate_percent': 100.0 * outlier_count / len(residuals) if residuals else 0.0,
                    'acceleration_rms_mps2': (math.sqrt(sum(v * v for v in accelerations) / len(accelerations))
                                              if accelerations else None),
                    'acceleration_p95_mps2': percentile([abs(v) for v in accelerations], 95),
                    'accelerations_mps2': accelerations,
                    'valid_residual_count': len(residuals),
                    'nominal_interval_seconds': nominal_dt,
                }
                sat_result[freq] = stats
                summary_rows.append({'sat_id': sat_id, 'system': sat_id[:1], 'freq': freq, **stats})

            frequencies = sorted(doppler_by_frequency)
            for first_index, freq1 in enumerate(frequencies):
                for freq2 in frequencies[first_index + 1:]:
                    common_times = sorted(set(doppler_by_frequency[freq1]) & set(doppler_by_frequency[freq2]))
                    differences = [doppler_by_frequency[freq1][t] - doppler_by_frequency[freq2][t]
                                   for t in common_times]
                    if differences:
                        cross_frequency.append({
                            'sat_id': sat_id, 'freq1': freq1, 'freq2': freq2,
                            'count': len(differences),
                            'bias_mps': statistics.mean(differences),
                            'rms_mps': math.sqrt(sum(v * v for v in differences) / len(differences)),
                            'p95_mps': percentile([abs(v) for v in differences], 95),
                        })
            if sat_result:
                results[sat_id] = sat_result

        results['summary'] = summary_rows
        results['cross_frequency'] = cross_frequency
        return results

    def calculate_observation_noise(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute pseudorange and carrier phase observation noise using third-order difference.

        Formulas (25)-(28):
            ΔP(t_k) = P_k - 3·P_{k-1} + 3·P_{k-2} - P_{k-3}
            σ_P = sqrt(mean(ΔP²) / 20)
            σ̄_P = mean(σ_P across satellites)

        Returns:
            dict per-satellite-per-freq with 'sigma_p', 'sigma_l', 'delta_p', 'delta_l', 'rms_p', 'rms_l'
            and a 'summary' key grouping by (system, freq)
        """
        obs = data.get('observations_meters', data) if isinstance(data, dict) else data
        if not obs:
            return {}

        result = {}
        system_names = {'G': 'GPS', 'R': 'GLO', 'E': 'Galileo', 'C': 'BDS', 'J': 'J-GPS', 'I': 'IRNSS'}
        summary_data = {}  # (system, freq) -> list of sigma_p, sigma_l

        for sat_id in sorted(obs.keys()):
            system = system_names.get(sat_id[0], sat_id[0])
            sat_result = {}
            for freq, freq_data in obs[sat_id].items():
                codes = freq_data.get('code', []) or []
                phases = freq_data.get('phase', []) or []
                phase_lli = freq_data.get('phase_lli', []) or []

                if len(codes) < 4 or len(phases) < 4:
                    continue

                # compute third-order differences
                delta_p = []
                delta_l = []
                for k in range(3, len(codes)):
                    p_k = codes[k]
                    p_k1 = codes[k - 1]
                    p_k2 = codes[k - 2]
                    p_k3 = codes[k - 3]
                    if None in (p_k, p_k1, p_k2, p_k3):
                        continue
                    dp = p_k - 3.0 * p_k1 + 3.0 * p_k2 - p_k3
                    delta_p.append(dp)

                    if k < len(phases):
                        l_k = phases[k]
                        l_k1 = phases[k - 1]
                        l_k2 = phases[k - 2]
                        l_k3 = phases[k - 3]
                        if None in (l_k, l_k1, l_k2, l_k3):
                            continue
                        # skip phase window if any epoch has LLI bit0 set (cycle slip)
                        lli_vals = []
                        if k < len(phase_lli):
                            lli_vals = [phase_lli[k], phase_lli[k-1], phase_lli[k-2], phase_lli[k-3]]
                        skip_phase = False
                        for lli in lli_vals:
                            if lli is not None and (int(lli) & 1) != 0:
                                skip_phase = True
                                break
                        if skip_phase:
                            continue
                        dl = l_k - 3.0 * l_k1 + 3.0 * l_k2 - l_k3
                        delta_l.append(dl)

                if not delta_p and not delta_l:
                    continue

                # outlier rejection: code 3σ + 100m abs, phase 2σ + 1m abs
                MAX_ABS_DELTA_P = 100.0
                MAX_ABS_DELTA_L = 1.0

                def _reject_outliers(values, n_sigma, abs_max):
                    if len(values) < 6:
                        return values
                    arr = [v for v in values if abs_max is None or abs(v) <= abs_max]
                    if len(arr) < 4:
                        return arr
                    while True:
                        m = sum(arr) / len(arr)
                        s = math.sqrt(sum((v - m) ** 2 for v in arr) / len(arr))
                        if s == 0:
                            break
                        filtered = [v for v in arr if abs(v - m) <= n_sigma * s]
                        if len(filtered) == len(arr) or len(filtered) < 4:
                            break
                        arr = filtered
                    return arr

                filtered_dp = _reject_outliers(delta_p, n_sigma=3.0, abs_max=MAX_ABS_DELTA_P)
                filtered_dl = _reject_outliers(delta_l, n_sigma=3.0, abs_max=MAX_ABS_DELTA_L) if delta_l else []

                n_p = len(filtered_dp)
                n_l = len(filtered_dl)

                if n_p < 4:
                    continue

                mean_p = sum(filtered_dp) / n_p if n_p > 0 else 0.0
                mean_l = sum(filtered_dl) / n_l if n_l > 0 else 0.0
                rms_p = math.sqrt(sum(v * v for v in filtered_dp) / n_p) if n_p > 0 else 0.0
                rms_l = math.sqrt(sum(v * v for v in filtered_dl) / n_l) if n_l > 0 else 0.0
                var_p = sum((v - mean_p) ** 2 for v in filtered_dp) / n_p if n_p > 0 else 0.0
                var_l = sum((v - mean_l) ** 2 for v in filtered_dl) / n_l if n_l > 0 else 0.0

                sigma_p = rms_p / math.sqrt(20.0) if rms_p > 0 else 0.0
                sigma_l = rms_l / math.sqrt(20.0) if rms_l > 0 else 0.0
                std_p = math.sqrt(var_p / 20.0) if var_p > 0 else 0.0
                std_l = math.sqrt(var_l / 20.0) if var_l > 0 else 0.0

                sat_result[freq] = {
                    'sigma_p': sigma_p,
                    'sigma_l': sigma_l,
                    'rms_p': rms_p,
                    'rms_l': rms_l,
                    'std_p': std_p,
                    'std_l': std_l,
                    'n_p': n_p,
                    'n_l': n_l,
                    'mean_p': mean_p,
                    'mean_l': mean_l,
                    'delta_p': delta_p,
                    'delta_l': delta_l,
                }

                key = (system, freq)
                if key not in summary_data:
                    summary_data.setdefault(key, {'sigma_p': [], 'sigma_l': [], 'rms_p': [], 'rms_l': [], 'sats': set()})
                summary_data[key]['sigma_p'].append(sigma_p)
                summary_data[key]['sigma_l'].append(sigma_l)
                summary_data[key]['rms_p'].append(rms_p)
                summary_data[key]['rms_l'].append(rms_l)
                summary_data[key]['sats'].add(sat_id)

            if sat_result:
                result[sat_id] = sat_result

        # aggregate summary
        summary = {}
        for (system, freq), vals in summary_data.items():
            n_sats = len(vals['sats'])
            avg_sigma_p = sum(vals['sigma_p']) / len(vals['sigma_p']) if vals['sigma_p'] else 0.0
            avg_sigma_l = sum(vals['sigma_l']) / len(vals['sigma_l']) if vals['sigma_l'] else 0.0
            avg_rms_p = sum(vals['rms_p']) / len(vals['rms_p']) if vals['rms_p'] else 0.0
            avg_rms_l = sum(vals['rms_l']) / len(vals['rms_l']) if vals['rms_l'] else 0.0
            summary[(system, freq)] = {
                'sigma_p': avg_sigma_p,
                'sigma_l': avg_sigma_l,
                'rms_p': avg_rms_p,
                'rms_l': avg_rms_l,
                'n_sats': n_sats,
            }

        result['summary'] = summary
        return result
