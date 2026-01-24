from typing import Dict, Any, List
import statistics
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
                    'doppler_hz': []
                }

                for i in range(1, len(times)):
                    if (phase_cycles[i - 1] is not None and doppler_mps[i - 1] is not None and i < len(doppler_mps)
                            and doppler_mps[i] is not None and phase_cycles[i] is not None and frequency is not None
                            and wavelength is not None):
                        dt = (times[i] - times[i - 1]).total_seconds()
                        doppler_now_hz = doppler_mps[i] / wavelength
                        doppler_old_hz = doppler_mps[i - 1] / wavelength
                        doppler_arith = (doppler_now_hz + doppler_old_hz) / 2
                        phase_change = -dt * doppler_arith / frequency
                        predicted_phase = phase_cycles[i - 1] + phase_change
                        error = (phase_cycles[i] - predicted_phase) * wavelength
                        freq_errors[freq]['times'].append(times[i])
                        freq_errors[freq]['actual_phase'].append(phase_cycles[i])
                        freq_errors[freq]['predicted_phase'].append(predicted_phase)
                        freq_errors[freq]['prediction_error'].append(error)
                        freq_errors[freq]['doppler_mps'].append((doppler_mps[i - 1] + doppler_mps[i]) / 2)
                        freq_errors[freq]['doppler_hz'].append(doppler_arith)

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
                n = len(code)
                if n < 3:
                    continue
                dd_code = []
                dd_phase = []
                dd_dop = []
                for i in range(n - 2):
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
                    'times': d.get('times', [])[2:],
                    'dd_code': dd_code,
                    'dd_phase': dd_phase,
                    'dd_doppler': dd_dop
                }
        return result

    def calculate_epoch_double_differences(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility wrapper for older callers expecting calculate_epoch_double_differences.

        Forwards to calculate_epoch_double_diffs (canonical implementation).
        """
        return self.calculate_epoch_double_diffs(data)
