from typing import Dict, Any, List
from .calculator import MetricCalculator
import statistics


class CoarseErrorProcessor:
    """Identify and mark gross errors; call writer to save cleaned files."""

    def process_cmc_threshold(self, observations_meters: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """Detect CMC (code - phase) changes exceeding threshold.

        observations_meters: {sat_id: {freq: {'times':[], 'code':[], 'phase':[]}}}
        Returns: cmc_flags {sat_id: {freq: [bool,...]}} where True indicates the epoch (1-based) should be removed.
        """
        cmc_flags: Dict[str, Dict[str, List[bool]]] = {}
        for sat_id, freqs in observations_meters.items():
            cmc_flags[sat_id] = {}
            for freq, data in freqs.items():
                codes = data.get('code', [])
                phases = data.get('phase', [])
                n = max(len(codes), len(phases))
                flags = [False] * n
                prev_cmc = None
                for i in range(n):
                    c = codes[i] if i < len(codes) else None
                    p = phases[i] if i < len(phases) else None
                    if c is None or p is None:
                        # can't compute cmc
                        cmc = None
                    else:
                        cmc = c - p
                    if prev_cmc is not None and cmc is not None:
                        if abs(cmc - prev_cmc) > threshold:
                            # mark current epoch (1-based index)
                            flags[i] = True
                    prev_cmc = cmc if cmc is not None else prev_cmc
                cmc_flags[sat_id][freq] = flags
        return cmc_flags

    def process_epoch_double_diff(self, observations_meters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute epoch-based double differences for all satellites/frequencies.

        Returns: {sat_id: {freq: {'times': [...], 'dd_code': [...], 'dd_phase': [...], 'dd_doppler': [...]}}}
        """
        mc = MetricCalculator()
        return mc.calculate_epoch_double_diffs({'observations_meters': observations_meters})

    def check_triple_median_error(self, double_diffs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute triple-median (3*sigma) thresholds and detect outliers in double differences.

        Input double_diffs format should be: {sat_id: {freq: {'times': [...], 'dd_code': [...], 'dd_phase': [...], 'dd_doppler': [...]}}}
        Returns: triple_errors {sat_id: {freq: {'code': {'threshold': float, 'outliers': [idx,...]}, ...}}}
        """
        triple_errors: Dict[str, Dict[str, Any]] = {}

        for sat_id, freq_data in double_diffs.items():
            triple_errors[sat_id] = {}
            for freq, dd_data in freq_data.items():
                # extract arrays and filter Nones
                dd_code = [v for v in dd_data.get('dd_code', []) if v is not None]
                dd_phase = [v for v in dd_data.get('dd_phase', []) if v is not None]
                dd_dop = [v for v in dd_data.get('dd_doppler', []) if v is not None]

                # helper to compute threshold and outliers
                def compute_threshold_and_outliers(arr, default):
                    if len(arr) > 1:
                        sigma = statistics.stdev(arr)
                        threshold = 3 * sigma if sigma != 0 else default
                    else:
                        threshold = 0
                    outliers = [i for i, v in enumerate(dd_data.get('dd_code' if arr is dd_code else ('dd_phase' if arr is dd_phase else 'dd_doppler'), [])) if v is not None and abs(v) > threshold] if threshold > 0 else []
                    return threshold, outliers

                # compute thresholds (use sensible defaults if sigma=0)
                # Note: defaults chosen to mirror legacy behavior
                sigma_code_def = 0.1
                sigma_phase_def = 0.01
                sigma_dop_def = 0.05

                # code
                if len(dd_code) > 1:
                    sigma_code = statistics.stdev(dd_code)
                    triple_code = 3 * sigma_code if sigma_code != 0 else sigma_code_def
                else:
                    triple_code = 0
                outliers_code = [i for i, v in enumerate(dd_data.get('dd_code', [])) if v is not None and triple_code > 0 and abs(v) > triple_code]

                # phase
                if len(dd_phase) > 1:
                    sigma_phase = statistics.stdev(dd_phase)
                    triple_phase = 3 * sigma_phase if sigma_phase != 0 else sigma_phase_def
                else:
                    triple_phase = 0
                outliers_phase = [i for i, v in enumerate(dd_data.get('dd_phase', [])) if v is not None and triple_phase > 0 and abs(v) > triple_phase]

                # doppler
                if len(dd_dop) > 1:
                    sigma_dop = statistics.stdev(dd_dop)
                    triple_dop = 3 * sigma_dop if sigma_dop != 0 else sigma_dop_def
                else:
                    triple_dop = 0
                outliers_dop = [i for i, v in enumerate(dd_data.get('dd_doppler', [])) if v is not None and triple_dop > 0 and abs(v) > triple_dop]

                triple_errors[sat_id][freq] = {
                    'code': {'threshold': triple_code, 'outliers': outliers_code},
                    'phase': {'threshold': triple_phase, 'outliers': outliers_phase},
                    'doppler': {'threshold': triple_dop, 'outliers': outliers_dop}
                }

        return triple_errors
