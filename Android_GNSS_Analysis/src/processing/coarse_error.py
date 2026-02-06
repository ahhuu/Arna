from typing import Dict, Any, List, Optional
from .calculator import MetricCalculator
import statistics


class CoarseErrorProcessor:
    """Identify and mark gross errors; call writer to save cleaned files."""

    def calculate_adaptive_threshold(self, values: List[float], multiplier: float = 2.0, 
                                      floor_threshold: float = 2.0, 
                                      sanity_limit: Optional[float] = None,
                                      sample_filter_limit: Optional[float] = None) -> Dict[str, float]:
        """Calculate adaptive threshold using quantile method with sanity check.
        
        Formula: T_adaptive = max(P99 × K, T_floor)
        
        Sanity Check: If P99 > sanity_limit, statistical calculation is deemed invalid,
                     and floor_threshold is used directly (circuit breaker triggered).
        
        Args:
            values: List of absolute difference values for threshold calculation
            multiplier: Multiplier for P99 (default 2.0)
            floor_threshold: Minimum threshold to prevent overly strict filtering
            sanity_limit: Maximum acceptable P99 value. If exceeded, circuit breaker triggers.
            
        Returns:
            Dict with:
                'threshold': Final adaptive threshold
                'p99': Calculated 99th percentile
                'circuit_breaker_triggered': Boolean indicating if sanity check failed
        """
        import numpy as np
        
        # Filter out None and NaN values
        valid_values = [abs(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        
        # Security Filter: Ignore extreme outliers that would skew percentiles
        if sample_filter_limit is not None:
            valid_values = [v for v in valid_values if v < sample_filter_limit]
        
        if len(valid_values) == 0:
            # No valid data, return floor threshold
            return {
                'threshold': floor_threshold, 
                'p99': 0.0,
                'circuit_breaker_triggered': False
            }
        
        # Calculate 99th percentile
        p99 = np.percentile(valid_values, 99)
        
        # Sanity check (Circuit breaker)
        if sanity_limit is not None and p99 > sanity_limit:
            # Statistical calculation failed, use floor threshold directly
            return {
                'threshold': floor_threshold,
                'p99': p99,
                'circuit_breaker_triggered': True
            }
        
        # Apply formula: max(P99 × K, T_floor)
        adaptive_threshold = max(p99 * multiplier, floor_threshold)
        
        return {
            'threshold': adaptive_threshold,
            'p99': p99,
            'circuit_breaker_triggered': False
        }

    def process_cmc_threshold(self, observations_meters: Dict[str, Any], threshold: float, mode: str = 'fixed') -> Dict[str, Any]:
        """Detect CMC (code - phase) changes exceeding threshold.

        observations_meters: {sat_id: {freq: {'times':[], 'code':[], 'phase':[]}}}
        threshold: Fixed threshold value or floor threshold for adaptive mode
        mode: 'fixed' or 'adaptive'
        
        Returns: {
            'cmc_flags': {sat_id: {freq: [bool,...]}},
            'calculated_thresholds': {sat_id: {freq: {'threshold': float, 'p99': float}}}  # Only for adaptive mode
        }
        """
        cmc_flags: Dict[str, Dict[str, List[bool]]] = {}
        calculated_thresholds: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        for sat_id, freqs in observations_meters.items():
            cmc_flags[sat_id] = {}
            if mode == 'adaptive':
                calculated_thresholds[sat_id] = {}
            
            for freq, data in freqs.items():
                codes = data.get('code', [])
                phases = data.get('phase', [])
                n = max(len(codes), len(phases))
                flags = [False] * n
                
                # Calculate CMC changes for adaptive threshold calculation
                cmc_changes = []
                prev_cmc = None
                
                for i in range(n):
                    c = codes[i] if i < len(codes) else None
                    p = phases[i] if i < len(phases) else None
                    if c is None or p is None:
                        cmc = None
                    else:
                        cmc = c - p
                    
                    if prev_cmc is not None and cmc is not None:
                        change = abs(cmc - prev_cmc)
                        cmc_changes.append(change)
                    
                    prev_cmc = cmc if cmc is not None else prev_cmc
                
                # Determine threshold to use
                if mode == 'adaptive':
                    # Calculate adaptive threshold for this sat/freq combination
                    thresh_info = self.calculate_adaptive_threshold(
                        cmc_changes, 
                        multiplier=2.0, 
                        floor_threshold=threshold,
                        sanity_limit=10.0,  # CMC sanity check: if P99 > 10m, use floor threshold
                        sample_filter_limit=100.0  # Filter out jumps > 100m in statistics
                    )
                    actual_threshold = thresh_info['threshold']
                    calculated_thresholds[sat_id][freq] = thresh_info
                else:
                    # Use fixed threshold
                    actual_threshold = threshold
                
                # Apply threshold to mark outliers
                prev_cmc = None
                for i in range(n):
                    c = codes[i] if i < len(codes) else None
                    p = phases[i] if i < len(phases) else None
                    if c is None or p is None:
                        cmc = None
                    else:
                        cmc = c - p
                    
                    if prev_cmc is not None and cmc is not None:
                        if abs(cmc - prev_cmc) > actual_threshold:
                            flags[i] = True
                    
                    prev_cmc = cmc if cmc is not None else prev_cmc
                
                cmc_flags[sat_id][freq] = flags
        
        result = {'cmc_flags': cmc_flags}
        if mode == 'adaptive':
            result['calculated_thresholds'] = calculated_thresholds
        
        return result

    def process_epoch_double_diff(self, observations_meters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute epoch-based double differences for all satellites/frequencies.

        Returns: {sat_id: {freq: {'times': [...], 'dd_code': [...], 'dd_phase': [...], 'dd_doppler': [...]}}}
        """
        mc = MetricCalculator()
        return mc.calculate_epoch_double_diffs({'observations_meters': observations_meters})

    def check_triple_median_error(self, double_diffs: Dict[str, Any], 
                                   use_triple_sigma: bool = False,
                                   mode: str = 'fixed',
                                   adaptive_floor_thresholds: Optional[Dict[str, float]] = None,
                                   max_threshold_limit: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Compute triple-median (3*sigma) thresholds and detect outliers in double differences.

        Args:
            double_diffs: {sat_id: {freq: {'times': [...], 'dd_code': [...], 'dd_phase': [...], 'dd_doppler': [...]}}}
            use_triple_sigma: Whether to calculate and use triple sigma (default False to align with original code)
            mode: 'fixed' or 'adaptive' threshold calculation mode
            adaptive_floor_thresholds: Floor thresholds for adaptive mode {'code': float, 'phase': float, 'doppler': float}
            max_threshold_limit: Maximum threshold limits {'code': float, 'phase': float, 'doppler': float}
            
        Returns: 
            triple_errors {sat_id: {freq: {'code': {'threshold': float, 'outliers': [idx,...]}, ...}}}
        """
        # Maximum threshold limits (use provided limits or defaults)
        if max_threshold_limit is None:
            max_threshold_limit = {
                'code': 10.0,   # meters
                'phase': 3.0,   # meters
                'doppler': 5.0  # m/s
            }
        
        MAX_CODE_THRESHOLD = max_threshold_limit['code']
        MAX_PHASE_THRESHOLD = max_threshold_limit['phase']
        MAX_DOPPLER_THRESHOLD = max_threshold_limit['doppler']
        
        # Set default floor thresholds for adaptive mode
        if adaptive_floor_thresholds is None:
            adaptive_floor_thresholds = {
                'code': 5.0,
                'phase': 1.5,
                'doppler': 3.0
            }
        
        triple_errors: Dict[str, Dict[str, Any]] = {}

        for sat_id, freq_data in double_diffs.items():
            triple_errors[sat_id] = {}
            for freq, dd_data in freq_data.items():
                if mode == 'adaptive':
                    # Adaptive mode: Calculate threshold using quantile method
                    dd_code = [v for v in dd_data.get('dd_code', []) if v is not None]
                    dd_phase = [v for v in dd_data.get('dd_phase', []) if v is not None]
                    dd_dop = [v for v in dd_data.get('dd_doppler', []) if v is not None]
                    
                    # Code - pseudorange
                    code_thresh_info = self.calculate_adaptive_threshold(
                        dd_code,
                        multiplier=2.0,
                        floor_threshold=adaptive_floor_thresholds['code'],
                        sanity_limit=20.0,  # Code sanity check: if P99 > 20m, use floor threshold
                        sample_filter_limit=200.0  # Filter out jumps > 200m in statistics
                    )
                    triple_code = min(code_thresh_info['threshold'], MAX_CODE_THRESHOLD)
                    outliers_code = [i for i, v in enumerate(dd_data.get('dd_code', [])) 
                                    if v is not None and abs(v) > triple_code]
                    
                    # Phase - carrier phase
                    phase_thresh_info = self.calculate_adaptive_threshold(
                        dd_phase,
                        multiplier=2.0,
                        floor_threshold=adaptive_floor_thresholds['phase'],
                        sanity_limit=5.0,  # Phase sanity check: if P99 > 5m, use floor threshold
                        sample_filter_limit=50.0  # Filter out jumps > 50m in statistics
                    )
                    triple_phase = min(phase_thresh_info['threshold'], MAX_PHASE_THRESHOLD)
                    outliers_phase = [i for i, v in enumerate(dd_data.get('dd_phase', [])) 
                                     if v is not None and abs(v) > triple_phase]
                    
                    # Doppler
                    doppler_thresh_info = self.calculate_adaptive_threshold(
                        dd_dop,
                        multiplier=2.0,
                        floor_threshold=adaptive_floor_thresholds['doppler'],
                        sanity_limit=10.0,  # Doppler sanity check: if P99 > 10m/s, use floor threshold
                        sample_filter_limit=100.0  # Filter out jumps > 100m/s in statistics
                    )
                    triple_dop = min(doppler_thresh_info['threshold'], MAX_DOPPLER_THRESHOLD)
                    outliers_dop = [i for i, v in enumerate(dd_data.get('dd_doppler', [])) 
                                   if v is not None and abs(v) > triple_dop]
                    
                    triple_errors[sat_id][freq] = {
                        'code': {
                            'threshold': triple_code, 
                            'outliers': outliers_code,
                            'p99': code_thresh_info['p99'],
                            'circuit_breaker_triggered': code_thresh_info['circuit_breaker_triggered']
                        },
                        'phase': {
                            'threshold': triple_phase, 
                            'outliers': outliers_phase,
                            'p99': phase_thresh_info['p99'],
                            'circuit_breaker_triggered': phase_thresh_info['circuit_breaker_triggered']
                        },
                        'doppler': {
                            'threshold': triple_dop, 
                            'outliers': outliers_dop,
                            'p99': doppler_thresh_info['p99'],
                            'circuit_breaker_triggered': doppler_thresh_info['circuit_breaker_triggered']
                        }
                    }
                elif not use_triple_sigma:
                    # Default mode: Don't calculate triple sigma, use max thresholds directly (align with original code)
                    # Original code's calculate_triple_median_error is commented out, returns threshold=0
                    # In remove_double_diff_outliers, when threshold<=0, it uses max thresholds
                    triple_errors[sat_id][freq] = {
                        'code': {'threshold': MAX_CODE_THRESHOLD, 'outliers': []},
                        'phase': {'threshold': MAX_PHASE_THRESHOLD, 'outliers': []},
                        'doppler': {'threshold': MAX_DOPPLER_THRESHOLD, 'outliers': []}
                    }
                else:
                    # Optional mode: Calculate triple sigma and apply max threshold limits
                    # Extract valid data (filter None)
                    dd_code = [v for v in dd_data.get('dd_code', []) if v is not None]
                    dd_phase = [v for v in dd_data.get('dd_phase', []) if v is not None]
                    dd_dop = [v for v in dd_data.get('dd_doppler', []) if v is not None]

                    # code - pseudorange
                    if len(dd_code) > 1:
                        sigma_code = statistics.stdev(dd_code)
                        if sigma_code != 0:
                            # Apply max threshold limit
                            triple_code = min(3 * sigma_code, MAX_CODE_THRESHOLD)
                        else:
                            triple_code = MAX_CODE_THRESHOLD
                    else:
                        # Use max threshold when insufficient data
                        triple_code = MAX_CODE_THRESHOLD

                    outliers_code = [i for i, v in enumerate(dd_data.get('dd_code', [])) 
                                    if v is not None and abs(v) > triple_code]

                    # phase - carrier phase
                    if len(dd_phase) > 1:
                        sigma_phase = statistics.stdev(dd_phase)
                        if sigma_phase != 0:
                            triple_phase = min(3 * sigma_phase, MAX_PHASE_THRESHOLD)
                        else:
                            triple_phase = MAX_PHASE_THRESHOLD
                    else:
                        triple_phase = MAX_PHASE_THRESHOLD

                    outliers_phase = [i for i, v in enumerate(dd_data.get('dd_phase', [])) 
                                     if v is not None and abs(v) > triple_phase]

                    # doppler
                    if len(dd_dop) > 1:
                        sigma_dop = statistics.stdev(dd_dop)
                        if sigma_dop != 0:
                            triple_dop = min(3 * sigma_dop, MAX_DOPPLER_THRESHOLD)
                        else:
                            triple_dop = MAX_DOPPLER_THRESHOLD
                    else:
                        triple_dop = MAX_DOPPLER_THRESHOLD

                    outliers_dop = [i for i, v in enumerate(dd_data.get('dd_doppler', [])) 
                                   if v is not None and abs(v) > triple_dop]

                    triple_errors[sat_id][freq] = {
                        'code': {'threshold': triple_code, 'outliers': outliers_code},
                        'phase': {'threshold': triple_phase, 'outliers': outliers_phase},
                        'doppler': {'threshold': triple_dop, 'outliers': outliers_dop}
                    }

        return triple_errors
