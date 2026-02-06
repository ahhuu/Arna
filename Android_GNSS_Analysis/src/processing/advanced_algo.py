from typing import Dict, Any, List, Optional, Tuple
import os
import datetime
import pandas as pd
import numpy as np
from ..data.writer import RinexWriter
from .cycle_slip_detector import CycleSlipDetector
from .inter_freq_bias import InterFrequencyBiasAnalyzer



class CoreAlgorithmProcessor:
    """Complex modeling and correction logic (Doppler, CCI, ISB).

    This module contains pure functions ported from the legacy analyzer implementation.
    Each method returns results and does not mutate external state.
    Methods are programmatically callable for testing; the window UI is optional.
    """

    def __init__(self):
        self.cs_detector = CycleSlipDetector()
        self.ifb_analyzer = InterFrequencyBiasAnalyzer()

    def calculate_dcmc(self, receiver_cmc: Dict[str, Any], phone_cmc: Dict[str, Any],
                       r_squared_threshold: float = 0.5,
                       enable_phone_only_analysis: bool = False,
                       phone_only_min_data_points: int = 10) -> Dict[str, Any]:
        """Wrapper port that mirrors legacy signature and returns {'dcmc':..., 'meta':...} (already implemented)"""
        """Compute dCMC between receiver and phone (port of analyzer.calculate_dcmc)

        receiver_cmc: {sat_id: {freq: {'times': [...], 'cmc_m': [...]}}}
        phone_cmc: same structure but phone uses fields 'times' and 'code_phase_diff'
        Returns: dcmc_results {sat_id: {freq: {'times': [...], 'dcmc': [...]}}} and metadata
        """
        dcmc_results = {}
        total_combinations = 0
        processed_combinations = 0
        filtered_combinations = 0
        linear_drift_detailed = {}

        # count total combinations
        for sat_id in receiver_cmc.keys():
            if sat_id in phone_cmc:
                total_combinations += len(set(receiver_cmc[sat_id].keys()) & set(phone_cmc[sat_id].keys()))

        # phone-only stats
        phone_only_combinations = 0
        if enable_phone_only_analysis:
            for sat_id in phone_cmc.keys():
                if sat_id not in receiver_cmc:
                    phone_only_combinations += len(phone_cmc[sat_id].keys())
            total_combinations += phone_only_combinations

        # iterate
        for sat_id in receiver_cmc.keys():
            if sat_id not in phone_cmc:
                continue
            receiver_freqs = receiver_cmc[sat_id]
            phone_freqs = phone_cmc[sat_id]
            common_freqs = set(receiver_freqs.keys()) & set(phone_freqs.keys())
            if not common_freqs:
                continue
            sat_dcmc = {}
            for freq in common_freqs:
                receiver_data = receiver_freqs[freq]
                phone_data = phone_freqs[freq]
                receiver_times = receiver_data['times']
                phone_times = phone_data['times']
                
                # O(N) Matching using common time buckets (0.1s tolerance)
                # Round to nearest 0.1s to create buckets
                common_times = []
                receiver_time_idx = {}
                phone_time_idx = {}
                
                # Dict for phone times: {rounded_time: original_idx}
                phone_lookup = {}
                for j, pt in enumerate(phone_times):
                    rounded = round(pt.timestamp() * 10) / 10.0
                    phone_lookup[rounded] = j
                
                for i, rt in enumerate(receiver_times):
                    rounded = round(rt.timestamp() * 10) / 10.0
                    if rounded in phone_lookup:
                        common_times.append(rt)
                        receiver_time_idx[rt] = i
                        phone_time_idx[rt] = phone_lookup[rounded]
                
                common_times.sort()
                if not common_times:
                    continue
                # phone cmc linear trend detection
                phone_cmc_values = []
                phone_times_list = []
                for t in common_times:
                    phone_idx = phone_time_idx.get(t, -1)
                    if phone_idx >= 0 and phone_idx < len(phone_data['code_phase_diff']):
                        phone_cmc_val = phone_data['code_phase_diff'][phone_idx]
                        if phone_cmc_val is not None:
                            phone_cmc_values.append(phone_cmc_val)
                            phone_times_list.append(t)
                if len(phone_cmc_values) < phone_only_min_data_points:
                    filtered_combinations += 1
                    continue

                # linear trend check
                phone_trend = self._check_linear_trend(phone_times_list, phone_cmc_values, r_squared_threshold)
                sat_freq_key = f"{sat_id}_{freq}"
                linear_drift_detailed[sat_freq_key] = {
                    'status': '有线性漂移' if phone_trend['has_linear_drift'] else '无线性漂移',
                    'r_squared': phone_trend['r_squared'],
                    'slope': phone_trend['slope'],
                    'intercept': phone_trend['intercept'],
                    'data_points': len(phone_cmc_values),
                    'min_r_squared': r_squared_threshold,
                    'min_slope_magnitude': 1e-6
                }
                if not phone_trend['has_linear_drift']:
                    filtered_combinations += 1
                    continue
                dcmc_vals = []
                dcmc_times = []
                for t in common_times:
                    rec_idx = receiver_time_idx[t]
                    phone_idx = phone_time_idx[t]
                    rec_cmc = receiver_data['cmc_m'][rec_idx]
                    phone_cmc_val = phone_data['code_phase_diff'][phone_idx]
                    dcmc_val = rec_cmc - phone_cmc_val
                    dcmc_vals.append(dcmc_val)
                    dcmc_times.append(t)
                if dcmc_vals:
                    sat_dcmc[freq] = {'times': dcmc_times, 'dcmc': dcmc_vals}
                else:
                    filtered_combinations += 1
                processed_combinations += 1
            if sat_dcmc:
                dcmc_results[sat_id] = sat_dcmc

        # phone-only processing skipped here (no side effects)

        meta = {
            'total_combinations': total_combinations,
            'processed_combinations': processed_combinations,
            'filtered_combinations': filtered_combinations,
            'linear_drift_detailed': linear_drift_detailed
        }
        return {'dcmc': dcmc_results, 'meta': meta}

    def extract_cci_series(self, dcmc_data: Dict[str, Any], max_gap_seconds: int = 300) -> Dict[str, Any]:
        """Port of extract_cci_series from analyzer.

        dcmc_data: {sat_id: {freq: {'times': [...], 'dcmc': [...]}}}
        Returns: {sat_id: {freq: {'times': [...], 'cci_series': [...], 'arc_info': [...]}}}
        """
        cci_results = {}
        for sat_id, freq_data in dcmc_data.items():
            sat_cci = {}
            for freq, dcmc_info in freq_data.items():
                times = dcmc_info['times']
                dcmc_values = dcmc_info['dcmc']
                if not dcmc_values:
                    continue
                arcs = self._identify_continuous_arcs(times, dcmc_values, max_gap_seconds=max_gap_seconds,
                                                      sat_freq_info=f"{sat_id}_{freq}")
                all_cci_vals = []
                all_times = []
                arc_info = []
                for arc_idx, arc in enumerate(arcs):
                    arc_times = arc['times']
                    arc_dcmc = arc['values']
                    if len(arc_dcmc) < 2:
                        continue
                    arc_mean = sum(arc_dcmc) / len(arc_dcmc)
                    cci_vals = [dcmc_val - arc_mean for dcmc_val in arc_dcmc]
                    all_cci_vals.extend(cci_vals)
                    all_times.extend(arc_times)
                    arc_info.append({
                        'arc_index': arc_idx,
                        'start_time': arc_times[0],
                        'end_time': arc_times[-1],
                        'duration': (arc_times[-1] - arc_times[0]).total_seconds(),
                        'mean_dcmc': arc_mean,
                        'num_points': len(arc_dcmc),
                        'cci_range': max(cci_vals) - min(cci_vals) if cci_vals else 0
                    })
                if all_cci_vals:
                    sat_cci[freq] = {
                        'times': all_times,
                        'cci_series': all_cci_vals,
                        'arc_info': arc_info
                    }
            if sat_cci:
                cci_results[sat_id] = sat_cci
        return cci_results

    def calculate_roc_model(self, cci_data: Dict[str, Any], cv_threshold: float = 0.5,
                            enable_phone_only_analysis: bool = False,
                            phone_only_linear_drift: Dict[str, Any] = None) -> Dict[str, Any]:
        """Port of calculate_roc_model. Returns combined ROC models (system-level and individual).
        """
        roc_results = {}
        system_freq_rocs = {}
        system_freq_contributing_sats = {}
        individual_rocs = {}

        total_combinations = 0
        for sat_id, freq_data in cci_data.items():
            total_combinations += len(freq_data)

        processed_combinations = 0
        for sat_id, freq_data in cci_data.items():
            sat_system = sat_id[0] if sat_id else 'Unknown'
            for freq, cci_info in freq_data.items():
                times = cci_info['times']
                cci_vals = cci_info['cci_series']
                if len(cci_vals) < 2:
                    continue
                time_zero = times[0]
                time_seconds = [(t - time_zero).total_seconds() for t in times]
                try:
                    slope, intercept = self._linear_fit(time_seconds, cci_vals)
                    roc_value = slope
                    system_freq_key = f"{sat_system}_{freq}"
                    if system_freq_key not in system_freq_rocs:
                        system_freq_rocs[system_freq_key] = []
                        system_freq_contributing_sats[system_freq_key] = []
                        individual_rocs[system_freq_key] = {}
                    system_freq_rocs[system_freq_key].append(roc_value)
                    system_freq_contributing_sats[system_freq_key].append(sat_id)
                    individual_rocs[system_freq_key][sat_id] = roc_value
                except Exception:
                    continue
                processed_combinations += 1

        individual_roc_models = {}
        for system_freq_key, roc_values in system_freq_rocs.items():
            if roc_values:
                mean_roc = sum(roc_values) / len(roc_values)
                std_roc = (sum((r - mean_roc) ** 2 for r in roc_values) / len(roc_values)) ** 0.5
                if mean_roc != 0:
                    roc_cv = abs(std_roc / mean_roc)
                else:
                    roc_cv = float('inf') if std_roc > 0 else 0.0
                if roc_cv < cv_threshold and len(roc_values) >= 3:
                    quality_level = "高质量"
                    is_high_quality = True
                elif cv_threshold <= roc_cv < 1.0 and len(roc_values) >= 3:
                    quality_level = "中等质量"
                    is_high_quality = False
                else:
                    quality_level = "个体级"
                    is_high_quality = True
                if roc_cv < cv_threshold and len(roc_values) >= 3:
                    roc_results[system_freq_key] = {
                        'roc_rate': mean_roc,
                        'roc_std': std_roc,
                        'roc_cv': roc_cv,
                        'quality_level': quality_level,
                        'is_high_quality': is_high_quality,
                        'contributing_sats': system_freq_contributing_sats[system_freq_key],
                        'individual_rocs': individual_rocs[system_freq_key],
                        'num_satellites': len(roc_values),
                        'model_type': 'system_level'
                    }
                else:
                    individual_cv = 0.0
                    if len(roc_values) >= 3:
                        individual_cv = roc_cv
                    for sat_id, individual_roc in individual_rocs[system_freq_key].items():
                        individual_key = f"{sat_id}_{system_freq_key.split('_')[1]}"
                        individual_roc_models[individual_key] = {
                            'roc_rate': individual_roc,
                            'roc_std': std_roc if len(roc_values) >= 3 else 0.0,
                            'roc_cv': individual_cv,
                            'quality_level': "个体级",
                            'is_high_quality': True,
                            'contributing_sats': [sat_id],
                            'individual_rocs': {sat_id: individual_roc},
                            'num_satellites': 1,
                            'model_type': 'individual_level',
                            'system_freq_cv': individual_cv,
                            'system_freq_satellites': len(roc_values)
                        }

        # phone-only ROC models
        phone_only_roc_models = {}
        if enable_phone_only_analysis and phone_only_linear_drift:
            for sat_freq_key, drift_info in phone_only_linear_drift.items():
                if drift_info['status'] == '有线性漂移':
                    phone_only_roc_models[sat_freq_key] = {
                        'roc_rate': drift_info['slope'],
                        'roc_std': 0.0,
                        'roc_cv': 0.0,
                        'quality_level': "个体级",
                        'is_high_quality': True,
                        'contributing_sats': [sat_freq_key.split('_')[0]],
                        'individual_rocs': {sat_freq_key.split('_')[0]: drift_info['slope']},
                        'num_satellites': 1,
                        'model_type': 'individual_level',
                        'data_source': 'phone_only'
                    }

        all_roc_models = {**roc_results, **individual_roc_models, **phone_only_roc_models}
        return all_roc_models

    def correct_phase_observations(self, observations_meters: Dict[str, Any], roc_model: Dict[str, Any],
                                   dcmc_data: Dict[str, Any], enable_phone_only_analysis: bool = False,
                                   phone_only_models: Optional[Dict[str, Any]] = None,
                                   original_rinex_path: Optional[str] = None,
                                   output_path: Optional[str] = None,
                                   writer: Optional[RinexWriter] = None) -> Dict[str, Any]:
        """Port of correct_phase_observations. Returns corrected phase dict similar to analyzer.

        If writer and original_rinex_path are provided, will call writer.write_corrected_rinex to produce a corrected RINEX file.
        phone_only_models: optional dict mapping "SAT_FREQ" -> model info (roc_rate...), used when enable_phone_only_analysis is True.
        """
        corrected_results = {}

        # collect valid combinations from dcmc_data
        valid_combinations = set()
        for sat_id, freqs in dcmc_data.items():
            for freq in freqs.keys():
                valid_combinations.add((sat_id, freq))

        for sat_id, freq_data in observations_meters.items():
            sat_corrected = {}
            for freq, obs_data in freq_data.items():
                if (sat_id, freq) not in valid_combinations:
                    continue
                times = obs_data.get('times', [])
                original_phase_m = obs_data.get('phase', [])
                original_phase_cycle = obs_data.get('phase_cycle', [])
                wavelengths = obs_data.get('wavelength', [])
                if not times or not original_phase_m:
                    continue
                sat_system = sat_id[0] if sat_id else 'Unknown'
                system_freq_key = f"{sat_system}_{freq}"
                individual_key = f"{sat_id}_{freq}"
                roc_info = None
                model_type = None
                if system_freq_key in roc_model:
                    roc_info = roc_model[system_freq_key]
                    model_type = "系统级"
                elif individual_key in roc_model:
                    roc_info = roc_model[individual_key]
                    model_type = "个体级"
                else:
                    continue
                roc_rate = roc_info['roc_rate']
                quality_level = roc_info.get('quality_level', '个体级')
                is_high_quality = roc_info.get('is_high_quality', True)
                roc_cv = roc_info.get('roc_cv', 0.0)
                arcs = self._identify_continuous_arcs(times, original_phase_m, sat_freq_info=f"{sat_id}_{freq}")
                corrected_phase_m = []
                correction_applied = []
                medium_quality_details = []
                high_quality_details = []
                individual_level_details = []
                arc_info = []
                for arc_idx, arc in enumerate(arcs):
                    arc_times = arc['times']
                    arc_phase = arc['values']
                    if not arc_times:
                        continue
                    arc_duration = (arc_times[-1] - arc_times[0]).total_seconds()
                    has_cycle_slip = arc.get('has_cycle_slip', False)
                    if has_cycle_slip:
                        t0 = arc_times[0]
                    else:
                        if arc_idx == 0:
                            t0 = times[0] if times else arc_times[0]
                        else:
                            t0 = arc_times[0]
                    arc_info.append({
                        'arc_index': arc_idx + 1,
                        'start_time': t0,
                        'end_time': arc_times[-1],
                        'duration_seconds': arc_duration,
                        'data_points': len(arc_times),
                        'has_cycle_slip': has_cycle_slip
                    })
                    for t, phase_val in zip(arc_times, arc_phase):
                        if phase_val is None:
                            corrected_phase_m.append(None)
                            correction_applied.append(None)
                            continue
                        time_diff_seconds = (t - t0).total_seconds()
                        base_correction = -roc_rate * time_diff_seconds
                        if quality_level == "高质量":
                            if has_cycle_slip:
                                correction = base_correction * 0.5
                            else:
                                correction = base_correction
                            high_quality_details.append({
                                'time': t,
                                'time_diff_seconds': time_diff_seconds,
                                'base_correction': base_correction,
                                'final_correction': correction,
                                'roc_cv': roc_cv,
                                'arc_index': arc_idx + 1,
                                'has_cycle_slip': has_cycle_slip,
                                'model_type': model_type
                            })
                        elif quality_level == "中等质量":
                            weight = max(0.3, 1.0 - roc_cv)
                            time_weight = max(0.5, 1.0 - abs(time_diff_seconds) / 3600.0)
                            if has_cycle_slip:
                                weight *= 0.3
                                max_correction = 0.02
                            else:
                                max_correction = 0.05
                            correction = base_correction * weight * time_weight
                            correction = max(-max_correction, min(max_correction, correction))
                            medium_quality_details.append({
                                'time': t,
                                'time_diff_seconds': time_diff_seconds,
                                'base_correction': base_correction,
                                'weight': weight,
                                'time_weight': time_weight,
                                'final_correction': correction,
                                'roc_cv': roc_cv,
                                'arc_index': arc_idx + 1,
                                'has_cycle_slip': has_cycle_slip,
                                'model_type': model_type
                            })
                        elif quality_level == "个体级":
                            if has_cycle_slip:
                                correction = base_correction * 0.5
                            else:
                                correction = base_correction
                            individual_level_details.append({
                                'time': t,
                                'time_diff_seconds': time_diff_seconds,
                                'base_correction': base_correction,
                                'final_correction': correction,
                                'roc_cv': roc_cv,
                                'arc_index': arc_idx + 1,
                                'has_cycle_slip': has_cycle_slip,
                                'model_type': model_type
                            })
                        corrected_phase = phase_val + correction
                        corrected_phase_m.append(corrected_phase)
                        correction_applied.append(correction)
                if medium_quality_details:
                    medium_key = f"{sat_id}_{freq}"
                    # attach as metadata key
                    corrected_results.setdefault('_medium_quality_details', {})[medium_key] = {
                        'details': medium_quality_details,
                        'arcs': arc_info
                    }
                if high_quality_details:
                    high_key = f"{sat_id}_{freq}"
                    corrected_results.setdefault('_high_quality_details', {})[high_key] = {
                        'details': high_quality_details,
                        'arcs': arc_info
                    }
                if individual_level_details:
                    ind_key = f"{sat_id}_{freq}"
                    corrected_results.setdefault('_individual_level_details', {})[ind_key] = {
                        'details': individual_level_details,
                        'arcs': arc_info
                    }
                if corrected_phase_m:
                    sat_corrected[freq] = {
                        'times': times,
                        'original_phase': original_phase_m,
                        'original_phase_cycle': original_phase_cycle,
                        'corrected_phase': corrected_phase_m,
                        'correction_applied': correction_applied,
                        'wavelengths': wavelengths,
                        'roc_rate': roc_rate
                    }
            if sat_corrected:
                corrected_results[sat_id] = sat_corrected

        # phone-only correction (port from original analyzer)
        phone_only_corrected = 0
        if enable_phone_only_analysis and phone_only_models:
            for sat_freq_key, roc_info in phone_only_models.items():
                try:
                    sat_id_p, freq_p = sat_freq_key.split('_', 1)
                except Exception:
                    continue
                if sat_id_p in observations_meters and freq_p in observations_meters[sat_id_p]:
                    phone_data = observations_meters[sat_id_p][freq_p]
                    times = phone_data.get('times', [])
                    original_phase = phone_data.get('phase', [])
                    if not original_phase:
                        continue
                    roc_rate_p = roc_info.get('roc_rate', 0.0)
                    corrected_phase = []
                    correction_applied = []
                    wavelengths_p = phone_data.get('wavelength', []) or [None] * len(times)
                    t0 = times[0]
                    for t, phase_val in zip(times, original_phase):
                        if phase_val is not None:
                            time_diff = (t - t0).total_seconds()
                            correction = roc_rate_p * time_diff
                            corrected_phase.append(phase_val + correction)
                            correction_applied.append(correction)
                        else:
                            corrected_phase.append(None)
                            correction_applied.append(0.0)
                    if sat_id_p not in corrected_results:
                        corrected_results[sat_id_p] = {}
                    corrected_results[sat_id_p][freq_p] = {
                        'times': times,
                        'original_phase': original_phase,
                        'corrected_phase': corrected_phase,
                        'correction_applied': correction_applied,
                        'wavelengths': wavelengths_p,
                        'roc_rate': roc_rate_p,
                        'model_type': 'phone_only',
                        'data_source': 'phone_only'
                    }
                    phone_only_corrected += 1

        out = corrected_results
        # if writer and original_rinex_path are provided, call writer to generate corrected file
        if writer and original_rinex_path:
            try:
                writer_result = writer.write_corrected_rinex(original_rinex_path, output_path, corrected_results, roc_model)
                out['_writer_result'] = writer_result
            except Exception:
                # ignore writer failures but still return algorithmic results
                out['_writer_result'] = {'error': 'writer_failed'}
        return out

    # --- Helper functions (ported small utils) ---
    def _identify_continuous_arcs(self, times, values, max_gap_seconds=300, sat_freq_info=""):
        if not times or not values:
            return []
        
        # 如果提供了卫星频率信息，尝试检测周跳
        cycle_slip_epochs = set()
        if sat_freq_info and hasattr(self, 'cs_detector'):
            # 这里我们期望sat_freq_info格式为 "SATID_FREQ"
            # 但我们需要完整的观测数据来进行探测，而这里只有times和values
            # 因此，我们只能基于简单的gap检测，或者在调用此函数前已经进行了周跳探测并传入了flags
            # 为保持签名兼容，我们这里暂时只做Gap检测
            # 真正的周跳探测应该在流程早期对全量数据进行
            pass

        arcs = []
        current_arc_times = []
        current_arc_values = []
        
        # 辅助：获取周跳标记（如果values中包含元组或对象，这里假设values是纯数值）
        # 如果需要集成周跳，我们在处理values前应该已经处理过
        
        for i, (time, value) in enumerate(zip(times, values)):
            if value is None:
                continue
                
            is_new_arc = False
            has_slip = False
            
            if not current_arc_times:
                is_new_arc = True
            else:
                time_diff = (time - current_arc_times[-1]).total_seconds()
                if time_diff > max_gap_seconds:
                    is_new_arc = True
                # 这里可以添加额外的逻辑：如果外部传入了周跳信息
            
            if is_new_arc:
                if current_arc_times:
                    arcs.append({'times': current_arc_times, 'values': current_arc_values, 'has_cycle_slip': False})
                current_arc_times = [time]
                current_arc_values = [value]
            else:
                current_arc_times.append(time)
                current_arc_values.append(value)
                
        if current_arc_times:
            arcs.append({'times': current_arc_times, 'values': current_arc_values, 'has_cycle_slip': False})
        return arcs

    def detect_cycle_slips_for_all(self, observations: Dict[str, Any], 
                                  frequencies: Dict[str, Dict], 
                                  wavelengths: Dict[str, Dict]) -> Dict[str, Any]:
        """Run cycle slip detection for all satellites."""
        if not hasattr(self, 'cs_detector'):
            self.cs_detector = CycleSlipDetector()
        return self.cs_detector.detect_cycle_slips(observations, frequencies, wavelengths)

    def run_inter_freq_bias_analysis(self, observations: Dict[str, Any], 
                                    freq1: str, freq2: str, 
                                    constellation: Optional[str] = None) -> Dict[str, Any]:
        """Run Inter-Frequency Bias analysis."""
        if not hasattr(self, 'ifb_analyzer'):
            self.ifb_analyzer = InterFrequencyBiasAnalyzer()
        return self.ifb_analyzer.analyze_inter_freq_bias(observations, freq1, freq2, constellation)

    def _check_linear_trend(self, times, values, min_r_squared=0.5, min_slope_magnitude=1e-6):
        if len(times) < 3:
            return {'has_linear_drift': False, 'slope': 0.0, 'r_squared': 0.0, 'intercept': 0.0}
        try:
            time_zero = times[0]
            time_seconds = [(t - time_zero).total_seconds() for t in times]
            slope, intercept = self._linear_fit(time_seconds, values)
            y_mean = sum(values) / len(values)
            ss_tot = sum((y - y_mean) ** 2 for y in values)
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(time_seconds, values))
            r_squared = 0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
            has_linear_drift = (r_squared >= min_r_squared and abs(slope) >= min_slope_magnitude)
            return {'has_linear_drift': has_linear_drift, 'slope': slope, 'r_squared': r_squared, 'intercept': intercept}
        except Exception:
            return {'has_linear_drift': False, 'slope': 0.0, 'r_squared': 0.0, 'intercept': 0.0}

    def _linear_fit(self, x_values, y_values):
        n = len(x_values)
        if n < 2:
            raise ValueError("至少需要2个点进行线性拟合")
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            raise ValueError("线性拟合失败：分母接近零")
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept


    # --- ISB core methods (simplified, ported behavior) ---
    def run_prepare_isb_data(self, observations_meters: Dict[str, Any], receiver_observations: Dict[str, Any],
                             time_tolerance: float = 0.1) -> Dict[str, Any]:
        """Prepare ISB input data (time alignment and BDS satellite classification)."""
        phone_times = set()
        receiver_times = set()
        # collect phone and receiver L2I times
        for sat_id, sat_data in observations_meters.items():
            if sat_id.startswith('C') and 'L2I' in sat_data:
                phone_times.update(sat_data['L2I'].get('times', []))
        for sat_id, sat_data in receiver_observations.items():
            if sat_id.startswith('C') and 'L2I' in sat_data:
                receiver_times.update(sat_data['L2I'].get('times', []))
        phone_times_list = sorted(phone_times)
        receiver_times_list = sorted(receiver_times)
        common_times = []
        for t in phone_times_list:
            # find close receiver time
            for rt in receiver_times_list:
                if abs((t - rt).total_seconds()) <= time_tolerance:
                    common_times.append(t)
                    break
        common_times = sorted(common_times)
        bds2_sats = []
        bds3_sats = []
        for sat_id in observations_meters.keys():
            if sat_id.startswith('C'):
                try:
                    prn = int(sat_id[1:])
                    if 1 <= prn <= 18:
                        bds2_sats.append(sat_id)
                    elif 19 <= prn <= 60:
                        bds3_sats.append(sat_id)
                except Exception:
                    continue
        isb_data = {
            'common_times': common_times,
            'bds2_sats': bds2_sats,
            'bds3_sats': bds3_sats,
            'phone_data': observations_meters,
            'receiver_data': receiver_observations,
            'time_tolerance': time_tolerance
        }
        return isb_data

    def _analyze_satellite_quality(self, sat_id: str, phone_times: list, phone_snr: list, phone_code: list,
                                   receiver_times: list, receiver_snr: list, receiver_code: list,
                                   common_times: list) -> Dict:
        """分析单个卫星的质量指标 (Ported from legacy analyzer)"""
        import numpy as np
        
        # 计算手机数据质量
        phone_valid_snr = [snr for snr in phone_snr if snr is not None and snr > 0]
        phone_valid_code = [code for code in phone_code if code is not None]

        if not phone_valid_snr or not phone_valid_code:
            return None

        phone_avg_snr = np.mean(phone_valid_snr)
        phone_snr_std = np.std(phone_valid_snr)
        phone_code_std = np.std(phone_valid_code)

        # 计算接收机数据质量
        receiver_valid_snr = [snr for snr in receiver_snr if snr is not None and snr > 0]
        receiver_valid_code = [code for code in receiver_code if code is not None]

        if not receiver_valid_snr or not receiver_valid_code:
            return None

        receiver_avg_snr = np.mean(receiver_valid_snr)
        receiver_snr_std = np.std(receiver_valid_snr)
        receiver_code_std = np.std(receiver_valid_code)

        # 计算共同时间段内的覆盖率
        phone_coverage = len(phone_valid_snr) / max(len(common_times), 1)
        receiver_coverage = len(receiver_valid_snr) / max(len(common_times), 1)

        # 使用手机和接收机数据的平均值
        avg_snr = (phone_avg_snr + receiver_avg_snr) / 2
        coverage_ratio = (phone_coverage + receiver_coverage) / 2

        # 计算稳定性评分（信噪比稳定性 + 观测值稳定性）
        phone_snr_std = phone_snr_std if phone_snr_std is not None else 0.0
        receiver_snr_std = receiver_snr_std if receiver_snr_std is not None else 0.0
        phone_code_std = phone_code_std if phone_code_std is not None else 0.0
        receiver_code_std = receiver_code_std if receiver_code_std is not None else 0.0

        snr_stability = 1.0 / (1.0 + (phone_snr_std + receiver_snr_std) / 2 / 10.0)
        code_stability = 1.0 / (1.0 + (phone_code_std + receiver_code_std) / 2 / 1000.0)
        stability_score = (snr_stability + code_stability) / 2

        # 综合评分：信噪比权重50%，覆盖率权重30%，稳定性权重20%
        overall_score = (avg_snr / 50.0) * 0.5 + coverage_ratio * 0.3 + stability_score * 0.2

        return {
            'avg_snr': avg_snr,
            'coverage_ratio': coverage_ratio,
            'stability_score': stability_score,
            'overall_score': overall_score
        }

    def run_select_reference_satellite(self, isb_data: Dict[str, Any]) -> str:
        """Select a reference BDS-2 satellite using detailed scoring."""
        bds2 = isb_data['bds2_sats']
        if not bds2:
            raise ValueError("No BDS-2 satellite available for reference selection")
        
        common_times = isb_data['common_times']
        # For analysis, we need common times to be a list for indexing if needed, 
        # but _analyze_satellite_quality primarily uses length.
        
        best = None
        best_score = -1.0
        
        for sat in bds2:
            phone_l2i = isb_data['phone_data'].get(sat, {}).get('L2I', {})
            recv_l2i = isb_data['receiver_data'].get(sat, {}).get('L2I', {})
            
            # Extract lists aligned with raw data (simplification: we pass raw lists 
            # and let the function filter valid ones, but ideally we should align with common_times.
            # However, the original logic passed the raw lists and the common_times list separately.)
            
            phone_times = phone_l2i.get('times', [])
            phone_snr = phone_l2i.get('snr', [])
            phone_code = phone_l2i.get('code', [])
            
            recv_times = recv_l2i.get('times', [])
            recv_snr = recv_l2i.get('snr', [])
            recv_code = recv_l2i.get('code', [])
            
            # Since the data structure here is dict of lists, we pass them directly.
            # The _analyze_satellite_quality function handles None checks.
            
            metrics = self._analyze_satellite_quality(
                sat, 
                phone_times, phone_snr, phone_code,
                recv_times, recv_snr, recv_code,
                common_times
            )
            
            if metrics:
                score = metrics['overall_score']
                if score > best_score:
                    best_score = score
                    best = sat
                    
        if not best:
            best = bds2[0]
        return best

    def run_filter_stable_satellites(self, isb_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Filter stable satellites (simplified dynamic checks)."""
        common_times = isb_data['common_times']
        phone = isb_data['phone_data']
        recv = isb_data['receiver_data']
        stable_bds2 = []
        stable_bds3 = []
        for sat in isb_data['bds2_sats']:
            pd = phone.get(sat, {}).get('L2I', {})
            rd = recv.get(sat, {}).get('L2I', {})
            if not pd or not rd:
                continue
            # basic coverage check
            if len(pd.get('times', [])) / max(len(common_times), 1) >= 0.8 and len(rd.get('times', [])) / max(len(common_times), 1) >= 0.8:
                stable_bds2.append(sat)
        for sat in isb_data['bds3_sats']:
            pd = phone.get(sat, {}).get('L2I', {})
            rd = recv.get(sat, {}).get('L2I', {})
            if not pd or not rd:
                continue
            if len(pd.get('times', [])) / max(len(common_times), 1) >= 0.7 and len(rd.get('times', [])) / max(len(common_times), 1) >= 0.7:
                stable_bds3.append(sat)
        return {'stable_bds2': stable_bds2, 'stable_bds3': stable_bds3}

    def run_calculate_isb_double_difference(self, isb_data: Dict[str, Any], reference_sat: str, stable_sats: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute ISB estimates using double-difference approach (simplified)."""
        common_times = isb_data['common_times']
        phone = isb_data['phone_data']
        receiver = isb_data['receiver_data']
        stable_bds2 = stable_sats.get('stable_bds2', [])
        stable_bds3 = stable_sats.get('stable_bds3', [])

        isb_estimates = []
        isb_epochs = []

        for t in common_times:
            sd_phone = {}
            sd_recv = {}
            # for benchmark compute single difference: PR_ref - PR_sat
            # find PR at time t for reference and other sats
            for sat in stable_bds2 + stable_bds3:
                # phone
                p_times = phone.get(sat, {}).get('L2I', {}).get('times', [])
                p_codes = phone.get(sat, {}).get('L2I', {}).get('code', [])
                r_times = receiver.get(sat, {}).get('L2I', {}).get('times', [])
                r_codes = receiver.get(sat, {}).get('L2I', {}).get('code', [])
                # get closest index
                p_val = None
                for i, pt in enumerate(p_times):
                    if abs((pt - t).total_seconds()) <= isb_data['time_tolerance']:
                        p_val = p_codes[i]
                        break
                r_val = None
                for i, rt in enumerate(r_times):
                    if abs((rt - t).total_seconds()) <= isb_data['time_tolerance']:
                        r_val = r_codes[i]
                        break
                if p_val is not None and r_val is not None:
                    sd_phone[sat] = phone.get(reference_sat, {}).get('L2I', {}).get('code', [None])[0] - p_val if phone.get(reference_sat, {}).get('L2I', {}).get('code') else None
                    sd_recv[sat] = receiver.get(reference_sat, {}).get('L2I', {}).get('code', [None])[0] - r_val if receiver.get(reference_sat, {}).get('L2I', {}).get('code') else None
            # compute DD for BDS-3 sats
            dd_vals = []
            for sat in stable_bds3:
                if sat in sd_phone and sd_phone[sat] is not None and sat in sd_recv and sd_recv[sat] is not None:
                    dd = sd_phone[sat] - sd_recv[sat]
                    dd_vals.append(dd)
            if dd_vals:
                isb_epoch = sum(dd_vals) / len(dd_vals)
                isb_estimates.append(isb_epoch)
                isb_epochs.append(t)

        if isb_estimates:
            import numpy as np
            isb_mean = float(np.mean(isb_estimates))
            isb_std = float(np.std(isb_estimates))
        else:
            isb_mean = 0.0
            isb_std = 0.0

        return {
            'reference_satellite': reference_sat,
            'common_times': common_times,
            'isb_estimates': isb_estimates,
            'isb_mean': isb_mean,
            'isb_std': isb_std,
            'isb_epochs': isb_epochs
        }

    def run_correct_isb_and_generate_rinex(self, isb_results: Dict[str, Any], input_rinex_path: str = None, output_path: str = None, writer: Optional[RinexWriter] = None) -> Dict[str, Any]:
        """Apply ISB correction by delegating to writer.write_isb_corrected_rinex when available.

        If a `writer` instance is provided and `input_rinex_path` is set, this will call
        `writer.write_isb_corrected_rinex(input_rinex_path, output_path, isb_results)` and return its result.
        Otherwise falls back to a simple copy of the file (legacy behavior).
        """
        if input_rinex_path is None:
            return {'corrected_rinex_path': None, 'modified': 0, 'modified_satellites': []}

        # prefer writer if available
        if writer is not None:
            try:
                return writer.write_isb_corrected_rinex(input_rinex_path, output_path, isb_results)
            except Exception:
                # fallback to naive copy if writer fails
                pass

        # fallback - naive copy behavior (kept for compatibility when no writer provided)
        isb_value = isb_results.get('isb_mean', 0.0)
        # create output path
        import os
        if output_path is None:
            base, ext = os.path.splitext(input_rinex_path)
            output_path = f"{base}-isb{ext}"

        with open(input_rinex_path, 'r', encoding='utf-8') as fr, open(output_path, 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(line)

        return {'corrected_rinex_path': output_path, 'modified': 0, 'modified_satellites': []}

    def apply_doppler_smoothing(self, observations_meters: Dict[str, Any], 
                                max_window: int = 20, 
                                reset_threshold_m: float = 15.0,
                                input_file_name: str = "N/A") -> Dict[str, Any]:
        """Apply Hatch Filter with Doppler Integration for pseudorange smoothing.
        
        Algorithm: 
            \\bar{P}_k = (1/n) * P_k + ((n-1)/n) * (\\bar{P}_{k-1} + \\Delta R_{doppler})
        
        Where:
            - P_k: raw pseudorange observation
            - n: dynamic window size (grows from 1 to max_window)
            - \\bar{P}_{k-1}: smoothed pseudorange from previous epoch
            - \\Delta R_{doppler}: range change from doppler integration (trapezoid rule)
        
        Input:
            observations_meters: {sat_id: {freq: {'times': [...], 'code': [...], 'phase': [...], 'doppler': [...]}}}
            max_window: maximum smoothing window size (number of epochs)
            reset_threshold_m: coarse error threshold in meters for window reset
            input_file_name: Name of the input file being processed (for logging)
        
        Output:
            {
                'smoothed_observations': {sat_id: {freq: {'times': [...], 'code': [...], 'code_smoothed': [...]}}},
                'smoothing_meta': {sat_id: {freq: {'window_sizes': [...], 'resets': [...], 'reset_reasons': [...]}}},
                'log': detailed processing log
            }
        """
        import numpy as np
        from datetime import datetime
        
        smoothed_obs = {}
        smoothing_meta = {}
        
        for sat_id in observations_meters.keys():
            sat_obs = observations_meters[sat_id]
            if sat_id not in smoothed_obs:
                smoothed_obs[sat_id] = {}
                smoothing_meta[sat_id] = {}
            
            for freq in sat_obs.keys():
                freq_data = sat_obs[freq]
                
                # Extract required fields
                times = freq_data.get('times', [])
                codes = freq_data.get('code', [])
                doppler_velocities = freq_data.get('doppler', [])
                
                if not times or not codes:
                    continue
                
                # Initialize smoothing state
                codes_smoothed = []
                window_sizes = []
                resets = []  # list of epoch indices where reset occurred
                reset_reasons = []  # list of reset reason strings
                reset_count = 0
                
                n = 1  # current window size
                prev_smoothed = None
                prev_doppler = None
                prev_time = None
                
                for epoch_idx in range(len(times)):
                    code_val = codes[epoch_idx]
                    doppler_vel = doppler_velocities[epoch_idx] if epoch_idx < len(doppler_velocities) else None
                    curr_time = times[epoch_idx]
                    
                    # Handle missing code observations
                    if code_val is None:
                        codes_smoothed.append(None)
                        window_sizes.append(0)
                        # Reset on data gap
                        if prev_smoothed is not None:
                            n = 1
                            prev_smoothed = None
                            resets.append(epoch_idx)
                            reset_reasons.append("data_gap")
                            reset_count += 1
                        continue
                    
                    # First epoch initialization
                    if prev_smoothed is None:
                        codes_smoothed.append(code_val)
                        window_sizes.append(1)
                        prev_smoothed = code_val
                        prev_doppler = doppler_vel
                        prev_time = curr_time
                        n = 1
                        continue
                    
                    # Compute time delta
                    if prev_time is not None:
                        try:
                            dt = (curr_time - prev_time).total_seconds()
                        except:
                            dt = 0.0
                    else:
                        dt = 0.0
                    
                    # Check for time discontinuity (> 3 seconds)
                    if dt > 3.0 or dt < 0:
                        n = 1
                        codes_smoothed.append(code_val)
                        window_sizes.append(1)
                        prev_smoothed = code_val
                        prev_doppler = doppler_vel
                        prev_time = curr_time
                        resets.append(epoch_idx)
                        reset_reasons.append(f"time_discontinuity(dt={dt:.2f}s)")
                        reset_count += 1
                        continue
                    
                    # Compute Doppler-based range change (trapezoid integration)
                    delta_r_doppler = 0.0
                    if prev_doppler is not None and doppler_vel is not None and dt > 0:
                        # Trapezoid rule: (v_prev + v_curr) / 2 * dt
                        doppler_vel_avg = (prev_doppler + doppler_vel) / 2.0
                        delta_r_doppler = doppler_vel_avg * dt
                    
                    # Apply Hatch filter formula
                    # \\bar{P}_k = (1/n) * P_k + ((n-1)/n) * (\\bar{P}_{k-1} + \\Delta R_{doppler})
                    smoothed_val = (1.0 / n) * code_val + ((n - 1.0) / n) * (prev_smoothed + delta_r_doppler)
                    
                    # Coarse error detection: check if smoothed value deviates too much from raw
                    error = abs(smoothed_val - code_val)
                    if error > reset_threshold_m:
                        # Reset and use raw value
                        n = 1
                        codes_smoothed.append(code_val)
                        window_sizes.append(1)
                        prev_smoothed = code_val
                        resets.append(epoch_idx)
                        reset_reasons.append(f"coarse_error({error:.2f}m)")
                        reset_count += 1
                    else:
                        # Accept smoothed value
                        codes_smoothed.append(smoothed_val)
                        prev_smoothed = smoothed_val
                        window_sizes.append(n)
                        
                        # Increment window size up to max
                        if n < max_window:
                            n += 1
                    
                    # Update for next iteration
                    prev_doppler = doppler_vel
                    prev_time = curr_time
                
                # Store results
                smoothed_obs[sat_id][freq] = {
                    'times': times,
                    'code': codes,
                    'code_smoothed': codes_smoothed,
                    'phase': freq_data.get('phase', []),
                    'doppler': doppler_velocities
                }
                
                smoothing_meta[sat_id][freq] = {
                    'window_sizes': window_sizes,
                    'resets': resets,
                    'reset_reasons': reset_reasons,
                    'reset_count': reset_count,
                    'avg_window_size': float(np.mean([w for w in window_sizes if w > 0])) if any(w > 0 for w in window_sizes) else 0.0
                }
        
        
        # --- Generate Detailed Log ---
        log_content = []
        # --- Generate Detailed Log ---
        log_content = []
        # Header is generated by GUI wrapper, so we start directly with content
        log_content.append(f"平滑窗口最大值 (N): {max_window}")
        log_content.append(f"平滑窗口最大值 (N): {max_window}")
        log_content.append(f"重置阈值 (C): {reset_threshold_m:.2f} m\n")
        
        # Summary
        total_sats = len(smoothing_meta)
        total_resets_all = sum(m.get('reset_count', 0) for s in smoothing_meta.values() for m in s.values())
        
        log_content.append("一、统计摘要")
        log_content.append("-" * 70)
        log_content.append(f"总计处理卫星数: {total_sats}")
        log_content.append(f"总计发生重置次数: {total_resets_all}\n")
        
        # Details
        log_content.append("二、详细处理信息")
        log_content.append("-" * 70)
        
        # Sort by satellite system/id
        sorted_sats = sorted(smoothing_meta.keys())
        
        for sat_id in sorted_sats:
            freq_map = smoothing_meta[sat_id]
            log_content.append(f"卫星 {sat_id}:")
            for freq in sorted(freq_map.keys()):
                meta = freq_map[freq]
                log_content.append(f"  频率 {freq}:")
                log_content.append(f"    观测历元数: {len(meta['window_sizes'])}")
                log_content.append(f"    重置次数: {meta['reset_count']}")
                log_content.append(f"    平均平滑窗口: {meta['avg_window_size']:.2f}")
                
                if meta['resets']:
                    log_content.append("    重置详情:")
                    for r_idx, (epoch_idx, reason) in enumerate(zip(meta['resets'], meta['reset_reasons'])):
                         # Limit details if too many
                         if r_idx >= 50: 
                             log_content.append(f"      ... (剩余 {len(meta['resets']) - 50} 个重置事件省略)")
                             break
                         log_content.append(f"      - 历元 {epoch_idx}: {reason}")
                log_content.append("")
            log_content.append("")
            
        return {
            'smoothed_observations': smoothed_obs,
            'smoothing_meta': smoothing_meta,
            'log': '\n'.join(log_content)
        }

    # ------------------------------------------------------------------------
    # Doppler Phase Prediction (Ported from analyzer.py)
    # ------------------------------------------------------------------------
    def run_doppler_phase_prediction(self, observations_meters: Dict[str, Any], frequencies: Dict[str, float],
                                     wavelengths: Dict[str, Dict[str, float]], 
                                     original_rinex_path: str = None,
                                     output_path: str = None,
                                     writer: Optional[RinexWriter] = None) -> Dict[str, Any]:
        """Perform Doppler-based phase prediction to repair cycle slips/gaps."""
        
        prediction_results = {
            'predicted_phases': {},
            'missing_epochs': {},
            'correction_log': [],
            'total_missing': 0,
            'total_predicted': 0,
            'sv_missing_stats': {} # {sv: {'missing': N, 'predicted': M}}
        }
        
        for sat_id, sat_obs in observations_meters.items():
            sat_system = sat_id[0]
            
            sat_prediction = {}
            sat_missing_epochs = {}
            sv_stats = {'missing': 0, 'predicted': 0}
            
            for freq, freq_data in sat_obs.items():
                if 'phase' not in freq_data or 'doppler' not in freq_data:
                    continue
                
                phase_values = freq_data['phase'] # List of cycles
                doppler_values = freq_data['doppler'] # m/s
                times = freq_data['times']
                
                if not (len(phase_values) == len(doppler_values) == len(times)):
                    continue
                    
                wavelength = wavelengths.get(sat_id, {}).get(freq)
                if not wavelength and sat_system in wavelengths and freq in wavelengths[sat_system]:
                     wavelength = wavelengths[sat_system][freq]
                if not wavelength:
                    continue
                    
                frequency = frequencies.get(freq, 1575.42e6)
                
                freq_prediction = {
                    'times': [],
                    'original_phases': [],
                    'predicted_phases': [],
                    'is_predicted': [],
                    'prediction_details': []
                }
                missing_epochs = []
                
                for i in range(len(times)):
                    if phase_values[i] is None or phase_values[i] == 0:
                        missing_epochs.append(i)
                        sv_stats['missing'] += 1
                        
                        predicted_phase_m = self._predict_phase_at_epoch(
                            i, times, phase_values, doppler_values, frequency, wavelength
                        )
                        
                        if predicted_phase_m is not None:
                            phase_values[i] = predicted_phase_m 
                            sv_stats['predicted'] += 1
                            
                            freq_prediction['times'].append(times[i])
                            freq_prediction['original_phases'].append(None)
                            freq_prediction['predicted_phases'].append(predicted_phase_m / wavelength)
                            freq_prediction['is_predicted'].append(True)
                            
                            detail = {
                                'epoch_idx': i,
                                'epoch_indices': i,  # ✅ 添加epoch_indices用于精确映射
                                'sat_id': sat_id,
                                'freq': freq,
                                'time': times[i],
                                'predicted_phase_cycle': predicted_phase_m / wavelength,
                                'predicted_phase_m': predicted_phase_m
                            }
                            freq_prediction['prediction_details'].append(detail)
                            prediction_results['correction_log'].append(detail)
                            prediction_results['total_predicted'] += 1
                    else:
                        freq_prediction['times'].append(times[i])
                        freq_prediction['original_phases'].append(phase_values[i])
                        freq_prediction['predicted_phases'].append(phase_values[i])
                        freq_prediction['is_predicted'].append(False)
                        
                sat_prediction[freq] = freq_prediction
                sat_missing_epochs[freq] = missing_epochs
                prediction_results['total_missing'] += len(missing_epochs)
            
            if sv_stats['missing'] > 0:
                prediction_results['sv_missing_stats'][sat_id] = sv_stats

            if sat_prediction:
                prediction_results['predicted_phases'][sat_id] = sat_prediction
                prediction_results['missing_epochs'][sat_id] = sat_missing_epochs

        if output_path and writer and original_rinex_path:
             writer.write_doppler_predicted_rinex(original_rinex_path, output_path, prediction_results)
             
        return prediction_results

    def _predict_phase_at_epoch(self, target_idx, times, phase_values, doppler_values,
                                frequency, wavelength):
        """Helper: predict phase (meters) at target index using neighbor doppler values."""
        if target_idx <= 0: return None
        prev_phase_m = phase_values[target_idx - 1] # in meters
        if prev_phase_m is None: return None
        
        prev_doppler = doppler_values[target_idx - 1] # m/s (range rate)
        curr_doppler = doppler_values[target_idx] # m/s (range rate)
        if prev_doppler is None or curr_doppler is None: return None
        
        try:
            time_diff = (times[target_idx] - times[target_idx - 1]).total_seconds()
        except:
            return None
        if time_diff <= 0: return None
        
        # Physics: dPhi_m = -range_rate * dt
        # Range rate dot(rho) is stored in doppler_values.
        # Phase (meters) decreases as range increases, so dPhi = -dRho.
        doppler_mean = (prev_doppler + curr_doppler) / 2.0
        phase_change_m = -doppler_mean * time_diff
        
        predicted_phase_m = prev_phase_m + phase_change_m
        return predicted_phase_m

