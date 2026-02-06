from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import os
import datetime
import pandas as pd
import numpy as np


class RinexWriter:
    """RINEX writing utilities for cleaned/corrected outputs.

    Methods try to mirror the behavior of the original Analyzer implementations
    but operate as pure functions that accept inputs and return results.
    """

    def write_corrected_rinex(self,
                              original_path: str,
                              output_path: Optional[str],
                              corrected_data: Dict[str, Any],
                              roc_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply phase corrections to a RINEX observation file and write a corrected copy.

        Returns a dict with keys: output_path, modification_details, total_modifications
        """
        if output_path is None:
            base_dir = os.path.dirname(original_path)
            bn = os.path.basename(original_path)
            name, ext = os.path.splitext(bn)
            output_path = os.path.join(base_dir, f"{name}_corrected{ext}")

        with open(original_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        # parse header to get obs types table
        rinex_data_header = {'header': {}}
        header_end = -1
        for i, line in enumerate(original_lines):
            if 'END OF HEADER' in line:
                header_end = i
                break
        if header_end > 0:
            for i in range(header_end):
                line = original_lines[i]
                if 'SYS / # / OBS TYPES' in line:
                    system = line[0] if line[0] != ' ' else None
                    tokens = line[1:].split()
                    num_types = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 0
                    obs_types_list = []
                    obs_types_list.extend(line[7:60].split())
                    j = i + 1
                    while j < header_end and len(obs_types_list) < num_types:
                        cont_line = original_lines[j]
                        if 'SYS / # / OBS TYPES' in cont_line:
                            obs_types_list.extend(cont_line[7:60].split())
                            j += 1
                        else:
                            break
                    if system:
                        rinex_data_header['header'][f'obs_types_{system}'] = obs_types_list[:num_types] if num_types > 0 else obs_types_list

        # parse epoch timestamps
        epoch_timestamps = {}
        current_epoch_idx = 0
        for line in original_lines:
            if line.startswith('>'):
                current_epoch_idx += 1
                parts = line[1:].split()
                if len(parts) >= 6:
                    second_float = float(parts[5])
                    epoch_timestamps[current_epoch_idx] = pd.Timestamp(
                        year=int(parts[0]), month=int(parts[1]), day=int(parts[2]),
                        hour=int(parts[3]), minute=int(parts[4]), second=int(second_float),
                        microsecond=int((second_float - int(second_float)) * 1000000)
                    )

        # build system obs info mapping
        system_obs_info = {}
        header_end_idx = next((i for i, line in enumerate(original_lines) if 'END OF HEADER' in line), -1)
        i = 0
        while i < len(original_lines) and i < header_end_idx:
            line = original_lines[i]
            if 'SYS / # / OBS TYPES' in line:
                system = line[0]
                try:
                    num_types = int(line[3:6].strip())
                except Exception:
                    num_types = 0
                obs_types = line.split()[2:]
                # Handle continuation lines
                j = i + 1
                while j < header_end_idx and len(obs_types) < num_types:
                    next_line = original_lines[j]
                    if 'SYS / # / OBS TYPES' in next_line and next_line[0] == ' ':
                        obs_types.extend(next_line.split())
                        j += 1
                    else:
                        break
                obs_types = obs_types[:num_types] if num_types > 0 else obs_types
                freq_to_indices = defaultdict(dict)
                for idx, obs in enumerate(obs_types):
                    if obs.startswith('C'):
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['code'] = idx
                    elif obs.startswith('L'):
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['phase'] = idx
                    elif obs.startswith('D'):
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['doppler'] = idx
                system_obs_info[system] = {'obs_types': obs_types, 'freq_to_indices': freq_to_indices}
            i += 1

        output_lines = original_lines.copy()
        modification_details = defaultdict(list)
        total_modifications = 0

        # Apply corrections
        for sat_id, freq_map in corrected_data.items():
            sat_system = sat_id[0]
            sat_prn = sat_id[1:].zfill(2)
            sys_info = system_obs_info.get(sat_system, {})
            freq_indices = sys_info.get('freq_to_indices', {})
            if not freq_indices:
                continue
            for freq, cdata in freq_map.items():
                times = cdata.get('times', [])
                corrected_phases = cdata.get('corrected_phase', [])
                wavelengths = cdata.get('wavelengths', [])
                if freq not in freq_indices or 'phase' not in freq_indices[freq]:
                    continue
                phase_field_idx = freq_indices[freq]['phase']
                for time_idx, (ct, corrected_phase_m, wavelength) in enumerate(zip(times, corrected_phases, wavelengths)):
                    if corrected_phase_m is None or wavelength is None:
                        continue
                    # locate epoch by matching timestamp with tolerance
                    epoch_start = -1
                    for i, line in enumerate(output_lines):
                        if line.startswith('>'):
                            parts = line[1:].split()
                            if len(parts) >= 6:
                                try:
                                    secf = float(parts[5])
                                    line_epoch = pd.Timestamp(
                                        year=int(parts[0]), month=int(parts[1]), day=int(parts[2]),
                                        hour=int(parts[3]), minute=int(parts[4]), second=int(secf),
                                        microsecond=int((secf - int(secf)) * 1000000)
                                    )
                                    if abs((ct - line_epoch).total_seconds()) < 0.1:
                                        epoch_start = i
                                        break
                                except Exception:
                                    continue
                    if epoch_start < 0:
                        continue
                    # find satellite line
                    sat_line_idx = -1
                    j = epoch_start + 1
                    while j < len(output_lines) and not output_lines[j].startswith('>'):
                        line = output_lines[j]
                        if len(line) >= 3 and line[0] == sat_system and line[1:3].strip().zfill(2) == sat_prn:
                            sat_line_idx = j
                            break
                        j += 1
                    if sat_line_idx < 0:
                        continue
                    # modify phase field
                    original_line = output_lines[sat_line_idx]
                    modified_line = list(original_line)
                    start_pos = 3 + phase_field_idx * 16
                    end_pos = start_pos + 16
                    if end_pos > len(modified_line):
                        continue
                    original_field = original_line[start_pos:end_pos].strip()
                    try:
                        original_phase_cycle = float(original_field) if original_field else None
                    except Exception:
                        original_phase_cycle = None
                    if original_phase_cycle is None:
                        # If original empty, skip
                        continue
                    original_phase_m = original_phase_cycle * wavelength

                    # compute correction_amount if possible (using roc_model)
                    correction_amount = 0.0
                    if roc_model:
                        key = f"{sat_system}_{freq}"
                        if key in roc_model:
                            roc_rate = roc_model[key].get('roc_rate', 0.0)
                            t0 = times[0] if times else ct
                            time_diff = (ct - t0).total_seconds()
                            correction_amount = -roc_rate * time_diff

                    corrected_phase_m2 = original_phase_m + correction_amount
                    corrected_phase_cycle2 = corrected_phase_m2 / wavelength
                    formatted_phase = f"{corrected_phase_cycle2:14.3f}"
                    modified_line[start_pos:end_pos] = formatted_phase.rjust(16)
                    output_lines[sat_line_idx] = ''.join(modified_line)

                    modification_info = {
                        'freq': freq,
                        'epoch': ct,
                        'original_phase_cycle': original_phase_cycle,
                        'original_phase_m': original_phase_m,
                        'corrected_phase_cycle': corrected_phase_cycle2,
                        'corrected_phase_m': corrected_phase_m2,
                        'wavelength': wavelength,
                        'correction_amount': correction_amount,
                        'formatted_phase': formatted_phase
                    }
                    modification_details[sat_id].append(modification_info)
                    total_modifications += 1

        # write output
        dirpath = os.path.dirname(output_path) or '.'
        os.makedirs(dirpath, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)

        return {
            'output_path': output_path,
            'modification_details': modification_details,
            'total_modifications': total_modifications
        }

    def write_cleaned_rinex(self,
                            original_path: str,
                            output_path: Optional[str],
                            double_diffs: Dict[str, Any],
                            triple_errors: Dict[str, Any],
                            max_threshold_limit: Optional[Dict[str, float]] = None,
                            enable_cci: bool = True,
                            cmc_flags: Dict[str, Dict[str, list]] = None,
                            observations_meters: Dict[str, Any] = None,
                            cmc_threshold: float = 5.0,
                            threshold_mode: str = 'fixed',
                            calculated_thresholds: Dict[str, Any] = None) -> Dict[str, Any]:
        """Remove outliers by clearing offending fields and write a cleaned RINEX.

        cmc_flags (optional): {sat_id: {freq: [bool,...]}} where True indicates the epoch index (0-based) flagged by CMC threshold.
        Returns dict with output_path and debug log path and stats.
        """        """Remove outliers by clearing offending fields and write a cleaned RINEX.

        Returns dict with output_path and debug log path and stats.
        """
        max_threshold_limit = max_threshold_limit or {'code': 50.0, 'phase': 5.0, 'doppler': 10.0}

        phone_file_name = os.path.basename(original_path)
        phone_file_name_no_ext = os.path.splitext(phone_file_name)[0]
        # Removed unconditional Coarse error directory creation here

        # select base file (for this pure writer we just use original_path)
        input_file = original_path
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # parse epochs and obs types similar to analyzer
        epoch_timestamps = {}
        current_epoch = 0
        for line in lines:
            if line.startswith('>'):
                current_epoch += 1
                parts = line[1:].split()
                if len(parts) >= 6:
                    try:
                        year = int(parts[0]); month = int(parts[1]); day = int(parts[2])
                        hour = int(parts[3]); minute = int(parts[4]); secf = float(parts[5])
                        epoch_timestamps[current_epoch] = pd.Timestamp(
                            year=year, month=month, day=day, hour=hour, minute=minute,
                            second=int(secf), microsecond=int((secf - int(secf)) * 1000000)
                        )
                    except Exception:
                        epoch_timestamps[current_epoch] = None

        system_obs_info = {}
        header_end_idx = next((i for i, line in enumerate(lines) if 'END OF HEADER' in line), -1)
        i = 0
        while i < len(lines) and i < header_end_idx:
            line = lines[i]
            if 'SYS / # / OBS TYPES' in line:
                system = line[0]
                try:
                    num_types = int(line[3:6].strip())
                except Exception:
                    num_types = 0
                obs_types = line.split()[2:]
                # Handle continuation lines
                j = i + 1
                while j < header_end_idx and len(obs_types) < num_types:
                    next_line = lines[j]
                    if 'SYS / # / OBS TYPES' in next_line and next_line[0] == ' ':
                        obs_types.extend(next_line.split())
                        j += 1
                    else:
                        break
                obs_types = obs_types[:num_types] if num_types > 0 else obs_types
                freq_to_indices = defaultdict(dict)
                for idx, obs in enumerate(obs_types):
                    if obs.startswith('C'):
                        freq = f"L{obs[1:]}"; freq_to_indices[freq]['code'] = idx
                    elif obs.startswith('L'):
                        freq = f"L{obs[1:]}"; freq_to_indices[freq]['phase'] = idx
                    elif obs.startswith('D'):
                        freq = f"L{obs[1:]}"; freq_to_indices[freq]['doppler'] = idx
                system_obs_info[system] = {'obs_types': obs_types, 'freq_to_indices': freq_to_indices}
            i += 1

        # identify outlier epochs
        outlier_epochs = defaultdict(lambda: defaultdict(list))
        outlier_details = defaultdict(list)
        
        # Build reverse mapping: timestamp -> epoch_idx for precise matching
        timestamp_to_epoch = {ts: idx for idx, ts in epoch_timestamps.items()}

        for sat_id, freq_data in double_diffs.items():
            for freq, dd_data in freq_data.items():
                # Get times from double_diff data (these are actual timestamps from observations_meters)
                dd_times = dd_data.get('times', [])
                
                for obs_type in ['code', 'phase', 'doppler']:
                    dd_key = f"dd_{obs_type}"
                    if dd_key not in dd_data:
                        continue
                    triple_sigma = triple_errors.get(sat_id, {}).get(freq, {}).get(obs_type, {}).get('threshold', 0)
                    if triple_sigma <= 0:
                        threshold = max_threshold_limit[obs_type]
                    else:
                        threshold = min(triple_sigma, max_threshold_limit[obs_type])
                        threshold = max(threshold, 0.01)
                    valid_dd = [(i, d) for i, d in enumerate(dd_data[dd_key]) if d is not None and not np.isnan(d)]
                    for orig_idx, dd_value in valid_dd:
                        if abs(dd_value) > threshold:
                            # Use timestamp matching instead of index mapping
                            if orig_idx < len(dd_times):
                                dd_timestamp = dd_times[orig_idx]
                                # Find matching epoch_idx by timestamp
                                epoch_idx = timestamp_to_epoch.get(dd_timestamp)
                                if epoch_idx is None:
                                    # Timestamp not found in RINEX file - skip this outlier
                                    continue
                            else:
                                # No timestamp available - skip
                                continue
                            
                            ts_obj = epoch_timestamps.get(epoch_idx)
                            timestamp = str(ts_obj) if ts_obj else f"未知时间戳(历元{epoch_idx})"
                            outlier_details[sat_id].append({'obs_type': obs_type, 'freq': freq, 'dd_value': dd_value, 'threshold_used': threshold, 'epoch_idx': epoch_idx, 'timestamp': timestamp})
                            outlier_epochs[sat_id][epoch_idx].append((obs_type, freq))


        # Apply CMC flags (if provided) and collect details
        if cmc_flags:
            pass
            
            for sat_id, freq_map in cmc_flags.items():
                for freq, flags in freq_map.items():
                    # Get observation data for calculating CMC change
                    codes, phases, times = [], [], []
                    if observations_meters and sat_id in observations_meters:
                        sat_data = observations_meters.get(sat_id, {}).get(freq, {})
                        codes = sat_data.get('code', [])
                        phases = sat_data.get('phase', [])
                        times = sat_data.get('times', [])
                    
                    for idx, flag in enumerate(flags):
                        if flag:
                            # idx is the array index (0-based) in this satellite's observation sequence
                            # We need to find the matching global epoch
                            # Note: idx对应observations_meters中该卫星/频率的第idx个历元（0-based）
                            
                            # Calculate CMC change value if data available
                            cmc_change = None
                            if len(codes) > idx and len(phases) > idx and idx > 0:
                                c_curr, p_curr = codes[idx], phases[idx]
                                
                                # Find the most recent valid previous observation
                                prev_idx = idx - 1
                                while prev_idx >= 0:
                                    if codes[prev_idx] is not None and phases[prev_idx] is not None:
                                        break
                                    prev_idx -= 1
                                
                                if prev_idx >= 0:
                                    c_prev, p_prev = codes[prev_idx], phases[prev_idx]
                                    
                                    if None not in [c_curr, p_curr, c_prev, p_prev]:
                                        cmc_curr = c_curr - p_curr
                                        cmc_prev = c_prev - p_prev
                                        cmc_change = abs(cmc_curr - cmc_prev)
                            
                            # Get timestamp from observations_meters times array
                            timestamp_obj = None
                            if idx < len(times) and times[idx] is not None:
                                timestamp_obj = times[idx]
                            
                            # Find matching global epoch by timestamp
                            epoch_idx = None
                            if timestamp_obj is not None:
                                # epoch_timestamps now contains pd.Timestamp objects directly
                                for e_idx, e_ts_obj in epoch_timestamps.items():
                                    if e_ts_obj is not None:
                                        # Compare with tolerance of 0.001 seconds
                                        if abs((timestamp_obj - e_ts_obj).total_seconds()) < 0.001:
                                            epoch_idx = e_idx
                                            break
                            
                            if epoch_idx is None:
                                continue
                            
                            # Get display timestamp
                            timestamp = str(timestamp_obj) if timestamp_obj else f"未知时间戳(观测序列索引{idx+1})"
                            
                            # Mark for clearing
                            outlier_epochs[sat_id][epoch_idx].append(('code', freq))
                            outlier_epochs[sat_id][epoch_idx].append(('phase', freq))
                            outlier_epochs[sat_id][epoch_idx].append(('doppler', freq))
                            
                            # Add to details
                            detail = {
                                'obs_type': 'cmc',
                                'freq': freq,
                                'epoch_idx': epoch_idx,
                                'timestamp': timestamp,
                                'threshold_used': cmc_threshold if cmc_threshold else 5.0,
                                'threshold': cmc_threshold if cmc_threshold else 5.0
                            }
                            if cmc_change is not None:
                                detail['dd_value'] = cmc_change  # Use dd_value for compatibility with logging
                                detail['diff_change'] = cmc_change
                            
                            outlier_details[sat_id].append(detail)
            
            # debug_log.close() - Removed


        # modify lines by clearing fields
        modified_count = defaultdict(int)
        modified_satellites = set()
        actually_modified_epochs = defaultdict(set)  # Track which epochs were actually modified
        skip_reasons = defaultdict(lambda: defaultdict(int))  # Track why modifications were skipped

        for sat_id, epoch_obs_map in outlier_epochs.items():
            sat_system = sat_id[0]; sat_prn = sat_id[1:].zfill(2)
            system_info = system_obs_info.get(sat_system, {})
            freq_indices = system_info.get('freq_to_indices', {})
            if not freq_indices:
                skip_reasons[sat_id]['no_freq_indices'] += len(epoch_obs_map)
                continue
            for epoch_idx, obs_freq_list in epoch_obs_map.items():
                target_ts = epoch_timestamps.get(epoch_idx)
                timestamp_str = str(target_ts) if target_ts else f"未知时间戳(历元{epoch_idx})"
                # find epoch start by matching timestamp object
                epoch_start = -1
                for i, line in enumerate(lines):
                    if line.startswith('>'):
                        parts = line[1:].split()
                        if len(parts) >= 6:
                            try:
                                y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
                                h, mi, s = int(parts[3]), int(parts[4]), float(parts[5])
                                line_ts = pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi,
                                                       second=int(s), microsecond=int((s - int(s)) * 1000000))
                                if target_ts and abs((target_ts - line_ts).total_seconds()) < 0.001:
                                    epoch_start = i
                                    break
                            except Exception:
                                continue
                if epoch_start < 0:
                    skip_reasons[sat_id]['epoch_not_found'] += 1
                    continue
                sat_line_idx = -1
                j = epoch_start + 1
                while j < len(lines) and not lines[j].startswith('>'):
                    line = lines[j]
                    if len(line) >= 3 and line[0] == sat_system and line[1:3].strip().zfill(2) == sat_prn:
                        sat_line_idx = j
                        break
                    j += 1
                if sat_line_idx < 0:
                    skip_reasons[sat_id]['sat_line_not_found'] += 1
                    continue
                original_line = lines[sat_line_idx]
                modified_line = list(original_line)
                field_modified = False
                modified_fields = []
                for obs_type, freq in obs_freq_list:
                    if freq not in freq_indices or obs_type not in freq_indices[freq]:
                        skip_reasons[sat_id]['freq_or_obstype_not_in_indices'] += 1
                        continue
                    field_idx = freq_indices[freq][obs_type]
                    start_pos = 3 + field_idx * 16
                    end_pos = start_pos + 16
                    if end_pos > len(modified_line):
                        skip_reasons[sat_id]['field_out_of_bounds'] += 1
                        continue
                    original_field = original_line[start_pos:end_pos].strip()
                    if original_field:
                        modified_line[start_pos:end_pos] = ' ' * 16
                        modified_count[obs_type] += 1
                        field_modified = True
                        modified_fields.append(f"{freq}({obs_type})")
                    else:
                        skip_reasons[sat_id]['field_already_empty'] += 1
                if field_modified:
                    lines[sat_line_idx] = ''.join(modified_line)
                    modified_satellites.add(sat_id)
                    actually_modified_epochs[sat_id].add(epoch_idx)  # Record this epoch was modified

        # determine output path
        if output_path:
            # Use provided path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            log_dir = os.path.dirname(output_path)
            # Decide log name based on operation type
            if cmc_flags:
                 debug_file_name = "code_phase_cleaning.log"
            else:
                 debug_file_name = "double_diffs_cleaning.log"
        else:
            # Fallback to default logic
            result_dir = os.path.dirname(original_path)
            coarse_error_dir = os.path.join(result_dir, "Coarse error")
            os.makedirs(coarse_error_dir, exist_ok=True)
            
            if enable_cci:
                base_ext = os.path.splitext(original_path)[1]
                modified_file_name = f"cleaned2-{phone_file_name_no_ext}-cc inconsistency{base_ext}"
            else:
                base_ext = os.path.splitext(original_path)[1]
                modified_file_name = f"cleaned2-{phone_file_name_no_ext}{base_ext}"
            output_path = os.path.join(coarse_error_dir, modified_file_name)
            log_dir = coarse_error_dir
            if cmc_flags:
                 debug_file_name = "code_phase_cleaning.log"
            else:
                 debug_file_name = "double_diffs_cleaning.log"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # write detailed debug log (Strict Chinese Format)
        debug_path = os.path.join(log_dir, debug_file_name)
        log_content = []
        
        # Header
        log_content.append("=" * 70 + "\n")
        if cmc_flags:
            log_content.append("基于伪距相位差值变化的异常观测值剔除日志 (CMC Cleaning)\n")
        else:
            log_content.append("RINEX 粗差处理详细日志 (Double Difference Cleaning)\n")
        log_content.append("=" * 70 + "\n\n")
        log_content.append(f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Paths & Methodology
        log_content.append(f"输入文件: {os.path.abspath(original_path)}\n")
        log_content.append(f"输出文件: {os.path.abspath(output_path)}\n\n")
        
        log_content.append("算法原理:\n")
        if cmc_flags:
            log_content.append("  基于伪距与相位观测值的组合 (Code-Minus-Carrier)，监测其随时间的变化率。\n")
            log_content.append("  若变化率超过预设阈值（默认 5.0m），则判定该历元存在粗差或严重的电离层变化，予以剔除。\n")
            log_content.append("\n  【自适应阈值机制】:\n")
            log_content.append("    1. 预过滤 (sample_filter_limit): 在计算P99之前，先剔除超过100.0m的极端值，\n")
            log_content.append("       避免离谱的跳变污染统计计算\n")
            log_content.append("    2. 熔断机制 (sanity_limit): 若P99 > 10.0m，判定统计计算失效，\n")
            log_content.append("       触发熔断器，强制使用保底阈值（GUI设定）\n")
            log_content.append("    3. 阈值计算: T_adaptive = max(P99 × 2.0, T_floor)，确保不过度剔除\n\n")
            
            # Add threshold mode information for CMC
            if threshold_mode == 'adaptive' and calculated_thresholds:
                log_content.append("阈值计算模式: 自适应 (基于分位数法)\n")
                log_content.append(f"  保底阈值 (T_floor): {cmc_threshold}m\n")
                log_content.append(f"  预过滤阈值 (sample_filter_limit): 100.0m (在计算P99前剔除)\n")
                log_content.append(f"  熔断阈值 (sanity_limit): 10.0m (P99超过此值时触发熔断)\n")
                log_content.append(f"  计算公式: T_adaptive = max(P99 × 2.0, T_floor)\n")
                log_content.append(f"  熔断机制: P99 > 10.0m 时判定统计失效，强制使用保底阈值\n\n")
                log_content.append("  各卫星频率自适应阈值计算结果:\n")
                for sat_id in sorted(calculated_thresholds.keys()):
                    for freq in sorted(calculated_thresholds[sat_id].keys()):
                        thresh_info = calculated_thresholds[sat_id][freq]
                        p99 = thresh_info.get('p99', 0)
                        threshold = thresh_info.get('threshold', 0)
                        circuit_breaker = thresh_info.get('circuit_breaker_triggered', False)
                        breaker_mark = " [熔断]" if circuit_breaker else ""
                        log_content.append(
                            f"    - {sat_id} {freq}: 阈值={threshold:.3f}m (P99={p99:.3f}m){breaker_mark}\n"
                        )
                log_content.append("\n")
            else:
                log_content.append(f"阈值计算模式: 固定\n")
                log_content.append(f"  CMC阈值={cmc_threshold}m\n\n")
        else:
            log_content.append("  利用参考站与流动站之间的站间、星间双差残差，通过三倍中误差 (3-sigma) 准则或预设的最大阈值，\n")
            log_content.append("  识别并剔除观测值中的离群点 (Outliers)，提高定位解算的可靠性。\n")
            log_content.append("\n  【自适应阈值机制】:\n")
            log_content.append("    1. 预过滤 (sample_filter_limit):\n")
            log_content.append("       - 伪距: 200.0m | 相位: 50.0m | 多普勒: 100.0m/s\n")
            log_content.append("       在计算P99之前剔除极端离群值，避免污染统计\n")
            log_content.append("    2. 熔断机制 (sanity_limit):\n")
            log_content.append("       - 伪距: 20.0m | 相位: 5.0m | 多普勒: 10.0m/s\n")
            log_content.append("       若P99超过此阈值，判定统计失效，触发熔断，强制使用保底阈值\n")
            log_content.append("    3. 阈值计算: T_adaptive = min(max(P99 × 2.0, T_floor), T_max)\n")
            log_content.append("       确保阈值在合理范围内，既不过度宽松也不过度严格\n\n")

        # Add adaptive threshold info for double diff
        if not cmc_flags:
            if threshold_mode == 'adaptive':
                log_content.append("阈值计算模式: 自适应 (基于分位数法)\n")
                log_content.append("  计算公式: T_adaptive = min(max(P99 × 2.0, T_floor), T_max)\n")
                log_content.append("  预过滤阈值 (sample_filter_limit):\n")
                log_content.append("    - 伪距: 200.0m | 相位: 50.0m | 多普勒: 100.0m/s\n")
                log_content.append("  熔断阈值 (sanity_limit):\n")
                log_content.append("    - 伪距: 20.0m | 相位: 5.0m | 多普勒: 10.0m/s\n\n")
            else:
                log_content.append("阈值计算模式: 固定\n\n")

            log_content.append(f"参数设置:\n")
            log_content.append(f"  观测值最大阈值限制: 伪距={max_threshold_limit['code']}m, 相位={max_threshold_limit['phase']}m, 多普勒={max_threshold_limit['doppler']}m/s\n\n")
            
            # Add threshold explanation for each satellite (only for double diff)
            log_content.append("  各卫星阈值应用情况:\n")
            for sat_id, freq_data in triple_errors.items():
                for freq, errors in freq_data.items():
                    for obs_type in ['code', 'phase', 'doppler']:
                        error_info = errors.get(obs_type, {})
                        triple_sigma = error_info.get('threshold', 0)
                        
                        if threshold_mode == 'adaptive' and 'p99' in error_info:
                            # Adaptive mode - show P99 and calculated threshold
                            p99_value = error_info.get('p99', 0)
                            circuit_breaker = error_info.get('circuit_breaker_triggered', False)
                            breaker_mark = " [熔断]" if circuit_breaker else ""
                            log_content.append(
                                f"    - 卫星 {sat_id} 频率 {freq} {obs_type}: 自适应阈值={triple_sigma:.4f}m (P99={p99_value:.4f}m){breaker_mark}\n"
                            )
                        else:
                            # Fixed mode or legacy format
                            if triple_sigma <= 0:
                                log_content.append(
                                    f"    - 卫星 {sat_id} 频率 {freq} {obs_type}: 无有效三倍中误差，使用最大阈值: {max_threshold_limit[obs_type]:.4f}\n"
                                )
                            else:
                                log_content.append(
                                    f"    - 卫星 {sat_id} 频率 {freq} {obs_type}: 使用三倍中误差阈值: {min(triple_sigma, max_threshold_limit[obs_type]):.4f}\n"
                                )
            log_content.append("\n")
        
        # 1. Summary
        total_modified = sum(modified_count.values())
        action_word = "剔除" if cmc_flags else "修改"
        log_content.append(f"\n一、{action_word}统计摘要\n")
        log_content.append("-" * 70 + "\n")
        log_content.append(f"总计{action_word}卫星数: {len(modified_satellites)}\n")
        log_content.append(f"总计{action_word}观测值: {total_modified}\n")
        log_content.append(f"{action_word}分类: 伪距={modified_count.get('code',0)}, 相位={modified_count.get('phase',0)}, 多普勒={modified_count.get('doppler',0)}\n\n")
        
        # 2. Details
        log_content.append("\n二、异常历元检测详情\n")
        log_content.append("-" * 70 + "\n")
        
        # Organize details
        system_satellites = defaultdict(list)
        for sat_id in outlier_details.keys():
            if sat_id: 
                 system = sat_id[0]
                 system_satellites[system].append(sat_id)
            
        for system in sorted(system_satellites.keys()):
            log_content.append(f"卫星系统 {system}:\n")
            for sat_id in sorted(system_satellites[system]):
                details = outlier_details[sat_id]
                log_content.append(f"  卫星 {sat_id} ({len(details)}个异常观测值):\n")
                
                if cmc_flags:
                    # CMC清洗：按频率分组
                    freq_groups = defaultdict(list)
                    for d in details:
                        freq = d.get('freq', 'unknown')
                        freq_groups[freq].append(d)
                    
                    for freq in sorted(freq_groups.keys()):
                        log_content.append(f"    频率 {freq}:\n")
                        for d in freq_groups[freq]:
                            epoch_idx = d.get('epoch_idx', 0)
                            ts = d.get('timestamp', f"Epoch {epoch_idx}")
                            val = d.get('dd_value', d.get('diff_change', 0))
                            th = d.get('threshold_used', d.get('threshold', 5.0))
                            # Check if this specific epoch was modified
                            epoch_modified = epoch_idx in actually_modified_epochs.get(sat_id, set())
                            status = "已剔除" if epoch_modified else "未剔除"
                            log_content.append(f"      - 历元 {epoch_idx} ({ts}): 差值变化={val:.6f}m, 阈值={th:.6f}m, 状态={status}\n")
                else:
                    # 双差清洗：按历元分组
                    epoch_groups = defaultdict(list)
                    for d in details:
                        epoch_groups[d['epoch_idx']].append(d)
                    
                    for epoch_idx in sorted(epoch_groups.keys()):
                        eds = epoch_groups[epoch_idx]
                        ts = eds[0].get('timestamp', f"Epoch {epoch_idx}")
                        # Check if this specific epoch was modified
                        epoch_modified = epoch_idx in actually_modified_epochs.get(sat_id, set())
                        log_content.append(f"    历元 {epoch_idx} ({ts}):\n")
                        for d in eds:
                            obs_type = d.get('obs_type', 'unknown')
                            freq = d.get('freq', 'unknown')
                            val = d.get('dd_value', 0)
                            th = d.get('threshold_used', 0)
                            status = "已修改" if epoch_modified else "未修改"
                            log_content.append(f"      - {obs_type}@{freq}: 双差值={val:.6f}m, 阈值={th:.6f}m, 状态={status}\n")
                log_content.append("\n")

        # Add skip reasons summary if there are any skipped modifications
        if skip_reasons:
            action_verb = "剔除" if cmc_flags else "修改"
            log_content.append(f"\n三、未{action_verb}原因统计\n")
            log_content.append("-" * 70 + "\n")
            log_content.append(f"以下卫星的部分或全部异常观测值未能成功{action_verb}，原因如下:\n\n")
            for sat_id in sorted(skip_reasons.keys()):
                reasons = skip_reasons[sat_id]
                if reasons:
                    log_content.append(f"  卫星 {sat_id}:\n")
                    if 'no_freq_indices' in reasons:
                        log_content.append(f"    - 未找到频率索引映射: {reasons['no_freq_indices']}个历元\n")
                    if 'epoch_not_found' in reasons:
                        log_content.append(f"    - 在RINEX文件中未找到对应历元: {reasons['epoch_not_found']}个\n")
                    if 'sat_line_not_found' in reasons:
                        log_content.append(f"    - 在历元中未找到卫星数据行: {reasons['sat_line_not_found']}个\n")
                    if 'freq_or_obstype_not_in_indices' in reasons:
                        log_content.append(f"    - 频率或观测类型不在索引映射中: {reasons['freq_or_obstype_not_in_indices']}个字段\n")
                    if 'field_out_of_bounds' in reasons:
                        log_content.append(f"    - 字段位置超出行长度: {reasons['field_out_of_bounds']}个字段\n")
                    if 'field_already_empty' in reasons:
                        log_content.append(f"    - 字段本身为空(已被其他步骤清除): {reasons['field_already_empty']}个字段\n")
            log_content.append("\n")

        with open(debug_path, 'w', encoding='utf-8') as f:
            f.writelines(log_content)

        return {'output_path': output_path, 'debug_log': debug_path, 'modified_counts': modified_count, 'modified_satellites': list(modified_satellites)}

    def write_doppler_predicted_rinex(self, original_path: str, output_path: Optional[str], prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Write a RINEX file with Doppler-predicted phase values applied.

        prediction_results expected structure: {'predicted_phases': {sat_id: {freq: {'prediction_details': [{'epoch_idx', 'predicted_phase_cycle', 'time'}]}}}}
        Returns dict with output_path, modification_details, total_modifications.
        """
        if output_path is None:
            base_dir = os.path.dirname(original_path)
            bn = os.path.basename(original_path)
            name, ext = os.path.splitext(bn)
            doppler_dir = os.path.join(base_dir, "doppler prediction")
            os.makedirs(doppler_dir, exist_ok=True)
            output_path = os.path.join(doppler_dir, f"{name}-doppler predicted{ext}")

        # load file
        with open(original_path, 'r', encoding='utf-8') as f:
            original_lines = [line.rstrip('\n') for line in f]

        # build system obs info mapping
        system_obs_info = {}
        header_end_idx = next((i for i, line in enumerate(original_lines) if 'END OF HEADER' in line), -1)
        i = 0
        while i < len(original_lines) and i < header_end_idx:
            line = original_lines[i]
            if 'SYS / # / OBS TYPES' in line:
                system = line[0]
                try:
                    num_types = int(line[3:6].strip())
                except Exception:
                    num_types = 0
                obs_types = line.split()[2:]
                # Handle continuation lines
                j = i + 1
                while j < header_end_idx and len(obs_types) < num_types:
                    next_line = original_lines[j]
                    if 'SYS / # / OBS TYPES' in next_line and next_line[0] == ' ':
                        obs_types.extend(next_line.split())
                        j += 1
                    else:
                        break
                obs_types = obs_types[:num_types] if num_types > 0 else obs_types
                freq_to_indices = {}
                for idx, obs_type in enumerate(obs_types):
                    if obs_type.startswith('L'):
                        freq_key = f"L{obs_type[1:]}"
                        if freq_key not in freq_to_indices:
                            freq_to_indices[freq_key] = {}
                        freq_to_indices[freq_key]['phase'] = idx
                system_obs_info[system] = {'obs_types': obs_types, 'freq_to_indices': freq_to_indices}
            i += 1

        output_lines = original_lines.copy()
        modification_details = []
        total_modifications = 0
        modified_satellites = set()

        predicted_phases = prediction_results.get('predicted_phases', {})
        # iterate predictions
        for sat_id, sat_data in predicted_phases.items():
            sat_system = sat_id[0]
            sat_prn = sat_id[1:].zfill(2)
            if sat_system not in system_obs_info:
                continue
            freq_to_indices = system_obs_info[sat_system]['freq_to_indices']
            for freq, freq_data in sat_data.items():
                if freq not in freq_to_indices or 'phase' not in freq_to_indices[freq]:
                    continue
                phase_field_idx = freq_to_indices[freq]['phase']
                wavelength = None
                # attempt to get wavelength from predicted data if present
                # prediction_details entries may include predicted_phase_cycle and optionally time; wavelengths may not be present
                for detail in freq_data.get('prediction_details', []):
                    epoch_indices = detail.get('epoch_indices')  # ✅ 使用epoch_indices替代计数器
                    epoch_idx = detail.get('epoch_idx')
                    predicted_phase_cycle = detail.get('predicted_phase_cycle')
                    epoch_time = detail.get('time')
                    # find epoch index (0-based) in file using epoch_indices
                    current_epoch = -1
                    for line_idx, line in enumerate(output_lines):
                        if line.startswith('>'):
                            current_epoch += 1
                            # ✅ 同时检查epoch_indices和时间戳，双重验证确保正确性
                            if current_epoch == epoch_indices:
                                # find satellite line
                                j = line_idx + 1
                                while j < len(output_lines) and not output_lines[j].startswith('>'):
                                    sat_line = output_lines[j]
                                    if len(sat_line) >= 3 and sat_line[0] == sat_system and sat_line[1:3].strip().zfill(2) == sat_prn:
                                        # apply predicted phase - similar to analyzer's helper
                                        line = sat_line.rstrip('\n')
                                        if len(line) < 3:
                                            break
                                        field_start = 3 + phase_field_idx * 16 + 2
                                        field_end = field_start + 14
                                        if len(line) < field_end:
                                            line = line.ljust(field_end + 2)
                                        phase_str = f"{predicted_phase_cycle:14.3f}"
                                        modified_line = line[:field_start] + phase_str + line[field_end:]
                                        output_lines[j] = modified_line + '\n'
                                        mod_detail = {
                                            'sat_id': sat_id,
                                            'freq': freq,
                                            'epoch_idx': epoch_idx,
                                            'predicted_phase_cycle': predicted_phase_cycle
                                        }
                                        modification_details.append(mod_detail)
                                        total_modifications += 1
                                        modified_satellites.add(sat_id)
                                        break
                                    j += 1
                                break

        # write output
        dirpath = os.path.dirname(output_path) or '.'
        os.makedirs(dirpath, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)

        return {'output_path': output_path, 'modification_details': modification_details, 'total_modifications': total_modifications, 'modified_satellites': list(modified_satellites)}

    def write_isb_corrected_rinex(self, original_path: str, output_path: Optional[str], isb_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ISB correction (scalar) to BDS-3 pseudorange (C2I) for epochs in isb_results['isb_epochs'].

        Returns dict with output_path, modification_details, total_modifications.
        """
        isb_correction = isb_results.get('isb_mean')
        isb_epochs = isb_results.get('isb_epochs', [])

        if isb_correction is None or not isb_epochs:
            return {'output_path': None, 'modification_details': {}, 'total_modifications': 0}

        if output_path is None:
            base_dir = os.path.dirname(original_path)
            bn = os.path.basename(original_path)
            name, ext = os.path.splitext(bn)
            isb_dir = os.path.join(base_dir, "BDS23_ISB")
            os.makedirs(isb_dir, exist_ok=True)
            output_path = os.path.join(isb_dir, f"{name}-isb{ext}")

        with open(original_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]

        # parse header
        header_end = 0
        for i, line in enumerate(lines):
            if 'END OF HEADER' in line:
                header_end = i + 1
                break

        output_lines = lines.copy()
        modification_details = {}
        total_modifications = 0
        modified_satellites = set()

        # iterate epochs
        data_lines = lines[header_end:]
        i = 0
        epoch_idx = 0
        while i < len(data_lines):
            line = data_lines[i]
            if line.startswith('>') and len(line) > 32:
                epoch_idx += 1
                parts = line[1:].split()
                if len(parts) >= 6:
                    try:
                        year = int(parts[0]); month = int(parts[1]); day = int(parts[2])
                        hour = int(parts[3]); minute = int(parts[4]); secf = float(parts[5])
                        epoch_time = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=int(secf), microsecond=int((secf - int(secf)) * 1000000))
                    except Exception:
                        epoch_time = None

                    # check if epoch_time matches any isb_epoch within 0.1s
                    match = False
                    if epoch_time is not None:
                        for ie in isb_epochs:
                            ie_dt = ie.to_pydatetime() if hasattr(ie, 'to_pydatetime') else ie
                            if abs((epoch_time.to_pydatetime() - ie_dt).total_seconds()) <= 0.1:
                                match = True
                                break

                    if match:
                        # collect satellite lines for this epoch
                        sat_lines = []
                        j = i + 1
                        while j < len(data_lines) and not data_lines[j].startswith('>'):
                            sat_lines.append(data_lines[j])
                            j += 1

                        # apply ISB to BDS-3 satellites (PRN >=19 and system 'C')
                        for idx, sat_line in enumerate(sat_lines):
                            if len(sat_line) >= 3:
                                sat_prn = sat_line[:3]
                                if sat_prn.startswith('C'):
                                    try:
                                        prn_num = int(sat_prn[1:])
                                    except Exception:
                                        prn_num = -1
                                    if prn_num >= 19:
                                        # apply correction to C2I field if present
                                        # get obs types for system C
                                        obs_types = []
                                        # try to get obs types from header
                                        for h in lines[:header_end]:
                                            if 'SYS / # / OBS TYPES' in h and h[0] == 'C':
                                                obs_types = h.split()[2:]
                                                break
                                        if not obs_types:
                                            obs_types = ['C1I', 'C2I', 'C5I', 'L1I', 'L2I']
                                        if 'C2I' in obs_types:
                                            code_idx = obs_types.index('C2I')
                                            start_pos = 3 + code_idx * 16
                                            end_pos = start_pos + 16
                                            if end_pos <= len(sat_line):
                                                original_field = sat_line[start_pos:end_pos].strip()
                                                if original_field:
                                                    try:
                                                        original_pr = float(original_field)
                                                        corrected_pr = original_pr - isb_correction
                                                        formatted_pr = f"{corrected_pr:14.3f}".rjust(16)
                                                        modified_line = list(sat_line)
                                                        modified_line[start_pos:end_pos] = formatted_pr
                                                        # write back into output_lines at correct index
                                                        out_idx = header_end + (i + 1) + idx
                                                        output_lines[out_idx] = ''.join(modified_line)
                                                        modification_details[sat_prn] = {
                                                            'epoch_idx': epoch_idx,
                                                            'original_pr': original_pr,
                                                            'corrected_pr': corrected_pr
                                                        }
                                                        total_modifications += 1
                                                        modified_satellites.add(sat_prn)
                                                    except Exception:
                                                        pass
                i = j
                continue
            i += 1

        # write output
        dirpath = os.path.dirname(output_path) or '.'
        os.makedirs(dirpath, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)

        return {'output_path': output_path, 'modification_details': modification_details, 'total_modifications': total_modifications, 'modified_satellites': list(modified_satellites)}

    def write_doppler_smoothed_rinex(self, original_path: str, output_path: Optional[str], smoothed_observations: Dict[str, Any]) -> Dict[str, Any]:
        """Write a RINEX file with Doppler-smoothed pseudorange values.

        smoothed_observations: {sat_id: {freq: {'times': [...], 'code_smoothed': [...]}}}
        Returns dict with output_path, modification_details, total_modifications.
        """
        if output_path is None:
            base_dir = os.path.dirname(original_path)
            bn = os.path.basename(original_path)
            name, ext = os.path.splitext(bn)
            doppler_dir = os.path.join(base_dir, "doppler smoothing")
            os.makedirs(doppler_dir, exist_ok=True)
            output_path = os.path.join(doppler_dir, f"{name}-doppler smoothed{ext}")

        with open(original_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]

        # parse header
        header_end = 0
        for i, line in enumerate(lines):
            if 'END OF HEADER' in line:
                header_end = i + 1
                break

        # build system obs info
        system_obs_info = {}
        i = 0
        while i < header_end:
            line = lines[i]
            if 'SYS / # / OBS TYPES' in line:
                system = line[0]
                try:
                    num_types = int(line[3:6].strip())
                except Exception:
                    num_types = 0
                obs_types = line.split()[2:]
                # Handle continuation lines
                j = i + 1
                while j < header_end and len(obs_types) < num_types:
                    next_line = lines[j]
                    if 'SYS / # / OBS TYPES' in next_line and next_line[0] == ' ':
                        obs_types.extend(next_line.split())
                        j += 1
                    else:
                        break
                obs_types = obs_types[:num_types] if num_types > 0 else obs_types
                freq_to_indices = {}
                for idx, obs_type in enumerate(obs_types):
                    if obs_type.startswith('C'): # Code/Pseudorange
                        freq_key = f"L{obs_type[1:]}"
                        if freq_key not in freq_to_indices:
                            freq_to_indices[freq_key] = {}
                        freq_to_indices[freq_key]['code'] = idx
                system_obs_info[system] = {'obs_types': obs_types, 'freq_to_indices': freq_to_indices}
            i += 1

        output_lines = lines.copy()
        modification_details = defaultdict(int)
        total_modifications = 0
        modified_satellites = set()

        # Iterate epochs
        data_lines = lines[header_end:]
        i = 0
        epoch_idx = 0
        while i < len(data_lines):
            line = data_lines[i]
            if line.startswith('>') and len(line) > 32:
                # Parse epoch time to match with smoothed data
                parts = line[1:].split()
                if len(parts) >= 6:
                    try:
                        year = int(parts[0]); month = int(parts[1]); day = int(parts[2])
                        hour = int(parts[3]); minute = int(parts[4]); secf = float(parts[5])
                        epoch_time = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=int(secf), microsecond=int((secf - int(secf)) * 1000000))
                    except Exception:
                        epoch_time = None
                    
                    if epoch_time is not None:
                        # Process satellites in this epoch
                        epoch_time_str = epoch_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # ✅ 格式化时间戳用于日志
                        j = i + 1
                        sat_lines_updated = []
                        while j < len(data_lines) and not data_lines[j].startswith('>'):
                            sat_line = data_lines[j]
                            if len(sat_line) >= 3:
                                sat_line_content = sat_line.rstrip('\n')
                                sat_sys = sat_line_content[0]
                                sat_prn = sat_line_content[:3] # e.g. G01
                                
                                updated_line = sat_line_content
                                if sat_sys in system_obs_info:
                                    freq_indices = system_obs_info[sat_sys]['freq_to_indices']
                                    
                                    # Check if we have smoothed data for this sat
                                    if sat_prn in smoothed_observations:
                                        sat_smooth_data = smoothed_observations[sat_prn]
                                        
                                        for freq, f_data in sat_smooth_data.items():
                                            if freq in freq_indices and 'code' in freq_indices[freq]:
                                                # Find value for this epoch - optimize by checking time range or simple search
                                                times = f_data.get('times', [])
                                                smooth_codes = f_data.get('code_smoothed', [])
                                                
                                                # Simple search (can be improved)
                                                found_val = None
                                                # Optimization: check last found index? 
                                                # For now, just linear scan.
                                                for k, t in enumerate(times):
                                                    if abs((t - epoch_time).total_seconds()) < 0.001:
                                                        found_val = smooth_codes[k]
                                                        break
                                                
                                                if found_val is not None:
                                                    code_idx = freq_indices[freq]['code']
                                                    start_pos = 3 + code_idx * 16
                                                    end_pos = start_pos + 16
                                                    # Extend line if short
                                                    if len(updated_line) < end_pos:
                                                        updated_line = updated_line.ljust(end_pos)
                                                    
                                                    formatted_code = f"{found_val:14.3f}".rjust(16)
                                                    updated_line = updated_line[:start_pos] + formatted_code + updated_line[end_pos:]
                                                    # ✅ 增加历元信息的日志输出
                                                    modification_details[f"{sat_prn} 历元{epoch_idx} ({epoch_time_str}) {freq}"] = found_val
                                                    total_modifications += 1
                                                    modified_satellites.add(sat_prn)
                                
                                # ✅ 修复索引：正确的全局索引应该是 header_end + i + (j - i_epoch_start)
                                output_lines[header_end + i + (j - i)] = updated_line  # Update proper index in output_lines
                            
                            j += 1
                        epoch_idx += 1  # ✅ 增加历元计数器
                        i = j
                        continue
            i += 1

        # write output
        dirpath = os.path.dirname(output_path) or '.'
        os.makedirs(dirpath, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines([l if l.endswith('\n') else l + '\n' for l in output_lines])

        return {'output_path': output_path, 'modification_details': dict(modification_details), 'total_modifications': total_modifications, 'modified_satellites': list(modified_satellites)}