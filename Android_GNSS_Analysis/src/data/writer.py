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
        for line in original_lines:
            if 'SYS / # / OBS TYPES' in line:
                system = line[0]
                obs_types = line.split()[2:]
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
                            cmc_flags: Dict[str, Dict[str, list]] = None) -> Dict[str, Any]:
        """Remove outliers by clearing offending fields and write a cleaned RINEX.

        cmc_flags (optional): {sat_id: {freq: [bool,...]}} where True indicates the epoch index (0-based) flagged by CMC threshold.
        Returns dict with output_path and debug log path and stats.
        """        """Remove outliers by clearing offending fields and write a cleaned RINEX.

        Returns dict with output_path and debug log path and stats.
        """
        max_threshold_limit = max_threshold_limit or {'code': 50.0, 'phase': 5.0, 'doppler': 10.0}

        phone_file_name = os.path.basename(original_path)
        phone_file_name_no_ext = os.path.splitext(phone_file_name)[0]
        result_dir = os.path.dirname(original_path)
        coarse_error_dir = os.path.join(result_dir, "Coarse error")
        os.makedirs(coarse_error_dir, exist_ok=True)

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
                    second_float = float(parts[5])
                    epoch_timestamps[current_epoch] = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {second_float}"

        system_obs_info = {}
        for line in lines:
            if 'SYS / # / OBS TYPES' in line:
                system = line[0]
                obs_types = line.split()[2:]
                freq_to_indices = defaultdict(dict)
                for idx, obs in enumerate(obs_types):
                    if obs.startswith('C'):
                        freq = f"L{obs[1:]}"; freq_to_indices[freq]['code'] = idx
                    elif obs.startswith('L'):
                        freq = f"L{obs[1:]}"; freq_to_indices[freq]['phase'] = idx
                    elif obs.startswith('D'):
                        freq = f"L{obs[1:]}"; freq_to_indices[freq]['doppler'] = idx
                system_obs_info[system] = {'obs_types': obs_types, 'freq_to_indices': freq_to_indices}

        # identify outlier epochs
        outlier_epochs = defaultdict(lambda: defaultdict(list))
        outlier_details = defaultdict(list)

        for sat_id, freq_data in double_diffs.items():
            for freq, dd_data in freq_data.items():
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
                            epoch_idx = orig_idx + 2
                            timestamp = epoch_timestamps.get(epoch_idx, f"未知时间戳(历元{epoch_idx})")
                            outlier_details[sat_id].append({'obs_type': obs_type, 'freq': freq, 'dd_value': dd_value, 'threshold_used': threshold, 'epoch_idx': epoch_idx, 'timestamp': timestamp})
                            outlier_epochs[sat_id][epoch_idx].append((obs_type, freq))

        # Apply CMC flags (if provided)
        if cmc_flags:
            for sat_id, freq_map in cmc_flags.items():
                for freq, flags in freq_map.items():
                    for idx, flag in enumerate(flags):
                        if flag:
                            epoch_idx = idx + 1  # cmc_flags indices are 0-based; epoch numbering here is 1-based
                            outlier_epochs[sat_id][epoch_idx].append(('code', freq))
                            outlier_epochs[sat_id][epoch_idx].append(('phase', freq))
                            outlier_details[sat_id].append({'obs_type': 'cmc', 'freq': freq, 'epoch_idx': epoch_idx})

        # modify lines by clearing fields
        modified_count = defaultdict(int)
        modified_satellites = set()

        for sat_id, epoch_obs_map in outlier_epochs.items():
            sat_system = sat_id[0]; sat_prn = sat_id[1:].zfill(2)
            system_info = system_obs_info.get(sat_system, {})
            freq_indices = system_info.get('freq_to_indices', {})
            if not freq_indices:
                continue
            for epoch_idx, obs_freq_list in epoch_obs_map.items():
                timestamp = epoch_timestamps.get(epoch_idx, f"未知时间戳(历元{epoch_idx})")
                # find epoch start by matching timestamp string
                epoch_start = -1
                for i, line in enumerate(lines):
                    if line.startswith('>') and timestamp in line:
                        epoch_start = i
                        break
                if epoch_start < 0:
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
                    continue
                original_line = lines[sat_line_idx]
                modified_line = list(original_line)
                field_modified = False
                modified_fields = []
                for obs_type, freq in obs_freq_list:
                    if freq not in freq_indices or obs_type not in freq_indices[freq]:
                        continue
                    field_idx = freq_indices[freq][obs_type]
                    start_pos = 3 + field_idx * 16
                    end_pos = start_pos + 16
                    if end_pos > len(modified_line):
                        continue
                    original_field = original_line[start_pos:end_pos].strip()
                    if original_field:
                        modified_line[start_pos:end_pos] = ' ' * 16
                        modified_count[obs_type] += 1
                        field_modified = True
                        modified_fields.append(f"{freq}({obs_type})")
                if field_modified:
                    lines[sat_line_idx] = ''.join(modified_line)
                    modified_satellites.add(sat_id)

        # determine output path
        if enable_cci:
            base_ext = os.path.splitext(original_path)[1]
            modified_file_name = f"cleaned2-{phone_file_name_no_ext}-cc inconsistency{base_ext}"
        else:
            base_ext = os.path.splitext(original_path)[1]
            modified_file_name = f"cleaned2-{phone_file_name_no_ext}{base_ext}"
        output_path = os.path.join(coarse_error_dir, modified_file_name)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # write debug log
        debug_file_name = "double_diffs_cleaning.log"
        debug_path = os.path.join(coarse_error_dir, debug_file_name)
        log_content = []
        log_content.append(f"总计修改观测值: {sum(modified_count.values())}\n")
        log_content.append(f"涉及卫星数: {len(modified_satellites)}\n")
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
        for line in original_lines:
            if 'SYS / # / OBS TYPES' in line:
                system = line[0]
                obs_types = line.split()[2:]
                freq_to_indices = {}
                for idx, obs_type in enumerate(obs_types):
                    if obs_type.startswith('L'):
                        freq_key = f"L{obs_type[1:]}"
                        if freq_key not in freq_to_indices:
                            freq_to_indices[freq_key] = {}
                        freq_to_indices[freq_key]['phase'] = idx
                system_obs_info[system] = {'obs_types': obs_types, 'freq_to_indices': freq_to_indices}

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
                    epoch_idx = detail.get('epoch_idx')
                    predicted_phase_cycle = detail.get('predicted_phase_cycle')
                    # find epoch index (0-based) in file
                    current_epoch = -1
                    for line_idx, line in enumerate(output_lines):
                        if line.startswith('>'):
                            current_epoch += 1
                            if current_epoch == epoch_idx:
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