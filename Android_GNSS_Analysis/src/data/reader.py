from typing import Dict, Any, Optional, List
from collections import defaultdict
import re
import pandas as pd
from ..core.config import GNSS_FREQUENCIES, GLONASS_K_MAP, SPEED_OF_LIGHT


class RinexReader:
    """RINEX reading utilities for phone and receiver files.

    Methods return dictionaries and do not modify external state.
    """
    _PHONE_PHASE_LLI_RE = re.compile(r'^([+-]?\d+\.\d{3})([23])$')

    @classmethod
    def _parse_phone_obs_field(cls, field_raw: str, obs_type: str) -> Dict[str, Any]:
        """Parse one 16-char RINEX observation field for phone files.

        For phase observations, supports both:
        1) Standard RINEX LLI at column 15 of the 16-char field.
        2) Phone-specific encoding where 4th decimal digit 2/3 indicates LLI.
        """
        field = (field_raw or '').ljust(16)[:16]
        value_str = field[:14].strip()

        standard_lli = int(field[14]) if field[14].isdigit() else None
        value = None
        mobile_lli = None

        if value_str:
            if obs_type.startswith('L'):
                m = cls._PHONE_PHASE_LLI_RE.match(value_str)
                if m:
                    try:
                        value = float(m.group(1))
                        mobile_lli = int(m.group(2))
                    except Exception:
                        value = None
                        mobile_lli = None
                else:
                    try:
                        value = float(value_str)
                    except Exception:
                        value = None
            else:
                try:
                    value = float(value_str)
                except Exception:
                    value = None

        lli = standard_lli if standard_lli is not None else mobile_lli
        return {'value': value, 'lli': lli}

    def read_phone_rinex(self,
                         file_path: str,
                         frequencies: Optional[Dict[str, Any]] = None,
                         wavelengths: Optional[Dict[str, Any]] = None,
                         glonass_k_map: Optional[Dict[str, int]] = None,
                         speed_of_light: float = SPEED_OF_LIGHT,
                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Parse a phone RINEX observation file.

        Returns dict with keys: 'data', 'observations_meters', 'satellite_wavelengths'.
        """
        frequencies = frequencies or GNSS_FREQUENCIES
        glonass_k_map = glonass_k_map or GLONASS_K_MAP
        wavelengths = wavelengths or {}
        # Pre-calculate base wavelengths if not provided
        if not wavelengths:
            for sys, freqs in frequencies.items():
                wavelengths[sys] = {fname: speed_of_light / fhz for fname, fhz in freqs.items() if fhz > 0}

        data = {'header': {}, 'epochs': []}
        observations_meters = {}
        final_satellite_wavelengths = {}

        header_parsed = False
        with open(file_path, 'r', encoding='utf-8') as f:
            # Step 1: Parse Header
            for line in f:
                line = line.rstrip('\n')
                if 'END OF HEADER' in line:
                    header_parsed = True
                    break
                if 'RINEX VERSION' in line:
                    version_str = line[:9].strip()
                    try:
                        data['header']['version'] = float(version_str) if version_str else None
                    except Exception:
                        data['header']['version'] = None
                elif 'MARKER NAME' in line:
                    data['header']['marker'] = line[:60].strip()
                elif 'OBS TYPES' in line:
                    system = line[0]
                    obs_types = line[6:60].split()
                    data['header'].setdefault(f'obs_types_{system}', []).extend(obs_types)

            if not header_parsed:
                return {'data': data, 'observations_meters': {}, 'satellite_wavelengths': {}}

            # Step 2: Parse Body (Streaming)
            current_epoch = None
            current_satellites = {}
            
            # Since we can't easily get line count for progress without reading twice, 
            # and performance is key, we skip progress calculate if not strictly needed 
            # or we could use file offset but simpler is better for now.
            
            for line in f:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                if line.startswith('>'):
                    if current_epoch is not None and current_satellites:
                        data['epochs'].append({'time': current_epoch, 'satellites': current_satellites.copy()})
                    parts = line[1:].split()
                    if len(parts) >= 6:
                        try:
                            year = int(parts[0]); month = int(parts[1]); day = int(parts[2])
                            hour = int(parts[3]); minute = int(parts[4])
                            second_float = float(parts[5])
                            
                            # Simple normalization
                            if second_float >= 60:
                                second_float -= 60; minute += 1
                                if minute >= 60: minute -= 60; hour += 1
                                    # Day wrap not handled here for simplicity in hot path, 
                                    # but timestamp will handle basic range
                            
                            current_epoch = pd.Timestamp(
                                year=year, month=month, day=day,
                                hour=hour, minute=minute, second=int(second_float),
                                microsecond=int((second_float - int(second_float)) * 1000000)
                            )
                            current_satellites = {}
                        except Exception:
                            continue
                    continue

                # satellite observation line
                if current_epoch is None or len(line) < 3:
                    continue
                
                sat_system = line[0]
                sat_prn = line[1:3].strip()
                if not sat_prn:
                    continue
                sat_id = f"{sat_system}{sat_prn}"

                # Optimization: Cache PRN check
                current_wavelengths = wavelengths.get(sat_system, {})
                if sat_system == 'R' and sat_prn.isdigit():
                    k = glonass_k_map.get(f"R{sat_prn.zfill(2)}", 0)
                    if not (-7 <= k <= 6): k = 0
                    l1c_freq = 1602e6 + k * 0.5625e6
                    # avoid copying the whole dict every line, use local override
                    current_wavelengths = current_wavelengths.copy()
                    current_wavelengths['L1C'] = speed_of_light / l1c_freq

                final_satellite_wavelengths[sat_id] = current_wavelengths

                obs_types = data['header'].get(f'obs_types_{sat_system}', [])
                if not obs_types:
                    continue

                observations = {}
                sat_data = line[3:]
                field_width = 16
                actual_fields = (len(sat_data) + field_width - 1) // field_width
                
                # Pre-calculate indices to avoid repeated work
                for j, obs_type in enumerate(obs_types):
                    if j < actual_fields:
                        field_raw = sat_data[j*16:(j+1)*16]
                        parsed = self._parse_phone_obs_field(field_raw, obs_type)
                        observations[obs_type] = parsed['value']
                        if obs_type.startswith('L'):
                            observations[f'{obs_type}_LLI'] = parsed['lli']
                    else:
                        observations[obs_type] = None
                        if obs_type.startswith('L'):
                            observations[f'{obs_type}_LLI'] = None

                current_satellites[sat_id] = observations

                if sat_id not in observations_meters:
                    observations_meters[sat_id] = {}

                for freq, wavelength in current_wavelengths.items():
                    suffix = freq[1:]
                    code_val = observations.get(f'C{suffix}')
                    phase_val = observations.get(f'L{suffix}')
                    phase_lli = observations.get(f'L{suffix}_LLI')
                    doppler_val = observations.get(f'D{suffix}')
                    snr_val = observations.get(f'S{suffix}')

                    if freq not in observations_meters[sat_id]:
                        observations_meters[sat_id][freq] = {
                            'times': [], 'code': [], 'phase': [], 'phase_cycle': [], 'doppler': [],
                            'wavelength': [], 'snr': [], 'phase_lli': []
                        }

                    obs_list = observations_meters[sat_id][freq]
                    obs_list['times'].append(current_epoch)
                    obs_list['code'].append(code_val)
                    obs_list['snr'].append(snr_val)
                    obs_list['phase_cycle'].append(phase_val)
                    obs_list['phase_lli'].append(phase_lli)
                    obs_list['wavelength'].append(wavelength)
                    
                    if phase_val is not None:
                        obs_list['phase'].append(phase_val * wavelength)
                    else:
                        obs_list['phase'].append(None)
                    
                    if doppler_val is not None:
                        obs_list['doppler'].append(-doppler_val * wavelength)
                    else:
                        obs_list['doppler'].append(None)

        if current_epoch is not None and current_satellites:
            data['epochs'].append({'time': current_epoch, 'satellites': current_satellites.copy()})

        if progress_callback:
            progress_callback(1.0)

        if current_epoch is not None and current_satellites:
            data['epochs'].append({'time': current_epoch, 'satellites': current_satellites.copy()})

        if progress_callback:
            progress_callback(1.0)

        return {'data': data, 'observations_meters': observations_meters, 'satellite_wavelengths': final_satellite_wavelengths}

    def read_receiver_rinex(self,
                             file_path: str,
                             frequencies: Optional[Dict[str, Any]] = None,
                             wavelengths: Optional[Dict[str, Any]] = None,
                             glonass_k_map: Optional[Dict[str, int]] = None,
                             speed_of_light: float = SPEED_OF_LIGHT,
                             progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Parse a receiver RINEX observation file.

        Returns dict with keys: 'data', 'receiver_observations', 'satellite_wavelengths'.
        """
        frequencies = frequencies or GNSS_FREQUENCIES
        glonass_k_map = glonass_k_map or GLONASS_K_MAP
        wavelengths = wavelengths or {}
        if not wavelengths:
            for sys, freqs in frequencies.items():
                wavelengths[sys] = {fname: speed_of_light / fhz for fname, fhz in freqs.items() if fhz > 0}

        data = {'header': {}, 'epochs': []}
        receiver_observations = {}
        final_satellite_wavelengths = {}

        header_parsed = False
        with open(file_path, 'r', encoding='utf-8') as f:
            # Step 1: Parse Header
            header_lines = []
            for line in f:
                line = line.rstrip('\n')
                header_lines.append(line)
                if 'END OF HEADER' in line:
                    header_parsed = True
                    break
            
            if not header_parsed:
                return {'data': data, 'receiver_observations': {}, 'satellite_wavelengths': {}}

            for line in header_lines:
                if 'RINEX VERSION' in line:
                    version_str = line[:9].strip()
                    try:
                        data['header']['version'] = float(version_str) if version_str else None
                    except Exception:
                        data['header']['version'] = None
                elif 'MARKER NAME' in line:
                    data['header']['marker'] = line[:60].strip()

            # parse multi-line SYS / # / OBS TYPES (from collected header lines)
            j = 0
            while j < len(header_lines):
                line = header_lines[j]
                if 'SYS / # / OBS TYPES' in line:
                    system = line[0].strip() if line[0].strip() else None
                    try:
                        num_types = int(line[3:6])
                    except Exception:
                        tokens = line.split()
                        system = system or (tokens[0] if tokens and len(tokens[0])==1 and tokens[0].isalpha() else None)
                        num_types = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 0
                    
                    obs_types_list = line[7:60].split()
                    j += 1
                    while j < len(header_lines) and len(obs_types_list) < num_types:
                        cont_line = header_lines[j]
                        if 'SYS / # / OBS TYPES' in cont_line:
                            obs_types_list.extend(cont_line[7:60].split())
                            j += 1
                        else:
                            break
                    if system:
                        data['header'][f'obs_types_{system}'] = obs_types_list[:num_types]
                    continue
                j += 1

            target_freqs = {'G': ['L1C', 'L5Q'], 'R': ['L1C'], 'E': ['L1C', 'L5Q', 'L7Q'], 'C': ['L2I', 'L1P', 'L5P']}

            # Step 2: Parse Body (Streaming)
            current_epoch = None
            current_satellites = {}
            
            # Helper to safely read field
            def read_field(sat_data, k, expected_fields, field_width=16):
                if k is None or k >= expected_fields:
                    return None
                start = k * field_width; end = start + field_width
                value_str = sat_data[start:end][:14].strip()
                try:
                    return float(value_str) if value_str else None
                except Exception:
                    return None

            # For streaming multi-line observations, we need a small buffer or state
            for line in f:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                if line.startswith('>'):
                    if current_epoch is not None:
                        data['epochs'].append({'time': current_epoch, 'satellites': current_satellites.copy()})
                    parts = line[1:].split()
                    if len(parts) >= 6:
                        try:
                            year, month, day, hour, minute = map(int, parts[:5])
                            second_float = float(parts[5])
                            current_epoch = pd.Timestamp(
                                year=year, month=month, day=day,
                                hour=hour, minute=minute, second=int(second_float),
                                microsecond=int((second_float - int(second_float)) * 1000000)
                            )
                            current_satellites = {}
                        except Exception:
                            continue
                    continue

                if current_epoch is None or len(line) < 3:
                    continue

                sat_system = line[0]
                sat_prn = line[1:3].strip()
                if not sat_prn:
                    continue
                sat_id = f"{sat_system}{sat_prn}"

                # Optimization: Cache PRN check
                current_wavelengths = wavelengths.get(sat_system, {})
                if sat_system == 'R' and sat_prn.isdigit():
                    k = glonass_k_map.get(f"R{sat_prn.zfill(2)}", 0)
                    if not (-7 <= k <= 6): k = 0
                    l1c_freq = 1602e6 + k * 0.5625e6
                    current_wavelengths = current_wavelengths.copy()
                    current_wavelengths['L1C'] = speed_of_light / l1c_freq
                
                final_satellite_wavelengths[sat_id] = current_wavelengths

                obs_types = data['header'].get(f'obs_types_{sat_system}', [])
                if not obs_types:
                    continue

                # Handle potential multi-line observation for a single satellite
                sat_data = line[3:]
                field_width = 16
                expected_fields = len(obs_types)
                
                # If fields are missing and next lines are not headers or new satellites, 
                # we should technically read more. But for simplicity and common files, 
                # let's assume one line first, or we'd need a more complex lookahead.
                # Given performance task, we assume standard RINEX 3 layouts.

                # Quick index map
                idx_map = {'code': {}, 'phase': {}, 'doppler': {}, 'snr': {}}
                for idx, obs in enumerate(obs_types):
                    tag = f"L{obs[1:]}"
                    if obs.startswith('C'): idx_map['code'][tag] = idx
                    elif obs.startswith('L'): idx_map['phase'][tag] = idx
                    elif obs.startswith('D'): idx_map['doppler'][tag] = idx
                    elif obs.startswith('S'): idx_map['snr'][tag] = idx

                if sat_id not in receiver_observations:
                    receiver_observations[sat_id] = {}

                for freq in target_freqs.get(sat_system, []):
                    if freq not in idx_map['code'] and freq not in idx_map['phase']:
                        continue
                    
                    if freq not in receiver_observations[sat_id]:
                        receiver_observations[sat_id][freq] = {
                            'times': [], 'code': [], 'phase': [], 'phase_cycle': [],
                            'doppler': [], 'doppler_hz': [], 'wavelength': [], 'snr': []
                        }

                    code_val = read_field(sat_data, idx_map['code'].get(freq), expected_fields)
                    phase_cycle_val = read_field(sat_data, idx_map['phase'].get(freq), expected_fields)
                    doppler_raw = read_field(sat_data, idx_map['doppler'].get(freq), expected_fields)
                    snr_val = read_field(sat_data, idx_map['snr'].get(freq), expected_fields)
                    wavelength = current_wavelengths.get(freq)

                    obs_list = receiver_observations[sat_id][freq]
                    obs_list['times'].append(current_epoch)
                    obs_list['code'].append(code_val)
                    obs_list['phase_cycle'].append(phase_cycle_val)
                    obs_list['wavelength'].append(wavelength)
                    obs_list['snr'].append(snr_val)
                    obs_list['doppler_hz'].append(doppler_raw)

                    if phase_cycle_val is not None and wavelength is not None:
                        obs_list['phase'].append(phase_cycle_val * wavelength)
                    else:
                        obs_list['phase'].append(None)
                    
                    if doppler_raw is not None and wavelength is not None:
                        obs_list['doppler'].append(-doppler_raw * wavelength)
                    else:
                        obs_list['doppler'].append(None)
                        
                # Note: current_satellites not used in receiver path for output, but populated for compatibility
                current_satellites[sat_id] = {} 

        if progress_callback:
            progress_callback(1.0)

        return {'data': data, 'receiver_observations': receiver_observations, 'satellite_wavelengths': final_satellite_wavelengths}
