from typing import Dict, Any, Optional, List
from collections import defaultdict
import pandas as pd
from ..core.config import GNSS_FREQUENCIES, GLONASS_K_MAP, SPEED_OF_LIGHT


class RinexReader:
    """RINEX reading utilities for phone and receiver files.

    Methods return dictionaries and do not modify external state.
    """

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
                wavelengths[sys] = {}
                for fname, fhz in freqs.items():
                     if fhz > 0:
                        wavelengths[sys][fname] = speed_of_light / fhz

        data = {'header': {}, 'epochs': []}
        observations_meters = {}

        # Store calculated specific satellite specific frequency wavelength {sat_id: {freq: wavelength}}
        # This is important for GLONASS satellites with different frequencies
        final_satellite_wavelengths = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]

        total_lines = len(lines)
        # header
        header_end = 0
        for i, line in enumerate(lines):
            if 'END OF HEADER' in line:
                header_end = i + 1
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
                data['header'][f'obs_types_{system}'] = obs_types

        # parse body
        i = header_end
        current_epoch = None
        current_satellites = {}
        satellite_wavelengths = {}

        while i < total_lines:
            if progress_callback and total_lines > 0 and i % max((total_lines // 80), 1) == 0:
                progress_callback(min(1.0, 0.2 + (i - header_end) / max(1, total_lines - header_end) * 0.8))

            line = lines[i]
            if not line.strip():
                i += 1
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
                        # normalize seconds/minutes/hours if necessary
                        if second_float >= 60:
                            extra_minutes = int(second_float // 60)
                            minute += extra_minutes
                            second_float = second_float % 60
                            if minute >= 60:
                                extra_hours = minute // 60
                                hour += extra_hours
                                minute = minute % 60
                                if hour >= 24:
                                    extra_days = hour // 24
                                    day += extra_days
                                    hour = hour % 24

                        current_epoch = pd.Timestamp(
                            year=year, month=month, day=day,
                            hour=hour, minute=minute, second=int(second_float),
                            microsecond=int((second_float - int(second_float)) * 1000000)
                        )
                        current_satellites = {}
                    except Exception:
                        i += 1
                        continue
                i += 1
                continue

            # satellite observation line
            if current_epoch is None or len(line) < 3:
                i += 1
                continue
            sat_system = line[0]
            sat_prn = line[1:3].strip()
            if not sat_prn:
                i += 1
                continue
            sat_id = f"{sat_system}{sat_prn}"

            current_freqs = frequencies.get(sat_system, {}).copy()
            current_wavelengths = wavelengths.get(sat_system, {}).copy()

            if sat_system == 'R' and sat_prn.isdigit():
                prn = f"R{sat_prn.zfill(2)}"
                k = glonass_k_map.get(prn, 0)
                if not (-7 <= k <= 6):
                    k = 0
                l1c_freq = 1602e6 + k * 0.5625e6
                current_freqs['L1C'] = l1c_freq
                current_wavelengths['L1C'] = speed_of_light / l1c_freq

            satellite_wavelengths[sat_id] = current_wavelengths.copy()
            final_satellite_wavelengths[sat_id] = current_wavelengths.copy()

            obs_types = data['header'].get(f'obs_types_{sat_system}', [])
            if not obs_types:
                i += 1
                continue

            observations = {}
            sat_data = line[3:]
            field_width = 16
            expected_fields = len(obs_types)
            actual_fields = (len(sat_data) + field_width - 1) // field_width
            for j in range(expected_fields):
                if j < actual_fields:
                    start_idx = j * field_width; end_idx = start_idx + field_width
                    field = sat_data[start_idx:end_idx].strip()
                    obs_type = obs_types[j]
                    try:
                        observations[obs_type] = float(field) if field else None
                    except ValueError:
                        observations[obs_type] = None
                else:
                    observations[obs_types[j]] = None

            current_satellites[sat_id] = observations

            if sat_id not in observations_meters:
                observations_meters[sat_id] = {}

            sat_wavelengths = satellite_wavelengths.get(sat_id, {})

            for freq in current_freqs:
                code_obs_type = f'C{freq[1:]}'
                phase_obs_type = f'L{freq[1:]}'
                doppler_obs_type = f'D{freq[1:]}'
                snr_obs_type = f'S{freq[1:]}'
                code_val = observations.get(code_obs_type)
                phase_val = observations.get(phase_obs_type)
                doppler_val = observations.get(doppler_obs_type)
                snr_val = observations.get(snr_obs_type)
                wavelength = sat_wavelengths.get(freq)

                if freq not in observations_meters[sat_id]:
                    observations_meters[sat_id][freq] = {
                        'times': [], 'code': [], 'phase': [], 'phase_cycle': [], 'doppler': [],
                        'wavelength': [], 'snr': []
                    }

                if wavelength is None:
                    global_wavelength = wavelengths.get(sat_system, {}).get(freq)
                    if global_wavelength is not None:
                        wavelength = global_wavelength
                    else:
                        wavelength = None

                observations_meters[sat_id][freq]['times'].append(current_epoch)
                observations_meters[sat_id][freq]['code'].append(code_val)
                observations_meters[sat_id][freq]['snr'].append(snr_val)
                if phase_val is not None and wavelength is not None:
                    observations_meters[sat_id][freq]['phase'].append(phase_val * wavelength)
                else:
                    observations_meters[sat_id][freq]['phase'].append(None)
                observations_meters[sat_id][freq]['phase_cycle'].append(phase_val)
                observations_meters[sat_id][freq]['wavelength'].append(wavelength)
                if doppler_val is not None and wavelength is not None:
                    observations_meters[sat_id][freq]['doppler'].append(-doppler_val * wavelength)
                else:
                    observations_meters[sat_id][freq]['doppler'].append(None)

            i += 1

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
                wavelengths[sys] = {}
                for fname, fhz in freqs.items():
                     if fhz > 0:
                        wavelengths[sys][fname] = speed_of_light / fhz

        data = {'header': {}, 'epochs': []}
        receiver_observations = {}
        
        final_satellite_wavelengths = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]

        total_lines = len(lines)
        header_end = 0
        for i, line in enumerate(lines):
            if 'END OF HEADER' in line:
                header_end = i + 1
                break
            if 'RINEX VERSION' in line:
                version_str = line[:9].strip()
                try:
                    data['header']['version'] = float(version_str) if version_str else None
                except Exception:
                    data['header']['version'] = None
            elif 'MARKER NAME' in line:
                data['header']['marker'] = line[:60].strip()

        # parse multi-line SYS / # / OBS TYPES
        j = 0
        while j < header_end:
            line = lines[j]
            if 'SYS / # / OBS TYPES' in line:
                system = line[0].strip() if line[0].strip() else None
                try:
                    num_types = int(line[3:6])
                except Exception:
                    tokens = line.split()
                    if system is None and tokens:
                        if len(tokens[0]) == 1 and tokens[0].isalpha():
                            system = tokens[0]
                    num_types = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 0
                obs_types_list = []
                obs_types_list.extend(line[7:60].split())
                j += 1
                while j < header_end and len(obs_types_list) < num_types:
                    cont_line = lines[j]
                    if 'SYS / # / OBS TYPES' in cont_line:
                        obs_types_list.extend(cont_line[7:60].split())
                        j += 1
                    else:
                        break
                if system:
                    data['header'][f'obs_types_{system}'] = obs_types_list[:num_types] if num_types > 0 else obs_types_list
                continue
            j += 1

        target_freqs = {
            'G': ['L1C', 'L5Q'], 'R': ['L1C'], 'E': ['L1C', 'L5Q', 'L7Q'], 'C': ['L2I', 'L1P', 'L5P']
        }

        i = header_end
        current_epoch = None
        current_satellites = {}
        while i < total_lines:
            if progress_callback and total_lines > 0 and i % max((total_lines // 80), 1) == 0:
                progress_callback(min(1.0, 0.2 + (i - header_end) / max(1, total_lines - header_end) * 0.8))

            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if line.startswith('>'):
                if current_epoch is not None:
                    data['epochs'].append({'time': current_epoch, 'satellites': current_satellites.copy()})
                parts = line[1:].split()
                if len(parts) >= 6:
                    try:
                        year = int(parts[0]); month = int(parts[1]); day = int(parts[2])
                        hour = int(parts[3]); minute = int(parts[4]); second_float = float(parts[5])
                        current_epoch = pd.Timestamp(
                            year=year, month=month, day=day,
                            hour=hour, minute=minute, second=int(second_float),
                            microsecond=int((second_float - int(second_float)) * 1000000)
                        )
                        current_satellites = {}
                    except Exception:
                        i += 1
                        continue
                i += 1
                continue

            if current_epoch is None or len(line) < 3:
                i += 1
                continue

            sat_system = line[0]
            sat_prn = line[1:3].strip()
            if not sat_prn:
                i += 1
                continue
            sat_id = f"{sat_system}{sat_prn}"

            current_freqs = frequencies.get(sat_system, {}).copy()
            
            current_wavelengths = wavelengths.get(sat_system, {}).copy()
            if sat_system == 'R' and sat_prn.isdigit():
                prn_key = f"R{sat_prn.zfill(2)}"
                k = glonass_k_map.get(prn_key, 0)
                if not (-7 <= k <= 6):
                    k = 0
                l1c_freq = 1602e6 + k * 0.5625e6
                current_freqs['L1C'] = l1c_freq
                current_wavelengths['L1C'] = speed_of_light / l1c_freq
            final_satellite_wavelengths[sat_id] = current_wavelengths.copy()

            if sat_id not in receiver_observations:
                receiver_observations[sat_id] = {}

            obs_types = data['header'].get(f'obs_types_{sat_system}', [])
            if not obs_types:
                i += 1
                continue

            sat_data = line[3:]
            field_width = 16
            expected_fields = len(obs_types)
            k = 1
            while (len(sat_data) // field_width) < expected_fields and (i + k) < total_lines:
                next_line = lines[i + k]
                if next_line.startswith('>'):
                    break
                if len(next_line) >= 3 and next_line[0].isalpha() and next_line[1:3].strip().isdigit():
                    break
                sat_data += next_line
                k += 1
            if k > 1:
                i += (k - 1)
            actual_fields = (len(sat_data) + field_width - 1) // field_width

            # build idx_map
            idx_map = defaultdict(dict)
            for idx, obs in enumerate(obs_types):
                if obs.startswith('C'):
                    f = f"L{obs[1:]}"; idx_map[f]['code'] = idx
                elif obs.startswith('L'):
                    f = f"L{obs[1:]}"; idx_map[f]['phase'] = idx
                elif obs.startswith('D'):
                    f = f"L{obs[1:]}"; idx_map[f]['doppler'] = idx
                elif obs.startswith('S'):
                    f = f"L{obs[1:]}"; idx_map[f]['snr'] = idx

            for freq in target_freqs.get(sat_system, []):
                if freq not in idx_map:
                    continue
                if freq not in receiver_observations[sat_id]:
                    receiver_observations[sat_id][freq] = {
                        'times': [], 'code': [], 'phase': [], 'phase_cycle': [],
                        'doppler': [], 'doppler_hz': [], 'wavelength': [], 'snr': []
                    }

                def read_field(k):
                    if k is None:
                        return None
                    if k >= expected_fields:
                        return None
                    start = k * field_width; end = start + field_width
                    field = sat_data[start:end]
                    value_str = field[:14].strip()
                    try:
                        return float(value_str) if value_str else None
                    except Exception:
                        return None

                code_idx = idx_map[freq].get('code')
                phase_idx = idx_map[freq].get('phase')
                doppler_idx = idx_map[freq].get('doppler')
                snr_idx = idx_map[freq].get('snr')

                code_val = read_field(code_idx)
                phase_cycle_val = read_field(phase_idx)
                doppler_raw = read_field(doppler_idx)
                snr_val = read_field(snr_idx)

                wavelength = current_wavelengths.get(freq)

                receiver_observations[sat_id][freq]['times'].append(current_epoch)
                receiver_observations[sat_id][freq]['code'].append(code_val)
                if phase_cycle_val is not None and wavelength is not None:
                    receiver_observations[sat_id][freq]['phase'].append(phase_cycle_val * wavelength)
                else:
                    receiver_observations[sat_id][freq]['phase'].append(None)
                receiver_observations[sat_id][freq]['phase_cycle'].append(phase_cycle_val)
                receiver_observations[sat_id][freq]['wavelength'].append(wavelength)
                if doppler_raw is not None and wavelength is not None:
                    receiver_observations[sat_id][freq]['doppler'].append(-doppler_raw * wavelength)
                else:
                    receiver_observations[sat_id][freq]['doppler'].append(None)
                receiver_observations[sat_id][freq]['doppler_hz'].append(doppler_raw)
                receiver_observations[sat_id][freq]['snr'].append(snr_val)

            i += 1

        if progress_callback:
            progress_callback(1.0)

        return {'data': data, 'receiver_observations': receiver_observations, 'satellite_wavelengths': final_satellite_wavelengths}