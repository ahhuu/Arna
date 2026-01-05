"""
RINEX文件读取模块
负责读取和解析RINEX观测文件
"""

import os
import datetime
import pandas as pd
import numpy as np
from typing import Dict
from collections import defaultdict

class RinexReader:
    """RINEX观测文件读取器"""
    
    def __init__(self, config=None):
        from ..core.config import GNSSConfig
        self.config = config if config else GNSSConfig()
        
        # 缓存观测数据
        self.observations_meters = {}
        self.receiver_observations = {}
        
        # 进度回调
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def update_progress(self, value):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(value)
    
    def read_rinex_obs(self, file_path: str) -> Dict:
        """读取手机RINEX观测文件并解析数据"""
        data = {
            'header': {},
            'epochs': []
        }
        
        self.observations_meters = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\\n') for line in f]
        
        # 解析头部
        header_end = 0
        for i, line in enumerate(lines):
            if 'END OF HEADER' in line:
                header_end = i + 1
                break
            # 提取头部信息
            if 'RINEX VERSION' in line:
                version_str = line[:9].strip()
                data['header']['version'] = float(version_str) if version_str else None
            elif 'MARKER NAME' in line:
                data['header']['marker'] = line[:60].strip()
            elif 'OBS TYPES' in line:
                system = line[0]
                obs_types = line[6:60].split()
                data['header'][f'obs_types_{system}'] = obs_types
        
        # 更新头部解析进度
        header_progress = min(int(header_end / len(lines) * 20), 20)
        self.update_progress(header_progress)
        
        # 解析观测数据
        current_epoch = None
        current_satellites = {}
        i = header_end
        total_lines = len(lines)
        
        # 为每个卫星维护独立的波长记录
        satellite_wavelengths = {}
        
        while i < total_lines:
            # 每处理1%的行更新一次进度
            if i % (total_lines // 80) == 0:
                progress = 20 + int((i - header_end) / (total_lines - header_end) * 80)
                self.update_progress(progress)
            
            line = lines[i]
            if not line.strip():
                i += 1
                continue
                
            if line.startswith('>'):
                # 新历元
                if current_epoch is not None:
                    data['epochs'].append({
                        'time': current_epoch,
                        'satellites': current_satellites.copy()
                    })
                
                try:
                    parts = line[1:].split()
                    if len(parts) >= 6:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        hour = int(parts[3])
                        minute = int(parts[4])
                        
                        # 处理秒的小数部分，保留精度避免历元合并
                        second_float = float(parts[5])
                        
                        # 处理边界情况：如果秒值>=60，需要进位到分钟
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
                        
                        # 创建精确的时间戳，保留秒的小数部分
                        current_epoch = pd.Timestamp(
                            year=year, month=month, day=day,
                            hour=hour, minute=minute, second=int(second_float),
                            microsecond=int((second_float - int(second_float)) * 1000000)
                        )
                        current_satellites = {}
                    else:
                        print(f"警告: 时间行格式错误 (行 {i + 1}): {line}")
                        i += 1
                        continue
                        
                except (ValueError, IndexError) as e:
                    print(f"时间解析错误 (行 {i + 1}): {line}")
                    print(f"错误详情: {str(e)}")
                    i += 1
                    continue
                    
                i += 1
                
            else:
                # 解析卫星观测值
                if current_epoch is None:
                    i += 1
                    continue
                if len(line) < 3:
                    i += 1
                    continue
                    
                sat_system = line[0]
                sat_prn = line[1:3].strip()
                if not sat_prn:
                    i += 1
                    continue
                sat_id = f"{sat_system}{sat_prn}"
                
                # 创建当前卫星的局部频率和波长字典
                current_freqs = self.config.frequencies.get(sat_system, {}).copy()
                current_wavelengths = self.config.wavelengths.get(sat_system, {}).copy()
                
                # GLONASS频率计算
                if sat_system == 'R' and sat_prn.isdigit():
                    prn = f"R{sat_prn.zfill(2)}"
                    k = self.config.get_glonass_k(prn)
                    if not (-7 <= k <= 6):
                        print(f"警告: GLONASS {prn} 的k值{k}超出有效范围")
                        k = 0
                    l1c_freq = 1602e6 + k * 0.5625e6
                    current_freqs['L1C'] = l1c_freq
                    current_wavelengths['L1C'] = self.config.speed_of_light / l1c_freq
                
                # 保存当前卫星的波长信息
                satellite_wavelengths[sat_id] = current_wavelengths.copy()
                
                # 获取该系统的观测类型
                obs_types = data['header'].get(f'obs_types_{sat_system}', [])
                if not obs_types:
                    i += 1
                    continue
                
                # 解析观测值
                observations = {}
                sat_data = line[3:]  # 跳过卫星标识
                field_width = 16
                expected_fields = len(obs_types)
                actual_fields = (len(sat_data) + field_width - 1) // field_width
                
                for j in range(expected_fields):
                    if j < actual_fields:
                        start_idx = j * field_width
                        end_idx = start_idx + field_width
                        field = sat_data[start_idx:end_idx].strip()
                        obs_type = obs_types[j]
                        try:
                            observations[obs_type] = float(field) if field else None
                        except ValueError:
                            observations[obs_type] = None
                    else:
                        obs_type = obs_types[j]
                        observations[obs_type] = None
                
                current_satellites[sat_id] = observations
                
                # 存储原始观测值（伪距、相位、多普勒）
                if sat_id not in self.observations_meters:
                    self.observations_meters[sat_id] = {}
                
                # 使用已保存的卫星特定波长值
                sat_wavelengths = satellite_wavelengths.get(sat_id, {})
                
                for freq in current_freqs:
                    code_obs_type = f'C{freq[1:]}'
                    phase_obs_type = f'L{freq[1:]}'
                    doppler_obs_type = f'D{freq[1:]}'
                    snr_obs_type = f'S{freq[1:]}'
                    
                    # 获取观测值
                    code_val = observations.get(code_obs_type)
                    phase_val = observations.get(phase_obs_type)
                    doppler_val = observations.get(doppler_obs_type)
                    snr_val = observations.get(snr_obs_type)
                    wavelength = sat_wavelengths.get(freq)
                    
                    # 初始化数据结构
                    if freq not in self.observations_meters[sat_id]:
                        self.observations_meters[sat_id][freq] = {
                            'times': [],
                            'code': [],
                            'phase': [],
                            'phase_cycle': [],
                            'doppler': [],
                            'wavelength': [],
                            'snr': []
                        }
                    
                    # 存储时间和观测值
                    self.observations_meters[sat_id][freq]['times'].append(current_epoch)
                    self.observations_meters[sat_id][freq]['code'].append(code_val)
                    self.observations_meters[sat_id][freq]['snr'].append(snr_val)
                    
                    if wavelength is None:
                        global_wavelength = self.config.wavelengths.get(sat_system, {}).get(freq)
                        if global_wavelength is not None:
                            wavelength = global_wavelength
                            print(f"警告: 卫星 {sat_id} 频率 {freq} 局部波长缺失，使用全局值 {wavelength}")
                        else:
                            print(f"错误: 卫星 {sat_id} 频率 {freq} 未找到波长定义")
                    
                    # 存储相位（米）
                    if phase_val is not None and wavelength is not None:
                        self.observations_meters[sat_id][freq]['phase'].append(phase_val * wavelength)
                    else:
                        self.observations_meters[sat_id][freq]['phase'].append(None)
                    
                    # 存储相位（周）和波长
                    self.observations_meters[sat_id][freq]['phase_cycle'].append(phase_val)
                    self.observations_meters[sat_id][freq]['wavelength'].append(wavelength)
                    
                    # 存储多普勒（米/秒）
                    if doppler_val is not None and wavelength is not None:
                        self.observations_meters[sat_id][freq]['doppler'].append(-doppler_val * wavelength)
                    else:
                        self.observations_meters[sat_id][freq]['doppler'].append(None)
                
                i += 1
        
        # 添加最后一个历元
        if current_epoch is not None and current_satellites:
            data['epochs'].append({
                'time': current_epoch,
                'satellites': current_satellites.copy()
            })
        
        self.update_progress(100)
        return data
    
    def read_receiver_rinex_obs(self, file_path: str) -> Dict:
        """读取接收机RINEX观测文件并解析数据"""
        data = {'header': {}, 'epochs': []}
        self.receiver_observations = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\\n') for line in f]
        
        # 解析头部
        header_end = 0
        for i, line in enumerate(lines):
            if 'END OF HEADER' in line:
                header_end = i + 1
                break
            if 'RINEX VERSION' in line:
                version_str = line[:9].strip()
                data['header']['version'] = float(version_str) if version_str else None
            elif 'MARKER NAME' in line:
                data['header']['marker'] = line[:60].strip()
        
        # 解析多行 SYS / # / OBS TYPES
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
        
        # 目标频点
        target_freqs = {
            'G': ['L1C', 'L5Q'],
            'R': ['L1C'],
            'E': ['L1C', 'L5Q', 'L7Q'],
            'C': ['L2I', 'L1P', 'L5P']
        }
        
        # 解析观测数据
        current_epoch = None
        current_satellites = {}
        total_lines = len(lines)
        i = header_end
        
        while i < total_lines:
            if i % (max((total_lines - header_end), 1) // 80 + 1) == 0:
                progress = 20 + int((i - header_end) / max((total_lines - header_end), 1) * 80)
                self.update_progress(min(progress, 99))
            
            line = lines[i]
            if not line.strip():
                i += 1
                continue
                
            if line.startswith('>'):
                if current_epoch is not None:
                    data['epochs'].append({'time': current_epoch, 'satellites': current_satellites.copy()})
                    
                try:
                    parts = line[1:].split()
                    if len(parts) >= 6:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        hour = int(parts[3])
                        minute = int(parts[4])
                        second_float = float(parts[5])
                        current_epoch = pd.Timestamp(
                            year=year, month=month, day=day,
                            hour=hour, minute=minute, second=int(second_float),
                            microsecond=int((second_float - int(second_float)) * 1000000)
                        )
                        current_satellites = {}
                    else:
                        i += 1
                        continue
                except Exception:
                    i += 1
                    continue
                    
                i += 1
                continue
            
            # 卫星行解析
            if current_epoch is None or len(line) < 3:
                i += 1
                continue
            
            sat_system = line[0]
            sat_prn = line[1:3].strip()
            if not sat_prn:
                i += 1
                continue
            sat_id = f"{sat_system}{sat_prn}"
            
            # 构建当前卫星频率/波长
            current_freqs = self.config.frequencies.get(sat_system, {}).copy()
            current_wavelengths = self.config.wavelengths.get(sat_system, {}).copy()
            
            if sat_system == 'R' and sat_prn.isdigit():
                prn_key = f"R{sat_prn.zfill(2)}"
                k = self.config.get_glonass_k(prn_key)
                if not (-7 <= k <= 6):
                    k = 0
                l1c_freq = 1602e6 + k * 0.5625e6
                current_freqs['L1C'] = l1c_freq
                current_wavelengths['L1C'] = self.config.speed_of_light / l1c_freq
            
            # 可用频点
            allowed_freqs = target_freqs.get(sat_system, [])
            if sat_id not in self.receiver_observations:
                self.receiver_observations[sat_id] = {}
            
            # 从头部获取该系统观测类型表
            obs_types = data['header'].get(f'obs_types_{sat_system}', [])
            if not obs_types:
                i += 1
                continue
            
            # 固定宽度字段解析
            sat_data = line[3:]
            field_width = 16
            expected_fields = len(obs_types)
            
            # 构造索引映射
            idx_map = defaultdict(dict)
            for idx, obs in enumerate(obs_types):
                if obs.startswith('C'):
                    f = f"L{obs[1:]}"
                    idx_map[f]['code'] = idx
                elif obs.startswith('L'):
                    f = f"L{obs[1:]}"
                    idx_map[f]['phase'] = idx
                elif obs.startswith('D'):
                    f = f"L{obs[1:]}"
                    idx_map[f]['doppler'] = idx
                elif obs.startswith('S'):
                    f = f"L{obs[1:]}"
                    idx_map[f]['snr'] = idx
            
            for freq in allowed_freqs:
                if freq not in idx_map:
                    continue
                    
                # 初始化频点容器
                if freq not in self.receiver_observations[sat_id]:
                    self.receiver_observations[sat_id][freq] = {
                        'times': [],
                        'code': [],
                        'phase': [],
                        'phase_cycle': [],
                        'doppler': [],
                        'doppler_hz': [],
                        'wavelength': [],
                        'snr': []
                    }
                
                # 取字段值
                code_idx = idx_map[freq].get('code')
                phase_idx = idx_map[freq].get('phase')
                doppler_idx = idx_map[freq].get('doppler')
                snr_idx = idx_map[freq].get('snr')
                
                def read_field(k):
                    if k is None or k >= expected_fields:
                        return None
                    start = k * field_width
                    end = start + field_width
                    field = sat_data[start:end]
                    value_str = field[:14].strip()
                    try:
                        return float(value_str) if value_str else None
                    except Exception:
                        return None
                
                code_val = read_field(code_idx)
                phase_cycle_val = read_field(phase_idx)
                doppler_raw = read_field(doppler_idx)
                snr_val = read_field(snr_idx)
                
                # 频率波长
                wavelength = current_wavelengths.get(freq)
                
                # 存储观测数据
                self.receiver_observations[sat_id][freq]['times'].append(current_epoch)
                self.receiver_observations[sat_id][freq]['code'].append(code_val)
                
                # 相位转米
                if phase_cycle_val is not None and wavelength is not None:
                    self.receiver_observations[sat_id][freq]['phase'].append(phase_cycle_val * wavelength)
                else:
                    self.receiver_observations[sat_id][freq]['phase'].append(None)
                    
                self.receiver_observations[sat_id][freq]['phase_cycle'].append(phase_cycle_val)
                self.receiver_observations[sat_id][freq]['wavelength'].append(wavelength)
                
                # 多普勒转 m/s
                if doppler_raw is not None and wavelength is not None:
                    self.receiver_observations[sat_id][freq]['doppler'].append(-doppler_raw * wavelength)
                else:
                    self.receiver_observations[sat_id][freq]['doppler'].append(None)
                    
                self.receiver_observations[sat_id][freq]['doppler_hz'].append(doppler_raw)
                self.receiver_observations[sat_id][freq]['snr'].append(snr_val)
            
            i += 1
        
        self.update_progress(100)
        return data