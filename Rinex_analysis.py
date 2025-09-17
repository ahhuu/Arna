import os
import datetime
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')  # 使用Tkinter兼容的交互式后端
import matplotlib.pyplot as plt
from typing import Dict
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from collections import defaultdict

# 设置中文字体（全局）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class GNSSAnalyzer:
    """GNSS观测数据分析与问题检测器"""

    def __init__(self):
        # 定义GNSS信号频率 (Hz)
        self.r_squared_threshold = 0.5  # R方阈值，默认0.5
        self.cv_threshold = 0.5  # CV值阈值，默认0.5
        # 历元间双差最大阈值限制
        self.max_threshold_limits = {
            'code': 10.0,    # 伪距（米）
            'phase': 3.0,    # 相位（米）
            'doppler': 5.0   # 多普勒（米/秒）
        }
        # 手机独有卫星分析配置
        self.enable_phone_only_analysis = False  # 是否启用手机独有卫星分析
        self.phone_only_min_data_points = 20     # 手机独有卫星最小数据点数
        self.frequencies = {
            'G': {'L1C': 1575.42e6, 'L5Q': 1176.45e6},  # GPS
            'R': {'L1C': 1602e6, 'L5Q': 1246e6},  # GLONASS
            'E': {'L1B': 1575.42e6, 'L1C': 1575.42e6, 'L5Q': 1176.45e6, 'L7Q': 1207.14e6},  # Galileo
            'C': {'L2I': 1561.098e6, 'L1P': 1575.42e6, 'L1D': 1575.42e6, 'L5P': 1176.45e6},  # BeiDou
            'J': {'L1C': 1575.42e6, 'L5Q': 1176.45e6},  # QZSS
            'I': {'L5Q': 1176.45e6, 'S': 2492.028e6},  # IRNSS/NavIC
            'S': {'L1C': 1575.42e6, 'L5Q': 1176.45e6},  # SBAS
        }

        # 北斗卫星系统分类（基于PRN号）
        self.beidou_systems = {
            'BDS-2': {  # 北斗二号系统 (GEO/IGSO/MEO)
                'GEO': ['C01', 'C02', 'C03', 'C04', 'C05'],  # 地球静止轨道
                'IGSO': ['C06', 'C07', 'C08', 'C09', 'C10', 'C13'],  # 倾斜地球同步轨道
                'MEO': ['C11', 'C12', 'C14']  # 中地球轨道
            },
            'BDS-3': {  # 北斗三号系统 (GEO/IGSO/MEO)
                'GEO': ['C59', 'C60', 'C61'],  # 地球静止轨道
                'IGSO': ['C38', 'C39', 'C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46'],  # 倾斜地球同步轨道
                'MEO': ['C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58']  # 中地球轨道
            }
        }

        # GLONASS PRN到k值的映射表
        self.glonass_k_map = {
            'R01': +1, 'R02': -4, 'R03': +5, 'R04': +6,
            'R05': +1, 'R06': -4, 'R07': +5, 'R08': +6,
            'R09': -2, 'R10': -7, 'R11': 0, 'R12': -1,
            'R13': -2, 'R14': -7, 'R15': 0, 'R16': -1,
            'R17': +4, 'R18': -3, 'R19': +3, 'R20': +2,
            'R21': +4, 'R22': -3, 'R23': +3, 'R24': +2
        }

        # 计算对应波长 (m)
        self.wavelengths = {}
        self.speed_of_light = 299792458  # m/s
        for system, freqs in self.frequencies.items():
            self.wavelengths[system] = {
                freq: self.speed_of_light / f for freq, f in freqs.items()
            }

        # 存储以米为单位的观测值（手机/Android来源）
        self.observations_meters = {}  # 结构: {sat_id: {freq: {'times': [], 'code': [], 'phase': []}}}
        # 存储接收机RINEX观测（与手机数据分离）
        self.receiver_observations = {}  # 结构: {sat_id: {freq: {'times': [], 'code': [], 'phase': [], 'phase_cycle': [], 'doppler': [], 'wavelength': []}}}
        self.receiver_input_file_path = None

        # 存储分析结果
        self.results = {
            'code_carrier_inconsistency': {},
            'observation_inconsistency': {},
            'phase_stagnation': {},
            'observable_derivatives': {},
            'code_phase_differences': {},
            'phase_prediction_errors': {},
            'beidou_system_bias': {},  # 北斗系统间偏差分析结果
            'receiver_cmc': {},  # 接收机CMC
            'phone_cmc': {},  # 手机CMC
            'dcmc': {},  # 站间单差CMC
            'cci_series': {},  # 码相不一致性时间序列
            'roc_model': {},  # 各频率的ROC模型参数
            'corrected_phase': {}  # 校正后的载波相位
        }

        # 结果根目录
        self.results_root = "results"
        self.results_dir = self.results_root
        os.makedirs(self.results_root, exist_ok=True)
        # 当前输入文件路径及对应结果子目录
        self.input_file_path = None
        self.current_result_dir = None
        # 定义各类图表的子文件夹名称
        self.plot_categories = {
            'raw_observations': '原始观测值',
            'derivatives': '观测值一阶差分',
            'code_phase_diffs': '伪距相位差值之差',
            'code_phase_diff_raw': '伪距相位原始差值',
            'phase_pred_errors': '相位预测误差',
            'double_differences': '历元间双差',
            'beidou_bias_analysis': '北斗系统间偏差分析',
            'receiver_cmc': '接收机CMC'
        }

        # 进度管理相关属性
        self.progress_callback = None
        self.current_stage = 0  # 当前阶段
        self.total_stages = 10  # 总阶段数（读取、计算差分、计算相位停滞、计算差值、计算预测误差、计算历元双差、北斗系统间偏差分析、保存新文件、保存报告、保存图表）
        self.stage_progress = 0  # 当前阶段的进度
        self.stage_max = 100  # 当前阶段的最大进度

        # 剔除粗差后的观测值文件
        self.output_format = "rinex"  # 输出文件格式，保持与原文件一致
        self.cleaned_observations = {}  # 存储清洗后的观测数据

    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback

    def update_progress(self, value):
        """更新当前阶段的进度"""
        self.stage_progress = value
        overall_progress = (self.current_stage / self.total_stages) + \
                           (value / self.stage_max / self.total_stages)
        if self.progress_callback:
            self.progress_callback(overall_progress)

    def start_stage(self, stage_index, stage_name, max_units=100):
        """开始一个新的处理阶段"""
        self.current_stage = stage_index
        self.stage_progress = 0
        self.stage_max = max_units
        print(f"开始阶段: {stage_name}")
        if self.progress_callback:
            self.progress_callback((self.current_stage) / self.total_stages)

    def read_rinex_obs(self, file_path: str) -> Dict:
        """读取RINEX观测文件并解析数据"""
        self.input_file_path = file_path
        filename = os.path.basename(file_path)
        self.current_result_dir = os.path.join(self.results_root, filename.split('.')[0])
        os.makedirs(self.current_result_dir, exist_ok=True)

        data = {
            'header': {},
            'epochs': []
        }

        self.start_stage(0, "读取RINEX文件", 100)
        self.observations_meters = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]

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
                        try:
                            second_float = float(parts[5])
                            
                            # 处理边界情况：如果秒值>=60，需要进位到分钟
                            if second_float >= 60:
                                # 进位到分钟
                                extra_minutes = int(second_float // 60)
                                minute += extra_minutes
                                second_float = second_float % 60
                                
                                # 如果分钟>=60，进位到小时
                                if minute >= 60:
                                    extra_hours = minute // 60
                                    hour += extra_hours
                                    minute = minute % 60
                                    
                                    # 如果小时>=24，进位到天
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
                        except ValueError as ve:
                            print(f"秒值解析错误 (行 {i + 1}): 秒值='{parts[5]}', 行内容='{line.strip()}'")
                            raise ValueError(f"无法解析秒值: {parts[5]}") from ve
                        current_satellites = {}
                    else:
                        print(f"警告: 时间行格式错误 (行 {i + 1}): {line}")
                        i += 1
                        continue
                except (ValueError, IndexError) as e:
                    print(f"时间解析错误 (行 {i + 1}): {line}")
                    print(f"错误详情: {str(e)}")
                    # 不抛出异常，继续处理下一行
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
                current_freqs = self.frequencies.get(sat_system, {}).copy()
                current_wavelengths = self.wavelengths.get(sat_system, {}).copy()

                # 修改后的GLONASS频率计算逻辑
                if sat_system == 'R' and sat_prn.isdigit():
                    prn = f"R{sat_prn.zfill(2)}"
                    # 从GLONASS k值映射表获取实时k值
                    k = self.glonass_k_map.get(prn, 0)
                    # 验证k值有效性
                    if not (-7 <= k <= 6):  # 根据ICD文档调整范围
                        print(f"警告: GLONASS {prn} 的k值{k}超出有效范围")
                        k = 0  # 使用默认频道
                    l1c_freq = 1602e6 + k * 0.5625e6
                    current_freqs['L1C'] = l1c_freq
                    current_wavelengths['L1C'] = self.speed_of_light / l1c_freq

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

                    # 获取观测值
                    code_val = observations.get(code_obs_type)
                    phase_val = observations.get(phase_obs_type)
                    doppler_val = observations.get(doppler_obs_type)
                    wavelength = sat_wavelengths.get(freq)

                    # 初始化数据结构（确保wavelength字段存在）
                    if freq not in self.observations_meters[sat_id]:
                        self.observations_meters[sat_id][freq] = {
                            'times': [],
                            'code': [],
                            'phase': [],
                            'phase_cycle': [],
                            'doppler': [],
                            'wavelength': []  # 强制初始化波长列表
                        }
                    # 确保wavelength字段已创建
                    if 'wavelength' not in self.observations_meters[sat_id][freq]:
                        self.observations_meters[sat_id][freq]['wavelength'] = []

                    # 存储时间和伪距
                    self.observations_meters[sat_id][freq]['times'].append(current_epoch)
                    self.observations_meters[sat_id][freq]['code'].append(code_val)

                    if wavelength is None:
                        global_wavelength = self.wavelengths.get(sat_system, {}).get(freq)
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

                    # 存储相位（周）和波长（关键修改：无论相位是否存在，都存储波长）
                    self.observations_meters[sat_id][freq]['phase_cycle'].append(phase_val)
                    self.observations_meters[sat_id][freq]['wavelength'].append(wavelength)  # 强制存储波长

                    # 存储多普勒（米/秒）
                    if doppler_val is not None and wavelength is not None:
                        self.observations_meters[sat_id][freq]['doppler'].append(-doppler_val * wavelength)
                    else:
                        self.observations_meters[sat_id][freq]['doppler'].append(None)

                i += 1  # 处理下一行

        # 添加最后一个历元
        if current_epoch is not None and current_satellites:
            data['epochs'].append({
                'time': current_epoch,
                'satellites': current_satellites.copy()
            })

        # # 输出调试信息
        # self.print_observation_debug(data)
        self.update_progress(100)
        return data

    def read_receiver_rinex_obs(self, file_path: str) -> Dict:
        """读取接收机RINEX观测文件并解析到 self.receiver_observations（与手机分离）

        仅提取以下频点的伪距/相位/多普勒：
        - G: L1C, L5Q
        - R: L1C
        - E: L1C, L5Q, L7Q
        - C: L2I
        """
        self.receiver_input_file_path = file_path
        filename = os.path.basename(file_path)
        # 单独结果目录
        self.current_result_dir = os.path.join(self.results_root, filename.split('.')[0])
        os.makedirs(self.current_result_dir, exist_ok=True)

        data = {'header': {}, 'epochs': []}
        self.start_stage(0, "读取接收机RINEX文件", 100)
        self.receiver_observations = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]

        # 头部结束行
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
        try:
            j = 0
            while j < header_end:
                line = lines[j]
                if 'SYS / # / OBS TYPES' in line:
                    # 系统标识可能在续行缺失，先记录首行的系统字母
                    system = line[0].strip() if line[0].strip() else None
                    try:
                        num_types = int(line[3:6])
                    except Exception:
                        tokens = line.split()
                        # tokens[0] 可能是系统字母，tokens[1]是数量
                        if system is None and tokens:
                            # 若首字符是空格，尝试从tokens[0]获取系统
                            if len(tokens[0]) == 1 and tokens[0].isalpha():
                                system = tokens[0]
                        num_types = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 0
                    obs_types_list = []
                    obs_types_list.extend(line[7:60].split())
                    j += 1
                    while j < header_end and len(obs_types_list) < num_types:
                        cont_line = lines[j]
                        if 'SYS / # / OBS TYPES' in cont_line:
                            # 续行不再强制要求同一系统字母（兼容空格开头续行）
                            obs_types_list.extend(cont_line[7:60].split())
                            j += 1
                        else:
                            break
                    if system:
                        data['header'][f'obs_types_{system}'] = obs_types_list[:num_types] if num_types > 0 else obs_types_list
                    continue
                j += 1
        except Exception:
            pass

        # 目标频点
        target_freqs = {
            'G': ['L1C', 'L5Q'],
            'R': ['L1C'],
            'E': ['L1C', 'L5Q', 'L7Q'],
            'C': ['L2I']
        }

        # 历元/数据体
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
                        year = int(parts[0]); month = int(parts[1]); day = int(parts[2])
                        hour = int(parts[3]); minute = int(parts[4])
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

            # 卫星行
            if current_epoch is None or len(line) < 3:
                i += 1
                continue

            sat_system = line[0]
            sat_prn = line[1:3].strip()
            if not sat_prn:
                i += 1
                continue
            sat_id = f"{sat_system}{sat_prn}"

            # 构建当前卫星频率/波长（动态GLONASS频道频点处理，与手机一致）
            current_freqs = self.frequencies.get(sat_system, {}).copy()
            current_wavelengths = self.wavelengths.get(sat_system, {}).copy()
            if sat_system == 'R' and sat_prn.isdigit():
                prn_key = f"R{sat_prn.zfill(2)}"
                k = self.glonass_k_map.get(prn_key, 0)
                if not (-7 <= k <= 6):
                    # 超出范围时回退默认频道
                    k = 0
                l1c_freq = 1602e6 + k * 0.5625e6
                current_freqs['L1C'] = l1c_freq
                current_wavelengths['L1C'] = self.speed_of_light / l1c_freq

            # 可用频点
            allowed_freqs = target_freqs.get(sat_system, [])
            if sat_id not in self.receiver_observations:
                self.receiver_observations[sat_id] = {}

            # 从头部获取该系统观测类型表
            obs_types = data['header'].get(f'obs_types_{sat_system}', [])
            if not obs_types:
                i += 1
                continue

            # 固定宽度字段 + 续行拼接
            sat_data = line[3:]
            field_width = 16
            expected_fields = len(obs_types)
            # 续行拼接，直到收够expected_fields或遇到下一卫星/历元
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

            # 为每个目标频率抽取对应的C/L/D
            # 构造索引映射
            idx_map = defaultdict(dict)
            for idx, obs in enumerate(obs_types):
                if obs.startswith('C'):
                    f = f"L{obs[1:]}"; idx_map[f]['code'] = idx
                elif obs.startswith('L'):
                    f = f"L{obs[1:]}"; idx_map[f]['phase'] = idx
                elif obs.startswith('D'):
                    f = f"L{obs[1:]}"; idx_map[f]['doppler'] = idx

            for freq in allowed_freqs:
                if freq not in idx_map:
                    continue
                # 初始化频点容器
                if freq not in self.receiver_observations[sat_id]:
                    self.receiver_observations[sat_id][freq] = {
                        'times': [],
                        'code': [],
                        'phase': [],            # 转米
                        'phase_cycle': [],      # 原始周
                        'doppler': [],          # 转 m/s
                        'doppler_hz': [],       # 原始 Hz
                        'wavelength': []
                    }

                # 取字段值
                code_idx = idx_map[freq].get('code')
                phase_idx = idx_map[freq].get('phase')
                doppler_idx = idx_map[freq].get('doppler')

                def read_field(k):
                    if k is None:
                        return None
                    if k >= expected_fields:
                        return None
                    start = k * field_width
                    end = start + field_width
                    field = sat_data[start:end]
                    # RINEX 3: 前14列为数值，15列为LLI，16列为SNR（可能为空）
                    value_str = field[:14].strip()
                    try:
                        return float(value_str) if value_str else None
                    except Exception:
                        return None

                code_val = read_field(code_idx)
                phase_cycle_val = read_field(phase_idx)  # 周
                doppler_raw = read_field(doppler_idx)  # Hz（RINEX多普勒是Hz）

                # 频率波长（优先使用当前卫星动态波长）
                wavelength = current_wavelengths.get(freq)

                # 存储
                self.receiver_observations[sat_id][freq]['times'].append(current_epoch)
                self.receiver_observations[sat_id][freq]['code'].append(code_val)
                # 相位转米
                if phase_cycle_val is not None and wavelength is not None:
                    self.receiver_observations[sat_id][freq]['phase'].append(phase_cycle_val * wavelength)
                else:
                    self.receiver_observations[sat_id][freq]['phase'].append(None)
                self.receiver_observations[sat_id][freq]['phase_cycle'].append(phase_cycle_val)
                self.receiver_observations[sat_id][freq]['wavelength'].append(wavelength)
                # 多普勒转 m/s（-D*lambda）
                if doppler_raw is not None and wavelength is not None:
                    self.receiver_observations[sat_id][freq]['doppler'].append(-doppler_raw * wavelength)
                else:
                    self.receiver_observations[sat_id][freq]['doppler'].append(None)
                # 存原始Hz
                self.receiver_observations[sat_id][freq]['doppler_hz'].append(doppler_raw)

            i += 1

        # 收尾
        if current_epoch is not None and current_satellites is not None:
            # 此处 data['epochs'] 对CMC不是刚需，仅保留结构一致性
            pass
        self.update_progress(100)
        return data

    def dump_receiver_observations_debug(self):
        """将接收机观测容器中的各卫星/频率/历元的伪距、相位、多普勒输出到文本，便于调试"""
        if not self.receiver_observations:
            print("[调试] receiver_observations 为空")
            return

        debug_path = os.path.join(self.current_result_dir or self.results_root, "receiver_observations_debug.txt")
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write("=== 接收机RINEX观测调试输出 ===\n")
            for sat_id, freqs in self.receiver_observations.items():
                f.write(f"\n## 卫星 {sat_id}\n")
                for freq, obs in freqs.items():
                    times = obs.get('times', [])
                    code = obs.get('code', [])
                    phase_m = obs.get('phase', [])
                    phase_cycle = obs.get('phase_cycle', [])
                    doppler_mps = obs.get('doppler', [])
                    doppler_hz = obs.get('doppler_hz', [])
                    f.write(f"  - 频率 {freq}: 历元数={len(times)}\n")
                    for idx in range(len(times)):
                        t = times[idx] if idx < len(times) else None
                        c = code[idx] if idx < len(code) else None
                        p_m = phase_m[idx] if idx < len(phase_m) else None
                        p_cyc = phase_cycle[idx] if idx < len(phase_cycle) else None
                        d_mps = doppler_mps[idx] if idx < len(doppler_mps) else None
                        d_hz = doppler_hz[idx] if idx < len(doppler_hz) else None
                        f.write(
                            f"    [{idx+1}] {t}: Code={c}, Phase(cyc)={p_cyc}, Phase(m)={p_m}, Doppler(Hz)={d_hz}, Doppler(m/s)={d_mps}\n"
                        )
        print(f"[调试] 接收机观测调试文件: {debug_path}")

    def print_observation_debug(self, data: Dict) -> None:
        """输出各卫星/频率的原始观测值调试信息并保存为txt文件"""
        print("\n=== 原始观测数据调试输出 ===")
        debug_file_path = os.path.join(self.current_result_dir, "observation_debug.txt")

        with open(debug_file_path, 'w') as debug_file:
            debug_file.write("=== 原始观测数据调试输出 ===\n\n")

            for epoch_idx, epoch in enumerate(data['epochs'], 1):
                # 历元标题
                epoch_header = f"\n>> 历元 {epoch_idx} - {epoch['time']} <<\n"
                debug_file.write(epoch_header)
                print(epoch_header, end='')

                for sat_id, obs in epoch['satellites'].items():
                    system = sat_id[0]
                    # 获取当前卫星的频率（使用存储的局部频率，而非全局）
                    available_freqs = self.observations_meters.get(sat_id, {}).keys()
                    # 卫星标题
                    sat_header = f"\n * 卫星 {sat_id}\n"
                    debug_file.write(sat_header)
                    print(sat_header, end='')

                    for freq in available_freqs:
                        # 获取观测类型和值
                        code_obs_type = f'C{freq[1:]}'
                        phase_obs_type = f'L{freq[1:]}'
                        doppler_obs_type = f'D{freq[1:]}'
                        snr_obs_type = f'S{freq[1:]}'
                        code_val = obs.get(code_obs_type)
                        phase_val = obs.get(phase_obs_type)
                        doppler_val = obs.get(doppler_obs_type)
                        snr_val = obs.get(snr_obs_type)

                        # 跳过全空观测
                        if all(v is None for v in [code_val, phase_val, doppler_val, snr_val]):
                            continue

                        # 从存储数据获取计算值和波长（关键修改：使用该卫星存储的波长）
                        stored_data = self.observations_meters.get(sat_id, {}).get(freq, {})
                        phase_in_meters = stored_data.get('phase', [None])[-1]
                        doppler_in_mps = stored_data.get('doppler', [None])[-1]
                        # 优先从存储的波长列表中取当前历元的值（若存在）
                        wavelength = stored_data.get('wavelength', [None])[-1] if 'wavelength' in stored_data else None

                        # 构建频率块输出
                        output = [
                            f"  频率 {freq}:",
                            f"    - 伪距 ({code_obs_type}): {code_val:.3f} m" if code_val is not None else "- 伪距: None",
                            f"    - 相位 ({phase_obs_type}): {phase_val:.3f} 周" if phase_val is not None else "- 相位: None",
                            f"    - 波长: {wavelength:.6f} m" if wavelength is not None else "- 波长: None",
                            f"    - 相位(米): {phase_in_meters:.3f} m" if phase_in_meters is not None else "- 相位(米): None",
                            f"    - 多普勒原始值 ({doppler_obs_type}): {doppler_val:.3f} Hz" if doppler_val is not None else "- 多普勒原始值: None",
                            f"    - 多普勒速度: {doppler_in_mps:.3f} m/s" if doppler_in_mps is not None else "- 多普勒速度: None",
                            f"    - 信噪比 ({snr_obs_type}): {snr_val:.1f} dBHz" if snr_val is not None else "- 信噪比: None",
                            ""
                        ]

                        # 输出到控制台和文件
                        block_output = "\n".join(output)
                        print(block_output)
                        debug_file.write(block_output + "\n")

                print(f"\n调试信息已保存至: {debug_file_path}")

    def calculate_observable_derivatives(self, data: Dict) -> Dict:
        """计算每个卫星每个频率伪距、相位与多普勒观测的一阶差分"""
        self.start_stage(1, "计算观测值一阶差分", 100)

        derivatives = {}
        total_sats = len(data['epochs'][0]['satellites'].keys())
        processed_sats = 0

        # 遍历所有卫星
        for sat_id in data['epochs'][0]['satellites'].keys():
            system = sat_id[0]
            freq_derivatives = {}
            available_freqs = self.frequencies.get(system, {})

            # 初始化频率数据结构
            for freq in available_freqs:
                freq_derivatives[freq] = {
                    'times': [],
                    'pr_derivative': [],  # (m/s)
                    'ph_derivative': [],  # (m/s)
                    'doppler': []  # (m/s)
                }

            # 收集该卫星在各历元的观测值
            for epoch in data['epochs']:
                if sat_id in epoch['satellites']:
                    obs = epoch['satellites'][sat_id]
                    time = epoch['time']
                    for freq in available_freqs:
                        code_obs_type = f'C{freq[1:]}'
                        phase_obs_type = f'L{freq[1:]}'
                        doppler_obs_type = f'D{freq[1:]}'
                        code_value = obs.get(code_obs_type)
                        phase_value = obs.get(phase_obs_type)
                        doppler_value = obs.get(doppler_obs_type)
                        if code_value is not None or phase_value is not None or doppler_value is not None:
                            freq_derivatives[freq]['times'].append(time)
                            freq_derivatives[freq]['pr_derivative'].append(code_value)
                            freq_derivatives[freq]['ph_derivative'].append(phase_value)
                            freq_derivatives[freq]['doppler'].append(doppler_value)

            # 计算一阶差分
            for freq in available_freqs:
                times = freq_derivatives[freq]['times']
                pr_values = freq_derivatives[freq]['pr_derivative']
                ph_values = freq_derivatives[freq]['ph_derivative']
                doppler_values = freq_derivatives[freq]['doppler']
                wavelength = self.wavelengths[system].get(freq)
                if wavelength is None:
                    continue

                # 伪距一阶差分 (m/s)
                pr_derivatives = []
                for i in range(1, len(times)):
                    time_diff = (times[i] - times[i - 1]).total_seconds()
                    if time_diff > 0 and pr_values[i] is not None and pr_values[i - 1] is not None:
                        pr_derivatives.append((pr_values[i] - pr_values[i - 1]) / time_diff)
                    else:
                        pr_derivatives.append(None)

                # 相位一阶差分 (m/s)
                ph_derivatives = []
                for i in range(1, len(times)):
                    time_diff = (times[i] - times[i - 1]).total_seconds()
                    if time_diff > 0 and ph_values[i] is not None and ph_values[i - 1] is not None:
                        phase_rate_cycles = (ph_values[i] - ph_values[i - 1]) / time_diff
                        phase_rate_meters = phase_rate_cycles * wavelength  # 转换为 m/s
                        ph_derivatives.append(phase_rate_meters)
                    else:
                        ph_derivatives.append(None)

                # 多普勒值转换为 m/s
                doppler_meters = []
                for doppler in doppler_values:
                    if doppler is not None:
                        doppler_meters.append(-doppler * wavelength)
                    else:
                        doppler_meters.append(None)

                # 保存结果（移除第一个时间点，因为差分计算少一个数据点）
                freq_derivatives[freq]['pr_derivative'] = pr_derivatives
                freq_derivatives[freq]['ph_derivative'] = ph_derivatives
                freq_derivatives[freq]['doppler'] = doppler_meters
                freq_derivatives[freq]['times'] = times[1:]

            # 保存该卫星的所有频率差分结果
            derivatives[sat_id] = freq_derivatives

            processed_sats += 1
            self.update_progress(int(processed_sats / total_sats * 100))

        # 存储结果
        self.results['observable_derivatives'] = derivatives
        return derivatives

    def detect_phase_stagnation(self, data: Dict, threshold_cycles: float = 0.1, min_consecutive: int = 5) -> Dict:
        """
        检测载波相位停滞
        参数:
            data: RINEX观测数据
            threshold_cycles: 相位变化阈值（周）
            min_consecutive: 最小连续停滞历元数
        返回:
            停滞检测结果字典
        """
        self.start_stage(2, "检测载波相位停滞", 100)

        stagnation_results = {}
        total_sats = len(self.observations_meters)
        processed_sats = 0

        for sat_id, freq_data in self.observations_meters.items():
            freq_stagnation = {}

            for freq, obs_data in freq_data.items():
                times = obs_data['times']
                phase_cycles = obs_data['phase_cycle']  # 相位（周）

                # 初始化停滞检测结果
                stagnant_epochs = []  # 停滞历元索引
                current_streak = 0  # 当前连续停滞计数
                max_streak = 0  # 最大连续停滞数

                for i in range(1, len(phase_cycles)):
                    # 检查当前历元和前一历元的相位变化
                    if (phase_cycles[i] is not None and
                            phase_cycles[i - 1] is not None and
                            abs(phase_cycles[i] - phase_cycles[i - 1]) < threshold_cycles):
                        current_streak += 1
                    else:
                        # 相位变化超过阈值，重置计数
                        current_streak = 0

                    # 更新最大连续停滞数
                    if current_streak > max_streak:
                        max_streak = current_streak

                    # 记录停滞历元（当连续停滞数达到最小阈值时）
                    if current_streak >= min_consecutive:
                        stagnant_epochs.append(i)

                # 存储该频率的停滞结果
                freq_stagnation[freq] = {
                    'is_stagnant': max_streak >= min_consecutive,
                    'max_stagnant_epochs': max_streak,
                    'stagnant_epochs': stagnant_epochs,
                    'threshold': threshold_cycles,
                    'min_consecutive': min_consecutive
                }

            # 存储该卫星的所有频率停滞结果
            stagnation_results[sat_id] = freq_stagnation

            processed_sats += 1
            self.update_progress(int(processed_sats / total_sats * 100))

        # 存储结果
        self.results['phase_stagnation'] = stagnation_results
        return stagnation_results

    def calculate_code_phase_differences(self, data: Dict) -> Dict:
        """计算每个卫星每个频率的伪距与相位观测值的差值及差值变化率"""
        self.start_stage(3, "计算伪距相位差值", 100)

        # 先执行相位停滞检测
        if not self.results.get('phase_stagnation'):
            self.detect_phase_stagnation(data)

        differences = {}
        total_sats = len(self.observations_meters)
        processed_sats = 0

        # 遍历所有卫星
        for sat_id, freq_data in self.observations_meters.items():
            freq_differences = {}

            # 获取该卫星的相位停滞结果
            sat_stagnation = self.results['phase_stagnation'].get(sat_id, {})

            # 为每个频率计算差值
            for freq, obs_data in freq_data.items():
                times = obs_data['times']
                code_values = obs_data['code']  # 米
                phase_values = obs_data['phase']  # 米

                # 获取该频率的停滞历元索引
                stagnant_epochs = sat_stagnation.get(freq, {}).get('stagnant_epochs', [])

                # 初始化结果存储
                freq_differences[freq] = {
                    'times': [],
                    'code_phase_diff': [],  # 伪距相位原始差值
                    'diff_changes': [],  # 差值的历元间变化
                    'original_epochs': len(times),
                    'filtered_epochs': 0,
                    'stagnant_epochs_removed': len(stagnant_epochs),
                    'missing_epochs': 0  # 新增：缺失观测值的历元数
                }

                prev_diff = None  # 上一历元的差值，用于计算变化
                missing_obs = 0  # 记录缺失观测值的历元数

                # 计算差值，跳过停滞历元
                for i in range(len(times)):
                    # 跳过停滞历元或观测值缺失的历元
                    if (i in stagnant_epochs or
                            code_values[i] is None or
                            phase_values[i] is None):
                        if code_values[i] is None or phase_values[i] is None:
                            missing_obs += 1
                        continue

                    diff = code_values[i] - phase_values[i]  # 直接计算（单位：米）
                    freq_differences[freq]['times'].append(times[i])
                    freq_differences[freq]['code_phase_diff'].append(diff)

                    # 计算差值的变化（仅当前后两个历元都有有效值时）
                    if prev_diff is not None:
                        diff_change = abs(diff - prev_diff)
                        freq_differences[freq]['diff_changes'].append(diff_change)
                    else:
                        freq_differences[freq]['diff_changes'].append(None)  # 第一个历元没有变化

                    prev_diff = diff

                # 更新统计信息
                freq_differences[freq]['filtered_epochs'] = len(freq_differences[freq]['code_phase_diff'])
                freq_differences[freq]['missing_epochs'] = missing_obs

            # 保存该卫星的所有频率差值结果
            differences[sat_id] = freq_differences

            processed_sats += 1
            self.update_progress(int(processed_sats / total_sats * 100))

        # 存储结果
        self.results['code_phase_differences'] = differences
        return differences

    def calculate_receiver_cmc(self) -> Dict:
        """基于 self.receiver_observations 计算接收机CMC(Code-Phase)，单位米。

        仅计算频点：G(L1C,L5Q)、R(L1C)、E(L1C,L5Q,L7Q)、C(L2I)。
        返回结构：{sat_id: {freq: {'times': [...], 'cmc_m': [...]}}}
        """
        self.start_stage(3, "计算接收机CMC", 100)

        target_freqs = {
            'G': ['L1C', 'L5Q'],
            'R': ['L1C'],
            'E': ['L1C', 'L5Q', 'L7Q'],
            'C': ['L2I']
        }

        cmc_results: Dict = {}
        source = self.receiver_observations
        total_sats = len(source)
        processed_sats = 0

        for sat_id, freq_data in source.items():
            system = sat_id[0]
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
            processed_sats += 1
            self.update_progress(int(processed_sats / max(total_sats, 1) * 100))

        self.results['receiver_cmc'] = cmc_results
        return cmc_results


    def calculate_dcmc(self, receiver_rinex_path: str) -> Dict:
        """计算接收机CMC和手机CMC的站间单差(dCMC)
        
        公式: dCMC[sat, freq, epoch] = CMC_receiver[sat, freq, epoch] - CMC_phone[sat, freq, epoch]
        使用现有的伪距相位原始差值作为手机CMC
        
        新增异常值处理：
        1. 过滤dCMC绝对值大于1m的异常值
        2. 检查手机CMC是否存在明显线性漂移，只有存在线性变化才进行后续计算
        
        参数:
            receiver_rinex_path: 接收机RINEX文件路径
            
        返回结构：{sat_id: {freq: {'times': [...], 'dcmc': [...]}}}
        """
        self.start_stage(5, "计算站间单差CMC", 100)
        
        # 确保已计算接收机CMC
        if not self.results.get('receiver_cmc'):
            self.read_receiver_rinex_obs(receiver_rinex_path)
            self.calculate_receiver_cmc()
            
        # 确保已计算手机伪距相位原始差值（作为手机CMC）
        if not self.results.get('code_phase_differences'):
            # 需要传入data参数，这里使用observations_meters
            data = {'observations_meters': self.observations_meters}
            self.calculate_code_phase_differences(data)
            
        receiver_cmc = self.results['receiver_cmc']
        phone_cmc = self.results['code_phase_differences']  # 使用伪距相位原始差值
        
        dcmc_results = {}
        total_combinations = 0
        processed_combinations = 0
        filtered_combinations = 0  # 被过滤的组合数
        outlier_count = 0  # 异常值数量
        
        # 存储线性漂移检测的详细信息
        linear_drift_detailed = {}
        
        # 统计总的卫星-频率组合数
        for sat_id in receiver_cmc.keys():
            if sat_id in phone_cmc:
                total_combinations += len(set(receiver_cmc[sat_id].keys()) & set(phone_cmc[sat_id].keys()))
        
        # 如果启用手机独有卫星分析，统计手机独有卫星
        phone_only_combinations = 0
        if self.enable_phone_only_analysis:
            for sat_id in phone_cmc.keys():
                if sat_id not in receiver_cmc:
                    phone_only_combinations += len(phone_cmc[sat_id].keys())
            total_combinations += phone_only_combinations
        
        for sat_id in receiver_cmc.keys():
            if sat_id not in phone_cmc:
                continue
                
            receiver_freqs = receiver_cmc[sat_id]
            phone_freqs = phone_cmc[sat_id]
            
            # 找到共同的频率
            common_freqs = set(receiver_freqs.keys()) & set(phone_freqs.keys())
            
            if not common_freqs:
                continue
                
            sat_dcmc = {}
            
            for freq in common_freqs:
                receiver_data = receiver_freqs[freq]
                phone_data = phone_freqs[freq]
                
                # 找到共同的时间历元（使用容差匹配）
                receiver_times = receiver_data['times']
                phone_times = phone_data['times']
                
                # 使用容差匹配找到共同时间点
                common_times = []
                receiver_time_idx = {}
                phone_time_idx = {}
                
                for i, rec_time in enumerate(receiver_times):
                    for j, phone_time in enumerate(phone_times):
                        # 使用0.1秒容差进行时间匹配
                        if abs((rec_time - phone_time).total_seconds()) < 0.1:
                            if rec_time not in common_times:
                                common_times.append(rec_time)
                                receiver_time_idx[rec_time] = i
                                phone_time_idx[rec_time] = j
                            break
                
                common_times = sorted(common_times)
                
                if not common_times:
                    continue
                
                # 检查手机CMC是否存在明显线性漂移
                phone_cmc_values = []
                phone_times_list = []
                for t in common_times:
                    phone_idx = phone_time_idx.get(t, -1)
                    if phone_idx >= 0 and phone_idx < len(phone_data['code_phase_diff']):
                        phone_cmc_val = phone_data['code_phase_diff'][phone_idx]
                        if phone_cmc_val is not None:
                            phone_cmc_values.append(phone_cmc_val)
                            phone_times_list.append(t)
                
                # 检查手机CMC线性漂移
                if len(phone_cmc_values) < 10:  # 至少需要10个点才能判断线性趋势
                    filtered_combinations += 1
                    continue
                
                # 计算手机CMC的线性趋势
                phone_cmc_linear_trend = self._check_linear_trend(phone_times_list, phone_cmc_values, self.r_squared_threshold)
                
                # 存储线性漂移检测的详细信息
                sat_freq_key = f"{sat_id}_{freq}"
                linear_drift_detailed[sat_freq_key] = {
                    'status': '有线性漂移' if phone_cmc_linear_trend['has_linear_drift'] else '无线性漂移',
                    'r_squared': phone_cmc_linear_trend['r_squared'],
                    'slope': phone_cmc_linear_trend['slope'],
                    'intercept': phone_cmc_linear_trend['intercept'],
                    'data_points': len(phone_cmc_values),
                    'min_r_squared': self.r_squared_threshold,  # 当前阈值
                    'min_slope_magnitude': 1e-6  # 当前阈值
                }
                
                # 检查是否有明显线性漂移
                if not phone_cmc_linear_trend['has_linear_drift']:
                    filtered_combinations += 1
                    continue
                
                dcmc_vals = []
                dcmc_times = []
                
                for t in common_times:
                    rec_idx = receiver_time_idx[t]
                    phone_idx = phone_time_idx[t]
                    
                    rec_cmc = receiver_data['cmc_m'][rec_idx]
                    phone_cmc_val = phone_data['code_phase_diff'][phone_idx]  # 使用code_phase_diff字段作为手机CMC
                    
                    # dCMC = CMC_receiver - CMC_phone
                    dcmc_val = rec_cmc - phone_cmc_val
                    
                    # 不再进行dCMC异常值过滤
                    # 大周跳的卫星-频率组合已在前面通过线性漂移检查被过滤
                    
                    dcmc_vals.append(dcmc_val)
                    dcmc_times.append(t)
                
                if dcmc_vals:
                    sat_dcmc[freq] = {'times': dcmc_times, 'dcmc': dcmc_vals}
                else:
                    filtered_combinations += 1
                    
                processed_combinations += 1
                self.update_progress(int(processed_combinations / max(total_combinations, 1) * 100))
                
            if sat_dcmc:
                dcmc_results[sat_id] = sat_dcmc
        
        # 处理手机独有卫星（如果启用）
        phone_only_processed = 0
        if self.enable_phone_only_analysis:
            print(f"开始处理手机独有卫星分析...")
            print(f"检测到线性漂移的手机独有卫星:")
            for sat_id in phone_cmc.keys():
                if sat_id not in receiver_cmc:
                    phone_freqs = phone_cmc[sat_id]
                    for freq, phone_data in phone_freqs.items():
                        self._handle_phone_only_satellite(sat_id, freq, phone_data)
                        phone_only_processed += 1
                        self.update_progress(int((processed_combinations + phone_only_processed) / max(total_combinations, 1) * 100))
        
        print(f"dCMC计算完成:")
        print(f"  总组合数: {total_combinations}")
        print(f"  共视卫星通过线性漂移检查: {processed_combinations}")
        if self.enable_phone_only_analysis:
            print(f"  手机独有卫星处理数: {phone_only_processed}")
        print(f"  被过滤组合数: {filtered_combinations}")
        
        # 存储线性漂移检测详细信息
        self.results['linear_drift_detailed'] = linear_drift_detailed
        
        self.results['dcmc'] = dcmc_results
        return dcmc_results

    def extract_cci_series(self, receiver_rinex_path: str = None) -> Dict:
        """从dCMC时间序列中提取码相不一致性误差(CCI)
        
        子步骤 2.1：去除常数偏差（Detrending）
        对于每个卫星每个频率的一个连续弧段 arc：
        arc_mean = mean(dCMC_arc)
        CCI_series[sat, freq, epoch] = dCMC_arc[epoch] - arc_mean
        
        参数:
            receiver_rinex_path: 接收机RINEX文件路径（可选，如果为None则使用已计算的结果）
            
        返回结构：{sat_id: {freq: {'times': [...], 'cci_series': [...], 'arc_info': {...}}}}
        """
        self.start_stage(6, "提取码相不一致性时间序列", 100)
        
        # 确保已计算dCMC（避免重复计算）
        dcmc_data = self.results.get('dcmc')
        if not dcmc_data or len(dcmc_data) == 0:
            if receiver_rinex_path is None:
                raise ValueError("dCMC数据不存在且未提供接收机RINEX文件路径")
            print("警告: dCMC数据不存在，正在计算...")
            self.calculate_dcmc(receiver_rinex_path)
            dcmc_data = self.results.get('dcmc')
        cci_results = {}
        total_sats = len(dcmc_data)
        processed_sats = 0
        
        print(f"  正在处理 {len(dcmc_data)} 个卫星的CCI序列提取...")
        
        for sat_id, freq_data in dcmc_data.items():
            sat_cci = {}
            
            for freq, dcmc_info in freq_data.items():
                times = dcmc_info['times']
                dcmc_values = dcmc_info['dcmc']
                
                if not dcmc_values:
                    continue
                    
                # 识别连续弧段（基于时间间隔）
                sat_freq_info = f"{sat_id}_{freq}"
                arcs = self._identify_continuous_arcs(times, dcmc_values, sat_freq_info=sat_freq_info)
                
                all_cci_vals = []
                all_times = []
                arc_info = []
                
                for arc_idx, arc in enumerate(arcs):
                    arc_times = arc['times']
                    arc_dcmc = arc['values']
                    
                    if len(arc_dcmc) < 2:  # 至少需要2个点才能去均值
                        continue
                        
                    # 计算弧段平均值
                    arc_mean = sum(arc_dcmc) / len(arc_dcmc)
                    
                    # 去除常数偏差：CCI = dCMC - arc_mean
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
                
            processed_sats += 1
            self.update_progress(int(processed_sats / max(total_sats, 1) * 100))
            
        self.results['cci_series'] = cci_results
        return cci_results


    def calculate_roc_model(self, receiver_rinex_path: str = None) -> Dict:
        """计算各卫星系统各频率的ROC模型参数
        
        子步骤 2.2：线性拟合计算ROC（Rate of Change）
        对CCI_series[sat, freq]进行线性拟合，得到斜率ROC_sat_freq
        
        子步骤 2.3：确定各卫星系统各频率的平均ROC模型
        ROC_model[system_freq] = mean(该卫星系统该频率的所有卫星的 ROC_sat_freq)
        
        参数:
            receiver_rinex_path: 接收机RINEX文件路径（可选，如果为None则使用已计算的结果）
            
        返回结构：{system_freq: {'roc_rate': float, 'contributing_sats': [...], 'individual_rocs': {...}}}
        """
        self.start_stage(7, "计算ROC模型参数", 100)
        
        # 确保已提取CCI序列（避免重复计算）
        if not self.results.get('cci_series'):
            if receiver_rinex_path is None:
                raise ValueError("CCI序列数据不存在且未提供接收机RINEX文件路径")
            print("警告: CCI序列数据不存在，正在计算...")
            self.extract_cci_series(receiver_rinex_path)
            
        cci_data = self.results['cci_series']
        roc_results = {}
        
        # 按卫星系统和频率分组收集ROC值
        system_freq_rocs = {}  # {system_freq: [roc_values]}
        system_freq_contributing_sats = {}  # {system_freq: [sat_ids]}
        individual_rocs = {}  # {system_freq: {sat_id: roc_value}}
        
        total_combinations = 0
        processed_combinations = 0
        
        # 统计总的卫星-频率组合数
        for sat_id, freq_data in cci_data.items():
            total_combinations += len(freq_data)
        
        for sat_id, freq_data in cci_data.items():
            # 获取卫星系统（卫星ID的第一个字符）
            sat_system = sat_id[0] if sat_id else 'Unknown'
            
            for freq, cci_info in freq_data.items():
                times = cci_info['times']
                cci_vals = cci_info['cci_series']
                
                if len(cci_vals) < 2:
                    continue
                    
                # 将时间转换为秒数（从第一个时间点开始）
                time_zero = times[0]
                time_seconds = [(t - time_zero).total_seconds() for t in times]
                
                # 线性拟合：CCI = slope * time + intercept
                try:
                    slope, intercept = self._linear_fit(time_seconds, cci_vals)
                    roc_value = slope  # 单位: m/s
                    
                    
                    # 创建卫星系统+频率的组合键
                    system_freq_key = f"{sat_system}_{freq}"
                    
                    # 收集到对应系统-频率组
                    if system_freq_key not in system_freq_rocs:
                        system_freq_rocs[system_freq_key] = []
                        system_freq_contributing_sats[system_freq_key] = []
                        individual_rocs[system_freq_key] = {}
                        
                    system_freq_rocs[system_freq_key].append(roc_value)
                    system_freq_contributing_sats[system_freq_key].append(sat_id)
                    individual_rocs[system_freq_key][sat_id] = roc_value
                    
                except Exception as e:
                    print(f"警告: 卫星 {sat_id} 频率 {freq} 线性拟合失败: {e}")
                    continue
                    
                processed_combinations += 1
                self.update_progress(int(processed_combinations / max(total_combinations, 1) * 100))
        
        # 计算各系统-频率的平均ROC和个体ROC
        individual_roc_models = {}  # 存储个体级ROC模型
        
        for system_freq_key, roc_values in system_freq_rocs.items():
            if roc_values:
                mean_roc = sum(roc_values) / len(roc_values)
                std_roc = (sum((r - mean_roc)**2 for r in roc_values) / len(roc_values))**0.5
                
                # 计算变异系数（CV）
                if mean_roc != 0:
                    roc_cv = abs(std_roc / mean_roc)
                else:
                    roc_cv = float('inf') if std_roc > 0 else 0.0
                
                # 质量等级判断（只针对系统级模型）
                if roc_cv < self.cv_threshold and len(roc_values) >= 3:
                    quality_level = "高质量"
                    is_high_quality = True
                elif self.cv_threshold <= roc_cv < 1.0 and len(roc_values) >= 3:
                    quality_level = "中等质量"
                    is_high_quality = False
                else:
                    # 这种情况会建立个体级模型，不需要质量等级
                    quality_level = "个体级"
                    is_high_quality = True
                
                # 根据CV值决定使用系统级还是个体级ROC模型
                if roc_cv < self.cv_threshold and len(roc_values) >= 3:
                    # CV<0.5且数据点≥3：使用系统级ROC模型
                    roc_results[system_freq_key] = {
                        'roc_rate': mean_roc,  # 单位: m/s
                        'roc_std': std_roc,
                        'roc_cv': roc_cv,  # 变异系数
                        'quality_level': quality_level,  # 质量等级
                        'is_high_quality': is_high_quality,  # 是否高质量
                        'contributing_sats': system_freq_contributing_sats[system_freq_key],
                        'individual_rocs': individual_rocs[system_freq_key],
                        'num_satellites': len(roc_values),
                        'model_type': 'system_level'  # 系统级模型
                    }
                else:
                    # CV≥阈值或数据点<3：为每个卫星建立个体ROC模型
                    if roc_cv >= self.cv_threshold:
                        print(f"系统-频率 {system_freq_key} CV值过高({roc_cv:.3f}≥{self.cv_threshold:.1f})，建立个体ROC模型")
                    else:
                        print(f"系统-频率 {system_freq_key} 数据点不足({len(roc_values)}<3)，建立个体ROC模型")
                    
                    # 为个体级模型计算CV值（如果卫星数>=3）
                    individual_cv = 0.0
                    if len(roc_values) >= 3:
                        individual_cv = roc_cv  # 使用已计算的CV值
                    
                    for sat_id, individual_roc in individual_rocs[system_freq_key].items():
                        individual_key = f"{sat_id}_{system_freq_key.split('_')[1]}"  # 如: E04_L7Q
                        individual_roc_models[individual_key] = {
                            'roc_rate': individual_roc,  # 单位: m/s
                            'roc_std': std_roc if len(roc_values) >= 3 else 0.0,  # 如果卫星数>=3，保存标准差
                            'roc_cv': individual_cv,  # 如果卫星数>=3，保存CV值
                            'quality_level': "个体级",  # 个体级质量
                            'is_high_quality': True,  # 个体级默认为高质量
                            'contributing_sats': [sat_id],
                            'individual_rocs': {sat_id: individual_roc},
                            'num_satellites': 1,
                            'model_type': 'individual_level',  # 个体级模型
                            'system_freq_cv': individual_cv,  # 系统-频率组合的CV值
                            'system_freq_satellites': len(roc_values)  # 系统-频率组合的卫星数
                        }
        
        # 处理手机独有卫星的ROC模型（如果启用）
        phone_only_roc_models = {}
        if self.enable_phone_only_analysis and 'phone_only_linear_drift' in self.results:
            print(f"开始建立手机独有卫星ROC模型...")
            phone_only_data = self.results['phone_only_linear_drift']
            
            for sat_freq_key, drift_info in phone_only_data.items():
                if drift_info['status'] == '有线性漂移':
                    # 为手机独有卫星建立个体ROC模型
                    phone_only_roc_models[sat_freq_key] = {
                        'roc_rate': drift_info['slope'],  # 使用斜率作为ROC
                        'roc_std': 0.0,  # 个体模型无标准差
                        'roc_cv': 0.0,   # 个体模型无变异系数
                        'quality_level': "个体级",  # 个体级质量
                        'is_high_quality': True,  # 个体级默认为高质量
                        'contributing_sats': [sat_freq_key.split('_')[0]],  # 单个卫星
                        'individual_rocs': {sat_freq_key.split('_')[0]: drift_info['slope']},
                        'num_satellites': 1,
                        'model_type': 'individual_level',  # 个体级模型
                        'data_source': 'phone_only'  # 标记数据来源
                    }
            
            print(f"手机独有卫星ROC模型建立完成: {len(phone_only_roc_models)} 个")
        
        # 合并系统级、个体级和手机独有卫星ROC模型
        all_roc_models = {**roc_results, **individual_roc_models, **phone_only_roc_models}
        self.results['roc_model'] = all_roc_models
        self.results['individual_roc_models'] = individual_roc_models
        self.results['phone_only_roc_models'] = phone_only_roc_models
        
        print(f"ROC模型计算完成: 系统级模型 {len(roc_results)} 个, 个体级模型 {len(individual_roc_models)} 个, 手机独有卫星模型 {len(phone_only_roc_models)} 个")
        return all_roc_models

    def _linear_fit(self, x_values, y_values):
        """简单线性拟合，返回斜率和截距"""
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

    def _handle_phone_only_satellite(self, sat_id: str, freq: str, phone_data: Dict):
        """处理手机独有卫星的码相不一致性检测
        
        参数:
            sat_id: 卫星ID
            freq: 频率
            phone_data: 手机观测数据
        """
        # 检查手机CMC是否存在明显线性漂移
        phone_times = phone_data['times']
        phone_cmc_values = phone_data['code_phase_diff']
        
        # 过滤None值
        valid_indices = [i for i, val in enumerate(phone_cmc_values) if val is not None]
        if len(valid_indices) < self.phone_only_min_data_points:
            return
            
        # 提取有效数据
        valid_times = [phone_times[i] for i in valid_indices]
        valid_cmc_values = [phone_cmc_values[i] for i in valid_indices]
        
        # 进行线性漂移检测
        try:
            phone_cmc_linear_trend = self._check_linear_trend(
                valid_times, valid_cmc_values, self.r_squared_threshold
            )
            
            # 存储手机独有卫星的线性漂移信息
            sat_freq_key = f"{sat_id}_{freq}"
            if 'phone_only_linear_drift' not in self.results:
                self.results['phone_only_linear_drift'] = {}
                
            self.results['phone_only_linear_drift'][sat_freq_key] = {
                'status': '有线性漂移' if phone_cmc_linear_trend['has_linear_drift'] else '无线性漂移',
                'r_squared': phone_cmc_linear_trend['r_squared'],
                'slope': phone_cmc_linear_trend['slope'],
                'intercept': phone_cmc_linear_trend['intercept'],
                'data_points': len(valid_cmc_values),
                'min_r_squared': self.r_squared_threshold,
                'min_slope_magnitude': 1e-6,
                'data_source': 'phone_only'  # 标记数据来源
            }
            
            if phone_cmc_linear_trend['has_linear_drift']:
                print(f"  {sat_freq_key}: R²={phone_cmc_linear_trend['r_squared']:.6f}, "
                      f"斜率={phone_cmc_linear_trend['slope']:.6e} m/s, "
                      f"数据点={len(valid_cmc_values)}")
                      
        except Exception as e:
            print(f"手机独有卫星 {sat_id}_{freq} 线性漂移检测失败: {e}")

    def _identify_continuous_arcs(self, times, values, max_gap_seconds=300, sat_freq_info=""):
        """识别连续弧段，基于时间间隔分割
        
        参数:
            times: 时间列表
            values: 对应的值列表
            max_gap_seconds: 最大允许时间间隔（秒），默认5分钟
            sat_freq_info: 卫星频率信息，用于调试输出
            
        返回:
            list: 连续弧段列表，每个弧段包含{'times': [...], 'values': [...], 'has_cycle_slip': False}
        """
        if not times or not values:
            return []
        
        arcs = []
        current_arc_times = []
        current_arc_values = []
        
        for i, (time, value) in enumerate(zip(times, values)):
            if value is None:
                # 遇到无效值，跳过但不分割弧段
                continue
            
            if not current_arc_times:
                # 开始新弧段
                current_arc_times.append(time)
                current_arc_values.append(value)
            else:
                # 检查时间间隔
                time_diff = (time - current_arc_times[-1]).total_seconds()
                
                # 弧段分割策略：仅基于时间间隔
                if time_diff > max_gap_seconds:
                    # 时间间隔过大（>5分钟），分割弧段
                    if sat_freq_info:  # 只在有卫星频率信息时输出
                        print(f"  弧段分割: {sat_freq_info} 时间间隔 {time_diff:.1f}s > {max_gap_seconds}s")
                    arcs.append({
                        'times': current_arc_times,
                        'values': current_arc_values,
                        'has_cycle_slip': False
                    })
                    current_arc_times = [time]
                    current_arc_values = [value]
                else:
                    # 时间间隔<5分钟，继续当前弧段
                    current_arc_times.append(time)
                    current_arc_values.append(value)
        
        # 添加最后一个弧段
        if current_arc_times:
            arcs.append({
                'times': current_arc_times,
                'values': current_arc_values,
                'has_cycle_slip': False
            })
        
        return arcs

    def _check_linear_trend(self, times, values, min_r_squared=0.5, min_slope_magnitude=1e-6):
        """检查时间序列是否存在明显线性漂移
        
        参数:
            times: 时间列表
            values: 对应的值列表
            min_r_squared: 最小R²阈值，默认0.5
            min_slope_magnitude: 最小斜率绝对值阈值，默认1e-6 m/s
            
        返回:
            dict: {'has_linear_drift': bool, 'slope': float, 'r_squared': float, 'intercept': float}
        """
        if len(times) < 3:
            return {'has_linear_drift': False, 'slope': 0.0, 'r_squared': 0.0, 'intercept': 0.0}
        
        try:
            # 将时间转换为秒数（从第一个时间点开始）
            time_zero = times[0]
            time_seconds = [(t - time_zero).total_seconds() for t in times]
            
            # 线性拟合
            slope, intercept = self._linear_fit(time_seconds, values)
            
            # 计算R²
            y_mean = sum(values) / len(values)
            ss_tot = sum((y - y_mean) ** 2 for y in values)  # 总平方和
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(time_seconds, values))  # 残差平方和
            
            if ss_tot == 0:
                r_squared = 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # 判断是否存在明显线性漂移
            has_linear_drift = (r_squared >= min_r_squared and abs(slope) >= min_slope_magnitude)
            
            return {
                'has_linear_drift': has_linear_drift,
                'slope': slope,
                'r_squared': r_squared,
                'intercept': intercept
            }
            
        except Exception as e:
            print(f"线性趋势检查失败: {e}")
            return {'has_linear_drift': False, 'slope': 0.0, 'r_squared': 0.0, 'intercept': 0.0}

    def correct_phase_observations(self, receiver_rinex_path: str = None) -> Dict:
        """使用ROC模型校正载波相位观测值
        
        第三步：校正载波相位观测值
        只对之前dCMC计算中通过线性漂移检查的卫星-频率组合进行校正
        
        对于每个历元、每个卫星、每个频率的原始载波相位观测值 L_phone_raw[sat, freq, t]：
        L_phone_corrected[sat, freq, t] = L_phone_raw[sat, freq, t] + ROC_model[freq] * (t - t0)
        
        参数:
            receiver_rinex_path: 接收机RINEX文件路径（可选，如果为None则使用已计算的结果）
            
        返回结构：{sat_id: {freq: {'times': [...], 'original_phase': [...], 'corrected_phase': [...], 'correction_applied': [...]}}}
        """
        self.start_stage(8, "校正载波相位观测值", 100)
        
        # 确保已计算ROC模型（避免重复计算）
        if not self.results.get('roc_model'):
            if receiver_rinex_path is None:
                raise ValueError("ROC模型数据不存在且未提供接收机RINEX文件路径")
            print("警告: ROC模型数据不存在，正在计算...")
            self.calculate_roc_model(receiver_rinex_path)
            
        roc_model = self.results['roc_model']
        corrected_results = {}
        
        # 获取有效的dCMC数据，这些是之前通过线性漂移检查的卫星-频率组合
        dcmc_data = self.results.get('dcmc', {})
        valid_combinations = set()
        
        # 从dCMC数据中提取有效的卫星-频率组合
        for sat_id, freq_data in dcmc_data.items():
            for freq in freq_data.keys():
                valid_combinations.add((sat_id, freq))
        
        print(f"对 {len(valid_combinations)} 个通过线性漂移检查的卫星-频率组合进行校正")
        
        # 显示ROC模型总结信息
        print(f"ROC模型包含 {len(roc_model)} 个系统-频率组合")
        # 只显示前5个ROC模型作为示例
        for i, (system_freq_key, roc_info) in enumerate(roc_model.items()):
            if i < 5:
                print(f"  {system_freq_key}: ROC = {roc_info['roc_rate']:.6e} m/s, 参与卫星数 = {roc_info['num_satellites']}")
            elif i == 5:
                print(f"  ... 还有 {len(roc_model) - 5} 个ROC模型")
                break
        
        processed_combinations = 0
        total_combinations = len(valid_combinations)
        
        for sat_id, freq_data in self.observations_meters.items():
            sat_corrected = {}
            
            for freq, obs_data in freq_data.items():
                # 只处理通过线性漂移检查的卫星-频率组合
                if (sat_id, freq) not in valid_combinations:
                    continue
                    
                times = obs_data.get('times', [])
                original_phase_m = obs_data.get('phase', [])  # 以米为单位
                original_phase_cycle = obs_data.get('phase_cycle', [])  # 以周为单位
                wavelengths = obs_data.get('wavelength', [])
                
                if not times or not original_phase_m:
                    continue
                
                    
                # 检查该频率是否有ROC模型（优先系统级，其次个体级）
                sat_system = sat_id[0] if sat_id else 'Unknown'
                system_freq_key = f"{sat_system}_{freq}"
                individual_key = f"{sat_id}_{freq}"
                
                roc_info = None
                model_type = None
                
                # 优先使用系统级ROC模型
                if system_freq_key in roc_model:
                    roc_info = roc_model[system_freq_key]
                    model_type = "系统级"
                # 其次使用个体级ROC模型
                elif individual_key in roc_model:
                    roc_info = roc_model[individual_key]
                    model_type = "个体级"
                else:
                    continue
                    
                roc_rate = roc_info['roc_rate']  # m/s
                quality_level = roc_info['quality_level']
                is_high_quality = roc_info['is_high_quality']
                roc_cv = roc_info['roc_cv']
                
                # 根据质量等级决定是否进行校正
                if quality_level == "低质量":
                    continue
                
                # 识别连续弧段（用于确定t0）
                sat_freq_info = f"{sat_id}_{freq}"
                arcs = self._identify_continuous_arcs(times, original_phase_m, sat_freq_info="")  # 不输出弧段分割信息
                
                corrected_phase_m = []
                correction_applied = []
                
                # 存储校正的详细信息
                medium_quality_details = []
                high_quality_details = []
                individual_level_details = []
                
                # 存储弧段信息用于报告
                arc_info = []
                
                for arc_idx, arc in enumerate(arcs):
                    arc_times = arc['times']
                    arc_phase = arc['values']
                    
                    if not arc_times:
                        continue
                        
                    # 记录弧段信息
                    arc_duration = (arc_times[-1] - arc_times[0]).total_seconds()
                    has_cycle_slip = arc.get('has_cycle_slip', False)
                    
                    # t0时间基准选择
                    if has_cycle_slip:
                        # 有周跳的弧段：以周跳历元为新的t0
                        t0 = arc_times[0]
                    else:
                        # 无周跳的弧段：以全局起始时间为t0（只有第一个弧段）
                        if arc_idx == 0:
                            # 第一个弧段：使用全局起始时间
                            t0 = times[0] if times else arc_times[0]
                        else:
                            # 后续弧段：使用当前弧段的起始时间
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
                            
                        # 计算时间差（秒）
                        time_diff_seconds = (t - t0).total_seconds()
                        
                        # 基础校正：L_corrected = L_original + ROC * (t - t0)
                        base_correction = -roc_rate * time_diff_seconds  # 米
                        
                        # 根据质量等级和弧段特征应用不同的校正策略
                        if quality_level == "高质量":
                            if has_cycle_slip:
                                # 有周跳的弧段：应用保守校正（减少50%）
                                correction = base_correction * 0.5
                            else:
                                # 无周跳的弧段：直接应用校正
                                correction = base_correction
                            
                            # 记录高质量校正详情
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
                            # 中等质量：应用加权校正和时间衰减
                            weight = max(0.3, 1.0 - roc_cv)  # 基于CV的权重
                            time_weight = max(0.5, 1.0 - abs(time_diff_seconds) / 3600.0)  # 时间衰减
                            
                            # 有周跳的弧段：进一步减少校正量
                            if has_cycle_slip:
                                weight *= 0.3  # 额外减少70%
                                max_correction = 0.02  # 最大校正量限制（2cm）
                            else:
                                max_correction = 0.05  # 最大校正量限制（5cm）
                            
                            correction = base_correction * weight * time_weight
                            correction = max(-max_correction, min(max_correction, correction))
                            
                            # 记录中等质量校正详情
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
                            # 个体级：直接应用校正（与高质量相同策略）
                            if has_cycle_slip:
                                # 有周跳的弧段：应用保守校正（减少50%）
                                correction = base_correction * 0.5
                            else:
                                # 无周跳的弧段：直接应用校正
                                correction = base_correction
                            
                            # 记录个体级校正详情
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
                
                # 存储校正详情
                if medium_quality_details:
                    if 'medium_quality_correction_details' not in self.results:
                        self.results['medium_quality_correction_details'] = {}
                    self.results['medium_quality_correction_details'][f"{sat_id}_{freq}"] = {
                        'details': medium_quality_details,
                        'arcs': arc_info
                    }
                
                if high_quality_details:
                    if 'high_quality_correction_details' not in self.results:
                        self.results['high_quality_correction_details'] = {}
                    self.results['high_quality_correction_details'][f"{sat_id}_{freq}"] = {
                        'details': high_quality_details,
                        'arcs': arc_info
                    }
                
                if individual_level_details:
                    if 'individual_level_correction_details' not in self.results:
                        self.results['individual_level_correction_details'] = {}
                    self.results['individual_level_correction_details'][f"{sat_id}_{freq}"] = {
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
                
                processed_combinations += 1
                self.update_progress(int(processed_combinations / max(total_combinations, 1) * 100))
            
            if sat_corrected:
                corrected_results[sat_id] = sat_corrected
        
        # 显示校正结果统计
        total_corrected_sats = len(corrected_results)
        total_corrected_combinations = sum(len(freq_data) for freq_data in corrected_results.values())
        total_corrected_obs = sum(
            len(freq_data['corrected_phase']) 
            for sat_data in corrected_results.values() 
            for freq_data in sat_data.values()
        )
        
        # 处理手机独有卫星的载波相位校正（如果启用）
        phone_only_corrected = 0
        if self.enable_phone_only_analysis and 'phone_only_roc_models' in self.results:
            print(f"\n开始校正手机独有卫星载波相位...")
            phone_only_models = self.results['phone_only_roc_models']
            
            for sat_freq_key, roc_info in phone_only_models.items():
                sat_id, freq = sat_freq_key.split('_', 1)
                
                # 检查手机观测数据中是否有该卫星-频率组合
                if sat_id in self.observations_meters and freq in self.observations_meters[sat_id]:
                    phone_data = self.observations_meters[sat_id][freq]
                    times = phone_data['times']
                    original_phase = phone_data['phase']
                    
                    if not original_phase:
                        continue
                    
                    # 使用手机独有卫星的ROC进行校正
                    roc_rate = roc_info['roc_rate']
                    corrected_phase = []
                    correction_applied = []
                    
                    # 获取波长信息
                    wavelengths = phone_data.get('wavelength', [])
                    if not wavelengths:
                        # 如果没有波长信息，使用默认波长
                        sat_system = sat_id[0]
                        default_wavelengths = self.wavelengths.get(sat_system, {})
                        wavelength = default_wavelengths.get(freq)
                        wavelengths = [wavelength] * len(times) if wavelength else [None] * len(times)
                    
                    # 使用第一个时间作为t0
                    t0 = times[0]
                    
                    for i, (t, phase_val) in enumerate(zip(times, original_phase)):
                        if phase_val is not None:
                            time_diff = (t - t0).total_seconds()
                            correction = roc_rate * time_diff
                            corrected_phase.append(phase_val + correction)
                            correction_applied.append(correction)
                        else:
                            corrected_phase.append(None)
                            correction_applied.append(0.0)
                    
                    # 存储校正结果
                    if sat_id not in corrected_results:
                        corrected_results[sat_id] = {}
                    
                    corrected_results[sat_id][freq] = {
                        'times': times,
                        'original_phase': original_phase,
                        'corrected_phase': corrected_phase,
                        'correction_applied': correction_applied,
                        'wavelengths': wavelengths,
                        'roc_rate': roc_rate,
                        'model_type': 'phone_only',
                        'data_source': 'phone_only'
                    }
                    
                    phone_only_corrected += 1
                    print(f"  校正手机独有卫星 {sat_freq_key}: ROC={roc_rate:.6e} m/s")
            
            print(f"手机独有卫星校正完成: {phone_only_corrected} 个组合")
        
        print(f"\n载波相位校正完成:")
        print(f"  校正卫星数: {total_corrected_sats}")
        print(f"  校正组合数: {total_corrected_combinations}")
        print(f"  校正观测值数: {total_corrected_obs}")
        if self.enable_phone_only_analysis:
            print(f"  手机独有卫星校正组合数: {phone_only_corrected}")
        
        # 显示个位数秒数校正统计
        if hasattr(self, '_debug_single_digit_count'):
            print(f"  个位数秒数校正数: {self._debug_single_digit_count}")
        
        self.results['corrected_phase'] = corrected_results
        return corrected_results

    def _generate_cci_modeling_report(self):
        """生成码相不一致性建模报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("码相不一致性建模报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. 线性漂移检测结果汇总
        if 'linear_drift_detailed' in self.results:
            report_lines.append("1. 线性漂移检测结果")
            report_lines.append("-" * 50)
            linear_drift_data = self.results['linear_drift_detailed']
            
            # 分离有线性漂移和无线性漂移的卫星-频率组合
            has_drift = []
            no_drift = []
            
            for sat_freq, drift_info in linear_drift_data.items():
                if drift_info['status'] == '有线性漂移':
                    has_drift.append((sat_freq, drift_info))
                else:
                    no_drift.append((sat_freq, drift_info))
            
            # 显示阈值信息（只显示一次）
            if linear_drift_data:
                sample_info = list(linear_drift_data.values())[0]
                report_lines.append(f"检测阈值: R²≥{sample_info['min_r_squared']}, |斜率|≥{sample_info['min_slope_magnitude']}")
                report_lines.append("")
            
            # 有线性漂移的卫星-频率组合
            report_lines.append(f"1.1 检测到线性漂移的卫星-频率组合 ({len(has_drift)}个):")
            report_lines.append("-" * 30)
            if has_drift:
                for sat_freq, drift_info in has_drift:
                    report_lines.append(f"  {sat_freq}: R²={drift_info['r_squared']:.6f}, 斜率={drift_info['slope']:.6e} m/s, 数据点={drift_info['data_points']}")
            else:
                report_lines.append("  无")
            report_lines.append("")
            
            # 无线性漂移的卫星-频率组合
            report_lines.append(f"1.2 未检测到线性漂移的卫星-频率组合 ({len(no_drift)}个):")
            report_lines.append("-" * 30)
            if no_drift:
                for sat_freq, drift_info in no_drift:
                    report_lines.append(f"  {sat_freq}: R²={drift_info['r_squared']:.6f}, 斜率={drift_info['slope']:.6e} m/s, 数据点={drift_info['data_points']}")
            else:
                report_lines.append("  无")
            report_lines.append("")
        
        # 1.3 手机独有卫星线性漂移检测结果
        if 'phone_only_linear_drift' in self.results and self.results['phone_only_linear_drift']:
            report_lines.append("1.3 手机独有卫星线性漂移检测结果")
            report_lines.append("-" * 50)
            phone_only_data = self.results['phone_only_linear_drift']
            
            # 分离有线性漂移和无线性漂移的手机独有卫星
            phone_has_drift = []
            phone_no_drift = []
            
            for sat_freq, drift_info in phone_only_data.items():
                if drift_info['status'] == '有线性漂移':
                    phone_has_drift.append((sat_freq, drift_info))
                else:
                    phone_no_drift.append((sat_freq, drift_info))
            
            # 有线性漂移的手机独有卫星
            report_lines.append(f"1.3.1 检测到线性漂移的手机独有卫星 ({len(phone_has_drift)}个):")
            report_lines.append("-" * 30)
            if phone_has_drift:
                for sat_freq, drift_info in phone_has_drift:
                    report_lines.append(f"  {sat_freq}: R²={drift_info['r_squared']:.6f}, 斜率={drift_info['slope']:.6e} m/s, 数据点={drift_info['data_points']}")
            else:
                report_lines.append("  无")
            report_lines.append("")
            
            # 无线性漂移的手机独有卫星
            report_lines.append(f"1.3.2 未检测到线性漂移的手机独有卫星 ({len(phone_no_drift)}个):")
            report_lines.append("-" * 30)
            if phone_no_drift:
                for sat_freq, drift_info in phone_no_drift:
                    report_lines.append(f"  {sat_freq}: R²={drift_info['r_squared']:.6f}, 斜率={drift_info['slope']:.6e} m/s, 数据点={drift_info['data_points']}")
            else:
                report_lines.append("  无")
            report_lines.append("")
        
        # 2. ROC模型质量分析
        if 'roc_model' in self.results:
            report_lines.append("2. ROC模型质量分析")
            report_lines.append("-" * 50)
            roc_model = self.results['roc_model']
            
            # 按模型类型分组
            system_models = []
            individual_models = []
            
            for system_freq, roc_info in roc_model.items():
                model_type = roc_info.get('model_type', 'system_level')
                
                if model_type == 'individual_level':
                    individual_models.append((system_freq, roc_info))
                else:
                    system_models.append((system_freq, roc_info))
            
            # 显示模型选择策略
            report_lines.append(f"模型选择策略: CV<{self.cv_threshold}且数据点≥3→系统级模型, CV≥{self.cv_threshold}或数据点<3→个体级模型")
            report_lines.append("")
            
            # 系统级ROC模型
            report_lines.append(f"2.1 系统级ROC模型 ({len(system_models)}个):")
            report_lines.append("-" * 30)
            if system_models:
                for system_freq, roc_info in system_models:
                    parts = system_freq.split('_', 1)
                    system_name = {'G': 'GPS', 'R': 'GLONASS', 'E': 'Galileo', 'C': 'BDS', 'J': 'QZSS', 'I': 'IRNSS'}.get(parts[0], parts[0])
                    contributing_sats = roc_info['contributing_sats']
                    quality = roc_info['quality_level']
                    report_lines.append(f"  {system_name} {parts[1]}: ROC={roc_info['roc_rate']:.6e} m/s, CV={roc_info['roc_cv']:.3f} ({quality})")
                    report_lines.append(f"    参与卫星 ({len(contributing_sats)}个): {', '.join(contributing_sats)}")
            else:
                report_lines.append("  无")
            report_lines.append("")
            
            # 个体级ROC模型
            report_lines.append(f"2.2 个体级ROC模型 ({len(individual_models)}个):")
            report_lines.append("-" * 30)
            if individual_models:
                # 按卫星系统分组显示
                individual_by_system = {}
                for sat_freq, roc_info in individual_models:
                    sat_id = sat_freq.split('_')[0]
                    freq = sat_freq.split('_')[1]
                    system = sat_id[0]
                    if system not in individual_by_system:
                        individual_by_system[system] = []
                    individual_by_system[system].append((sat_id, freq, roc_info))
                
                for system, sats in individual_by_system.items():
                    system_name = {'G': 'GPS', 'R': 'GLONASS', 'E': 'Galileo', 'C': 'BDS', 'J': 'QZSS', 'I': 'IRNSS'}.get(system, system)
                    report_lines.append(f"  {system_name}系统:")
                    
                    # 按频率分组，显示CV值
                    freq_groups = {}
                    for sat_id, freq, roc_info in sats:
                        if freq not in freq_groups:
                            freq_groups[freq] = []
                        freq_groups[freq].append((sat_id, roc_info))
                    
                    for freq, freq_sats in freq_groups.items():
                        # 检查是否有CV值信息
                        has_cv_info = any('system_freq_cv' in roc_info and roc_info['system_freq_cv'] > 0 for _, roc_info in freq_sats)
                        
                        if has_cv_info:
                            # 获取CV值（所有卫星的CV值应该相同）
                            cv_value = next(roc_info['system_freq_cv'] for _, roc_info in freq_sats if 'system_freq_cv' in roc_info and roc_info['system_freq_cv'] > 0)
                            sat_count = next(roc_info['system_freq_satellites'] for _, roc_info in freq_sats if 'system_freq_satellites' in roc_info)
                            report_lines.append(f"    {freq} (CV={cv_value:.3f}, {sat_count}个卫星):")
                        else:
                            report_lines.append(f"    {freq}:")
                        
                        for sat_id, roc_info in freq_sats:
                            report_lines.append(f"      {sat_id}: ROC={roc_info['roc_rate']:.6e} m/s")
            else:
                report_lines.append("  无")
            report_lines.append("")
            
            # 手机独有卫星ROC模型
            if 'phone_only_roc_models' in self.results and self.results['phone_only_roc_models']:
                report_lines.append(f"2.4 手机独有卫星ROC模型 ({len(self.results['phone_only_roc_models'])}个):")
                report_lines.append("-" * 30)
                phone_only_models = self.results['phone_only_roc_models']
                
                # 按卫星系统分组显示
                phone_only_by_system = {}
                for sat_freq, roc_info in phone_only_models.items():
                    sat_id = sat_freq.split('_')[0]
                    freq = sat_freq.split('_')[1]
                    system = sat_id[0]
                    if system not in phone_only_by_system:
                        phone_only_by_system[system] = []
                    phone_only_by_system[system].append((sat_id, freq, roc_info))
                
                for system, sats in phone_only_by_system.items():
                    system_name = {'G': 'GPS', 'R': 'GLONASS', 'E': 'Galileo', 'C': 'BDS', 'J': 'QZSS', 'I': 'IRNSS'}.get(system, system)
                    report_lines.append(f"  {system_name}系统:")
                    for sat_id, freq, roc_info in sats:
                        report_lines.append(f"    {sat_id} {freq}: ROC={roc_info['roc_rate']:.6e} m/s (手机独有)")
                report_lines.append("")
            
            report_lines.append("")
        
        # 3. 载波相位校正详细信息
        has_high_quality = 'high_quality_correction_details' in self.results
        has_medium_quality = 'medium_quality_correction_details' in self.results
        has_individual_level = 'individual_level_correction_details' in self.results
        
        if has_high_quality or has_medium_quality or has_individual_level:
            report_lines.append("3. 载波相位校正详细信息")
            report_lines.append("-" * 50)
            
            # 分离系统级和个体级校正信息
            system_level_high = {}
            individual_level_high = {}
            system_level_medium = {}
            individual_level_medium = {}
            
            # 处理系统级校正信息
            if has_high_quality:
                system_level_high = self.results['high_quality_correction_details']
            
            if has_medium_quality:
                system_level_medium = self.results['medium_quality_correction_details']
            
            # 处理个体级校正详情
            if has_individual_level:
                individual_level_high = self.results['individual_level_correction_details']
            
            # 3.1 系统级校正信息
            if system_level_high or system_level_medium:
                report_lines.append("3.1 系统级校正信息")
                report_lines.append("-" * 30)
                
                # 合并所有系统级校正信息
                all_system = {**system_level_high, **system_level_medium}
                report_lines.append(f"系统级校正 ({len(all_system)}个卫星-频率组合):")
                report_lines.append("-" * 35)
                
                for sat_freq, data in all_system.items():
                    details = data['details'] if isinstance(data, dict) and 'details' in data else data
                    arcs = data['arcs'] if isinstance(data, dict) and 'arcs' in data else []
                    # 计算校正统计信息
                    corrections = [d['final_correction'] for d in details if d['final_correction'] is not None]
                    if corrections:
                        max_correction = max(corrections)
                        min_correction = min(corrections)
                        avg_correction = sum(corrections) / len(corrections)
                        max_abs_correction = max(abs(max_correction), abs(min_correction))
                    else:
                        max_correction = min_correction = avg_correction = max_abs_correction = 0.0
                    
                    report_lines.append(f"  {sat_freq}:")
                    report_lines.append(f"    校正次数: {len(details)}")
                    report_lines.append(f"    最大校正: {max_correction:.6f}m")
                    report_lines.append(f"    最小校正: {min_correction:.6f}m")
                    report_lines.append(f"    平均校正: {avg_correction:.6f}m")
                    report_lines.append(f"    最大绝对校正: {max_abs_correction:.6f}m")
                    report_lines.append(f"    ROC CV: {details[0]['roc_cv']:.3f}")
                    
                    # 显示前5个校正详情
                    report_lines.append("    校正详情 (前5个):")
                    for i, detail in enumerate(details[:5]):
                        report_lines.append(f"      {i+1}. 时间: {detail['time']}, 时间差: {detail['time_diff_seconds']:.1f}s, 校正: {detail['final_correction']:.6f}m")
                    if len(details) > 5:
                        report_lines.append(f"      ... 还有 {len(details) - 5} 个校正记录")
                    
                    # 显示后5个校正详情
                    if len(details) > 5:
                        report_lines.append("    校正详情 (后5个):")
                        for i, detail in enumerate(details[-5:], len(details) - 4):
                            report_lines.append(f"      {i}. 时间: {detail['time']}, 时间差: {detail['time_diff_seconds']:.1f}s, 校正: {detail['final_correction']:.6f}m")
                    report_lines.append("")
            else:
                report_lines.append("3.1 系统级校正信息: 无")
                report_lines.append("")
            
            # 3.2 个体级校正信息
            if individual_level_high or individual_level_medium:
                report_lines.append("3.2 个体级校正信息")
                report_lines.append("-" * 30)
                
                # 合并所有个体级校正信息
                all_individual = {**individual_level_high, **individual_level_medium}
                report_lines.append(f"个体级校正 ({len(all_individual)}个卫星-频率组合):")
                report_lines.append("-" * 35)
                
                for sat_freq, data in all_individual.items():
                    details = data['details'] if isinstance(data, dict) and 'details' in data else data
                    arcs = data['arcs'] if isinstance(data, dict) and 'arcs' in data else []
                    # 计算校正统计信息
                    corrections = [d['final_correction'] for d in details if d['final_correction'] is not None]
                    if corrections:
                        max_correction = max(corrections)
                        min_correction = min(corrections)
                        avg_correction = sum(corrections) / len(corrections)
                        max_abs_correction = max(abs(max_correction), abs(min_correction))
                    else:
                        max_correction = min_correction = avg_correction = max_abs_correction = 0.0
                    
                    report_lines.append(f"  {sat_freq}:")
                    report_lines.append(f"    校正次数: {len(details)}")
                    report_lines.append(f"    最大校正: {max_correction:.6f}m")
                    report_lines.append(f"    最小校正: {min_correction:.6f}m")
                    report_lines.append(f"    平均校正: {avg_correction:.6f}m")
                    report_lines.append(f"    最大绝对校正: {max_abs_correction:.6f}m")
                    report_lines.append(f"    ROC CV: {details[0]['roc_cv']:.3f}")
                    
                    # 显示前5个校正详情
                    report_lines.append("    校正详情 (前5个):")
                    for i, detail in enumerate(details[:5]):
                        report_lines.append(f"      {i+1}. 时间: {detail['time']}, 时间差: {detail['time_diff_seconds']:.1f}s, 校正: {detail['final_correction']:.6f}m")
                    if len(details) > 5:
                        report_lines.append(f"      ... 还有 {len(details) - 5} 个校正记录")
                    
                    # 显示后5个校正详情
                    if len(details) > 5:
                        report_lines.append("    校正详情 (后5个):")
                        for i, detail in enumerate(details[-5:], len(details) - 4):
                            report_lines.append(f"      {i}. 时间: {detail['time']}, 时间差: {detail['time_diff_seconds']:.1f}s, 校正: {detail['final_correction']:.6f}m")
                    report_lines.append("")
            else:
                report_lines.append("3.2 个体级校正信息: 无")
                report_lines.append("")
            
            # 3.3 手机独有卫星校正信息
            if 'corrected_phase' in self.results:
                corrected_data = self.results['corrected_phase']
                phone_only_corrections = {}
                
                # 筛选手机独有卫星的校正信息
                for sat_id, freq_data in corrected_data.items():
                    for freq, correction_info in freq_data.items():
                        if correction_info.get('data_source') == 'phone_only':
                            sat_freq_key = f"{sat_id}_{freq}"
                            phone_only_corrections[sat_freq_key] = correction_info
                
                if phone_only_corrections:
                    report_lines.append("3.3 手机独有卫星校正信息")
                    report_lines.append("-" * 30)
                    
                    for sat_freq, correction_info in phone_only_corrections.items():
                        corrections = correction_info.get('correction_applied', [])
                        times = correction_info.get('times', [])
                        
                        if corrections:
                            max_correction = max(corrections)
                            min_correction = min(corrections)
                            avg_correction = sum(corrections) / len(corrections)
                            max_abs_correction = max(abs(max_correction), abs(min_correction))
                        else:
                            max_correction = min_correction = avg_correction = max_abs_correction = 0.0
                        
                        roc_rate = correction_info.get('roc_rate', 0.0)
                        roc_cv = correction_info.get('roc_cv', 0.0)
                        
                        report_lines.append(f"  {sat_freq}:")
                        report_lines.append(f"    校正次数: {len(corrections)}")
                        report_lines.append(f"    最大校正: {max_correction:.6f}m")
                        report_lines.append(f"    最小校正: {min_correction:.6f}m")
                        report_lines.append(f"    平均校正: {avg_correction:.6f}m")
                        report_lines.append(f"    最大绝对校正: {max_abs_correction:.6f}m")
                        report_lines.append(f"    ROC: {roc_rate:.6e} m/s")
                        if roc_cv > 0:
                            report_lines.append(f"    ROC CV: {roc_cv:.3f}")
                        
                        # 显示前5个校正详情
                        if len(corrections) > 0 and len(times) > 0:
                            report_lines.append("    校正详情 (前5个):")
                            for i in range(min(5, len(corrections))):
                                time_diff = 0.0
                                if i > 0 and len(times) > i:
                                    time_diff = (times[i] - times[0]).total_seconds()
                                report_lines.append(f"      {i+1}. 时间: {times[i]}, 时间差: {time_diff:.1f}s, 校正: {corrections[i]:.6f}m")
                            
                            if len(corrections) > 5:
                                report_lines.append(f"      ... 还有 {len(corrections) - 5} 个校正记录")
                                
                                # 显示后5个校正详情
                                report_lines.append("    校正详情 (后5个):")
                                for i in range(max(0, len(corrections) - 5), len(corrections)):
                                    time_diff = 0.0
                                    if i > 0 and len(times) > i:
                                        time_diff = (times[i] - times[0]).total_seconds()
                                    report_lines.append(f"      {i+1}. 时间: {times[i]}, 时间差: {time_diff:.1f}s, 校正: {corrections[i]:.6f}m")
                        report_lines.append("")
                else:
                    report_lines.append("3.3 手机独有卫星校正信息: 无")
                    report_lines.append("")
            
                report_lines.append("")
        
        return "\n".join(report_lines)


    def generate_corrected_rinex_file(self, receiver_rinex_path: str, output_path: str = None) -> str:
        """生成校正后的手机RINEX数据文件
        
        参数:
            receiver_rinex_path: 接收机RINEX文件路径
            output_path: 输出文件路径，如果为None则自动生成
            
        返回:
            生成的RINEX文件路径
        """
        self.start_stage(9, "生成校正后的RINEX文件", 100)
        
        # 确保已校正载波相位
        if not self.results.get('corrected_phase'):
            self.correct_phase_observations(receiver_rinex_path)
            
        corrected_data = self.results['corrected_phase']
        
        # 显示校正数据总结信息
        print(f"校正数据包含 {len(corrected_data)} 颗卫星")
        
        if not self.input_file_path:
            raise ValueError("未设置输入文件路径，无法生成校正后的RINEX文件")
            
        # 确定输入文件路径（优先使用经过两次剔除后的cleaned2文件）
        input_file = self.input_file_path
        # 使用手机RINEX文件的文件夹作为基础路径（而不是接收机文件的文件夹）
        phone_file_name = os.path.basename(self.input_file_path)
        phone_file_name_no_ext = os.path.splitext(phone_file_name)[0]
        phone_result_dir = os.path.join("results", phone_file_name_no_ext)
        
        if os.path.exists(phone_result_dir):
            cleaned2_file_name = f"cleaned2-{phone_file_name}"
            cleaned2_file_path = os.path.join(phone_result_dir, cleaned2_file_name)
            if os.path.exists(cleaned2_file_path):
                input_file = cleaned2_file_path
                print(f"使用经过两次剔除后的文件: {cleaned2_file_name}")
            else:
                print(f"cleaned2文件不存在，使用原始文件")
        else:
            print(f"手机文件结果目录不存在，使用原始文件")
            
        # 生成输出文件路径（保存到手机文件results文件夹下的code-carrier inconsistency子文件夹）
        if output_path is None:
            # 使用手机文件的结果目录
            if os.path.exists(phone_result_dir):
                # 创建code-carrier inconsistency子文件夹
                cci_dir = os.path.join(phone_result_dir, "code-carrier inconsistency")
                if not os.path.exists(cci_dir):
                    os.makedirs(cci_dir)
                
                input_basename = os.path.basename(input_file)
                input_name, input_ext = os.path.splitext(input_basename)
                output_path = os.path.join(cci_dir, f"{input_name}_corrected{input_ext}")
            else:
                input_dir = os.path.dirname(input_file)
                input_basename = os.path.basename(input_file)
                input_name, input_ext = os.path.splitext(input_basename)
                output_path = os.path.join(input_dir, f"{input_name}_corrected{input_ext}")
            
        # 读取RINEX文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
            
        # 保存RINEX数据以供_apply_phase_correction_to_line使用
        # 这里我们需要读取文件头信息来获取观测类型
        self.rinex_data = {'header': {}}
        
        # 解析文件头以获取观测类型信息
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
                    if system is None and tokens:
                        if len(tokens[0]) == 1 and tokens[0].isalpha():
                            system = tokens[0]
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
                        self.rinex_data['header'][f'obs_types_{system}'] = obs_types_list[:num_types] if num_types > 0 else obs_types_list
            
        # 解析历元时间戳（参考remove_outliers_and_save函数）
        epoch_timestamps = {}
        current_epoch = 0
        for line in original_lines:
            if line.startswith('>'):
                current_epoch += 1
                parts = line[1:].split()
                if len(parts) >= 6:
                    # 保留秒的小数部分精度，避免历元合并
                    year, month, day, hour, minute, second = parts[:6]
                    second_float = float(second)
                    # 保留原始秒数格式，不进行四舍五入
                    timestamp = f"{year} {month} {day} {hour} {minute} {second_float}"
                    epoch_timestamps[current_epoch] = timestamp
        
        # 解析观测类型信息（参考remove_outliers_and_save函数）
        system_obs_info = {}  # {系统: {'obs_types': [], 'freq_to_indices': {频率: {观测类型: 字段索引}}}}
        for line in original_lines:
            if 'SYS / # / OBS TYPES' in line:
                system = line[0]
                obs_types = line.split()[2:]
                freq_to_indices = defaultdict(dict)  # {频率: {'code': 索引, 'phase': 索引, 'doppler': 索引}}

                for idx, obs in enumerate(obs_types):
                    if obs.startswith('C'):  # 伪距
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['code'] = idx
                    elif obs.startswith('L'):  # 相位
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['phase'] = idx
                    elif obs.startswith('D'):  # 多普勒
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['doppler'] = idx

                system_obs_info[system] = {
                    'obs_types': obs_types,
                    'freq_to_indices': freq_to_indices
                }
        
        # 复制原始文件内容到输出
        output_lines = original_lines.copy()
            
        
        # 记录修改详情
        modification_details = defaultdict(list)  # {sat_id: [{epoch, freq, original_phase, corrected_phase, correction_amount}]}
        total_modifications = 0
        modified_satellites = set()
        first_epoch_processed = False  # 标记是否已处理第一个历元
        
        # 处理每个有校正数据的卫星（参考remove_outliers_and_save函数）
        for sat_id, freq_data in corrected_data.items():
            sat_system = sat_id[0]
            sat_prn = sat_id[1:].zfill(2)
            system_info = system_obs_info.get(sat_system, {})
            freq_indices = system_info.get('freq_to_indices', {})
            
            if not freq_indices:
                continue  # 跳过无观测类型信息的卫星
            
            # 处理每个频率的校正数据
            for freq, correction_data in freq_data.items():
                times = correction_data['times']
                corrected_phases = correction_data['corrected_phase']
                wavelengths = correction_data['wavelengths']
                
                if freq not in freq_indices or 'phase' not in freq_indices[freq]:
                    continue  # 跳过无相位索引的频率
                
                phase_field_idx = freq_indices[freq]['phase']
                
                # 对每个时间点应用校正
                for time_idx, (correction_time, corrected_phase_m, wavelength) in enumerate(zip(times, corrected_phases, wavelengths)):
                    if corrected_phase_m is None or wavelength is None:
                        continue
                    
                    # 确保wavelength不为None
                    if wavelength is None:
                        continue
                    
                    # 查找对应的历元（假设时间顺序一致）
                    epoch_idx = time_idx + 1  # 历元编号从1开始
                    if epoch_idx not in epoch_timestamps:
                        continue
                    
                    timestamp = epoch_timestamps[epoch_idx]
                    
                    # 定位历元行（使用时间戳比较而不是字符串匹配）
                    epoch_start = -1
                    for i, line in enumerate(output_lines):
                        if line.startswith('>'):
                            parts = line[1:].split()
                            if len(parts) >= 6:
                                try:
                                    year = int(parts[0]); month = int(parts[1]); day = int(parts[2])
                                    hour = int(parts[3]); minute = int(parts[4]); second_float = float(parts[5])
                                    line_epoch = pd.Timestamp(
                                        year=year, month=month, day=day,
                                        hour=hour, minute=minute, second=int(second_float),
                                        microsecond=int((second_float - int(second_float)) * 1000000)
                                    )
                                    # 使用0.1秒容差进行时间匹配
                                    if abs((correction_time - line_epoch).total_seconds()) < 0.1:
                                        epoch_start = i
                                        break
                                except (ValueError, IndexError):
                                    continue
                    if epoch_start < 0:
                        continue
                    
                    # 定位该卫星在历元中的数据行
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
                    
                    # 修改相位观测值字段
                    original_line = output_lines[sat_line_idx]
                    modified_line = list(original_line)
                    
                    # 计算校正量（从ROC模型获取）
                    correction_amount = 0
                    if self.results.get('roc_model'):
                        system_freq_key = f"{sat_system}_{freq}"
                        if system_freq_key in self.results['roc_model']:
                            roc_info = self.results['roc_model'][system_freq_key]
                            roc_rate = roc_info['roc_rate']
                            
                            # 计算校正量：ROC * (t - t0)
                            # 使用与correct_phase_observations相同的逻辑
                            if times:
                                # 找到当前时间在times中的索引
                                time_idx = None
                                for idx, t in enumerate(times):
                                    if abs((t - correction_time).total_seconds()) < 0.1:  # 0.1秒容差
                                        time_idx = idx
                                        break
                                
                                if time_idx is not None and 'correction_applied' in freq_data:
                                    # 直接使用已计算的校正量
                                    correction_amount = freq_data['correction_applied'][time_idx]
                                else:
                                    # 备用计算方式
                                    t0 = times[0]
                                    time_diff_seconds = (correction_time - t0).total_seconds()
                                    correction_amount = -roc_rate * time_diff_seconds
                    
                    # 定位相位字段位置
                    start_pos = 3 + phase_field_idx * 16  # 3: 卫星标识长度
                    end_pos = start_pos + 16
                    
                    if end_pos > len(modified_line):
                        continue  # 防止越界
                    
                    # 读取原始相位值
                    original_field = original_line[start_pos:end_pos].strip()
                    if original_field:
                        try:
                            original_phase_cycle = float(original_field)
                            original_phase_m = original_phase_cycle * wavelength
                            
                            # 重新计算校正后相位，确保与校正量一致
                            corrected_phase_m = original_phase_m + correction_amount
                            
                            # 将校正后的相位值转换为周
                            corrected_phase_cycle = corrected_phase_m / wavelength
                            
                            # 格式化相位观测值（保持原有格式，只保留3位小数）
                            formatted_phase = f"{corrected_phase_cycle:14.3f}"
                            
                            # 更新观测值字段
                            modified_line[start_pos:end_pos] = formatted_phase.rjust(16)
                            output_lines[sat_line_idx] = ''.join(modified_line)
                            
                            # 记录修改详情
                            modification_info = {
                                'freq': freq,
                                'epoch': correction_time,
                                'original_phase_cycle': original_phase_cycle,
                                'original_phase_m': original_phase_m,
                                'corrected_phase_cycle': corrected_phase_cycle,
                                'corrected_phase_m': corrected_phase_m,
                                'wavelength': wavelength,
                                'correction_amount': correction_amount,
                                'formatted_phase': formatted_phase
                            }
                            modification_details[sat_id].append(modification_info)
                            total_modifications += 1
                            modified_satellites.add(sat_id)
                            
                            # 记录详细修改信息到日志（不输出到终端）
                            if not first_epoch_processed:
                                # 解析具体历元时间
                                epoch_timestamp = epoch_timestamps.get(epoch_idx, f"历元{epoch_idx}")
                                modification_details[sat_id].append({
                                    'epoch_timestamp': epoch_timestamp,
                                    'debug_info': f"历元 {epoch_timestamp}，卫星 {sat_id}，频率 {freq}\n  原始相位: {original_phase_cycle:.6f} 周 ({original_phase_m:.6f} 米)\n  校正后相位: {corrected_phase_cycle:.6f} 周 ({corrected_phase_m:.6f} 米)\n  校正量: {correction_amount:.6f} 米"
                                })
                                first_epoch_processed = True
                                
                        except ValueError:
                            continue  # 跳过无法解析的观测值
                
        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
            
        # 保存修改详情到结果中
        self.results['phase_modification_details'] = {
            'modification_details': modification_details,
            'total_modifications': total_modifications,
            'modified_satellites': list(modified_satellites)
        }
        
        self.update_progress(100)
        print(f"校正后的RINEX文件已生成: {output_path}")
        return output_path

    def _apply_phase_correction_to_line(self, obs_line: str, sat_id: str, epoch: datetime, corrected_freq_data: Dict) -> tuple:
        """对单行观测数据应用相位校正
        
        参数:
            obs_line: 原始观测数据行
            sat_id: 卫星ID
            epoch: 当前历元时间
            corrected_freq_data: 该卫星的校正数据
            
        返回:
            (校正后的观测数据行, 修改详情列表)
        """
        # 解析观测值（假设固定宽度格式）
        line = obs_line.rstrip('\n')
        if len(line) < 80:
            return obs_line + "\n", []
            
        # 提取观测值字段（每16字符一个观测值）
        obs_values = []
        for i in range(80, len(line), 16):
            field = line[i:i+16].strip()
            obs_values.append(field if field else None)
            
        corrected_line = line[:80]  # 保持前80字符不变
        modifications = []
        
        # 获取该卫星系统的观测类型信息
        sat_system = sat_id[0] if sat_id else 'Unknown'
        
        # 从RINEX数据中获取观测类型信息
        obs_types = []
        if hasattr(self, 'rinex_data') and self.rinex_data and 'header' in self.rinex_data:
            obs_types = self.rinex_data['header'].get(f'obs_types_{sat_system}', [])
            
        # 调试信息：显示观测类型（只在第一次调用时显示）
        if not hasattr(self, '_debug_shown_obs_types'):
            print(f"    调试信息：卫星系统 {sat_system} 的观测类型: {obs_types}")
            print(f"    调试信息：观测值字段数: {len(obs_values)}")
            self._debug_shown_obs_types = True
        
        # 对于每个频率的校正数据
        for freq, freq_data in corrected_freq_data.items():
            times = freq_data['times']
            corrected_phases = freq_data['corrected_phase']
            wavelengths = freq_data['wavelengths']
            
            # 找到最接近当前历元的时间索引
            closest_idx = 0
            min_time_diff = float('inf')
            
            for idx, t in enumerate(times):
                time_diff = abs((t - epoch).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_idx = idx
                    
            # 如果时间差太大（超过0.1秒），跳过校正
            # 使用更小的容差以确保精确匹配
            if min_time_diff > 0.1:
                # 添加调试信息
                if not hasattr(self, '_debug_time_mismatch_shown'):
                    print(f"    调试信息：时间匹配失败 - 最小时间差: {min_time_diff:.6f}秒")
                    print(f"    调试信息：当前历元: {epoch}")
                    print(f"    调试信息：可用时间范围: {times[0] if times else 'None'} 到 {times[-1] if times else 'None'}")
                    self._debug_time_mismatch_shown = True
                continue
                
            # 获取波长
            wavelength = wavelengths[closest_idx] if closest_idx < len(wavelengths) else None
            
            if wavelength is not None:
                # 获取原始相位值
                original_phase_cycle = None
                original_phase_m = None
                
                # 根据观测类型找到对应的相位观测值
                phase_obs_type = freq  # 频率名称就是观测类型，例如：L1C -> L1C
                if not hasattr(self, '_debug_shown_phase_parsing'):
                    print(f"      调试信息：查找观测类型 {phase_obs_type}")
                    self._debug_shown_phase_parsing = True
                
                if phase_obs_type in obs_types:
                    phase_idx = obs_types.index(phase_obs_type)
                    if phase_idx < len(obs_values) and obs_values[phase_idx]:
                        try:
                            original_phase_cycle = float(obs_values[phase_idx])
                            original_phase_m = original_phase_cycle * wavelength
                            if not hasattr(self, '_debug_shown_phase_parsing'):
                                print(f"      调试信息：解析到原始相位值 {original_phase_cycle} 周 ({original_phase_m} 米)")
                                self._debug_shown_phase_parsing = True
                        except ValueError as e:
                            if not hasattr(self, '_debug_shown_phase_parsing'):
                                print(f"      调试信息：解析相位值失败: {e}")
                                self._debug_shown_phase_parsing = True
                else:
                    if not hasattr(self, '_debug_shown_phase_parsing'):
                        print(f"      调试信息：未找到观测类型 {phase_obs_type} 在观测类型列表中")
                        self._debug_shown_phase_parsing = True
                
                # 重新计算校正后相位，确保与校正量一致
                if original_phase_m is not None:
                    corrected_phase_m = original_phase_m + correction_amount
                    # 将米转换为周
                    corrected_phase_cycle = corrected_phase_m / wavelength
                
                # 格式化相位观测值（保持原有格式，只保留3位小数）
                formatted_phase = f"{corrected_phase_cycle:14.3f}"
                
                # 计算校正量（从ROC模型获取）
                correction_amount = 0
                if self.results.get('roc_model'):
                    system_freq_key = f"{sat_system}_{freq}"
                    
                    if system_freq_key in self.results['roc_model']:
                        roc_info = self.results['roc_model'][system_freq_key]
                        roc_rate = roc_info['roc_rate']
                        
                        # 计算校正量：ROC * (t - t0)
                        # 使用与correct_phase_observations相同的逻辑
                        if times:
                            # 找到当前时间在times中的索引
                            time_idx = None
                            for idx, t in enumerate(times):
                                if abs((t - epoch).total_seconds()) < 0.1:  # 0.1秒容差
                                    time_idx = idx
                                    break
                            
                            if time_idx is not None and 'correction_applied' in freq_data:
                                # 直接使用已计算的校正量
                                correction_amount = freq_data['correction_applied'][time_idx]
                            else:
                                # 备用计算方式
                                t0 = times[0]
                                time_diff_seconds = (epoch - t0).total_seconds()
                                correction_amount = -roc_rate * time_diff_seconds
                
                # 原始相位值已在上面解析
                
                # 记录修改详情
                modification_info = {
                    'freq': freq,
                    'epoch': epoch,
                    'original_phase_cycle': original_phase_cycle,
                    'original_phase_m': original_phase_m,
                    'corrected_phase_cycle': corrected_phase_cycle,
                    'corrected_phase_m': corrected_phase_m,
                    'wavelength': wavelength,
                    'correction_amount': correction_amount,
                    'formatted_phase': formatted_phase
                }
                modifications.append(modification_info)
                
                # 这里需要根据实际的观测类型顺序来确定相位观测值的位置
                # 简化处理：假设相位观测值在特定位置
                # 实际应用中需要根据RINEX文件头信息来确定
                
        return corrected_line + "\n", modifications

    def perform_code_phase_inconsistency_modeling(self, receiver_rinex_path: str, output_path: str = None) -> Dict:
        """执行完整的码相不一致性建模和校正流程
        
        参数:
            receiver_rinex_path: 接收机RINEX文件路径
            output_path: 校正后的RINEX文件输出路径
            
        返回:
            包含所有分析结果的字典
        """
        print("开始码相不一致性建模和校正流程...")
        
        # 确保已读取手机观测数据
        if not self.observations_meters:
            raise ValueError("请先读取手机RINEX观测数据")
            
        # 第一步：读取接收机观测数据
        print("步骤1: 读取接收机RINEX观测数据...")
        self.read_receiver_rinex_obs(receiver_rinex_path)
        
        # 第二步：计算接收机CMC（手机CMC使用现有的伪距相位原始差值）
        print("步骤2: 计算接收机CMC...")
        receiver_cmc = self.calculate_receiver_cmc()
        
        # 第三步：计算站间单差dCMC
        print("步骤3: 计算站间单差dCMC...")
        dcmc = self.calculate_dcmc(receiver_rinex_path)
        
        # 第四步：提取CCI时间序列
        print("步骤4: 提取码相不一致性时间序列...")
        cci_series = self.extract_cci_series(None)  # 不传递路径，使用已计算的结果
        
        # 第五步：计算ROC模型参数
        print("步骤5: 计算ROC模型参数...")
        roc_model = self.calculate_roc_model(None)  # 不传递路径，使用已计算的结果
        
        # 第六步：校正载波相位观测值
        print("步骤6: 校正载波相位观测值...")
        corrected_phase = self.correct_phase_observations(None)  # 不传递路径，使用已计算的结果
        
        # 第七步：生成校正后的RINEX文件
        print("步骤7: 生成校正后的RINEX文件...")
        corrected_rinex_path = self.generate_corrected_rinex_file(receiver_rinex_path, output_path)
        
        # 生成分析报告
        print("生成分析报告...")
        report = self._generate_cci_modeling_report()
        
        # 保存报告和日志到文件（使用手机文件的结果目录）
        phone_file_name = os.path.basename(self.input_file_path)
        phone_file_name_no_ext = os.path.splitext(phone_file_name)[0]
        phone_result_dir = os.path.join("results", phone_file_name_no_ext)
        
        if os.path.exists(phone_result_dir):
            # 创建code-carrier inconsistency子文件夹
            cci_dir = os.path.join(phone_result_dir, "code-carrier inconsistency")
            if not os.path.exists(cci_dir):
                os.makedirs(cci_dir)
            report_path = os.path.join(cci_dir, "码相不一致性建模报告.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"分析报告已保存: {report_path}")
            
            # 保存处理日志
            log_path = self.save_cci_processing_log(receiver_rinex_path)
            if log_path:
                print(f"处理日志已保存: {log_path}")
        
        print("码相不一致性建模和校正流程完成!")
        
        return {
            'receiver_cmc': receiver_cmc,
            'phone_cmc': self.results.get('code_phase_differences', {}),  # 使用伪距相位原始差值
            'dcmc': dcmc,
            'cci_series': cci_series,
            'roc_model': roc_model,
            'corrected_phase': corrected_phase,
            'corrected_rinex_path': corrected_rinex_path,
            'report': report
        }

    def save_cci_processing_log(self, receiver_rinex_path: str) -> str:
        """保存码相不一致性处理日志，格式仿照code_phase_cleaning.log
        
        参数:
            receiver_rinex_path: 接收机RINEX文件路径
            
        返回:
            日志文件路径
        """
        # 使用手机文件的结果目录
        phone_file_name = os.path.basename(self.input_file_path)
        phone_file_name_no_ext = os.path.splitext(phone_file_name)[0]
        phone_result_dir = os.path.join("results", phone_file_name_no_ext)
        
        if not os.path.exists(phone_result_dir):
            return None
            
        # 保存日志文件到code-carrier inconsistency子文件夹
        cci_dir = os.path.join(phone_result_dir, "code-carrier inconsistency")
        if not os.path.exists(cci_dir):
            os.makedirs(cci_dir)
        log_path = os.path.join(cci_dir, "code_phase_inconsistency_processing.log")
        
        # 初始化日志内容
        log_content = [
            "=" * 70 + "\n",
            "码相不一致性建模和校正处理日志\n",
            "=" * 70 + "\n\n",
            f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"手机RINEX文件: {self.input_file_path}\n",
            f"接收机RINEX文件: {receiver_rinex_path}\n\n"
        ]
        
        # 载波相位修改详情
        modification_details = self.results.get('phase_modification_details', {})
        if modification_details:
            total_modifications = modification_details.get('total_modifications', 0)
            modified_satellites = modification_details.get('modified_satellites', [])
            sat_details = modification_details.get('modification_details', {})
            
            log_content.append(f"载波相位校正统计:\n")
            log_content.append(f"- 总计修改 {total_modifications} 个载波相位观测值\n")
            log_content.append(f"- 涉及 {len(modified_satellites)} 颗卫星\n")
            log_content.append(f"- 修改的卫星: {', '.join(modified_satellites)}\n\n")
            
            # 详细修改信息
            log_content.append("各卫星载波相位修改详情:\n")
            log_content.append("-" * 50 + "\n")
            
            for sat_id in modified_satellites:
                if sat_id in sat_details:
                    details = sat_details[sat_id]
                    log_content.append(f"卫星 {sat_id}:\n")
                    
                    # 先添加调试信息（如果存在）
                    debug_info_added = False
                    for detail in details:
                        if 'debug_info' in detail:
                            log_content.append(f"  {detail['debug_info']}\n")
                            debug_info_added = True
                            break
                    
                    # 按历元分组
                    epoch_groups = defaultdict(list)
                    for detail in details:
                        if 'epoch' in detail:  # 只处理有epoch字段的详情
                            epoch_groups[detail['epoch']].append(detail)
                    
                    for epoch in sorted(epoch_groups.keys()):
                        epoch_details = epoch_groups[epoch]
                        log_content.append(f"  历元 {epoch}:\n")
                        
                        for detail in epoch_details:
                            # 获取原始相位值
                            original_phase_m = detail.get('original_phase_m')
                            original_phase_cycle = detail.get('original_phase_cycle')
                            
                            if original_phase_m is not None:
                                log_content.append(f"    - {detail['freq']}: "
                                                 f"原始相位={original_phase_m:.6f}m ({original_phase_cycle:.6f}周), "
                                                 f"校正后相位={detail['corrected_phase_m']:.6f}m ({detail['corrected_phase_cycle']:.6f}周), "
                                                 f"校正量={detail['correction_amount']:.6f}m, "
                                                 f"波长={detail['wavelength']:.4f}m\n")
                            else:
                                log_content.append(f"    - {detail['freq']}: "
                                                 f"原始相位=无法解析, "
                                                 f"校正后相位={detail['corrected_phase_m']:.6f}m ({detail['corrected_phase_cycle']:.6f}周), "
                                                 f"校正量={detail['correction_amount']:.6f}m, "
                                                 f"波长={detail['wavelength']:.4f}m\n")
                    log_content.append("\n")
        
        # ROC模型参数
        if self.results.get('roc_model'):
            log_content.append("ROC模型参数（按卫星系统分别计算）:\n")
            log_content.append("-" * 40 + "\n")
            for system_freq_key, roc_info in self.results['roc_model'].items():
                # 解析系统-频率键
                parts = system_freq_key.split('_', 1)
                if len(parts) == 2:
                    sat_system = parts[0]
                    freq = parts[1]
                    system_name = {'G': 'GPS', 'R': 'GLONASS', 'E': 'Galileo', 'C': 'BDS', 'J': 'QZSS', 'I': 'IRNSS'}.get(sat_system, sat_system)
                    log_content.append(f"{system_name} 系统 频率 {freq}:\n")
                else:
                    log_content.append(f"系统-频率 {system_freq_key}:\n")
                log_content.append(f"  ROC变化率: {roc_info['roc_rate']:.6e} m/s\n")
                log_content.append(f"  标准差: {roc_info['roc_std']:.6e} m/s\n")
                log_content.append(f"  参与卫星数: {roc_info['num_satellites']}\n")
                log_content.append(f"  参与卫星: {', '.join(roc_info['contributing_sats'])}\n\n")
    
        # dCMC统计信息
        if self.results.get('dcmc'):
            log_content.append("dCMC统计信息:\n")
            log_content.append("-" * 40 + "\n")
            dcmc_data = self.results['dcmc']
            total_combinations = 0
            total_points = 0
            
            for sat_id, freq_data in dcmc_data.items():
                for freq, dcmc_info in freq_data.items():
                    total_combinations += 1
                    total_points += len(dcmc_info['dcmc'])
                    
            log_content.append(f"卫星-频率组合数: {total_combinations}\n")
            log_content.append(f"总观测点数: {total_points}\n\n")
            
        # CCI分析结果
        if self.results.get('cci_series'):
            log_content.append("CCI时间序列分析:\n")
            log_content.append("-" * 40 + "\n")
            cci_data = self.results['cci_series']
            
            for sat_id, freq_data in cci_data.items():
                log_content.append(f"卫星 {sat_id}:\n")
                for freq, cci_info in freq_data.items():
                    arc_info = cci_info['arc_info']
                    log_content.append(f"  频率 {freq}: {len(arc_info)} 个连续弧段\n")
                    for arc in arc_info:
                        log_content.append(f"    弧段 {arc['arc_index']+1}: 时长 {arc['duration']:.1f}s, "
                                          f"点数 {arc['num_points']}, CCI范围 {arc['cci_range']:.4f}m\n")
                log_content.append("\n")
            
        # 校正效果统计
        if self.results.get('corrected_phase'):
            log_content.append("校正效果统计:\n")
            log_content.append("-" * 40 + "\n")
            corrected_data = self.results['corrected_phase']
            
            for sat_id, freq_data in corrected_data.items():
                log_content.append(f"卫星 {sat_id}:\n")
                for freq, freq_data in freq_data.items():
                    corrections = [c for c in freq_data['correction_applied'] if c is not None]
                    if corrections:
                        max_corr = max(corrections)
                        min_corr = min(corrections)
                        mean_corr = sum(corrections) / len(corrections)
                        log_content.append(f"  频率 {freq}: 校正范围 [{min_corr:.4f}, {max_corr:.4f}]m, "
                                          f"平均校正 {mean_corr:.4f}m\n")
                log_content.append("\n")
        
        log_content.append("=" * 70 + "\n")
        log_content.append("处理完成\n")
        log_content.append("=" * 70 + "\n")
        
        # 写入日志文件
        with open(log_path, 'w', encoding='utf-8') as f:
            f.writelines(log_content)
            
        return log_path

    def remove_code_phase_outliers(self, data: Dict, threshold: float = 5.0) -> str:
        """
        基于伪距相位差值变化阈值，剔除异常观测值并生成剔除后的文件
        
        参数:
            data: RINEX观测数据
            threshold: 差值变化阈值（米），默认5米
            
        返回:
            剔除后的文件路径
        """
        self.start_stage(6, "基于伪距相位差值剔除异常观测值", 100)

        # 确保已经计算了伪距相位差值
        if not self.results.get('code_phase_differences'):
            self.calculate_code_phase_differences(data)

        # 初始化日志内容
        log_content = [
            "=" * 70 + "\n",
            "基于伪距相位差值变化的异常观测值剔除日志\n",
            "=" * 70 + "\n\n",
            f"差值变化阈值: {threshold} 米\n\n"
        ]

        # 1. 识别需要剔除的历元
        outlier_epochs = defaultdict(lambda: defaultdict(list))  # {sat_id: {freq: [历元索引]}}
        outlier_details = defaultdict(list)  # {sat_id: [异常详情]}

        code_phase_diffs = self.results['code_phase_differences']
        total_sats = len(code_phase_diffs)
        processed_sats = 0

        for sat_id, freq_data in code_phase_diffs.items():
            sat_outliers = {}

            for freq, diff_data in freq_data.items():
                times = diff_data['times']
                diff_changes = diff_data['diff_changes']

                # 找出超过阈值的差值变化对应的历元
                outlier_indices = []
                for i, change in enumerate(diff_changes):
                    if change is not None and change > threshold:
                        # 注意：diff_changes[i] 对应的是 times[i+1] 历元（后一个历元）
                        # 因为 diff_changes 计算的是后一个值减前一个值
                        outlier_epoch_idx = i + 1  # 后一个历元
                        if outlier_epoch_idx < len(times):
                            outlier_indices.append(outlier_epoch_idx)

                            # 记录异常详情
                            outlier_info = {
                                'freq': freq,
                                'epoch_idx': outlier_epoch_idx,
                                'time': times[outlier_epoch_idx],
                                'diff_change': change,
                                'threshold': threshold
                            }
                            outlier_details[sat_id].append(outlier_info)

                if outlier_indices:
                    sat_outliers[freq] = outlier_indices
                    outlier_epochs[sat_id][freq] = outlier_indices

            processed_sats += 1
            self.update_progress(int(processed_sats / total_sats * 50))

        # 2. 读取RINEX文件内容
        with open(self.input_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 3. 解析观测类型映射
        system_obs_info = {}  # {系统: {'obs_types': [], 'freq_to_indices': {频率: {观测类型: 字段索引}}}}
        for line in lines:
            if 'SYS / # / OBS TYPES' in line:
                system = line[0]
                obs_types = line.split()[2:]
                freq_to_indices = defaultdict(dict)  # {频率: {'code': 索引, 'phase': 索引, 'doppler': 索引}}

                for idx, obs in enumerate(obs_types):
                    if obs.startswith('C'):  # 伪距
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['code'] = idx
                    elif obs.startswith('L'):  # 相位
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['phase'] = idx
                    elif obs.startswith('D'):  # 多普勒
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['doppler'] = idx

                system_obs_info[system] = {
                    'obs_types': obs_types,
                    'freq_to_indices': freq_to_indices
                }

        # 4. 修改异常历元的观测值
        modified_count = defaultdict(int)  # {观测类型: 修改数量}
        modified_satellites = set()
        satellite_modify_details = []

        for sat_id, freq_outliers in outlier_epochs.items():
            sat_system = sat_id[0]
            sat_prn = sat_id[1:].zfill(2)
            system_info = system_obs_info.get(sat_system, {})
            freq_indices = system_info.get('freq_to_indices', {})

            if not freq_indices:
                continue

            satellite_modifications = []

            # 处理每个频率的异常历元
            for freq, epoch_indices in freq_outliers.items():
                for epoch_idx in epoch_indices:
                    # 定位历元行
                    epoch_start = -1
                    for i, line in enumerate(lines):
                        if line.startswith('>'):
                            # 解析历元时间
                            parts = line[1:].split()
                            if len(parts) >= 6:
                                try:
                                    year = int(parts[0])
                                    month = int(parts[1])
                                    day = int(parts[2])
                                    hour = int(parts[3])
                                    minute = int(parts[4])
                                    second_float = float(parts[5])
                                    epoch_time = pd.Timestamp(
                                        year=year, month=month, day=day,
                                        hour=hour, minute=minute, second=int(second_float),
                                        microsecond=int((second_float - int(second_float)) * 1000000)
                                    )

                                    # 从code_phase_diffs中获取对应的时间
                                    if sat_id in code_phase_diffs and freq in code_phase_diffs[sat_id]:
                                        diff_times = code_phase_diffs[sat_id][freq]['times']
                                        if epoch_idx < len(diff_times) and epoch_time == diff_times[epoch_idx]:
                                            epoch_start = i
                                            break
                                except (ValueError, IndexError):
                                    continue

                    if epoch_start < 0:
                        continue

                    # 定位该卫星在历元中的数据行
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

                    # 修改异常观测值字段
                    original_line = lines[sat_line_idx]
                    modified_line = list(original_line)
                    field_modified = False
                    modified_fields = []

                    # 剔除伪距、相位、多普勒观测值
                    for obs_type in ['code', 'phase', 'doppler']:
                        if freq in freq_indices and obs_type in freq_indices[freq]:
                            field_idx = freq_indices[freq][obs_type]
                            start_pos = 3 + field_idx * 16  # 3: 卫星标识长度
                            end_pos = start_pos + 16

                            if end_pos <= len(modified_line):
                                original_field = original_line[start_pos:end_pos].strip()
                                if original_field:
                                    modified_line[start_pos:end_pos] = ' ' * 16
                                    modified_count[obs_type] += 1
                                    field_modified = True
                                    modified_fields.append(f"{freq}({obs_type})")

                    if field_modified:
                        lines[sat_line_idx] = ''.join(modified_line)
                        modified_satellites.add(sat_id)
                        satellite_modifications.append(
                            f"  历元 {epoch_idx} ({epoch_time}): 已剔除 {', '.join(modified_fields)}")

            if satellite_modifications:
                satellite_modify_details.append(f"卫星 {sat_id} 的剔除详情:")
                satellite_modify_details.extend(satellite_modifications)
                satellite_modifications.append("")

        # 5. 保存修改后的文件
        results_dir = self.current_result_dir
        file_name = os.path.basename(self.input_file_path)
        cleaned_file_name = f"cleaned1-{file_name}"
        output_path = os.path.join(results_dir, cleaned_file_name)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # 6. 生成详细日志
        total_modified = sum(modified_count.values())
        log_content.append("\n一、剔除统计摘要\n")
        log_content.append("-" * 70 + "\n")
        log_content.append(f"总计剔除卫星数: {len(modified_satellites)}\n")
        log_content.append(f"总计剔除观测值: {total_modified}\n")
        log_content.append(
            f"剔除分类: 伪距={modified_count['code']}, 相位={modified_count['phase']}, 多普勒={modified_count['doppler']}\n\n")

        # 添加异常历元详情到日志
        log_content.append("\n二、异常历元检测详情\n")
        log_content.append("-" * 70 + "\n")

        # 按系统组织卫星信息
        system_satellites = defaultdict(list)
        for sat_id in outlier_details.keys():
            system_satellites[sat_id[0]].append(sat_id)

        # 按系统顺序输出
        for system, satellites in system_satellites.items():
            log_content.append(f"卫星系统 {system}:\n")

            for sat_id in sorted(satellites):
                details = outlier_details[sat_id]
                log_content.append(f"  卫星 {sat_id} ({len(details)}个异常观测值):\n")

                # 按频率分组异常信息
                freq_groups = defaultdict(list)
                for detail in details:
                    freq_groups[detail['freq']].append(detail)

                for freq in sorted(freq_groups.keys()):
                    freq_details = freq_groups[freq]
                    log_content.append(f"    频率 {freq}:\n")

                    for detail in freq_details:
                        log_content.append(f"      - 历元 {detail['epoch_idx']} ({detail['time']}): "
                                           f"差值变化={detail['diff_change']:.6f}m, "
                                           f"阈值={detail['threshold']:.6f}m\n")
                log_content.append("\n")

        # 写入日志文件
        debug_file_name = "code_phase_cleaning.log"
        debug_path = os.path.join(results_dir, debug_file_name)
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.writelines(log_content)

        # 控制台输出
        print(f"--基于伪距相位差值变化检测到 {len(outlier_epochs)} 颗卫星存在异常历元")
        print(f"--总计剔除 {total_modified} 个观测值，涉及 {len(modified_satellites)} 颗卫星")
        print(f"   - 伪距: {modified_count['code']}")
        print(f"   - 相位: {modified_count['phase']}")
        print(f"   - 多普勒: {modified_count['doppler']}")
        print(f"--剔除异常观测值后的文件保存至: {output_path}")
        print(f"--剔除详细信息保存至: {debug_path}")

        # 更新进度
        self.update_progress(100)

        return output_path

    def calculate_phase_prediction_errors(self, data: Dict) -> Dict:
        """计算相位预测误差
        
        使用公式：φˆrs,j(k + 1) = φrs,j(k) + (Drs,j(k+1)+Drs,j(k))/2 * T
        其中：
        - φˆrs,j(k + 1): 预测的相位值（周）
        - φrs,j(k): 前一历元的相位值（周）
        - T: 时间差（秒）
        - Drs,j(k+1): 当前历元的多普勒频率（Hz）
        - Drs,j(k): 前一历元的多普勒频率（Hz）
        - (Drs,j(k+1)+Drs,j(k))/2: 多普勒频率的算术平均值
        
        载波相位预测误差计算为：ˆΦs r,j − Φrs,j = λj (ˆφrs,j − φrs,j)（米）
        其中 λj 是载波波长。
        """
        self.start_stage(4, "计算相位预测误差", 100)

        errors = {}
        total_sats = len(self.observations_meters)
        processed_sats = 0

        for sat_id, freq_data in self.observations_meters.items():
            freq_errors = {}

            for freq, obs_data in freq_data.items():
                times = obs_data['times']
                phase_values = obs_data['phase_cycle']  # 周
                doppler_values = obs_data['doppler']  # 米/秒（存储时已转换为速度单位）

                # 获取对应频率(Hz)和波长(米)
                frequency = self.frequencies[sat_id[0]].get(freq)
                wavelength = self.wavelengths[sat_id[0]].get(freq)

                # 初始化结果存储
                freq_errors[freq] = {
                    'times': [],
                    'actual_phase': [],  # 周
                    'predicted_phase': [],  # 周
                    'prediction_error': [],  # 米
                    'doppler_mps': [],  # 多普勒速度（米/秒）
                    'doppler_hz': []   # 多普勒频率（Hz）
                }

                # 计算预测误差
                for i in range(1, len(times)):
                    # 检查数据有效性
                    if (phase_values[i - 1] is not None and
                            doppler_values[i - 1] is not None and
                            i < len(doppler_values) and
                            doppler_values[i] is not None and
                            phase_values[i] is not None and
                            frequency is not None and
                            wavelength is not None):
                        
                        time_diff = (times[i] - times[i - 1]).total_seconds()
                        
                        # 获取当前历元和前一历元的多普勒频率（Hz）
                        # 将存储的多普勒速度（米/秒）转换回多普勒频率（Hz）
                        doppler_now_hz = doppler_values[i] / wavelength      # 当前历元多普勒频率
                        doppler_old_hz = doppler_values[i - 1] / wavelength  # 前一历元多普勒频率
                        
                        # 使用新公式：φˆrs,j(k + 1) = φrs,j(k) + (Drs,j(k+1)+Drs,j(k))/2 * T
                        # 其中 Drs,j(k+1) 和 Drs,j(k) 是多普勒频率（Hz），T 是时间差（秒）
                        # (Drs,j(k+1)+Drs,j(k))/2 是多普勒频率的算术平均值
                        doppler_arithmetic_mean = (doppler_now_hz + doppler_old_hz) / 2
                        
                        # 预测相位（周）
                        # 注意：多普勒频率与相位变化率的关系是 dΦ/dt = -f_d / f_c
                        # 所以相位变化量 = -dt * f_d / f_c
                        phase_change = -time_diff * doppler_arithmetic_mean / frequency
                        predicted_phase = phase_values[i - 1] + phase_change  # 周
                        
                        # 计算预测误差（米）
                        # ˆΦs r,j − Φrs,j = λj (ˆφrs,j − φrs,j)
                        error = (phase_values[i] - predicted_phase) * wavelength  # 周 * 米/周 = 米

                        # 保存结果
                        freq_errors[freq]['times'].append(times[i])
                        freq_errors[freq]['actual_phase'].append(phase_values[i])
                        freq_errors[freq]['predicted_phase'].append(predicted_phase)
                        freq_errors[freq]['prediction_error'].append(error)
                        freq_errors[freq]['doppler_mps'].append((doppler_values[i - 1] + doppler_values[i]) / 2)  # 保存速度值（算术平均）
                        freq_errors[freq]['doppler_hz'].append(doppler_arithmetic_mean)    # 保存频率值（算术平均）

                errors[sat_id] = freq_errors
                processed_sats += 1
                self.update_progress(int(processed_sats / total_sats * 100))

        self.results['phase_prediction_errors'] = errors
        return errors

    def calculate_epoch_double_differences(self):
        """计算各卫星各频率的历元间双差（伪距、相位、多普勒）"""
        self.start_stage(5, "计算历元间双差", 100)
        double_diffs = {}  # 存储双差结果 {sat_id: {freq: {dd_code: [], dd_phase: [], dd_doppler: []}}}

        for sat_id, freq_data in self.observations_meters.items():
            double_diffs[sat_id] = {}
            for freq, data in freq_data.items():
                code = np.array(data['code'], dtype=float)  # 伪距（米）
                phase = np.array(data['phase'], dtype=float)  # 载波相位（米）
                doppler = np.array(data['doppler'], dtype=float)  # 多普勒（米/秒）

                # 计算双差（i>2时，dd = x[i+2] - 2x[i+1] + x[i]）
                n = len(code)
                if n < 3:
                    continue  # 至少需要3个历元才能计算双差

                dd_code = np.zeros(n - 2)
                dd_phase = np.zeros(n - 2)
                dd_doppler = np.zeros(n - 2)

                for i in range(n - 2):
                    dd_code[i] = code[i + 2] - 2 * code[i + 1] + code[i]
                    dd_phase[i] = phase[i + 2] - 2 * phase[i + 1] + phase[i]
                    dd_doppler[i] = doppler[i + 2] - 2 * doppler[i + 1] + doppler[i]

                # 存储结果（剔除前两个历元，双差结果长度为n-2）
                double_diffs[sat_id][freq] = {
                    'times': data['times'][2:],  # 双差对应的时间为第3个历元起
                    'dd_code': dd_code.tolist(),
                    'dd_phase': dd_phase.tolist(),
                    'dd_doppler': dd_doppler.tolist()
                }

        self.results['double_differences'] = double_diffs  # 保存双差结果
        self.update_progress(100)
        return double_diffs

    def calculate_triple_median_error(self, double_diffs):
        """计算双差结果的三倍中误差并检测超限值（包含伪距、相位、多普勒）"""
        triple_errors = {}  # 存储三倍中误差及超限值 {sat_id: {freq: {threshold: float, outliers: list}}}

        for sat_id, freq_data in double_diffs.items():
            triple_errors[sat_id] = {}
            for freq, dd_data in freq_data.items():
                # 提取双差观测值（过滤无效值）
                code = np.array([v for v in dd_data['dd_code'] if v is not None and not np.isnan(v)])
                phase = np.array([v for v in dd_data['dd_phase'] if v is not None and not np.isnan(v)])
                doppler = np.array([v for v in dd_data['dd_doppler'] if v is not None and not np.isnan(v)])

                # 计算中误差（使用标准差）
                def std_error(arr):
                    if len(arr) < 2:  # 至少需要两个观测值计算标准差
                        return 0
                    return np.std(arr, ddof=1)  # 样本标准差（自由度n-1）

                # 伪距双差
                if len(code) > 1:
                    sigma_code = std_error(code)
                    triple_sigma_code = 3 * sigma_code if sigma_code != 0 else 0.1  # 三倍中误差
                    outliers_code = np.where(np.abs(dd_data['dd_code']) > triple_sigma_code)[0].tolist()  # 超限索引
                else:
                    triple_sigma_code, outliers_code = 0, []

                # 载波相位双差
                if len(phase) > 1:
                    sigma_phase = std_error(phase)
                    triple_sigma_phase = 3 * sigma_phase if sigma_phase != 0 else 0.01  # 相位精度更高，阈值更小
                    outliers_phase = np.where(np.abs(dd_data['dd_phase']) > triple_sigma_phase)[0].tolist()
                else:
                    triple_sigma_phase, outliers_phase = 0, []

                # 多普勒双差
                if len(doppler) > 1:
                    sigma_doppler = std_error(doppler)
                    triple_sigma_doppler = 3 * sigma_doppler if sigma_doppler != 0 else 0.05  # 多普勒阈值
                    outliers_doppler = np.where(np.abs(dd_data['dd_doppler']) > triple_sigma_doppler)[0].tolist()
                else:
                    triple_sigma_doppler, outliers_doppler = 0, []

                # 合并结果
                triple_errors[sat_id][freq] = {
                    'code': {'threshold': triple_sigma_code, 'outliers': outliers_code},
                    'phase': {'threshold': triple_sigma_phase, 'outliers': outliers_phase},
                    'doppler': {'threshold': triple_sigma_doppler, 'outliers': outliers_doppler}
                }

        self.results['triple_median_errors'] = triple_errors
        return triple_errors

    def analyze_beidou_system_bias(self):
        """分析北斗二号和北斗三号系统间的偏差"""
        self.start_stage(6, "分析北斗系统间偏差", 100)
        
        # 初始化结果存储
        bias_results = {
            'BDS-2': {},
            'BDS-3': {},
            'system_comparison': {},
            'frequency_bias': {},
            'orbit_type_bias': {}
        }
        
        # 统计北斗卫星数量
        bds2_sats = []
        bds3_sats = []
        
        for sat_id in self.observations_meters.keys():
            if sat_id.startswith('C'):
                prn = sat_id[1:]
                prn_num = int(prn) if prn.isdigit() else 0
                if 1 <= prn_num <= 14:  # C01-C14
                    bds2_sats.append(sat_id)
                elif 19 <= prn_num <= 61:  # C19-C61
                    bds3_sats.append(sat_id)
        
        self.update_progress(20)
        
        # 分析各系统的观测值统计特性
        for system_name, sat_list in [('BDS-2', bds2_sats), ('BDS-3', bds3_sats)]:
            if not sat_list:
                continue
                
            system_stats = {}
            for freq in ['L1D', 'L1P', 'L2I', 'L5P']:
                freq_stats = {
                    'satellites': [],
                    'code_mean': [],
                    'code_std': [],
                    'phase_mean': [],
                    'phase_std': [],
                    'snr_mean': [],
                    'snr_std': []
                }
                
                for sat_id in sat_list:
                    if sat_id in self.observations_meters and freq in self.observations_meters[sat_id]:
                        sat_data = self.observations_meters[sat_id][freq]
                        
                        # 计算统计量
                        code_vals = [v for v in sat_data['code'] if v is not None]
                        phase_vals = [v for v in sat_data['phase'] if v is not None]
                        
                        if code_vals and phase_vals:
                            freq_stats['satellites'].append(sat_id)
                            freq_stats['code_mean'].append(np.mean(code_vals))
                            freq_stats['code_std'].append(np.std(code_vals))
                            freq_stats['phase_mean'].append(np.mean(phase_vals))
                            freq_stats['phase_std'].append(np.std(phase_vals))
                
                if freq_stats['satellites']:
                    system_stats[freq] = freq_stats
            
            bias_results[system_name] = system_stats
        
        self.update_progress(50)
        
        # 计算系统间偏差
        common_freqs = set(bias_results['BDS-2'].keys()) & set(bias_results['BDS-3'].keys())
        
        for freq in common_freqs:
            bds2_data = bias_results['BDS-2'][freq]
            bds3_data = bias_results['BDS-3'][freq]
            
            if bds2_data['satellites'] and bds3_data['satellites']:
                # 计算伪距偏差
                code_bias = np.mean(bds3_data['code_mean']) - np.mean(bds2_data['code_mean'])
                code_bias_std = np.sqrt(np.var(bds3_data['code_mean']) + np.var(bds2_data['code_mean']))
                
                # 计算相位偏差
                phase_bias = np.mean(bds3_data['phase_mean']) - np.mean(bds2_data['phase_mean'])
                phase_bias_std = np.sqrt(np.var(bds3_data['phase_mean']) + np.var(bds2_data['phase_mean']))
                
                bias_results['system_comparison'][freq] = {
                    'code_bias': code_bias,
                    'code_bias_std': code_bias_std,
                    'phase_bias': phase_bias,
                    'phase_bias_std': phase_bias_std,
                    'bds2_count': len(bds2_data['satellites']),
                    'bds3_count': len(bds3_data['satellites'])
                }
        
        self.update_progress(80)
        
        # 分析不同轨道类型的偏差
        for system_name in ['BDS-2', 'BDS-3']:
            orbit_bias = {}
            for orbit_type in ['GEO', 'IGSO', 'MEO']:
                orbit_sats = self.beidou_systems[system_name][orbit_type]
                orbit_stats = {}
                
                for freq in ['L1D', 'L1P', 'L2I', 'L5P']:
                    freq_data = []
                    for sat_id in orbit_sats:
                        if sat_id in self.observations_meters and freq in self.observations_meters[sat_id]:
                            sat_data = self.observations_meters[sat_id][freq]
                            code_vals = [v for v in sat_data['code'] if v is not None]
                            if code_vals:
                                freq_data.extend(code_vals)
                    
                    if freq_data:
                        orbit_stats[freq] = {
                            'mean': np.mean(freq_data),
                            'std': np.std(freq_data),
                            'count': len(freq_data)
                        }
                
                if orbit_stats:
                    orbit_bias[orbit_type] = orbit_stats
            
            bias_results['orbit_type_bias'][system_name] = orbit_bias
        
        self.update_progress(100)
        
        # 保存结果
        self.results['beidou_system_bias'] = bias_results
        
        # 生成分析报告
        self._generate_beidou_bias_report(bias_results)
        
        return bias_results

    def _generate_beidou_bias_report(self, bias_results):
        """生成北斗系统间偏差分析报告"""
        report_path = os.path.join(self.current_result_dir, 'beidou_system_bias_analysis.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("北斗二号与北斗三号系统间偏差分析报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 卫星数量统计
            f.write("1. 卫星数量统计\n")
            f.write("-" * 40 + "\n")
            bds2_count = sum(len(sats) for sats in bias_results['BDS-2'].values()) if bias_results['BDS-2'] else 0
            bds3_count = sum(len(sats) for sats in bias_results['BDS-3'].values()) if bias_results['BDS-3'] else 0
            f.write(f"北斗二号系统卫星数量: {bds2_count}\n")
            f.write(f"北斗三号系统卫星数量: {bds3_count}\n\n")
            
            # 系统间偏差
            f.write("2. 系统间偏差分析\n")
            f.write("-" * 40 + "\n")
            for freq, bias_data in bias_results['system_comparison'].items():
                f.write(f"频率 {freq}:\n")
                f.write(f"  伪距偏差: {bias_data['code_bias']:.4f} ± {bias_data['code_bias_std']:.4f} m\n")
                f.write(f"  相位偏差: {bias_data['phase_bias']:.4f} ± {bias_data['phase_bias_std']:.4f} m\n")
                f.write(f"  北斗二号卫星数: {bias_data['bds2_count']}\n")
                f.write(f"  北斗三号卫星数: {bias_data['bds3_count']}\n\n")
            
            # 轨道类型偏差
            f.write("3. 不同轨道类型偏差分析\n")
            f.write("-" * 40 + "\n")
            for system_name in ['BDS-2', 'BDS-3']:
                f.write(f"\n{system_name} 系统:\n")
                for orbit_type, orbit_data in bias_results['orbit_type_bias'].get(system_name, {}).items():
                    f.write(f"  {orbit_type} 轨道:\n")
                    for freq, freq_data in orbit_data.items():
                        f.write(f"    {freq}: 均值={freq_data['mean']:.4f}m, 标准差={freq_data['std']:.4f}m\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("分析完成时间: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 80 + "\n")

    # 剔除粗差保存新文件
    def remove_outliers_and_save(self, double_diffs, triple_errors):
        """
        基于伪距、相位、多普勒双差检测结果，修改RINEX文件中的异常观测值
        """
        # 各观测类型的最大阈值限制（使用用户设置的值）
        max_threshold_limit = self.max_threshold_limits

        # 初始化日志内容
        log_content = [
            "=" * 70 + "\n",
            "RINEX 粗差处理详细日志\n",
            "=" * 70 + "\n\n",
            f"观测值最大阈值限制: 伪距={max_threshold_limit['code']}m, 相位={max_threshold_limit['phase']}m, 多普勒={max_threshold_limit['doppler']}m/s\n\n"
        ]

        # 1. 整理每个卫星每个频率的阈值
        # 存储结构：{sat_id: {freq: {obs_type: threshold}}}
        satellite_freq_thresholds = defaultdict(lambda: defaultdict(dict))

        for sat_id, freq_data in triple_errors.items():
            for freq, errors in freq_data.items():
                for obs_type in ['code', 'phase', 'doppler']:
                    # 从triple_errors中获取已计算的三倍中误差
                    triple_sigma = errors.get(obs_type, {}).get('threshold', 0)

                    if triple_sigma <= 0:
                        # 无有效三倍中误差时使用最大阈值限制
                        threshold = max_threshold_limit[obs_type]
                        log_content.append(
                            f"卫星 {sat_id} 频率 {freq} {obs_type} 无有效三倍中误差，使用最大阈值: {threshold:.4f}m\n"
                        )
                    else:
                        # 确保阈值不超过最大限制
                        threshold = min(triple_sigma, max_threshold_limit[obs_type])
                        # 确保阈值为正数
                        threshold = max(threshold, 0.01)  # 避免阈值过小导致误判
                        log_content.append(
                            f"卫星 {sat_id} 频率 {freq} {obs_type} 使用三倍中误差作为阈值: "
                            f"计算值={triple_sigma:.4f}m, 应用阈值={threshold:.4f}m\n"
                        )
                    satellite_freq_thresholds[sat_id][freq][obs_type] = threshold

        # 2. 读取RINEX文件内容（优先读取基于伪距相位差值剔除后的文件）
        input_file = self.input_file_path
        # 检查是否存在基于伪距相位差值剔除后的文件
        results_dir = self.current_result_dir
        file_name = os.path.basename(self.input_file_path)
        cleaned_file_name = f"cleaned1-{file_name}"
        cleaned_file_path = os.path.join(results_dir, cleaned_file_name)

        if os.path.exists(cleaned_file_path):
            input_file = cleaned_file_path
            log_content.append(f"使用基于伪距相位差值剔除后的文件: {cleaned_file_name}\n")
        else:
            log_content.append(f"使用原始文件: {file_name}\n")

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 3. 解析历元时间戳
        epoch_timestamps = {}
        current_epoch = 0
        for line in lines:
            if line.startswith('>'):
                current_epoch += 1
                parts = line[1:].split()
                if len(parts) >= 6:
                    # 保留秒的小数部分精度，避免历元合并
                    year, month, day, hour, minute, second = parts[:6]
                    second_float = float(second)
                    # 保留原始秒数格式，不进行四舍五入
                    timestamp = f"{year} {month} {day} {hour} {minute} {second_float}"
                    epoch_timestamps[current_epoch] = timestamp
        log_content.append(f"成功解析 {len(epoch_timestamps)} 个历元时间戳\n\n")

        # 4. 解析观测类型（同时处理伪距、相位、多普勒）
        system_obs_info = {}  # {系统: {'obs_types': [], 'freq_to_indices': {频率: {观测类型: 字段索引}}}}
        for line in lines:
            if 'SYS / # / OBS TYPES' in line:
                system = line[0]
                obs_types = line.split()[2:]
                freq_to_indices = defaultdict(dict)  # {频率: {'code': 索引, 'phase': 索引, 'doppler': 索引}}

                for idx, obs in enumerate(obs_types):
                    if obs.startswith('C'):  # 伪距
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['code'] = idx
                    elif obs.startswith('L'):  # 相位
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['phase'] = idx
                    elif obs.startswith('D'):  # 多普勒
                        freq = f"L{obs[1:]}"
                        freq_to_indices[freq]['doppler'] = idx

                system_obs_info[system] = {
                    'obs_types': obs_types,
                    'freq_to_indices': freq_to_indices
                }
                log_content.append(f"解析系统 {system} 观测类型: {len(obs_types)} 种\n")

        # 5. 识别异常历元（使用每个卫星每个频率的阈值）
        outlier_epochs = defaultdict(lambda: defaultdict(list))  # {sat_id: {历元: [(观测类型, 频率)]}}
        outlier_details = defaultdict(list)  # {sat_id: [异常详情]}

        for sat_id, freq_data in double_diffs.items():
            for freq, dd_data in freq_data.items():
                # 检查三种观测值的双差超限情况
                for obs_type in ['code', 'phase', 'doppler']:
                    dd_key = f"dd_{obs_type}"
                    if dd_key not in dd_data:
                        continue  # 跳过无数据的观测类型

                    # 获取该卫星该频率的阈值（已考虑最大限制）
                    threshold = satellite_freq_thresholds[sat_id][freq].get(
                        obs_type, max_threshold_limit[obs_type]
                    )

                    # 检测超限值
                    valid_dd = [(i, d) for i, d in enumerate(dd_data[dd_key])
                                if d is not None and not np.isnan(d)]
                    for orig_idx, dd_value in valid_dd:
                        if abs(dd_value) > threshold:
                            epoch_idx = orig_idx + 2  # 双差结果对应原历元（滞后2个）
                            timestamp = epoch_timestamps.get(epoch_idx, f"未知时间戳(历元{epoch_idx})")
                            outlier_info = {
                                'obs_type': obs_type,
                                'freq': freq,
                                'dd_value': dd_value,
                                'threshold_used': threshold,
                                'epoch_idx': epoch_idx,
                                'timestamp': timestamp
                            }
                            outlier_details[sat_id].append(outlier_info)
                            outlier_epochs[sat_id][epoch_idx].append((obs_type, freq))

        # 6. 修改异常历元的观测值（同时处理三种观测类型）
        modified_count = defaultdict(int)  # {观测类型: 修改数量}
        modified_satellites = set()
        satellite_modify_details = []

        for sat_id, epoch_obs_map in outlier_epochs.items():
            sat_system = sat_id[0]
            sat_prn = sat_id[1:].zfill(2)
            system_info = system_obs_info.get(sat_system, {})
            freq_indices = system_info.get('freq_to_indices', {})  # 频率到字段索引的映射

            if not freq_indices:
                continue  # 跳过无观测类型信息的卫星

            satellite_modifications = []

            # 处理每个异常历元
            for epoch_idx, obs_freq_list in epoch_obs_map.items():
                timestamp = epoch_timestamps.get(epoch_idx, f"未知时间戳(历元{epoch_idx})")

                # 定位历元行
                epoch_start = -1
                for i, line in enumerate(lines):
                    if line.startswith('>') and timestamp in line:
                        epoch_start = i
                        break
                if epoch_start < 0:
                    continue

                # 定位该卫星在历元中的数据行
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

                # 修改异常观测值字段
                original_line = lines[sat_line_idx]
                modified_line = list(original_line)
                field_modified = False
                modified_fields = []

                for obs_type, freq in obs_freq_list:
                    # 定位该观测值在数据行中的位置
                    if freq not in freq_indices or obs_type not in freq_indices[freq]:
                        continue  # 跳过无索引的观测值
                    field_idx = freq_indices[freq][obs_type]
                    start_pos = 3 + field_idx * 16  # 3: 卫星标识长度
                    end_pos = start_pos + 16

                    if end_pos > len(modified_line):
                        continue  # 防止越界

                    # 清除该字段（设为空格）
                    original_field = original_line[start_pos:end_pos].strip()
                    if original_field:
                        modified_line[start_pos:end_pos] = ' ' * 16  # RINEX标准字段宽度为16字符
                        modified_count[obs_type] += 1
                        field_modified = True
                        modified_fields.append(f"{freq}({obs_type})")

                if field_modified:
                    lines[sat_line_idx] = ''.join(modified_line)
                    modified_satellites.add(sat_id)
                    satellite_modifications.append(
                        f"  历元 {epoch_idx} ({timestamp}): 已修改 {', '.join(modified_fields)}")

            if satellite_modifications:
                satellite_modify_details.append(f"卫星 {sat_id} 的修改详情:")
                satellite_modify_details.extend(satellite_modifications)
                satellite_modify_details.append("")

        # 7. 保存修改后的文件
        results_dir = self.current_result_dir
        file_name = os.path.basename(self.input_file_path)
        modified_file_name = f"cleaned2-{file_name}"
        output_path = os.path.join(results_dir, modified_file_name)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # 8. 生成详细日志
        total_modified = sum(modified_count.values())
        log_content.append("\n一、修改统计摘要\n")
        log_content.append("-" * 70 + "\n")
        log_content.append(f"总计修改卫星数: {len(modified_satellites)}\n")
        log_content.append(f"总计修改观测值: {total_modified}\n")
        log_content.append(
            f"修改分类: 伪距={modified_count['code']}, 相位={modified_count['phase']}, 多普勒={modified_count['doppler']}\n\n")

        # 添加异常历元详情到日志
        log_content.append("\n二、异常历元检测详情\n")
        log_content.append("-" * 70 + "\n")

        # 按系统组织卫星信息
        system_satellites = defaultdict(list)
        for sat_id in outlier_details.keys():
            system_satellites[sat_id[0]].append(sat_id)

        # 按系统顺序输出
        for system, satellites in system_satellites.items():
            log_content.append(f"卫星系统 {system}:\n")

            for sat_id in sorted(satellites):
                details = outlier_details[sat_id]
                log_content.append(f"  卫星 {sat_id} ({len(details)}个异常观测值):\n")

                # 按历元分组异常信息
                epoch_groups = defaultdict(list)
                for detail in details:
                    epoch_groups[detail['epoch_idx']].append(detail)

                for epoch_idx in sorted(epoch_groups.keys()):
                    details = epoch_groups[epoch_idx]
                    timestamp = details[0]['timestamp']
                    log_content.append(f"    历元 {epoch_idx} ({timestamp}):\n")

                    for detail in details:
                        log_content.append(f"      - {detail['obs_type']}@{detail['freq']}: "
                                           f"双差值={detail['dd_value']:.6f}m, "
                                           f"阈值={detail['threshold_used']:.6f}m, "
                                           f"状态={'已修改' if sat_id in modified_satellites else '未修改'}\n")
                log_content.append("\n")

        # 写入日志文件
        debug_file_name = "double_diffs_cleaning.log"
        debug_path = os.path.join(results_dir, debug_file_name)
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.writelines(log_content)

        # 控制台输出
        print(f"--成功解析 {len(epoch_timestamps)} 个历元时间戳")
        print(f"--检测到 {len(outlier_epochs)} 颗卫星存在异常历元")
        print(f"--总计修改 {total_modified} 个观测值，涉及 {len(modified_satellites)} 颗卫星")
        print(f"   - 伪距: {modified_count['code']}")
        print(f"   - 相位: {modified_count['phase']}")
        print(f"   - 多普勒: {modified_count['doppler']}")
        print(f"--剔除粗差后的文件保存至: {output_path}")
        print(f"--剔除粗差详细信息保存至: {debug_path}")

        return output_path

    def plot_raw_observations(self, sat_id: str, save=True) -> None:
        """绘制指定卫星的原始观测值图"""
        if sat_id not in self.observations_meters:
            print(f"错误: 未找到卫星 {sat_id} 的观测数据")
            return

        satellite_data = self.observations_meters[sat_id]
        valid_freqs = [freq for freq, data in satellite_data.items()
                       if len(data['code']) > 0 or len(data['phase']) > 0]

        if not valid_freqs:
            print(f"错误: 卫星 {sat_id} 没有有效的观测数据")
            return

        try:
            plt.figure(figsize=(12, 6))

            # 定义样式字典，为每个频率分配不同的颜色、线型和标记
            style_dict = {
                'L1C': {'color': 'blue', 'linestyle': '-', 'marker': 's', 'label_code': 'L1C 伪距',
                        'label_phase': 'L1C 相位', 'phase_marker': 'o'},
                'L1D': {'color': 'cyan', 'linestyle': '-', 'marker': '^', 'label_code': 'L1D 伪距',
                        'label_phase': 'L1D 相位', 'phase_marker': 'x'},
                'L1P': {'color': 'red', 'linestyle': '-', 'marker': 'D', 'label_code': 'L1P 伪距',
                        'label_phase': 'L1P 相位', 'phase_marker': '*'},
                'L2I': {'color': 'blue', 'linestyle': '-', 'marker': 'o', 'label_code': 'L2I 伪距',
                        'label_phase': 'L2I 相位', 'phase_marker': '^'},
                'L5Q': {'color': 'cyan', 'linestyle': '-', 'marker': '^', 'label_code': 'L5Q 伪距',
                        'label_phase': 'L5Q 相位', 'phase_marker': 's'},
                'L7Q': {'color': 'magenta', 'linestyle': '-', 'marker': '^', 'label_code': 'L7Q 伪距',
                        'label_phase': 'L7Q 相位', 'phase_marker': 'D'},
                'L5P': {'color': 'magenta', 'linestyle': '-', 'marker': 'D', 'label_code': 'L5P 伪距',
                        'label_phase': 'L5P 相位', 'phase_marker': '^'}
            }

            # 记录是否有绘制的线条（用于判断是否添加图例）
            has_plotted_lines = False

            # 绘制所有频率的伪距和相位
            for idx, freq in enumerate(valid_freqs):
                data = satellite_data[freq]
                times = data['times']
                code_values = data['code']
                phase_values = data['phase']

                if not times or (not code_values and not phase_values):
                    continue

                epochs = list(range(1, len(times) + 1))
                style = style_dict.get(freq, {'color': 'gray', 'linestyle': '-', 'marker': 'o', 'phase_marker': 'o'})

                # 计算调整常数（统计分析法）
                system = sat_id[0]
                wavelength = self.wavelengths[system].get(freq)
                if wavelength is None:
                    continue

                # 过滤掉None值
                valid_indices = []
                for i in range(len(code_values)):
                    if code_values[i] is not None and phase_values[i] is not None:
                        valid_indices.append(i)

                if len(valid_indices) < 10:
                    print(f"--警告: 卫星 {sat_id} 频率 {freq} 的有效数据点太少，无法计算调整常数")
                    continue

                # 创建仅包含有效值的数组
                valid_code_values = np.array([code_values[i] for i in valid_indices])
                valid_phase_values = np.array([phase_values[i] for i in valid_indices])

                # 计算调整常数
                differences = valid_code_values - valid_phase_values * wavelength
                adjustment_constant = np.mean(differences)

                # 调整载波相位值
                adjusted_phase_values = [None] * len(phase_values)
                for i in valid_indices:
                    adjusted_phase_values[i] = phase_values[i] * wavelength + adjustment_constant

                # ---------------------- 新增标记间隔控制逻辑 ----------------------
                # 生成每隔200历元的索引
                mark_indices = np.arange(0, len(epochs), 200)
                # ---------------------- 伪距绘图（控制标记显示） ----------------------
                plt.plot(
                    epochs, code_values,
                    linestyle=style['linestyle'],
                    color=style['color'],
                    label=style.get('label_code', f'{freq} 伪距 (m)'),
                    linewidth=1,  # 保留线宽
                    marker=style['marker'],  # 启用标记
                    markevery=mark_indices,  # 仅在指定索引显示标记
                    markersize=10,  # 适当增大标记尺寸
                    markeredgewidth=1  # 标记边框宽度
                )
                has_plotted_lines = True  # 标记有线条绘制
                # ---------------------- 相位绘图（控制标记显示） ----------------------
                plt.plot(
                    epochs, adjusted_phase_values,
                    linestyle=style['linestyle'],
                    color=style['color'],
                    label=style.get('label_phase', f'{freq} 相位 (m)'),
                    linewidth=1,
                    marker=style['phase_marker'],  # 相位单独标记
                    markevery=mark_indices,  # 相位也使用相同标记间隔
                    markersize=8,  # 相位标记稍小
                    markeredgewidth=1
                )

            # 只有在有绘制线条时才添加图例
            if has_plotted_lines:
                plt.legend(loc='upper right', fontsize=10)

            plt.xlabel('历元')
            plt.ylabel('观测值 (m)')
            plt.title(f'{sat_id} 原始观测值')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            if save:
                category_dir = os.path.join(self.current_result_dir, "raw_observations")
                os.makedirs(category_dir, exist_ok=True)
                filename = f"{sat_id}_raw_observations.png"
                full_path = os.path.join(category_dir, filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                print(f"--原始观测值图表已保存至: {full_path}")
            else:
                # 确保图表显示在最上层
                plt.show()

        except Exception as e:
            print(f"绘制原始观测值图表时出现错误: {str(e)}")
        finally:
            plt.close()  # 确保关闭图表，释放内存

    def plot_observable_derivatives(self, derivatives: Dict, sat_id: str, freq: str, save=True) -> None:
        """绘制指定卫星和频率的观测值一阶差分变化图"""
        if sat_id in derivatives and freq in derivatives[sat_id]:
            data = derivatives[sat_id][freq]

            system = sat_id[0]
            wavelength = self.wavelengths[system].get(freq)
            if wavelength is None:
                print(f"错误: 无法获取 {sat_id} 的 {freq} 频率波长")
                return

            try:
                plt.figure(figsize=(14, 12))

                # 第一部分：将伪距、相位一阶差分和多普勒观测值绘制在一个图中
                plt.subplot(2, 1, 1)

                # 找出所有三种观测值都有效的历元索引
                valid_indices = []
                for i in range(len(data['times'])):
                    if (data['pr_derivative'][i] is not None and
                            data['ph_derivative'][i] is not None and
                            data['doppler'][i] is not None):
                        valid_indices.append(i)

                # 使用相同的有效索引提取数据
                valid_pr_derivatives = [data['pr_derivative'][i] for i in valid_indices]
                valid_ph_derivatives = [data['ph_derivative'][i] for i in valid_indices]
                valid_doppler = [data['doppler'][i] for i in valid_indices]

                epochs = list(range(1, len(valid_indices) + 1))

                plt.plot(epochs, valid_pr_derivatives, 'b-', label='伪距一阶差分 (m/s)')
                plt.plot(epochs, valid_ph_derivatives, 'g-', label='相位一阶差分 (m/s)')
                plt.plot(epochs, valid_doppler, 'r-', label='多普勒观测值 (m/s)')

                plt.xlabel('历元')
                plt.ylabel('速度 (m/s)')
                plt.title(f'{sat_id} - {freq} 观测值一阶差分对比')
                plt.grid(True)
                plt.legend()

                # 第二部分：绘制多普勒观测值减去伪距和相位一阶差分后的值
                plt.subplot(2, 1, 2)

                # 计算 -λ·D - dP/dt
                doppler_minus_pr = [
                    valid_doppler[i] - valid_pr_derivatives[i]
                    for i in range(len(valid_indices))
                ]

                # 计算 -λ·D - dΦ/dt
                doppler_minus_ph = [
                    valid_doppler[i] - valid_ph_derivatives[i]
                    for i in range(len(valid_indices))
                ]

                plt.plot(epochs, doppler_minus_pr, 'm-', label='-λ·D - dP/dt')
                plt.plot(epochs, doppler_minus_ph, 'c-', label='-λ·D - dΦ/dt')

                plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                plt.xlabel('历元')
                plt.ylabel('差值 (m/s)')
                plt.title(f'{sat_id} - {freq} 多普勒与观测值一阶差分差值')
                plt.grid(True)
                plt.legend()

                plt.tight_layout()

                if save:
                    category_dir = os.path.join(self.current_result_dir, "derivatives")
                    os.makedirs(category_dir, exist_ok=True)
                    filename = f"{sat_id}_{freq}_derivatives_comparison.png"
                    full_path = os.path.join(category_dir, filename)
                    plt.savefig(full_path, dpi=300, bbox_inches='tight')
                    print(f"--一阶差分图表已保存至: {full_path}")
                else:
                    # 确保图表显示在最上层
                    plt.show()

            except Exception as e:
                print(f"绘制观测值一阶差分图表时出现错误: {str(e)}")
            finally:
                plt.close()  # 确保关闭图表，释放内存

    def plot_code_phase_raw_differences(self, differences: Dict, sat_id: str, freq: str, save=True) -> None:
        """绘制指定卫星和频率的伪距与载波相位原始差值图"""
        if sat_id in differences and freq in differences[sat_id]:
            data = differences[sat_id][freq]

            try:
                plt.figure(figsize=(12, 6))

                # 绘制原始差值
                valid_diffs = [d for d in data['code_phase_diff'] if d is not None]
                epochs = list(range(1, len(valid_diffs) + 1))

                plt.plot(epochs, valid_diffs, 'b-', label='伪距-相位差值')

                plt.xlabel('历元')
                plt.ylabel('差值 (m)')
                plt.title(f'{sat_id} - {freq} 伪距与载波相位原始差值')
                plt.grid(True)
                plt.legend()

                plt.tight_layout()

                if save:
                    category_dir = os.path.join(self.current_result_dir, "code_phase_diff_raw")
                    os.makedirs(category_dir, exist_ok=True)
                    filename = f"{sat_id}_{freq}_code_phase_raw_diff.png"
                    full_path = os.path.join(category_dir, filename)
                    plt.savefig(full_path, dpi=300, bbox_inches='tight')
                    print(f"--码相原始差值图表已保存至: {full_path}")
                else:
                    # 确保图表显示在最上层
                    plt.show()

            except Exception as e:
                print(f"绘制码相原始差值图表时出现错误: {str(e)}")
            finally:
                plt.close()  # 确保关闭图表，释放内存

    def plot_code_phase_differences(self, differences: Dict, sat_id: str, freq: str, save=True) -> None:
        """绘制指定卫星和频率的伪距与载波相位差值变化图（横坐标为历元序号）"""
        if sat_id in differences and freq in differences[sat_id]:
            data = differences[sat_id][freq]

            try:
                # 绘制差值变化率
                valid_changes = [c for c in data['diff_changes'] if c is not None]
                epochs = list(range(1, len(valid_changes) + 1))  # 生成历元序号列表

                plt.figure(figsize=(12, 6))
                plt.plot(epochs, valid_changes, 'b-', label='伪距-相位差变化率')

                # # 添加阈值线（10米）
                # plt.axhline(y=10.0, color='r', linestyle='--', alpha=0.5, label='阈值线 (10米)')

                plt.xlabel('历元')
                plt.ylabel('差值变化 (m/历元)')
                plt.title(f'{sat_id} - {freq} 伪距相位差变化率')
                plt.grid(True)
                plt.legend()

                plt.tight_layout()

                if save:
                    category_dir = os.path.join(self.current_result_dir, "code_phase_diffs")
                    os.makedirs(category_dir, exist_ok=True)
                    filename = f"{sat_id}_{freq}_code_phase_diff_changes.png"
                    full_path = os.path.join(category_dir, filename)
                    plt.savefig(full_path, dpi=300, bbox_inches='tight')
                    print(f"--码相差值变化图表已保存至: {full_path}")
                else:
                    # 确保图表显示在最上层
                    plt.show()

            except Exception as e:
                print(f"绘制码相差值变化图表时出现错误: {str(e)}")
            finally:
                plt.close()  # 确保关闭图表，释放内存

    def plot_phase_prediction_errors(self, errors: Dict, sat_id: str, freq: str, save=True) -> None:
        """绘制指定卫星和频率的载波相位预测误差图（横坐标为历元）"""
        if sat_id in errors and freq in errors[sat_id]:
            data = errors[sat_id][freq]

            try:
                plt.figure(figsize=(12, 8))

                # 绘制预测误差
                valid_errors = [e for e in data['prediction_error'] if e is not None]
                epochs = list(range(1, len(valid_errors) + 1))  # 生成历元序号

                plt.plot(epochs, valid_errors, 'b-', label='预测误差')
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.xlabel('历元')
                plt.ylabel('预测误差 (m)')
                plt.title(f'{sat_id} - {freq} 载波相位预测误差 (米)')
                plt.grid(True)
                plt.legend()

                plt.tight_layout()

                if save:
                    category_dir = os.path.join(self.current_result_dir, "phase_pred_errors")
                    os.makedirs(category_dir, exist_ok=True)
                    filename = f"{sat_id}_{freq}_phase_pred_error.png"
                    full_path = os.path.join(category_dir, filename)
                    plt.savefig(full_path, dpi=300, bbox_inches='tight')
                    print(f"--相位预测图表已保存至: {full_path}")
                else:
                    # 确保图表显示在最上层
                    plt.show()

            except Exception as e:
                print(f"绘制相位预测误差图表时出现错误: {str(e)}")
            finally:
                plt.close()  # 确保关闭图表，释放内存

    def plot_double_differences(self, double_diffs, triple_errors, sat_id, freq, save=True):
        """绘制伪距、相位和多普勒双差结果并显示各自的阈值"""
        if sat_id not in double_diffs or freq not in double_diffs[sat_id]:
            print(f"错误: 卫星 {sat_id} 或频率 {freq} 无双差数据")
            return

        try:
            data = double_diffs[sat_id][freq]
            errors = triple_errors[sat_id][freq]
            epochs = list(range(1, len(data['dd_code']) + 1))  # 历元序号

            # 创建子图
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f'{sat_id} - {freq} 历元间双差', fontsize=14)

            # 1. 伪距双差
            ax1.plot(epochs, data['dd_code'], 'b-', label='伪距双差 (m)')
            ax1.axhline(y=errors['code']['threshold'], color='r', linestyle='--', alpha=0.7,
                        label=f'阈值: {errors["code"]["threshold"]:.2f} m')
            ax1.axhline(y=-errors['code']['threshold'], color='r', linestyle='--', alpha=0.7)
            ax1.set_ylabel('伪距双差 (m)')
            ax1.grid(True)
            ax1.legend()

            # 2. 载波相位双差
            ax2.plot(epochs, data['dd_phase'], 'g-', label='载波相位双差 (m)')
            ax2.axhline(y=errors['phase']['threshold'], color='r', linestyle='--', alpha=0.7,
                        label=f'阈值: {errors["phase"]["threshold"]:.2f} m')
            ax2.axhline(y=-errors['phase']['threshold'], color='r', linestyle='--', alpha=0.7)
            ax2.set_ylabel('载波相位双差 (m)')
            ax2.grid(True)
            ax2.legend()

            # 3. 多普勒双差
            ax3.plot(epochs, data['dd_doppler'], 'm-', label='多普勒双差 (m/s)')
            ax3.axhline(y=errors['doppler']['threshold'], color='r', linestyle='--', alpha=0.7,
                        label=f'阈值: {errors["doppler"]["threshold"]:.2f} m/s')
            ax3.axhline(y=-errors['doppler']['threshold'], color='r', linestyle='--', alpha=0.7)
            ax3.set_xlabel('历元')
            ax3.set_ylabel('多普勒双差 (m/s)')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend(loc='upper right')

            # 调整子图间距
            plt.tight_layout()

            if save:
                category_dir = os.path.join(self.current_result_dir, "double_differences")
                os.makedirs(category_dir, exist_ok=True)
                filename = f"{sat_id}_{freq}_double_differences.png"
                full_path = os.path.join(category_dir, filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                print(f"--双差图表已保存至: {full_path}")
            else:
                # 确保图表显示在最上层
                plt.show()

        except Exception as e:
            print(f"绘制双差图表时出现错误: {str(e)}")
        finally:
            plt.close()  # 确保关闭图表，释放内存

    def plot_beidou_system_bias_analysis(self, bias_results, save=True):
        """绘制北斗系统间偏差分析图表"""
        try:
            # 创建图表目录
            category_dir = os.path.join(self.current_result_dir, "beidou_bias_analysis")
            os.makedirs(category_dir, exist_ok=True)
            
            # 1. 系统间偏差对比图
            if bias_results['system_comparison']:
                plt.figure(figsize=(15, 10))
                
                # 伪距偏差对比
                plt.subplot(2, 2, 1)
                freqs = list(bias_results['system_comparison'].keys())
                code_biases = [bias_results['system_comparison'][f]['code_bias'] for f in freqs]
                code_bias_stds = [bias_results['system_comparison'][f]['code_bias_std'] for f in freqs]
                
                x_pos = np.arange(len(freqs))
                plt.bar(x_pos, code_biases, yerr=code_bias_stds, capsize=5, 
                       color='skyblue', alpha=0.7, label='伪距偏差')
                plt.xlabel('频率')
                plt.ylabel('偏差 (m)')
                plt.title('北斗二号与北斗三号系统间伪距偏差')
                plt.xticks(x_pos, freqs)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # 相位偏差对比
                plt.subplot(2, 2, 2)
                phase_biases = [bias_results['system_comparison'][f]['phase_bias'] for f in freqs]
                phase_bias_stds = [bias_results['system_comparison'][f]['phase_bias_std'] for f in freqs]
                
                plt.bar(x_pos, phase_biases, yerr=phase_bias_stds, capsize=5, 
                       color='lightcoral', alpha=0.7, label='相位偏差')
                plt.xlabel('频率')
                plt.ylabel('偏差 (m)')
                plt.title('北斗二号与北斗三号系统间相位偏差')
                plt.xticks(x_pos, freqs)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # 卫星数量对比
                plt.subplot(2, 2, 3)
                bds2_counts = [bias_results['system_comparison'][f]['bds2_count'] for f in freqs]
                bds3_counts = [bias_results['system_comparison'][f]['bds3_count'] for f in freqs]
                
                x_pos = np.arange(len(freqs))
                width = 0.35
                plt.bar(x_pos - width/2, bds2_counts, width, label='北斗二号', color='blue', alpha=0.7)
                plt.bar(x_pos + width/2, bds3_counts, width, label='北斗三号', color='red', alpha=0.7)
                plt.xlabel('频率')
                plt.ylabel('卫星数量')
                plt.title('各频率北斗二号与北斗三号卫星数量对比')
                plt.xticks(x_pos, freqs)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 偏差统计分布
                plt.subplot(2, 2, 4)
                all_code_biases = []
                all_phase_biases = []
                for freq_data in bias_results['system_comparison'].values():
                    all_code_biases.append(freq_data['code_bias'])
                    all_phase_biases.append(freq_data['phase_bias'])
                
                plt.hist(all_code_biases, bins=10, alpha=0.7, label='伪距偏差', color='skyblue')
                plt.hist(all_phase_biases, bins=10, alpha=0.7, label='相位偏差', color='lightcoral')
                plt.xlabel('偏差值 (m)')
                plt.ylabel('频次')
                plt.title('系统间偏差分布')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save:
                    filename = "beidou_system_bias_overview.png"
                    full_path = os.path.join(category_dir, filename)
                    plt.savefig(full_path, dpi=300, bbox_inches='tight')
                    print(f"--北斗系统间偏差总览图已保存至: {full_path}")
                
                plt.close()
            
            # 2. 轨道类型偏差分析图
            if bias_results['orbit_type_bias']:
                plt.figure(figsize=(16, 12))
                
                orbit_types = ['GEO', 'IGSO', 'MEO']
                systems = ['BDS-2', 'BDS-3']
                freqs = ['L1D', 'L1P', 'L2I', 'L5P']
                
                for i, freq in enumerate(freqs):
                    plt.subplot(2, 2, i+1)
                    
                    x_pos = np.arange(len(orbit_types))
                    width = 0.35
                    
                    for j, system in enumerate(systems):
                        means = []
                        for orbit_type in orbit_types:
                            orbit_data = bias_results['orbit_type_bias'].get(system, {}).get(orbit_type, {})
                            freq_data = orbit_data.get(freq, {})
                            means.append(freq_data.get('mean', 0) if freq_data else 0)
                        
                        plt.bar(x_pos + j*width, means, width, 
                               label=system, alpha=0.7)
                    
                    plt.xlabel('轨道类型')
                    plt.ylabel('观测值均值 (m)')
                    plt.title(f'{freq} 频率不同轨道类型观测值对比')
                    plt.xticks(x_pos + width/2, orbit_types)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save:
                    filename = "beidou_orbit_type_bias_analysis.png"
                    full_path = os.path.join(category_dir, filename)
                    plt.savefig(full_path, dpi=300, bbox_inches='tight')
                    print(f"--北斗轨道类型偏差分析图已保存至: {full_path}")
                
                plt.close()
            
            # 3. 系统性能对比雷达图
            if bias_results['BDS-2'] and bias_results['BDS-3']:
                self._plot_beidou_radar_chart(bias_results, category_dir, save)
                
        except Exception as e:
            print(f"绘制北斗系统间偏差分析图表时出现错误: {str(e)}")

    def _plot_beidou_radar_chart(self, bias_results, category_dir, save):
        """绘制北斗系统性能对比雷达图"""
        try:
            # 计算各系统的性能指标
            systems = ['BDS-2', 'BDS-3']
            metrics = ['code_stability', 'phase_stability', 'frequency_coverage', 'satellite_count']
            
            # 计算性能指标
            performance_data = {}
            for system in systems:
                system_data = bias_results.get(system, {})
                if not system_data:
                    continue
                
                # 伪距稳定性（标准差越小越好，取倒数）
                code_stability = 0
                code_stds = []
                for freq_data in system_data.values():
                    if freq_data['satellites']:
                        code_stds.extend(freq_data['code_std'])
                if code_stds:
                    code_stability = 1 / (np.mean(code_stds) + 1e-6)
                
                # 相位稳定性
                phase_stability = 0
                phase_stds = []
                for freq_data in system_data.values():
                    if freq_data['satellites']:
                        phase_stds.extend(freq_data['phase_std'])
                if phase_stds:
                    phase_stability = 1 / (np.mean(phase_stds) + 0.001)
                
                # 频率覆盖度
                frequency_coverage = len(system_data) / 4.0  # 4个主要频率
                
                # 卫星数量（归一化）
                total_sats = sum(len(freq_data['satellites']) for freq_data in system_data.values())
                satellite_count = min(total_sats / 20.0, 1.0)  # 假设20颗卫星为满分
                
                performance_data[system] = [code_stability, phase_stability, frequency_coverage, satellite_count]
            
            if len(performance_data) >= 2:
                # 绘制雷达图
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
                
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                angles += angles[:1]  # 闭合图形
                
                colors = ['blue', 'red']
                for i, (system, values) in enumerate(performance_data.items()):
                    values += values[:1]  # 闭合图形
                    ax.plot(angles, values, 'o-', linewidth=2, label=system, color=colors[i])
                    ax.fill(angles, values, alpha=0.25, color=colors[i])
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(['伪距稳定性', '相位稳定性', '频率覆盖度', '卫星数量'])
                ax.set_ylim(0, 1)
                ax.set_title('北斗二号与北斗三号系统性能对比雷达图', size=15, pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                ax.grid(True)
                
                if save:
                    filename = "beidou_performance_radar_chart.png"
                    full_path = os.path.join(category_dir, filename)
                    plt.savefig(full_path, dpi=300, bbox_inches='tight')
                    print(f"--北斗系统性能对比雷达图已保存至: {full_path}")
                
                plt.close()
                
        except Exception as e:
            print(f"绘制北斗系统性能对比雷达图时出现错误: {str(e)}")

    def save_report(self) -> None:
        """保存分析报告到当前文件结果目录"""
        self.start_stage(7, "保存分析报告", 100)

        report = self.generate_report()
        filename = "analysis_report.txt"
        full_path = os.path.join(self.current_result_dir, filename)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(report)

        self.update_progress(100)
        print(f"--报告已保存至: {full_path}")

    def save_all_plots(self) -> None:
        """保存所有分析结果的图表"""
        import gc
        import matplotlib.pyplot as plt

        self.start_stage(8, "保存分析图表", 100)
        print("--开始保存各图表...")
        # 创建结果目录
        os.makedirs(self.current_result_dir, exist_ok=True)
        # 创建类别子文件夹
        for category in self.plot_categories:
            os.makedirs(os.path.join(self.current_result_dir, category), exist_ok=True)

        try:
            # 保存原始观测值图表
            if self.observations_meters:
                total_sats = len(self.observations_meters)
                processed_sats = 0

                for sat_id in self.observations_meters:
                    has_data = False
                    for freq, data in self.observations_meters[sat_id].items():
                        if (len([c for c in data['code'] if c is not None]) > 0 or
                                len([p for p in data['phase'] if p is not None]) > 0):
                            has_data = True
                            break
                    if has_data:
                        self.plot_raw_observations(sat_id, save=True)
                        processed_sats += 1
                        self.update_progress(int(processed_sats / total_sats * 16))
                    else:
                        print(f"--跳过保存: {sat_id} 原始观测数据不足")

                # 清理内存
                plt.close('all')
                gc.collect()
                print("--原始观测值图表保存完成，内存已清理")

            # 保存观测值一阶差分图表
            if 'observable_derivatives' in self.results:
                total_sats = len(self.results['observable_derivatives'])
                processed_sats = 0

                for sat_id in self.results['observable_derivatives']:
                    has_data = False
                    for freq, data in self.results['observable_derivatives'][sat_id].items():
                        valid_pr = [d for d in data['pr_derivative'] if d is not None]
                        valid_ph = [d for d in data['ph_derivative'] if d is not None]
                        valid_doppler = [d for d in data['doppler'] if d is not None]
                        if len(valid_pr) > 0 and len(valid_ph) > 0 and len(valid_doppler) > 0:
                            has_data = True
                            break
                    if has_data:
                        for freq in self.results['observable_derivatives'][sat_id]:
                            self.plot_observable_derivatives(
                                self.results['observable_derivatives'],
                                sat_id, freq, save=True
                            )
                        processed_sats += 1
                        self.update_progress(16 + int(processed_sats / total_sats * 16))
                    else:
                        print(f"--跳过保存: {sat_id} 观测值一阶差分数据不足")

                # 清理内存
                plt.close('all')
                gc.collect()
                print("--观测值一阶差分图表保存完成，内存已清理")

            # 保存伪距相位原始差值图表
            if 'code_phase_diffs' in self.results:
                total_sats = len(self.results['code_phase_diffs'])
                processed_sats = 0

                for sat_id in self.results['code_phase_diffs']:
                    has_data = False
                    for freq, data in self.results['code_phase_diffs'][sat_id].items():
                        valid_diffs = [d for d in data['code_phase_diff'] if d is not None]
                        if len(valid_diffs) > 0:
                            has_data = True
                            break
                    if has_data:
                        for freq in self.results['code_phase_diffs'][sat_id]:
                            # 调用新的绘图函数
                            self.plot_code_phase_raw_differences(
                                self.results['code_phase_diffs'],
                                sat_id, freq, save=True
                            )
                        processed_sats += 1
                        self.update_progress(32 + int(processed_sats / total_sats * 16))
                    else:
                        print(f"--跳过保存: {sat_id} 伪距相位原始差值数据不足")

                # 清理内存
                plt.close('all')
                gc.collect()
                print("--伪距相位原始差值图表保存完成，内存已清理")

            # 保存伪距相位差值变化图表
            if 'code_phase_diffs' in self.results:
                total_sats = len(self.results['code_phase_diffs'])
                processed_sats = 0

                for sat_id in self.results['code_phase_diffs']:
                    has_data = False
                    for freq, data in self.results['code_phase_diffs'][sat_id].items():
                        valid_diffs = [d for d in data['code_phase_diff'] if d is not None]
                        if len(valid_diffs) > 0:
                            has_data = True
                            break
                    if has_data:
                        for freq in self.results['code_phase_diffs'][sat_id]:
                            self.plot_code_phase_differences(
                                self.results['code_phase_diffs'],
                                sat_id, freq, save=True
                            )
                        processed_sats += 1
                        self.update_progress(47 + int(processed_sats / total_sats * 16))
                    else:
                        print(f"--跳过保存: {sat_id} 伪距相位差值数据不足")

                # 清理内存
                plt.close('all')
                gc.collect()
                print("--伪距相位差值变化图表保存完成，内存已清理")

            # 保存相位预测误差图表
            if 'phase_prediction_errors' in self.results:
                total_sats = len(self.results['phase_prediction_errors'])
                processed_sats = 0

                for sat_id in self.results['phase_prediction_errors']:
                    has_data = False
                    for freq, data in self.results['phase_prediction_errors'][sat_id].items():
                        valid_errors = [e for e in data['prediction_error'] if e is not None]
                        if len(valid_errors) > 0:
                            has_data = True
                            break
                    if has_data:
                        for freq in self.results['phase_prediction_errors'][sat_id]:
                            self.plot_phase_prediction_errors(
                                self.results['phase_prediction_errors'],
                                sat_id, freq, save=True
                            )
                        processed_sats += 1
                        self.update_progress(63 + int(processed_sats / total_sats * 16))
                    else:
                        print(f"--跳过保存: {sat_id} 相位预测误差数据不足")

                # 清理内存
                plt.close('all')
                gc.collect()
                print("--相位预测误差图表保存完成，内存已清理")

            # 保存双差图表
            if 'double_differences' in self.results and 'triple_median_errors' in self.results:
                double_diffs = self.results['double_differences']
                triple_errors = self.results['triple_median_errors']
                total_sats = len(double_diffs)
                processed_sats = 0

                for sat_id in double_diffs:
                    has_data = any(len(freq_data['dd_code']) > 0 for freq_data in double_diffs[sat_id].values())
                    if has_data:
                        for freq in double_diffs[sat_id]:
                            self.plot_double_differences(double_diffs, triple_errors, sat_id, freq, save=True)
                        processed_sats += 1
                        self.update_progress(79 + int(processed_sats / total_sats * 21))
                    else:
                        print(f"--跳过保存: {sat_id} 双差数据不足")

                # 清理内存
                plt.close('all')
                gc.collect()
                print("--双差图表保存完成，内存已清理")

            # 保存北斗系统间偏差分析图表
            if 'beidou_system_bias' in self.results and self.results['beidou_system_bias']:
                print("--开始保存北斗系统间偏差分析图表...")
                self.plot_beidou_system_bias_analysis(
                    self.results['beidou_system_bias'], save=True
                )
                self.update_progress(95)
                
                # 清理内存
                plt.close('all')
                gc.collect()
                print("--北斗系统间偏差分析图表保存完成，内存已清理")

            # 完成进度
            self.update_progress(100)
            print("所有图表保存完成")

        except Exception as e:
            print(f"保存图表过程中出现错误: {str(e)}")
            # 即使出错也要清理内存
            plt.close('all')
            gc.collect()
            raise e

    def generate_report(self) -> str:
        """生成检测结果报告"""
        report = "=== GNSS观测数据分析报告 ===\n\n"

        # 观测值一阶差分变化与多普勒值不一致性检测
        report += "1. 观测值一阶差分变化与多普勒值不一致性检测:\n"
        inconsistent_derivatives_doppler = {}
        for sat_id, freq_data in self.results['observable_derivatives'].items():
            for freq, data in freq_data.items():
                valid_indices = []
                for i in range(len(data['times'])):
                    if (data['pr_derivative'][i] is not None and
                            data['ph_derivative'][i] is not None and
                            data['doppler'][i] is not None):
                        valid_indices.append(i)
                valid_pr_derivatives = [data['pr_derivative'][i] for i in valid_indices]
                valid_ph_derivatives = [data['ph_derivative'][i] * self.wavelengths[sat_id[0]].get(freq) for i in
                                        valid_indices]
                valid_doppler = [-data['doppler'][i] * self.wavelengths[sat_id[0]].get(freq) for i in valid_indices]

                doppler_minus_pr = [valid_doppler[i] - valid_pr_derivatives[i] for i in range(len(valid_indices))]
                doppler_minus_ph = [valid_doppler[i] - valid_ph_derivatives[i] for i in range(len(valid_indices))]

                max_diff_pr = max(doppler_minus_pr) if doppler_minus_pr else 0
                max_diff_ph = max(doppler_minus_ph) if doppler_minus_ph else 0
                threshold_pr = 10  # 可根据实际情况调整阈值，单位m/s
                threshold_ph = 10  # 可根据实际情况调整阈值，单位m/s
                if max_diff_pr > threshold_pr or max_diff_ph > threshold_ph:
                    inconsistent_derivatives_doppler.setdefault(sat_id, {})[freq] = {
                        'max_diff_pr': max_diff_pr,
                        'max_diff_ph': max_diff_ph,
                        'threshold_pr': threshold_pr,
                        'threshold_ph': threshold_ph
                    }

        if inconsistent_derivatives_doppler:
            report += f"  发现 {len(inconsistent_derivatives_doppler)} 颗卫星在部分频率上存在观测值一阶差分变化与多普勒值不一致性:\n"
            for sat_id, freq_info in inconsistent_derivatives_doppler.items():
                for freq, info in freq_info.items():
                    report += f"    - 卫星 {sat_id}，频率 {freq}:\n"
                    report += f"      - 多普勒值减去伪距一阶差分之最大差值: {info['max_diff_pr']:.2f} m/s (阈值: {info['threshold_pr']} m/s)\n"
                    report += f"      - 多普勒值减去相位一阶差分之最大差值: {info['max_diff_ph']:.2f} m/s (阈值: {info['threshold_ph']} m/s)\n"
                    report += "      - 潜在影响: 可能表明卫星信号存在异常，或在信号传播过程中受到干扰，导致观测值与理论计算值偏差较大，影响定位精度。\n"
        else:
            report += "  未发现观测值一阶差分变化与多普勒值不一致性问题\n"
        report += "\n"

        # 伪距相位差值不一致检测
        report += "2. 伪距相位差值不一致检测:\n"
        # 设置阈值(单位:米)
        threshold = 10.0  # 差值变化阈值（米）
        inconsistent_code_phase = []
        for sat, freq_data in self.results['code_phase_differences'].items():
            for freq, data in freq_data.items():
                # 使用diff_changes而非code_phase_diff
                changes = [c for c in data['diff_changes'] if c is not None]
                if len(changes) < 3:  # 数据点太少，跳过
                    continue

                # 计算变化率的统计特征
                mean_change = np.mean(changes)
                max_change = np.max(changes)
                std_change = np.std(changes)

                # 数据过滤信息
                original_epochs = data.get('original_epochs', 0)
                filtered_epochs = data.get('filtered_epochs', 0)
                stagnant_removed = data.get('stagnant_epochs_removed', 0)
                missing_obs = data.get('missing_epochs', 0)

                # 判断是否异常
                if max_change > threshold and std_change > 1.0:
                    inconsistent_code_phase.append({
                        'sat': sat,
                        'freq': freq,
                        'max_change': max_change,
                        'mean_change': mean_change,
                        'std_change': std_change,
                        'threshold': threshold,
                        'original_epochs': original_epochs,
                        'filtered_epochs': filtered_epochs,
                        'stagnant_removed': stagnant_removed,
                        'missing_obs': missing_obs
                    })

        if inconsistent_code_phase:
            report += f"  发现 {len(inconsistent_code_phase)} 颗卫星存在伪距相位差值波动异常:\n"
            for item in inconsistent_code_phase:
                report += f"    - 卫星 {item['sat']}，频率 {item['freq']}:\n"
                report += f"      - 差值变化统计: 最大值={item['max_change']:.3f} m, 均值={item['mean_change']:.3f} m, 标准差={item['std_change']:.3f} m\n"
                report += f"      - 阈值: {item['threshold']} m\n"
                report += f"      - 数据过滤: 原始历元={item['original_epochs']}, 缺失观测值={item['missing_obs']}, 剔除停滞历元={item['stagnant_removed']}, 有效历元={item['filtered_epochs']}\n"

                # 分析差值异常可能原因
                if item['missing_obs'] > item['filtered_epochs'] * 0.3:  # 缺失率超过30%
                    report += "      - 潜在原因: 大量观测值缺失导致计算不稳定，可能是信号遮挡或接收机问题\n"
                elif item['max_change'] > 1000:  # 异常大的变化值
                    report += "      - 潜在原因: 整周模糊度变化、信号失锁或接收机钟跳变\n"
                else:
                    report += "      - 潜在原因: 多径效应、接收机噪声或卫星钟差异常\n"
                report += "      - 潜在影响: 可能导致定位结果偏差，影响导航和定位服务的可靠性\n"
        else:
            report += "  未发现伪距相位差值波动异常（所有波动均在10米以内）\n"
        report += "\n"

        # 相位停滞统计
        report += "\n3. 相位停滞统计:\n"

        if 'phase_stagnation' in self.results:
            stagnant_sats = set()
            total_stagnant_freqs = 0
            for sat_id, freq_data in self.results['phase_stagnation'].items():
                for freq, stats in freq_data.items():
                    if stats.get('is_stagnant', False):
                        stagnant_sats.add(sat_id)
                        total_stagnant_freqs += 1

            report += f"  检测到 {len(stagnant_sats)} 颗卫星存在相位停滞现象，涉及 {total_stagnant_freqs} 个频率通道\n"
            if stagnant_sats:
                report += "  停滞卫星列表:\n"
                for sat in stagnant_sats:
                    sat_freqs = [freq for freq, stats in self.results['phase_stagnation'][sat].items()
                                 if stats.get('is_stagnant', False)]
                    report += f"    - {sat}: 频率 {', '.join(sat_freqs)}\n"
            else:
                report += "  未检测到相位停滞现象\n"
        else:
            report += "  未执行相位停滞检测\n"

        # 粗差检测：历元间双差超限检测
        report += "\n4. 历元间双差超限检测:\n"
        double_diff_outliers = self.results.get('triple_median_errors', {})  # 从results中获取双差误差结果
        total_outliers = 0

        for sat_id, freq_data in double_diff_outliers.items():
            for freq, errors in freq_data.items():
                code_outliers = errors['code']['outliers']  # 伪距双差超限历元索引
                phase_outliers = errors['phase']['outliers']  # 载波相位双差超限历元索引
                doppler_outliers = errors['doppler']['outliers']  # 多普勒双差超限历元索引

                if code_outliers or phase_outliers or doppler_outliers:
                    total_outliers += 1
                    report += f"  - 卫星 {sat_id}, 频率 {freq}:\n"
                    if code_outliers:
                        report += f"    ▶ 伪距双差超限历元: {code_outliers}（阈值: {errors['code']['threshold']:.2f} m）\n"
                    if phase_outliers:
                        report += f"    ▶ 载波相位双差超限历元: {phase_outliers}（阈值: {errors['phase']['threshold']:.2f} m）\n"
                    if doppler_outliers:
                        report += f"    ▶ 多普勒双差超限历元: {doppler_outliers}（阈值: {errors['doppler']['threshold']:.2f} m/s）\n"
                    report += "    - 潜在原因: 多径效应、卫星信号失锁或接收机钟差异常\n"
                    report += "    - 潜在影响: 可能导致载波相位差分（RTK）解算失败或定位精度骤降\n"

        if total_outliers == 0:
            report += "  未检测到双差超限值（所有双差值均在3×MAD阈值内）\n"
        else:
            report += f"  共检测到 {total_outliers} 个频率通道存在双差超限值，建议结合图表分析具体历元\n"

        # 北斗系统间偏差分析
        if 'beidou_system_bias' in self.results and self.results['beidou_system_bias']:
            report += "\n4. 北斗系统间偏差分析:\n"
            bias_results = self.results['beidou_system_bias']
            
            # 卫星数量统计
            bds2_count = sum(len(sats) for sats in bias_results['BDS-2'].values()) if bias_results['BDS-2'] else 0
            bds3_count = sum(len(sats) for sats in bias_results['BDS-3'].values()) if bias_results['BDS-3'] else 0
            report += f"  北斗二号系统卫星数量: {bds2_count}\n"
            report += f"  北斗三号系统卫星数量: {bds3_count}\n\n"
            
            # 系统间偏差
            if bias_results['system_comparison']:
                report += "  系统间偏差分析结果:\n"
                for freq, bias_data in bias_results['system_comparison'].items():
                    report += f"    - 频率 {freq}:\n"
                    report += f"      ▶ 伪距偏差: {bias_data['code_bias']:.4f} ± {bias_data['code_bias_std']:.4f} m\n"
                    report += f"      ▶ 相位偏差: {bias_data['phase_bias']:.4f} ± {bias_data['phase_bias_std']:.4f} m\n"
                    report += f"      ▶ 北斗二号卫星数: {bias_data['bds2_count']}\n"
                    report += f"      ▶ 北斗三号卫星数: {bias_data['bds3_count']}\n"
                    
                    # 偏差影响分析
                    if abs(bias_data['code_bias']) > 1.0:
                        report += f"      ▶ 伪距偏差较大，可能影响单点定位精度\n"
                    if abs(bias_data['phase_bias']) > 0.1:
                        report += f"      ▶ 相位偏差较大，可能影响高精度定位\n"
                    report += "\n"
            
            # 轨道类型偏差
            if bias_results['orbit_type_bias']:
                report += "  不同轨道类型偏差分析:\n"
                for system_name in ['BDS-2', 'BDS-3']:
                    report += f"    - {system_name} 系统:\n"
                    for orbit_type, orbit_data in bias_results['orbit_type_bias'].get(system_name, {}).items():
                        report += f"      ▶ {orbit_type} 轨道:\n"
                        for freq, freq_data in orbit_data.items():
                            report += f"        {freq}: 均值={freq_data['mean']:.4f}m, 标准差={freq_data['std']:.4f}m\n"
            
            # 总结和建议
            report += "\n  北斗系统间偏差分析总结:\n"
            if bds2_count > 0 and bds3_count > 0:
                report += "    - 同时接收到北斗二号和北斗三号信号，可进行系统间偏差分析\n"
                report += "    - 建议在定位解算中考虑系统间偏差的影响\n"
                report += "    - 可通过双频观测值组合减少系统间偏差的影响\n"
            else:
                report += "    - 仅接收到单一北斗系统信号，无法进行系统间偏差分析\n"
                report += "    - 建议在开阔环境下进行观测以获得更多北斗卫星信号\n"

        return report

    def reset_analysis(self):
        """重置分析器状态，准备处理新文件"""
        self.observations_meters = {}
        self.results = {
            'code_carrier_inconsistency': {},
            'observation_inconsistency': {},
            'phase_stagnation': {},
            'observable_derivatives': {},
            'code_phase_differences': {},
            'phase_prediction_errors': {},
            'beidou_system_bias': {}  # 北斗系统间偏差分析结果
        }
        # 重置当前文件相关路径
        self.input_file_path = None
        self.current_result_dir = None


# 主函数
def center_window(window, width=None, height=None):
    """将窗口居中显示在屏幕上"""
    window.update_idletasks()
    
    # 获取窗口尺寸
    if width is None or height is None:
        window_width = window.winfo_width()
        window_height = window.winfo_height()
    else:
        window_width = width
        window_height = height
    
    # 获取屏幕尺寸
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    
    # 计算居中位置
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    # 设置窗口位置
    window.geometry(f"{window_width}x{window_height}+{x}+{y}")

def main():
    root = tk.Tk()
    root.title("GNSS数据分析器")
    root.geometry("800x600")
    root.resizable(True, True)
    
    # 居中显示主窗口
    center_window(root, 800, 600)

    # 创建主菜单栏
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # 预处理菜单
    cleaning_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="预处理", menu=cleaning_menu)
    cleaning_menu.add_command(label="执行预处理", command=lambda: show_cleaning_window(root))

    # 图表菜单
    charts_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="图表", menu=charts_menu)
    charts_menu.add_command(label="选择图表类型", command=lambda: show_charts_window(root))

    # 报告菜单
    report_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="报告", menu=report_menu)
    report_menu.add_command(label="生成分析报告", command=lambda: show_report_window(root))

    # 主界面
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 欢迎信息
    welcome_label = ttk.Label(main_frame, text="ANDROID RINEX数据分析器",
                              font=("Microsoft YaHei", 16, "bold"))
    welcome_label.pack(pady=20)

    # 功能说明
    desc_frame = ttk.LabelFrame(main_frame, text="功能说明", padding="20")
    desc_frame.pack(fill=tk.X, pady=20)

    ttk.Label(desc_frame, text="• 预处理：基于历元间双差和CMC变化阈值剔除异常观测值，码相不一致性建模和校正",
              font=("Microsoft YaHei", 10)).pack(anchor=tk.W, pady=2)
    ttk.Label(desc_frame, text="• 图表：生成各类分析图表，支持单独保存和批量保存",
              font=("Microsoft YaHei", 10)).pack(anchor=tk.W, pady=2)
    ttk.Label(desc_frame, text="• 报告：生成完整的分析报告，包含北斗系统间偏差分析",
              font=("Microsoft YaHei", 10)).pack(anchor=tk.W, pady=2)

    # 快速操作按钮
    quick_frame = ttk.LabelFrame(main_frame, text="快速操作", padding="20")
    quick_frame.pack(fill=tk.X, pady=20)

    quick_btn_frame = ttk.Frame(quick_frame)
    quick_btn_frame.pack()

    ttk.Button(quick_btn_frame, text="开始预处理",
               command=lambda: show_cleaning_window(root)).pack(side=tk.LEFT, padx=10)
    ttk.Button(quick_btn_frame, text="查看图表",
               command=lambda: show_charts_window(root)).pack(side=tk.LEFT, padx=10)
    ttk.Button(quick_btn_frame, text="生成报告",
               command=lambda: show_report_window(root)).pack(side=tk.LEFT, padx=10)

    def on_closing():
        """程序关闭时的清理函数"""
        try:
            # 关闭所有Matplotlib图表窗口
            plt.close('all')
            # 清理Matplotlib资源
            import matplotlib
            matplotlib.pyplot.close('all')
            # 销毁主窗口
            root.destroy()
        except Exception as e:
            print(f"关闭程序时出现错误: {str(e)}")
            # 强制退出
            root.quit()

    # 绑定关闭事件
    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()


def show_cleaning_window(parent):
    """显示数据预处理功能窗口"""
    cleaning_window = tk.Toplevel(parent)
    cleaning_window.title("数据预处理")
    cleaning_window.geometry("700x600")
    cleaning_window.resizable(True, True)
    cleaning_window.transient(parent)
    cleaning_window.grab_set()
    
    # 居中显示窗口
    center_window(cleaning_window, 700, 600)

    # 主框架
    main_frame = ttk.Frame(cleaning_window, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 文件选择
    file_frame = ttk.LabelFrame(main_frame, text="选择数据文件", padding="10")
    file_frame.pack(fill=tk.X, pady=10)

    # 手机RINEX文件选择
    phone_frame = ttk.Frame(file_frame)
    phone_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(phone_frame, text="手机RINEX文件:").pack(side=tk.LEFT)
    phone_file_var = tk.StringVar()
    phone_file_entry = ttk.Entry(phone_frame, textvariable=phone_file_var, width=50)
    phone_file_entry.pack(side=tk.LEFT, padx=(10, 10), fill=tk.X, expand=True)

    def select_phone_file():
        file_types = [
            ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
            ("All Files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="选择手机RINEX观测文件",
            filetypes=file_types
        )
        if filename:
            phone_file_var.set(filename)

    ttk.Button(phone_frame, text="浏览", command=select_phone_file).pack(side=tk.RIGHT)
    
    # 接收机RINEX文件选择（可选，用于码相不一致性建模）
    receiver_frame = ttk.Frame(file_frame)
    receiver_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(receiver_frame, text="接收机RINEX文件(可选):").pack(side=tk.LEFT)
    receiver_file_var = tk.StringVar()
    receiver_file_entry = ttk.Entry(receiver_frame, textvariable=receiver_file_var, width=50)
    receiver_file_entry.pack(side=tk.LEFT, padx=(10, 10), fill=tk.X, expand=True)

    def select_receiver_file():
        file_types = [
            ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
            ("All Files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="选择接收机RINEX观测文件",
            filetypes=file_types
        )
        if filename:
            receiver_file_var.set(filename)

    ttk.Button(receiver_frame, text="浏览", command=select_receiver_file).pack(side=tk.RIGHT)
    
    # 兼容性：保持原有的file_var变量
    file_var = phone_file_var

    # 参数设置
    param_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="10")
    param_frame.pack(fill=tk.X, pady=10)

    # 历元间双差最大阈值设置
    double_diff_frame = ttk.Frame(param_frame)
    double_diff_frame.pack(fill=tk.X, pady=5)

    ttk.Label(double_diff_frame, text="历元间双差最大阈值:").pack(side=tk.LEFT)
    
    # 伪距阈值
    ttk.Label(double_diff_frame, text="伪距(米):").pack(side=tk.LEFT, padx=(10, 0))
    code_threshold_var = tk.DoubleVar(value=10.0)
    code_threshold_entry = ttk.Entry(double_diff_frame, textvariable=code_threshold_var, width=8)
    code_threshold_entry.pack(side=tk.LEFT, padx=(5, 0))
    
    # 相位阈值
    ttk.Label(double_diff_frame, text="相位(米):").pack(side=tk.LEFT, padx=(10, 0))
    phase_threshold_var = tk.DoubleVar(value=3.0)
    phase_threshold_entry = ttk.Entry(double_diff_frame, textvariable=phase_threshold_var, width=8)
    phase_threshold_entry.pack(side=tk.LEFT, padx=(5, 0))
    
    # 多普勒阈值
    ttk.Label(double_diff_frame, text="多普勒(米/秒):").pack(side=tk.LEFT, padx=(10, 0))
    doppler_threshold_var = tk.DoubleVar(value=5.0)
    doppler_threshold_entry = ttk.Entry(double_diff_frame, textvariable=doppler_threshold_var, width=8)
    doppler_threshold_entry.pack(side=tk.LEFT, padx=(5, 0))

    # CMC变化阈值设置
    threshold_frame = ttk.Frame(param_frame)
    threshold_frame.pack(fill=tk.X, pady=5)

    ttk.Label(threshold_frame, text="CMC变化阈值(米):").pack(side=tk.LEFT)
    threshold_var = tk.DoubleVar(value=5.0)
    threshold_entry = ttk.Entry(threshold_frame, textvariable=threshold_var, width=10)
    threshold_entry.pack(side=tk.LEFT, padx=(10, 0))

    # R方阈值和CV值阈值设置（同一行）
    threshold_row_frame = ttk.Frame(param_frame)
    threshold_row_frame.pack(fill=tk.X, pady=5)

    # R方阈值
    ttk.Label(threshold_row_frame, text="R方阈值:").pack(side=tk.LEFT)
    r_squared_var = tk.DoubleVar(value=0.5)
    r_squared_entry = ttk.Entry(threshold_row_frame, textvariable=r_squared_var, width=10)
    r_squared_entry.pack(side=tk.LEFT, padx=(10, 0))

    # CV值阈值
    ttk.Label(threshold_row_frame, text="CV阈值:").pack(side=tk.LEFT, padx=(20, 0))
    cv_threshold_var = tk.DoubleVar(value=0.5)
    cv_threshold_entry = ttk.Entry(threshold_row_frame, textvariable=cv_threshold_var, width=10)
    cv_threshold_entry.pack(side=tk.LEFT, padx=(10, 0))
    ttk.Label(threshold_row_frame, text="(默认: 0.5)").pack(side=tk.LEFT, padx=(5, 0))

    # 手机独有卫星分析设置
    phone_only_frame = ttk.Frame(param_frame)
    phone_only_frame.pack(fill=tk.X, pady=5)

    phone_only_var = tk.BooleanVar(value=False)
    phone_only_checkbox = ttk.Checkbutton(phone_only_frame, text="启用手机独有卫星分析", 
                                         variable=phone_only_var)
    phone_only_checkbox.pack(side=tk.LEFT)

    ttk.Label(phone_only_frame, text="(检测手机独有卫星的码相不一致性)").pack(side=tk.LEFT, padx=(10, 0))

    # 进度显示
    progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
    progress_frame.pack(fill=tk.X, pady=10)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL,
                                   variable=progress_var, mode='determinate')
    progress_bar.pack(fill=tk.X, pady=5)

    status_var = tk.StringVar(value="等待开始...")
    status_label = ttk.Label(progress_frame, textvariable=status_var)
    status_label.pack()

    # 操作按钮
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)

    def start_cleaning():
        if not file_var.get():
            tk.messagebox.showerror("错误", "请先选择数据文件")
            return

        # 禁用按钮
        start_btn.config(state='disabled')
        select_btn.config(state='disabled')

        def cleaning_process():
            try:
                analyzer = GNSSAnalyzer()

                # 设置文件路径
                file_path = file_var.get()
                analyzer.input_file_path = file_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(file_path), "analysis_results")
                
                # 设置R方阈值
                analyzer.r_squared_threshold = r_squared_var.get()
                
                # 设置CV值阈值
                analyzer.cv_threshold = cv_threshold_var.get()
                
                # 设置历元间双差最大阈值
                analyzer.max_threshold_limits = {
                    'code': code_threshold_var.get(),
                    'phase': phase_threshold_var.get(),
                    'doppler': doppler_threshold_var.get()
                }
                
                # 设置手机独有卫星分析
                analyzer.enable_phone_only_analysis = phone_only_var.get()

                # 更新状态
                status_var.set("正在读取RINEX文件...")
                progress_var.set(10)
                cleaning_window.update_idletasks()

                # 读取文件
                data = analyzer.read_rinex_obs(file_path)

                # 计算伪距相位差值
                status_var.set("正在计算伪距相位差值...")
                progress_var.set(30)
                cleaning_window.update_idletasks()

                code_phase_diffs = analyzer.calculate_code_phase_differences(data)

                # 第一阶段剔除
                status_var.set("正在执行第一阶段剔除...")
                progress_var.set(50)
                cleaning_window.update_idletasks()

                cleaned_file_path = analyzer.remove_code_phase_outliers(data, threshold_var.get())

                # 计算历元间双差
                status_var.set("正在计算历元间双差...")
                progress_var.set(70)
                cleaning_window.update_idletasks()

                double_diffs = analyzer.calculate_epoch_double_differences()
                triple_errors = analyzer.calculate_triple_median_error(double_diffs)

                # 第二阶段剔除
                status_var.set("正在执行第二阶段剔除...")
                progress_var.set(70)
                cleaning_window.update_idletasks()

                analyzer.remove_outliers_and_save(double_diffs, triple_errors)

                # 码相不一致性建模和校正（如果提供了接收机文件）
                if receiver_file_var.get():
                    status_var.set("正在进行码相不一致性建模和校正...")
                    progress_var.set(85)
                    cleaning_window.update_idletasks()
                    
                    try:
                        # 执行码相不一致性建模和校正
                        cci_results = analyzer.perform_code_phase_inconsistency_modeling(
                            receiver_rinex_path=receiver_file_var.get()
                        )
                    except Exception as e:
                        print(f"码相不一致性建模和校正失败: {e}")
                        # 继续执行其他步骤，不中断整个流程

                # 北斗系统间偏差分析
                status_var.set("正在分析北斗系统间偏差...")
                progress_var.set(90)
                cleaning_window.update_idletasks()

                beidou_bias_results = analyzer.analyze_beidou_system_bias()

                # 完成
                status_var.set("处理完成!")
                progress_var.set(100)
                cleaning_window.update_idletasks()

                # 显示结果
                phone_file_name = os.path.basename(file_path)
                phone_file_name_no_ext = os.path.splitext(phone_file_name)[0]
                phone_result_dir = os.path.join("results", phone_file_name_no_ext)
                message = f"数据预处理完成！\n结果保存在：{phone_result_dir}"
                if receiver_file_var.get() and 'cci_results' in locals():
                    # 显示码相不一致性建模和校正的实际文件路径
                    corrected_file_path = cci_results.get('corrected_rinex_path', '')
                    if corrected_file_path:
                        message += f"\n\n码相不一致性建模和校正已完成\n校正后的RINEX文件：{corrected_file_path}"
                    else:
                        message += "\n\n码相不一致性建模和校正已完成"
                tk.messagebox.showinfo("完成", message)

            except Exception as e:
                tk.messagebox.showerror("错误", f"处理过程中出现错误：\n{str(e)}")
                status_var.set("处理失败")
            finally:
                # 恢复按钮
                start_btn.config(state='normal')
                select_btn.config(state='normal')

        # 在新线程中执行
        import threading
        thread = threading.Thread(target=cleaning_process)
        thread.daemon = True
        thread.start()

    start_btn = ttk.Button(button_frame, text="开始预处理", command=start_cleaning)
    start_btn.pack(side=tk.LEFT, padx=10)

    select_btn = ttk.Button(button_frame, text="选择文件", command=select_phone_file)
    select_btn.pack(side=tk.LEFT, padx=10)

    def close_cleaning_window():
        """关闭剔除窗口时的清理函数"""
        try:
            # 关闭所有Matplotlib图表窗口
            plt.close('all')
        except Exception as e:
            print(f"关闭剔除窗口时清理图表出错: {str(e)}")
        finally:
            cleaning_window.destroy()

    ttk.Button(button_frame, text="关闭",
               command=close_cleaning_window).pack(side=tk.LEFT, padx=10)


def show_charts_window(parent):
    """显示图表功能窗口"""
    charts_window = tk.Toplevel(parent)
    charts_window.title("图表生成")
    charts_window.geometry("700x800")
    charts_window.resizable(True, True)
    charts_window.transient(parent)
    charts_window.grab_set()
    
    # 居中显示窗口
    center_window(charts_window, 700, 800)

    # 主框架
    main_frame = ttk.Frame(charts_window, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 文件选择
    file_frame = ttk.LabelFrame(main_frame, text="选择数据文件", padding="10")
    file_frame.pack(fill=tk.X, pady=10)

    file_var = tk.StringVar()
    file_entry = ttk.Entry(file_frame, textvariable=file_var, width=50)
    file_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)

    def select_file():
        file_types = [
            ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
            ("All Files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="选择RINEX观测文件",
            filetypes=file_types
        )
        if filename:
            file_var.set(filename)
            # 自动加载预定义的卫星和频率信息
            load_satellite_info()

    ttk.Button(file_frame, text="浏览", command=select_file).pack(side=tk.RIGHT)

    # 图表类型选择
    chart_frame = ttk.LabelFrame(main_frame, text="图表类型", padding="10")
    chart_frame.pack(fill=tk.X, pady=10)

    chart_types = [
        ("原始观测值", "raw_observations"),
        ("观测值一阶差分", "derivatives"),
        ("伪距相位差值之差", "code_phase_diffs"),
        ("伪距相位原始差值", "code_phase_diff_raw"),
        ("相位预测误差", "phase_pred_errors"),
        ("历元间双差", "double_differences"),
        ("北斗系统间偏差分析", "beidou_system_bias"),
        ("接收机CMC", "receiver_cmc")
    ]

    chart_var = tk.StringVar(value="raw_observations")
    for text, value in chart_types:
        ttk.Radiobutton(chart_frame, text=text, variable=chart_var,
                        value=value).pack(anchor=tk.W, pady=2)

    # 接收机RINEX文件选择（用于接收机CMC）
    rx_file_frame = ttk.LabelFrame(main_frame, text="选择接收机RINEX文件(仅用于接收机CMC)", padding="10")
    rx_file_frame.pack(fill=tk.X, pady=10)

    rx_file_var = tk.StringVar()
    rx_file_entry = ttk.Entry(rx_file_frame, textvariable=rx_file_var, width=50)
    rx_file_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)

    def select_rx_file():
        file_types = [
            ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
            ("All Files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="选择接收机RINEX观测文件",
            filetypes=file_types
        )
        if filename:
            rx_file_var.set(filename)
            # 自动加载接收机RINEX的卫星与频率到下拉框
            try:
                load_receiver_satellite_info()
            except Exception as e:
                messagebox.showwarning("警告", f"加载接收机RINEX信息失败:\n{str(e)}")

    ttk.Button(rx_file_frame, text="浏览", command=select_rx_file).pack(side=tk.RIGHT)

    # 卫星系统、PRN和频率选择
    sat_frame = ttk.LabelFrame(main_frame, text="卫星系统、PRN和频率选择", padding="10")
    sat_frame.pack(fill=tk.X, pady=10)

    sat_frame_inner = ttk.Frame(sat_frame)
    sat_frame_inner.pack(fill=tk.X)

    ttk.Label(sat_frame_inner, text="卫星系统:").pack(side=tk.LEFT)
    sat_system_var = tk.StringVar()
    sat_system_combo = ttk.Combobox(sat_frame_inner, textvariable=sat_system_var, width=15)
    sat_system_combo.pack(side=tk.LEFT, padx=(10, 20))

    ttk.Label(sat_frame_inner, text="卫星PRN:").pack(side=tk.LEFT)
    sat_prn_var = tk.StringVar()
    sat_prn_combo = ttk.Combobox(sat_frame_inner, textvariable=sat_prn_var, width=15)
    sat_prn_combo.pack(side=tk.LEFT, padx=(10, 20))

    ttk.Label(sat_frame_inner, text="频率:").pack(side=tk.LEFT)
    freq_var = tk.StringVar()
    freq_combo = ttk.Combobox(sat_frame_inner, textvariable=freq_var, width=15)
    freq_combo.pack(side=tk.LEFT, padx=(10, 0))

    # 保存选项已移除 - 图表窗口自带保存功能，操作更方便

    # 进度显示
    progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
    progress_frame.pack(fill=tk.X, pady=10)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL,
                                   variable=progress_var, mode='determinate')
    progress_bar.pack(fill=tk.X, pady=5)

    status_var = tk.StringVar(value="等待开始...")
    status_label = ttk.Label(progress_frame, textvariable=status_var)
    status_label.pack()

    # 操作按钮
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)

    def load_satellite_info():
        """从输入文件中加载实际的卫星和频率信息"""
        try:
            if not file_var.get():
                return

            # 创建分析器并读取文件
            analyzer = GNSSAnalyzer()
            file_path = file_var.get()
            analyzer.input_file_path = file_path

            # 更新状态
            status_var.set("正在读取文件...")
            charts_window.update_idletasks()

            # 读取文件
            data = analyzer.read_rinex_obs(file_path)

            # 从实际数据中获取卫星系统
            if hasattr(analyzer, 'observations_meters') and analyzer.observations_meters:
                # 获取所有可用的卫星PRN
                available_satellites = list(analyzer.observations_meters.keys())

                # 从PRN中提取卫星系统
                satellite_systems = set()
                for sat in available_satellites:
                    if sat and len(sat) > 0:
                        satellite_systems.add(sat[0])  # 第一个字符是卫星系统标识

                satellite_systems = list(satellite_systems)

                # 设置卫星系统列表
                sat_system_combo['values'] = satellite_systems
                if satellite_systems:
                    sat_system_var.set(satellite_systems[0])  # 默认选择第一个卫星系统

                # 设置卫星PRN列表（从实际数据中获取）
                sat_prn_combo['values'] = available_satellites
                if available_satellites:
                    sat_prn_var.set(available_satellites[0])  # 默认选择第一个PRN

                # 根据选择的卫星PRN设置对应的频率列表
                if available_satellites:
                    selected_prn = available_satellites[0]
                    if selected_prn in analyzer.observations_meters:
                        frequencies = list(analyzer.observations_meters[selected_prn].keys())
                        freq_combo['values'] = frequencies
                        if frequencies:
                            freq_var.set(frequencies[0])  # 默认选择第一个频率

                status_var.set("文件读取完成")
            else:
                messagebox.showwarning("警告", "文件中未找到观测数据")
                status_var.set("未找到数据")

        except Exception as e:
            messagebox.showerror("错误", f"加载卫星和频率信息失败：\n{str(e)}")
            status_var.set("加载失败")

    def load_receiver_satellite_info():
        """从接收机RINEX文件加载卫星与频率信息，填充PRN与频率下拉框"""
        try:
            if not rx_file_var.get():
                return
            analyzer = GNSSAnalyzer()
            rx_path = rx_file_var.get()
            analyzer.read_receiver_rinex_obs(rx_path)

            if analyzer.receiver_observations:
                available_satellites = list(analyzer.receiver_observations.keys())

                # 推断卫星系统集合
                satellite_systems = sorted({sid[0] for sid in available_satellites if sid})
                sat_system_combo['values'] = satellite_systems
                if satellite_systems:
                    sat_system_var.set(satellite_systems[0])

                # 所有卫星列表
                sat_prn_combo['values'] = available_satellites
                if available_satellites:
                    sat_prn_var.set(available_satellites[0])

                # 对应频率列表
                first = sat_prn_var.get()
                if first in analyzer.receiver_observations:
                    freqs = list(analyzer.receiver_observations[first].keys())
                    freq_combo['values'] = freqs
                    if freqs:
                        freq_var.set(freqs[0])
                status_var.set("接收机文件加载完成")
            else:
                messagebox.showwarning("警告", "接收机RINEX中未找到观测数据")
                status_var.set("未找到数据")
        except Exception as e:
            messagebox.showerror("错误", f"加载接收机RINEX信息失败：\n{str(e)}")
            status_var.set("加载失败")

    def get_prn_list_for_system(system):
        """根据卫星系统获取对应的PRN列表"""
        prn_mapping = {
            'G': [f'G{i:02d}' for i in range(1, 33)],  # G01-G32
            'R': [f'R{i:02d}' for i in range(2, 23)],  # R02-R22
            'E': [f'E{i:02d}' for i in range(2, 37)],  # E02-E36
            'C': [f'C{i:02d}' for i in range(1, 63)],  # C01-C62
            'J': [f'J{i:02d}' for i in range(2, 8)]  # J02-J07
        }
        return prn_mapping.get(system, [])

    def on_satellite_system_change(*args):
        """卫星系统改变时更新PRN和频率列表"""
        if sat_system_var.get():
            try:
                # 获取当前选择的卫星系统
                selected_system = sat_system_var.get()

                # 从实际数据中筛选属于该系统的卫星PRN
                # 需要重新读取文件来获取数据
                if rx_file_var.get():
                    # 优先处理接收机RINEX
                    analyzer = GNSSAnalyzer()
                    analyzer.read_receiver_rinex_obs(rx_file_var.get())
                    if analyzer.receiver_observations:
                        available_satellites = list(analyzer.receiver_observations.keys())
                        system_satellites = [sat for sat in available_satellites if sat.startswith(selected_system)]

                        # 更新PRN列表
                        sat_prn_combo['values'] = system_satellites
                        if system_satellites:
                            sat_prn_var.set(system_satellites[0])

                            # 更新频率列表
                            selected_prn = system_satellites[0]
                            if selected_prn in analyzer.receiver_observations:
                                frequencies = list(analyzer.receiver_observations[selected_prn].keys())
                                freq_combo['values'] = frequencies
                                if frequencies:
                                    freq_var.set(frequencies[0])
                        return

                if file_var.get():
                    analyzer = GNSSAnalyzer()
                    file_path = file_var.get()
                    analyzer.input_file_path = file_path
                    data = analyzer.read_rinex_obs(file_path)

                    if hasattr(analyzer, 'observations_meters') and analyzer.observations_meters:
                        available_satellites = list(analyzer.observations_meters.keys())
                        system_satellites = [sat for sat in available_satellites if sat.startswith(selected_system)]

                        # 更新PRN列表
                        sat_prn_combo['values'] = system_satellites
                        if system_satellites:
                            sat_prn_var.set(system_satellites[0])  # 选择第一个PRN

                            # 更新频率列表
                            selected_prn = system_satellites[0]
                            if selected_prn in analyzer.observations_meters:
                                frequencies = list(analyzer.observations_meters[selected_prn].keys())
                                freq_combo['values'] = frequencies
                                if frequencies:
                                    freq_var.set(frequencies[0])  # 选择第一个频率
            except Exception as e:
                print(f"更新PRN和频率列表时出错: {e}")

    def on_satellite_prn_change(*args):
        """卫星PRN改变时更新频率列表"""
        if sat_prn_var.get():
            try:
                selected_prn = sat_prn_var.get()

                # 从实际数据中获取该卫星的频率列表
                # 需要重新读取文件来获取数据
                if rx_file_var.get():
                    analyzer = GNSSAnalyzer()
                    analyzer.read_receiver_rinex_obs(rx_file_var.get())
                    if analyzer.receiver_observations:
                        if selected_prn in analyzer.receiver_observations:
                            frequencies = list(analyzer.receiver_observations[selected_prn].keys())
                            freq_combo['values'] = frequencies
                            if frequencies:
                                freq_var.set(frequencies[0])
                        return

                if file_var.get():
                    analyzer = GNSSAnalyzer()
                    file_path = file_var.get()
                    analyzer.input_file_path = file_path
                    data = analyzer.read_rinex_obs(file_path)

                    if hasattr(analyzer, 'observations_meters') and analyzer.observations_meters:
                        if selected_prn in analyzer.observations_meters:
                            frequencies = list(analyzer.observations_meters[selected_prn].keys())
                            freq_combo['values'] = frequencies
                            if frequencies:
                                freq_var.set(frequencies[0])  # 选择第一个频率
            except Exception as e:
                print(f"更新频率列表时出错: {e}")

    # 绑定卫星系统选择变化事件
    sat_system_var.trace('w', on_satellite_system_change)

    # 绑定卫星PRN选择变化事件
    sat_prn_var.trace('w', on_satellite_prn_change)

    def generate_chart():
        """生成选中的图表"""
        selected = chart_var.get()
        # 按图表类型判断所需输入文件
        if selected == "receiver_cmc":
            if not rx_file_var.get():
                messagebox.showerror("错误", "请先选择接收机RINEX文件")
                return
        else:
            if not file_var.get():
                messagebox.showerror("错误", "请先选择数据文件")
                return

        # 禁用按钮
        generate_btn.config(state='disabled')

        def chart_process():
            try:
                analyzer = GNSSAnalyzer()

                # 设置文件路径
                file_path = file_var.get()
                analyzer.input_file_path = file_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(file_path), "analysis_results")

                # 更新状态
                status_var.set("正在读取RINEX文件...")
                progress_var.set(20)
                charts_window.update_idletasks()

                # 读取文件（接收机CMC不需要手机文件解析）
                data = None
                if selected != "receiver_cmc":
                    data = analyzer.read_rinex_obs(file_path)

                # 检查选择的卫星PRN和频率是否在数据中存在（接收机CMC走接收机数据容器）
                sat_prn = sat_prn_var.get()
                freq = freq_var.get()

                if selected != "receiver_cmc":
                    if sat_prn not in analyzer.observations_meters:
                        messagebox.showerror("错误", f"在输入文件中未找到卫星 {sat_prn} 的观测数据")
                        status_var.set("生成失败")
                        return
                    if freq not in analyzer.observations_meters[sat_prn]:
                        messagebox.showerror("错误", f"卫星 {sat_prn} 在输入文件中未找到频率 {freq} 的观测数据")
                        status_var.set("生成失败")
                        return
                    obs_data = analyzer.observations_meters[sat_prn][freq]
                    if not obs_data or len(obs_data) < 2:
                        messagebox.showerror("错误", f"卫星 {sat_prn} 频率 {freq} 的观测数据不足，至少需要2个历元的数据")
                        status_var.set("生成失败")
                        return
                    valid_obs = [obs for obs in obs_data if obs is not None]
                    if len(valid_obs) < 2:
                        messagebox.showerror("错误", f"卫星 {sat_prn} 频率 {freq} 的有效观测数据不足，至少需要2个有效观测值")
                        status_var.set("生成失败")
                        return
                else:
                    # 接收机CMC路径：读取接收机并立即dump调试
                    analyzer.read_receiver_rinex_obs(rx_file_var.get())
                    analyzer.dump_receiver_observations_debug()

                # 根据图表类型生成相应的图表
                chart_type = chart_var.get()

                status_var.set("正在生成图表...")
                progress_var.set(60)
                charts_window.update_idletasks()

                # 使用after方法在主线程中执行绘图操作
                def plot_on_main_thread():
                    try:
                        # 直接调用相应的绘图函数 - 总是显示图表，不自动保存
                        if chart_type == "raw_observations":
                            analyzer.plot_raw_observations(sat_prn, False)
                        elif chart_type == "derivatives":
                            # 计算观测值一阶差分
                            derivatives = analyzer.calculate_observable_derivatives(data)
                            analyzer.plot_observable_derivatives(derivatives, sat_prn, freq, False)
                        elif chart_type == "code_phase_diffs":
                            # 计算伪距相位差值之差
                            code_phase_diffs = analyzer.calculate_code_phase_differences(data)
                            analyzer.plot_code_phase_differences(code_phase_diffs, sat_prn, freq, False)
                        elif chart_type == "code_phase_diff_raw":
                            # 计算伪距相位原始差值
                            code_phase_diffs = analyzer.calculate_code_phase_differences(data)
                            analyzer.plot_code_phase_raw_differences(code_phase_diffs, sat_prn, freq, False)
                        elif chart_type == "phase_pred_errors":
                            # 计算相位预测误差
                            phase_pred_errors = analyzer.calculate_phase_prediction_errors(data)
                            analyzer.plot_phase_prediction_errors(phase_pred_errors, sat_prn, freq, False)
                        elif chart_type == "double_differences":
                            # 计算历元间双差
                            double_diffs = analyzer.calculate_epoch_double_differences()
                            triple_errors = analyzer.calculate_triple_median_error(double_diffs)
                            analyzer.plot_double_differences(double_diffs, triple_errors, sat_prn, freq, False)
                        elif chart_type == "receiver_cmc":
                            # 读取接收机RINEX并绘制CMC
                            if not rx_file_var.get():
                                messagebox.showerror("错误", "请先选择接收机RINEX文件")
                                return
                            analyzer.read_receiver_rinex_obs(rx_file_var.get())
                            cmc_results = analyzer.calculate_receiver_cmc()
                            if sat_prn not in cmc_results or freq not in cmc_results.get(sat_prn, {}):
                                messagebox.showerror("错误", f"接收机CMC无数据: {sat_prn} {freq}\n请确认RINEX包含所选卫星与频率")
                                return
                            # 轻量级绘制
                            vals = cmc_results[sat_prn][freq]['cmc_m']
                            epochs = list(range(1, len(vals)+1))
                            plt.figure(figsize=(12,6))
                            plt.plot(epochs, vals, 'b-', label='CMC (m)')
                            plt.axhline(0, color='k', linestyle='--', alpha=0.4)
                            plt.xlabel('历元'); plt.ylabel('CMC (m)'); plt.title(f'{sat_prn}-{freq} 接收机CMC')
                            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
                        elif chart_type == "beidou_system_bias":
                            # 分析北斗系统间偏差
                            beidou_bias_results = analyzer.analyze_beidou_system_bias()
                            analyzer.plot_beidou_system_bias_analysis(beidou_bias_results, save=False)

                        # 完成
                        status_var.set("图表生成完成!")
                        progress_var.set(100)
                        charts_window.update_idletasks()
                    except Exception as e:
                        messagebox.showerror("错误", f"生成图表过程中出现错误：\n{str(e)}")
                        status_var.set("生成失败")
                    finally:
                        # 恢复按钮
                        generate_btn.config(state='normal')

                # 在主线程中执行绘图操作
                charts_window.after(100, plot_on_main_thread)

            except Exception as e:
                messagebox.showerror("错误", f"准备图表数据过程中出现错误：\n{str(e)}")
                status_var.set("生成失败")
                # 恢复按钮
                generate_btn.config(state='normal')

        # 在新线程中执行数据准备
        import threading
        thread = threading.Thread(target=chart_process)
        thread.daemon = True
        thread.start()

    generate_btn = ttk.Button(button_frame, text="生成图表", command=generate_chart)
    generate_btn.pack(side=tk.LEFT, padx=10)

    ttk.Button(button_frame, text="批量保存所有图表",
               command=lambda: batch_save_all_charts()).pack(side=tk.LEFT, padx=10)

    def batch_save_all_charts():
        """批量保存所有类型的图表"""
        # receiver_cmc 允许没有手机文件
        if chart_var.get() != "receiver_cmc" and not file_var.get():
            messagebox.showerror("错误", "请先选择数据文件")
            return

        # 禁用按钮
        batch_btn.config(state='disabled')

        def batch_process():
            """在后台线程中准备数据"""
            try:
                analyzer = GNSSAnalyzer()

                # 选择基准路径（接收机CMC走接收机文件，其它走手机文件）
                selected_type = chart_var.get()
                base_path = rx_file_var.get() if selected_type == "receiver_cmc" else file_var.get()
                if not base_path or not os.path.isfile(base_path):
                    charts_window.after(0, lambda: messagebox.showerror("错误", f"未找到文件: {base_path}"))
                    batch_btn.config(state='normal')
                    return
                analyzer.input_file_path = base_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(base_path), "analysis_results")
                os.makedirs(analyzer.current_result_dir, exist_ok=True)

                # 更新状态
                charts_window.after(0, lambda: status_var.set("正在读取RINEX文件..."))
                charts_window.after(0, lambda: progress_var.set(20))
                charts_window.after(0, lambda: charts_window.update_idletasks())

                # 读取文件（接收机CMC不依赖手机文件）
                data = None
                if selected_type != "receiver_cmc":
                    data = analyzer.read_rinex_obs(base_path)

                # 更新状态
                charts_window.after(0, lambda: status_var.set("正在计算分析数据..."))
                charts_window.after(0, lambda: progress_var.set(40))
                charts_window.after(0, lambda: charts_window.update_idletasks())

                # 计算所有必要的数据分析结果（非接收机CMC）
                if selected_type != "receiver_cmc":
                    derivatives = analyzer.calculate_observable_derivatives(data)
                    analyzer.results['observable_derivatives'] = derivatives
                    code_phase_diffs = analyzer.calculate_code_phase_differences(data)
                    analyzer.results['code_phase_diffs'] = code_phase_diffs
                    phase_pred_errors = analyzer.calculate_phase_prediction_errors(data)
                    analyzer.results['phase_prediction_errors'] = phase_pred_errors
                    analyzer.calculate_epoch_double_differences()
                    analyzer.calculate_triple_median_error(analyzer.results['double_differences'])
                else:
                    # 接收机CMC预先载入接收机RINEX
                    if not rx_file_var.get():
                        charts_window.after(0, lambda: messagebox.showerror("错误", "请先选择接收机RINEX文件"))
                        return
                    analyzer.read_receiver_rinex_obs(base_path)
                    analyzer.calculate_receiver_cmc()

                # 分析北斗系统间偏差
                beidou_bias_results = analyzer.analyze_beidou_system_bias()
                analyzer.results['beidou_system_bias'] = beidou_bias_results

                # 更新状态
                charts_window.after(0, lambda: status_var.set("正在生成所有图表..."))
                charts_window.after(0, lambda: progress_var.set(60))
                charts_window.after(0, lambda: charts_window.update_idletasks())

                # 在主线程中执行绘图操作
                def plot_on_main_thread():
                    try:
                        # 保存所有图表
                        analyzer.save_all_plots()

                        # 完成
                        status_var.set("批量保存完成!")
                        progress_var.set(100)
                        charts_window.update_idletasks()

                        # 显示结果
                        result_dir = analyzer.current_result_dir
                        messagebox.showinfo("完成", f"批量保存完成！\n结果保存在：{result_dir}")

                    except Exception as e:
                        messagebox.showerror("错误", f"生成图表过程中出现错误：\n{str(e)}")
                        status_var.set("生成失败")
                    finally:
                        # 恢复按钮
                        batch_btn.config(state='normal')

                # 在主线程中执行绘图操作
                charts_window.after(100, plot_on_main_thread)

            except Exception as e:
                messagebox.showerror("错误", f"准备图表数据过程中出现错误：\n{str(e)}")
                status_var.set("准备失败")
                # 恢复按钮
                batch_btn.config(state='normal')

        # 在新线程中执行数据准备
        import threading
        thread = threading.Thread(target=batch_process)
        thread.daemon = True
        thread.start()

    def batch_save_selected_chart_type():
        """批量保存选中的图表类型"""
        selected_chart_type = chart_var.get()
        if selected_chart_type != "receiver_cmc" and not file_var.get():
            messagebox.showerror("错误", "请先选择数据文件")
            return
        if selected_chart_type == "receiver_cmc" and not rx_file_var.get():
            messagebox.showerror("错误", "请先选择接收机RINEX文件")
            return

        # 获取用户选择的图表类型（已在前面拿到）
        if not selected_chart_type:
            messagebox.showerror("错误", "请先选择要保存的图表类型")
            return

        # 禁用按钮
        batch_btn.config(state='disabled')

        def batch_process():
            """在后台线程中准备数据"""
            try:
                analyzer = GNSSAnalyzer()

                # 选择基准路径（接收机CMC走接收机文件，其它走手机文件）
                base_path = rx_file_var.get() if selected_chart_type == "receiver_cmc" else file_var.get()
                if not base_path or not os.path.isfile(base_path):
                    charts_window.after(0, lambda: messagebox.showerror("错误", f"未找到文件: {base_path}"))
                    batch_btn.config(state='normal')
                    return
                analyzer.input_file_path = base_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(base_path), "analysis_results")
                os.makedirs(analyzer.current_result_dir, exist_ok=True)

                # 更新状态
                charts_window.after(0, lambda: status_var.set("正在读取RINEX文件..."))
                charts_window.after(0, lambda: progress_var.set(20))
                charts_window.after(0, lambda: charts_window.update_idletasks())

                # 读取文件（接收机CMC不依赖手机文件）
                data = None
                if selected_chart_type != "receiver_cmc":
                    data = analyzer.read_rinex_obs(base_path)

                # 更新状态
                charts_window.after(0, lambda: status_var.set(f"正在生成{selected_chart_type}图表..."))
                charts_window.after(0, lambda: progress_var.set(60))
                charts_window.after(0, lambda: charts_window.update_idletasks())

                # 准备数据
                if selected_chart_type == "raw_observations":
                    satellites = list(analyzer.observations_meters.keys())
                elif selected_chart_type == "derivatives":
                    derivatives = analyzer.calculate_observable_derivatives(data)
                    satellites = list(derivatives.keys())
                elif selected_chart_type == "code_phase_diff_raw":
                    differences = analyzer.calculate_code_phase_differences(data)
                    satellites = list(differences.keys())
                elif selected_chart_type == "code_phase_diffs":
                    differences = analyzer.calculate_code_phase_differences(data)
                    satellites = list(differences.keys())
                elif selected_chart_type == "phase_pred_errors":
                    errors = analyzer.calculate_phase_prediction_errors(data)
                    satellites = list(errors.keys())
                elif selected_chart_type == "receiver_cmc":
                    analyzer.read_receiver_rinex_obs(base_path)
                    cmc_results = analyzer.calculate_receiver_cmc()
                    satellites = list(cmc_results.keys())
                elif selected_chart_type == "double_differences":
                    analyzer.calculate_epoch_double_differences()
                    analyzer.calculate_triple_median_error(analyzer.results['double_differences'])
                    satellites = list(analyzer.observations_meters.keys())
                elif selected_chart_type == "beidou_system_bias":
                    beidou_bias_results = analyzer.analyze_beidou_system_bias()
                    satellites = ["beidou_system"]  # 北斗系统间偏差分析不需要按卫星分别处理
                else:
                    satellites = []

                # 计算总图表数量
                total_charts = 0
                for sat_id in satellites:
                    if selected_chart_type == "raw_observations":
                        total_charts += len(analyzer.observations_meters[sat_id])
                    elif selected_chart_type in ["derivatives", "code_phase_diff_raw", "code_phase_diffs",
                                                 "phase_pred_errors"]:
                        if sat_id in locals().get(selected_chart_type.replace("_", ""), {}):
                            total_charts += len(locals()[selected_chart_type.replace("_", "")][sat_id])
                    elif selected_chart_type == "double_differences":
                        total_charts += len(analyzer.observations_meters[sat_id])
                    elif selected_chart_type == "beidou_system_bias":
                        total_charts = 1  # 北斗系统间偏差分析只生成一组图表

                # 在主线程中执行绘图操作
                def plot_on_main_thread():
                    try:
                        saved_charts = 0

                        # 根据选择的图表类型保存对应的图表
                        if selected_chart_type == "raw_observations":
                            for sat_id in satellites:
                                for freq in analyzer.observations_meters[sat_id]:
                                    try:
                                        analyzer.plot_raw_observations(sat_id, save=True)
                                        saved_charts += 1
                                        # 更新进度
                                        progress = 60 + (saved_charts / total_charts) * 30
                                        progress_var.set(int(progress))
                                        charts_window.update_idletasks()
                                        # 强制清理内存
                                        plt.close('all')
                                        import gc
                                        gc.collect()
                                        # 添加短暂延迟，让系统有时间释放资源
                                        import time
                                        time.sleep(0.1)
                                    except Exception as e:
                                        print(f"保存{sat_id} {freq}原始观测值图表时出错: {str(e)}")
                                        continue

                        elif selected_chart_type == "derivatives":
                            for sat_id in satellites:
                                if sat_id in derivatives:
                                    for freq in derivatives[sat_id]:
                                        try:
                                            analyzer.plot_observable_derivatives(derivatives, sat_id, freq, save=True)
                                            saved_charts += 1
                                            progress = 60 + (saved_charts / total_charts) * 30
                                            progress_var.set(int(progress))
                                            charts_window.update_idletasks()
                                            plt.close('all')
                                            import gc
                                            gc.collect()
                                            import time
                                            time.sleep(0.1)
                                        except Exception as e:
                                            print(f"保存{sat_id} {freq}导数图表时出错: {str(e)}")
                                            continue

                        elif selected_chart_type == "code_phase_diff_raw":
                            for sat_id in satellites:
                                if sat_id in differences:
                                    for freq in differences[sat_id]:
                                        try:
                                            analyzer.plot_code_phase_raw_differences(differences, sat_id, freq,
                                                                                     save=True)
                                            saved_charts += 1
                                            progress = 60 + (saved_charts / total_charts) * 30
                                            progress_var.set(int(progress))
                                            charts_window.update_idletasks()
                                            plt.close('all')
                                            import gc
                                            gc.collect()
                                            import time
                                            time.sleep(0.1)
                                        except Exception as e:
                                            print(f"保存{sat_id} {freq}伪距相位原始差值图表时出错: {str(e)}")
                                            continue

                        elif selected_chart_type == "code_phase_diffs":
                            for sat_id in satellites:
                                if sat_id in differences:
                                    for freq in differences[sat_id]:
                                        try:
                                            analyzer.plot_code_phase_differences(differences, sat_id, freq, save=True)
                                            saved_charts += 1
                                            progress = 60 + (saved_charts / total_charts) * 30
                                            progress_var.set(int(progress))
                                            charts_window.update_idletasks()
                                            plt.close('all')
                                            import gc
                                            gc.collect()
                                            import time
                                            time.sleep(0.1)
                                        except Exception as e:
                                            print(f"保存{sat_id} {freq}伪距相位差值图表时出错: {str(e)}")
                                            continue

                        elif selected_chart_type == "phase_pred_errors":
                            for sat_id in satellites:
                                if sat_id in errors:
                                    for freq in errors[sat_id]:
                                        try:
                                            analyzer.plot_phase_prediction_errors(errors, sat_id, freq, save=True)
                                            saved_charts += 1
                                            progress = 60 + (saved_charts / total_charts) * 30
                                            progress_var.set(int(progress))
                                            charts_window.update_idletasks()
                                            plt.close('all')
                                            import gc
                                            gc.collect()
                                            import time
                                            time.sleep(0.1)
                                        except Exception as e:
                                            print(f"保存{sat_id} {freq}相位预测误差图表时出错: {str(e)}")
                                            continue

                        elif selected_chart_type == "double_differences":
                            for sat_id in satellites:
                                for freq in analyzer.observations_meters[sat_id]:
                                    try:
                                        analyzer.plot_double_differences(analyzer.results['double_differences'],
                                                                         analyzer.results['triple_median_errors'],
                                                                         sat_id, freq, save=True)
                                        saved_charts += 1
                                        progress = 60 + (saved_charts / total_charts) * 30
                                        progress_var.set(int(progress))
                                        charts_window.update_idletasks()
                                        plt.close('all')
                                        import gc
                                        gc.collect()
                                        import time
                                        time.sleep(0.1)
                                    except Exception as e:
                                        print(f"保存{sat_id} {freq}双差图表时出错: {str(e)}")
                                        continue

                        elif selected_chart_type == "receiver_cmc":
                            for sat_id in satellites:
                                for freq in cmc_results.get(sat_id, {}):
                                    try:
                                        vals = cmc_results[sat_id][freq]['cmc_m']
                                        epochs = list(range(1, len(vals)+1))
                                        plt.figure(figsize=(12,6))
                                        plt.plot(epochs, vals, 'b-', label='CMC (m)')
                                        plt.axhline(0, color='k', linestyle='--', alpha=0.4)
                                        plt.xlabel('历元'); plt.ylabel('CMC (m)'); plt.title(f'{sat_id}-{freq} 接收机CMC')
                                        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
                                        category_dir = os.path.join(analyzer.current_result_dir, 'receiver_cmc')
                                        os.makedirs(category_dir, exist_ok=True)
                                        out_path = os.path.join(category_dir, f"{sat_id}_{freq}_receiver_cmc.png")
                                        plt.savefig(out_path, dpi=300, bbox_inches='tight')
                                        plt.close()
                                        saved_charts += 1
                                        progress = 60 + (saved_charts / total_charts) * 30
                                        progress_var.set(int(progress))
                                        charts_window.update_idletasks()
                                        import gc; gc.collect(); import time; time.sleep(0.05)
                                    except Exception as e:
                                        print(f"保存{sat_id} {freq} CMC图表时出错: {str(e)}")
                                        continue

                        elif selected_chart_type == "beidou_system_bias":
                            try:
                                analyzer.plot_beidou_system_bias_analysis(beidou_bias_results, save=True)
                                saved_charts += 1
                                progress = 60 + (saved_charts / total_charts) * 30
                                progress_var.set(int(progress))
                                charts_window.update_idletasks()
                                plt.close('all')
                                import gc
                                gc.collect()
                                import time
                                time.sleep(0.1)
                            except Exception as e:
                                print(f"保存北斗系统间偏差分析图表时出错: {str(e)}")

                        # 完成
                        status_var.set(f"{selected_chart_type}图表保存完成!")
                        progress_var.set(100)
                        charts_window.update_idletasks()

                        # 显示结果
                        result_dir = analyzer.current_result_dir
                        messagebox.showinfo("完成", f"{selected_chart_type}图表保存完成！\n结果保存在：{result_dir}")

                    except Exception as e:
                        messagebox.showerror("错误", f"生成图表过程中出现错误：\n{str(e)}")
                        status_var.set("生成失败")
                    finally:
                        # 恢复按钮
                        batch_btn.config(state='normal')

                # 在主线程中执行绘图操作
                charts_window.after(100, plot_on_main_thread)

            except Exception as e:
                messagebox.showerror("错误", f"准备图表数据过程中出现错误：\n{str(e)}")
                status_var.set("准备失败")
                # 恢复按钮
                batch_btn.config(state='normal')

        # 在新线程中执行数据准备
        import threading
        thread = threading.Thread(target=batch_process)
        thread.daemon = True
        thread.start()

    batch_btn = ttk.Button(button_frame, text="批量保存选中类型", command=batch_save_selected_chart_type)
    batch_btn.pack(side=tk.LEFT, padx=10)

    def close_charts_window():
        """关闭图表窗口时的清理函数"""
        try:
            # 关闭所有Matplotlib图表窗口
            plt.close('all')
        except Exception as e:
            print(f"关闭图表窗口时清理图表出错: {str(e)}")
        finally:
            charts_window.destroy()

    ttk.Button(button_frame, text="关闭", command=close_charts_window).pack(side=tk.LEFT, padx=10)

    # 初始化时加载预定义的卫星和频率信息
    load_satellite_info()


def show_chart_window(analyzer, data, chart_type, sat_prn, freq, auto_save):
    """显示图表窗口"""
    chart_window = tk.Toplevel()
    chart_window.title(f"图表显示 - {chart_type} - {sat_prn} - {freq}")
    chart_window.geometry("1000x700")
    chart_window.resizable(True, True)
    
    # 居中显示窗口
    center_window(chart_window, 1000, 700)

    # 主框架
    main_frame = ttk.Frame(chart_window, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 图表信息
    info_frame = ttk.LabelFrame(main_frame, text="图表信息", padding="10")
    info_frame.pack(fill=tk.X, pady=10)

    ttk.Label(info_frame, text=f"卫星PRN: {sat_prn}").pack(side=tk.LEFT, padx=(0, 20))
    ttk.Label(info_frame, text=f"频率: {freq}").pack(side=tk.LEFT, padx=(0, 20))
    ttk.Label(info_frame, text=f"图表类型: {chart_type}").pack(side=tk.LEFT, padx=(0, 20))

    # 图表显示区域
    chart_frame = ttk.LabelFrame(main_frame, text="图表显示", padding="10")
    chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    # 创建matplotlib图形
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 8))
    canvas = FigureCanvasTkAgg(fig, chart_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # 生成图表
    try:
        if chart_type == "raw_observations":
            generate_raw_observations_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "derivatives":
            generate_derivatives_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "code_phase_diffs":
            generate_code_phase_diffs_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "code_phase_diff_raw":
            generate_code_phase_raw_diffs_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "phase_pred_errors":
            generate_phase_pred_errors_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "double_differences":
            generate_double_differences_plot(analyzer, data, sat_prn, freq, fig)
        elif chart_type == "beidou_system_bias":
            # 北斗系统间偏差分析图表
            generate_beidou_bias_plot(analyzer, data, fig)

        # 刷新画布
        canvas.draw()

    except Exception as e:
        # 显示错误信息
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"生成图表时出错：\n{str(e)}",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        canvas.draw()

    # 保存按钮
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=10)

    def save_chart():
        try:
            file_types = [
                ("PNG Files", "*.png"),
                ("PDF Files", "*.pdf"),
                ("SVG Files", "*.svg"),
                ("All Files", "*.*")
            ]
            filename = filedialog.asksaveasfilename(
                title="保存图表",
                filetypes=file_types,
                defaultextension=".png"
            )
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("成功", f"图表已保存到：{filename}")
        except Exception as e:
            messagebox.showerror("错误", f"保存图表失败：{str(e)}")

    ttk.Button(button_frame, text="保存图表", command=save_chart).pack(side=tk.LEFT, padx=10)

    def close_chart_window():
        """关闭图表窗口时的清理函数"""
        try:
            # 关闭当前图表
            plt.close(fig)
        except Exception as e:
            print(f"关闭图表窗口时清理图表出错: {str(e)}")
        finally:
            chart_window.destroy()

    ttk.Button(button_frame, text="关闭", command=close_chart_window).pack(side=tk.LEFT, padx=10)

    # 如果自动保存开启，则自动保存
    if auto_save:
        try:
            results_dir = analyzer.current_result_dir
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{chart_type}_{sat_prn}_{freq}_{timestamp}.png"
            filepath = os.path.join(results_dir, filename)

            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("自动保存", f"图表已自动保存到：{filepath}")
        except Exception as e:
            messagebox.showerror("自动保存失败", f"自动保存图表失败：{str(e)}")


def generate_raw_observations_plot(analyzer, data, sat_prn, freq, fig):
    """生成原始观测值图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 从处理后的观测数据中获取数据
    if sat_prn in analyzer.observations_meters and freq in analyzer.observations_meters[sat_prn]:
        obs_data = analyzer.observations_meters[sat_prn][freq]
        times = obs_data['times']
        code_values = obs_data['code']
        phase_values = obs_data['phase']
        doppler_values = obs_data['doppler']

        # 绘制图表
        ax.plot(times, code_values, 'b-', label='伪距 (Code)', alpha=0.7)
        ax.plot(times, phase_values, 'r-', label='载波相位 (Phase)', alpha=0.7)
        ax.plot(times, doppler_values, 'g-', label='多普勒 (Doppler)', alpha=0.7)

        ax.set_xlabel('时间')
        ax.set_ylabel('观测值')
        ax.set_title(f'原始观测值 - {sat_prn} - {freq}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        import matplotlib.pyplot as plt
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, f'未找到卫星 {sat_prn} 频率 {freq} 的数据',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('数据缺失')


def generate_derivatives_plot(analyzer, data, sat_prn, freq, fig):
    """生成观测值一阶差分图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 计算一阶差分
    derivatives = analyzer.calculate_observable_derivatives(data)

    if sat_prn in derivatives and freq in derivatives[sat_prn]:
        deriv_data = derivatives[sat_prn][freq]
        epochs = list(deriv_data.keys())

        # 提取差分值
        c_derivs = [deriv_data[epoch].get('C', []) for epoch in epochs]
        l_derivs = [deriv_data[epoch].get('L', []) for epoch in epochs]
        d_derivs = [deriv_data[epoch].get('D', []) for epoch in epochs]

        # 绘制图表
        ax.plot(epochs, c_derivs, 'b-', label='伪距差分', alpha=0.7)
        ax.plot(epochs, l_derivs, 'r-', label='载波相位差分', alpha=0.7)
        ax.plot(epochs, d_derivs, 'g-', label='多普勒差分', alpha=0.7)

        ax.set_xlabel('历元')
        ax.set_ylabel('一阶差分')
        ax.set_title(f'观测值一阶差分 - {sat_prn} - {freq}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的一阶差分数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def generate_code_phase_diffs_plot(analyzer, data, sat_prn, freq, fig):
    """生成伪距相位差值图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 计算伪距相位差值
    code_phase_diffs = analyzer.calculate_code_phase_differences(data)

    if sat_prn in code_phase_diffs and freq in code_phase_diffs[sat_prn]:
        diff_data = code_phase_diffs[sat_prn][freq]
        epochs = list(diff_data.keys())

        # 提取差值
        diff_values = [diff_data[epoch].get('diff', []) for epoch in epochs]

        # 绘制图表
        ax.plot(epochs, diff_values, 'b-', label='伪距相位差值', alpha=0.7)

        ax.set_xlabel('历元')
        ax.set_ylabel('差值 (米)')
        ax.set_title(f'伪距相位差值 - {sat_prn} - {freq}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的伪距相位差值数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def generate_code_phase_raw_diffs_plot(analyzer, data, sat_prn, freq, fig):
    """生成伪距相位原始差值图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 计算伪距相位差值
    code_phase_diffs = analyzer.calculate_code_phase_differences(data)

    if sat_prn in code_phase_diffs and freq in code_phase_diffs[sat_prn]:
        diff_data = code_phase_diffs[sat_prn][freq]
        epochs = list(diff_data.keys())

        # 提取原始差值
        raw_diff_values = [diff_data[epoch].get('raw_diff', []) for epoch in epochs]

        # 绘制图表
        ax.plot(epochs, raw_diff_values, 'r-', label='伪距相位原始差值', alpha=0.7)

        ax.set_xlabel('历元')
        ax.set_ylabel('原始差值 (米)')
        ax.set_title(f'伪距相位原始差值 - {sat_prn} - {freq}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的伪距相位原始差值数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def generate_phase_pred_errors_plot(analyzer, data, sat_prn, freq, fig):
    """生成相位预测误差图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 计算相位预测误差
    phase_pred_errors = analyzer.calculate_phase_prediction_errors(data)

    if sat_prn in phase_pred_errors and freq in phase_pred_errors[sat_prn]:
        error_data = phase_pred_errors[sat_prn][freq]
        
        # 检查数据结构
        if 'prediction_error' in error_data and len(error_data['prediction_error']) > 0:
            # 提取误差值和时间
            error_values = error_data['prediction_error']
            times = error_data['times']
            
            # 生成历元序号（从1开始）
            epochs = list(range(1, len(error_values) + 1))
            
            # 绘制图表
            ax.plot(epochs, error_values, 'm-', label='相位预测误差', alpha=0.7)
            
            ax.set_xlabel('历元')
            ax.set_ylabel('预测误差 (米)')
            ax.set_title(f'相位预测误差 - {sat_prn} - {freq}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的有效相位预测误差数据",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的相位预测误差数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def generate_double_differences_plot(analyzer, data, sat_prn, freq, fig):
    """生成历元间双差图表"""
    fig.clear()
    ax = fig.add_subplot(111)

    # 计算历元间双差
    double_diffs = analyzer.calculate_epoch_double_differences()
    triple_errors = analyzer.calculate_triple_median_error(double_diffs)

    if sat_prn in double_diffs and freq in double_diffs[sat_prn]:
        diff_data = double_diffs[sat_prn][freq]
        epochs = list(diff_data.keys())

        # 提取双差值
        diff_values = [diff_data[epoch].get('double_diff', []) for epoch in epochs]

        # 绘制图表
        ax.plot(epochs, diff_values, 'c-', label='历元间双差', alpha=0.7)

        ax.set_xlabel('历元')
        ax.set_ylabel('双差值 (米)')
        ax.set_title(f'历元间双差 - {sat_prn} - {freq}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, f"未找到 {sat_prn} - {freq} 的历元间双差数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def generate_beidou_bias_plot(analyzer, data, fig):
    """生成北斗系统间偏差分析图表"""
    try:
        # 分析北斗系统间偏差
        beidou_bias_results = analyzer.analyze_beidou_system_bias()
        
        if not beidou_bias_results or not beidou_bias_results.get('bds2_sats') or not beidou_bias_results.get('bds3_sats'):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "没有足够的北斗二号和三号卫星数据进行系统间偏差分析",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return

        # 创建子图
        fig.clear()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        # 1. 卫星数量对比
        bds2_count = len(beidou_bias_results['bds2_sats'])
        bds3_count = len(beidou_bias_results['bds3_sats'])
        
        systems = ['北斗二号', '北斗三号']
        counts = [bds2_count, bds3_count]
        colors = ['#FF6B6B', '#4ECDC4']
        
        ax1.bar(systems, counts, color=colors, alpha=0.7)
        ax1.set_title('北斗系统卫星数量对比')
        ax1.set_ylabel('卫星数量')
        ax1.set_ylim(0, max(counts) * 1.2)
        
        # 在柱状图上添加数值标签
        for i, count in enumerate(counts):
            ax1.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')

        # 2. 系统间偏差分布
        if 'system_bias' in beidou_bias_results:
            bias_data = beidou_bias_results['system_bias']
            frequencies = list(bias_data.keys())
            code_biases = [bias_data[freq]['code_bias'] for freq in frequencies]
            phase_biases = [bias_data[freq]['phase_bias'] for freq in frequencies]
            
            x = range(len(frequencies))
            width = 0.35
            
            ax2.bar([i - width/2 for i in x], code_biases, width, label='码观测值偏差', alpha=0.7)
            ax2.bar([i + width/2 for i in x], phase_biases, width, label='相位观测值偏差', alpha=0.7)
            ax2.set_title('系统间偏差分布')
            ax2.set_xlabel('频率')
            ax2.set_ylabel('偏差 (m)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(frequencies)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. 轨道类型偏差分析
        if 'orbit_bias' in beidou_bias_results:
            orbit_data = beidou_bias_results['orbit_bias']
            orbit_types = list(orbit_data.keys())
            orbit_biases = [orbit_data[orbit]['mean_bias'] for orbit in orbit_types]
            
            ax3.bar(orbit_types, orbit_biases, color='#95E1D3', alpha=0.7)
            ax3.set_title('轨道类型偏差分析')
            ax3.set_ylabel('平均偏差 (m)')
            ax3.grid(True, alpha=0.3)

        # 4. 雷达图 - 系统性能对比
        if 'bds2_count' in beidou_bias_results and 'bds3_count' in beidou_bias_results:
            # 简化的雷达图
            categories = ['卫星数量', '码稳定性', '相位稳定性', '频率覆盖']
            bds2_scores = [bds2_count, 0.8, 0.85, 0.9]  # 示例值
            bds3_scores = [bds3_count, 0.9, 0.95, 0.95]  # 示例值
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            bds2_scores += bds2_scores[:1]
            bds3_scores += bds3_scores[:1]
            
            ax4.plot(angles, bds2_scores, 'o-', linewidth=2, label='北斗二号', color='#FF6B6B')
            ax4.fill(angles, bds2_scores, alpha=0.25, color='#FF6B6B')
            ax4.plot(angles, bds3_scores, 'o-', linewidth=2, label='北斗三号', color='#4ECDC4')
            ax4.fill(angles, bds3_scores, alpha=0.25, color='#4ECDC4')
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories)
            ax4.set_title('系统性能对比雷达图')
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax4.grid(True)

        # 调整布局
        fig.tight_layout()

    except Exception as e:
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"生成北斗系统间偏差分析图表时出错：\n{str(e)}",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)


def show_report_window(parent):
    """显示报告功能窗口"""
    report_window = tk.Toplevel(parent)
    report_window.title("分析报告")
    report_window.geometry("600x500")
    report_window.resizable(True, True)
    report_window.transient(parent)
    report_window.grab_set()
    
    # 居中显示窗口
    center_window(report_window, 600, 500)

    # 主框架
    main_frame = ttk.Frame(report_window, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 文件选择
    file_frame = ttk.LabelFrame(main_frame, text="选择数据文件", padding="10")
    file_frame.pack(fill=tk.X, pady=10)

    file_var = tk.StringVar()
    file_entry = ttk.Entry(file_frame, textvariable=file_var, width=50)
    file_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)

    def select_file():
        file_types = [
            ("RINEX Files", "*.??O *.??o *.RNX *.rnx"),
            ("All Files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="选择RINEX观测文件",
            filetypes=file_types
        )
        if filename:
            file_var.set(filename)

    ttk.Button(file_frame, text="浏览", command=select_file).pack(side=tk.RIGHT)

    # 报告选项
    report_frame = ttk.LabelFrame(main_frame, text="报告选项", padding="10")
    report_frame.pack(fill=tk.X, pady=10)

    include_plots_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(report_frame, text="包含图表", variable=include_plots_var).pack(anchor=tk.W)

    include_stats_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(report_frame, text="包含统计信息", variable=include_stats_var).pack(anchor=tk.W)

    # 进度显示
    progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
    progress_frame.pack(fill=tk.X, pady=10)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL,
                                   variable=progress_var, mode='determinate')
    progress_bar.pack(fill=tk.X, pady=5)

    status_var = tk.StringVar(value="等待开始...")
    status_label = ttk.Label(progress_frame, textvariable=status_var)
    status_label.pack()

    # 操作按钮
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)

    def generate_report():
        if not file_var.get():
            tk.messagebox.showerror("错误", "请先选择数据文件")
            return

        # 禁用按钮
        generate_btn.config(state='disabled')

        def report_process():
            try:
                analyzer = GNSSAnalyzer()

                # 设置文件路径
                file_path = file_var.get()
                analyzer.input_file_path = file_path
                analyzer.current_result_dir = os.path.join(os.path.dirname(file_path), "analysis_results")

                # 更新状态
                status_var.set("正在读取RINEX文件...")
                progress_var.set(20)
                report_window.update_idletasks()

                # 读取文件
                data = analyzer.read_rinex_obs(file_path)

                # 更新状态
                status_var.set("正在生成分析报告...")
                progress_var.set(60)
                report_window.update_idletasks()

                # 生成报告
                analyzer.save_report()

                # 完成
                status_var.set("报告生成完成!")
                progress_var.set(100)
                report_window.update_idletasks()

                # 显示结果
                result_dir = analyzer.current_result_dir
                tk.messagebox.showinfo("完成", f"分析报告生成完成！\n结果保存在：{result_dir}")

            except Exception as e:
                tk.messagebox.showerror("错误", f"生成报告过程中出现错误：\n{str(e)}")
                status_var.set("生成失败")
            finally:
                # 恢复按钮
                generate_btn.config(state='normal')

        # 在新线程中执行
        import threading
        thread = threading.Thread(target=report_process)
        thread.daemon = True
        thread.start()

    generate_btn = ttk.Button(button_frame, text="生成报告", command=generate_report)
    generate_btn.pack(side=tk.LEFT, padx=10)

    def close_report_window():
        """关闭报告窗口时的清理函数"""
        try:
            # 关闭所有Matplotlib图表窗口
            plt.close('all')
        except Exception as e:
            print(f"关闭报告窗口时清理图表出错: {str(e)}")
        finally:
            report_window.destroy()

    ttk.Button(button_frame, text="关闭",
               command=close_report_window).pack(side=tk.LEFT, padx=10)


if __name__ == "__main__":
    main()
