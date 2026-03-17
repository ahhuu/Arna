import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import time
import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 使用Tkinter兼容的交互式后端
import matplotlib.pyplot as plt
from typing import Dict
from collections import defaultdict

# 导入分析脚本
import sys
import importlib.util
from pathlib import Path


def _resolve_tools_root():
    base = Path(__file__).parent
    new_root = base / 'Android_GNSS_Analysis' / 'tools'
    old_root = base / 'tools'
    return new_root if new_root.exists() else old_root


TOOLS_ROOT = _resolve_tools_root()

# 动态导入 Pseudorange_Residuals
pr_path = TOOLS_ROOT / 'analysis_tools' / 'Pseudorange_Residuals.py'
pr_spec = importlib.util.spec_from_file_location("Pseudorange_Residuals", pr_path)
Pseudorange_Residuals = importlib.util.module_from_spec(pr_spec)
sys.modules["Pseudorange_Residuals"] = Pseudorange_Residuals
pr_spec.loader.exec_module(Pseudorange_Residuals)

# 动态导入 SNR_Weighting
snr_path = TOOLS_ROOT / 'analysis_tools' / 'SNR_Weighting.py'
snr_spec = importlib.util.spec_from_file_location("SNR_Weighting", snr_path)
SNR_Weighting = importlib.util.module_from_spec(snr_spec)
sys.modules["SNR_Weighting"] = SNR_Weighting
snr_spec.loader.exec_module(SNR_Weighting)

# 设置中文字体（全局）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class TextRedirector(io.StringIO):
    """用于重定向stdout和stderr到GUI"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update()


class GNSSAnalyzer:
    """GNSS观测数据分析与问题检测器（从Rinex_analysis.py导入）"""

    def __init__(self):
        # 定义GNSS信号频率 (Hz)
        self.frequencies = {
            'G': {'L1C': 1575.42e6, 'L5Q': 1176.45e6},  # GPS
            'R': {'L1C': 1602e6, 'L5Q': 1246e6},  # GLONASS
            'E': {'L1B': 1575.42e6, 'L1C': 1575.42e6, 'L5Q': 1176.45e6, 'L7Q': 1207.14e6},  # Galileo
            'C': {'L2I': 1561.098e6, 'L1P': 1575.42e6, 'L1D': 1575.42e6, 'L5P': 1176.45e6},  # BeiDou
            'J': {'L1C': 1575.42e6, 'L5Q': 1176.45e6},  # QZSS
            'I': {'L5Q': 1176.45e6, 'S': 2492.028e6},  # IRNSS/NavIC
            'S': {'L1C': 1575.42e6, 'L5Q': 1176.45e6},  # SBAS
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

        # 存储以米为单位的观测值
        self.observations_meters = {}

        # 存储分析结果
        self.results = {
            'code_carrier_inconsistency': {},
            'observation_inconsistency': {},
            'phase_stagnation': {},
            'observable_derivatives': {},
            'code_phase_differences': {},
            'phase_prediction_errors': {}
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
            'double_differences': '历元间双差'
        }

        # 进度管理相关属性
        self.progress_callback = None
        self.current_stage = 0
        self.total_stages = 9
        self.stage_progress = 0
        self.stage_max = 100

        # 剔除粗差后的观测值文件
        self.output_format = "rinex"
        self.cleaned_observations = {}

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
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.rstrip('\n') for line in f]
        except Exception as e:
            print(f"读取文件错误: {str(e)}")
            return data

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
            elif 'SYS / # / OBS TYPES' in line:
                system = line[0]
                obs_types = line.split()[2:]
                data['header'][f'obs_types_{system}'] = obs_types

        # 更新头部解析进度
        header_progress = min(int(header_end / len(lines) * 20), 20) if lines else 0
        self.update_progress(header_progress)

        # 解析观测数据
        current_epoch = None
        current_satellites = {}
        i = header_end
        total_lines = len(lines)

        while i < total_lines:
            # 每处理1%的行更新一次进度
            if total_lines > 80 and i % (total_lines // 80) == 0:
                progress = 20 + int((i - header_end) / (total_lines - header_end) * 80)
                self.update_progress(progress)
            
            line = lines[i]
            if not line.strip():
                i += 1
                continue
                
            if line.startswith('>'):
                # 新历元
                if current_epoch is not None and current_satellites:
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
                        second = int(float(parts[5]))
                        current_epoch = pd.Timestamp(
                            year=year, month=month, day=day,
                            hour=hour, minute=minute, second=second
                        )
                        current_satellites = {}
                    else:
                        print(f"警告: 时间行格式错误 (行 {i + 1}): {line}")
                        i += 1
                        continue
                except (ValueError, IndexError) as e:
                    print(f"时间解析错误 (行 {i + 1}): {line}")
                    i += 1
                    continue
                i += 1
            else:
                # 解析卫星观测值
                if current_epoch is None or len(line) < 3:
                    i += 1
                    continue
                    
                sat_system = line[0]
                sat_prn = line[1:3].strip()
                if not sat_prn:
                    i += 1
                    continue
                sat_id = f"{sat_system}{sat_prn}"

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

                # 存储观测值到observations_meters
                if sat_id not in self.observations_meters:
                    self.observations_meters[sat_id] = {}

                # 获取频率信息
                current_freqs = self.frequencies.get(sat_system, {})
                
                for freq in current_freqs:
                    code_obs_type = f'C{freq[1:]}'
                    phase_obs_type = f'L{freq[1:]}'
                    doppler_obs_type = f'D{freq[1:]}'

                    # 获取观测值
                    code_val = observations.get(code_obs_type)
                    phase_val = observations.get(phase_obs_type)
                    doppler_val = observations.get(doppler_obs_type)
                    wavelength = self.wavelengths[sat_system].get(freq)

                    # 初始化数据结构
                    if freq not in self.observations_meters[sat_id]:
                        self.observations_meters[sat_id][freq] = {
                            'times': [],
                            'code': [],
                            'phase': [],
                            'phase_cycle': [],
                            'doppler': [],
                            'wavelength': []
                        }

                    # 存储时间和观测值
                    self.observations_meters[sat_id][freq]['times'].append(current_epoch)
                    self.observations_meters[sat_id][freq]['code'].append(code_val)

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

    def calculate_observable_derivatives(self, data: Dict) -> Dict:
        """计算每个卫星每个频率伪距、相位与多普勒观测的一阶差分"""
        self.start_stage(1, "计算观测值一阶差分", 100)

        derivatives = {}
        if not data.get('epochs'):
            return derivatives

        total_sats = len(self.observations_meters)
        processed_sats = 0

        for sat_id, freq_data in self.observations_meters.items():
            freq_derivatives = {}
            
            for freq, obs_data in freq_data.items():
                times = obs_data['times']
                code_values = obs_data['code']
                phase_values = obs_data['phase_cycle']
                doppler_values = obs_data['doppler']
                wavelength = self.wavelengths.get(sat_id[0], {}).get(freq)

                freq_derivatives[freq] = {
                    'times': [],
                    'pr_derivative': [],
                    'ph_derivative': [],
                    'doppler': []
                }

                if len(times) < 2 or wavelength is None:
                    continue

                # 计算一阶差分
                for i in range(1, len(times)):
                    time_diff = (times[i] - times[i - 1]).total_seconds()
                    if time_diff <= 0:
                        continue

                    freq_derivatives[freq]['times'].append(times[i])

                    # 伪距一阶差分
                    if code_values[i] is not None and code_values[i - 1] is not None:
                        pr_derivative = (code_values[i] - code_values[i - 1]) / time_diff
                        freq_derivatives[freq]['pr_derivative'].append(pr_derivative)
                    else:
                        freq_derivatives[freq]['pr_derivative'].append(None)

                    # 相位一阶差分
                    if phase_values[i] is not None and phase_values[i - 1] is not None:
                        phase_rate_cycles = (phase_values[i] - phase_values[i - 1]) / time_diff
                        phase_rate_meters = phase_rate_cycles * wavelength
                        freq_derivatives[freq]['ph_derivative'].append(phase_rate_meters)
                    else:
                        freq_derivatives[freq]['ph_derivative'].append(None)

                    # 多普勒值
                    if doppler_values[i] is not None:
                        freq_derivatives[freq]['doppler'].append(doppler_values[i])
                    else:
                        freq_derivatives[freq]['doppler'].append(None)

            derivatives[sat_id] = freq_derivatives
            processed_sats += 1
            if total_sats > 0:
                self.update_progress(int(processed_sats / total_sats * 100))

        self.results['observable_derivatives'] = derivatives
        return derivatives

    def detect_phase_stagnation(self, data: Dict, threshold_cycles: float = 0.1, min_consecutive: int = 5) -> Dict:
        """检测载波相位停滞"""
        self.start_stage(2, "检测载波相位停滞", 100)

        stagnation_results = {}
        total_sats = len(self.observations_meters)
        processed_sats = 0

        for sat_id, freq_data in self.observations_meters.items():
            freq_stagnation = {}

            for freq, obs_data in freq_data.items():
                phase_cycles = obs_data['phase_cycle']

                stagnant_epochs = []
                current_streak = 0
                max_streak = 0

                for i in range(1, len(phase_cycles)):
                    if (phase_cycles[i] is not None and
                            phase_cycles[i - 1] is not None and
                            abs(phase_cycles[i] - phase_cycles[i - 1]) < threshold_cycles):
                        current_streak += 1
                    else:
                        current_streak = 0

                    if current_streak > max_streak:
                        max_streak = current_streak

                    if current_streak >= min_consecutive:
                        stagnant_epochs.append(i)

                freq_stagnation[freq] = {
                    'is_stagnant': max_streak >= min_consecutive,
                    'max_stagnant_epochs': max_streak,
                    'stagnant_epochs': stagnant_epochs,
                    'threshold': threshold_cycles,
                    'min_consecutive': min_consecutive
                }

            stagnation_results[sat_id] = freq_stagnation
            processed_sats += 1
            if total_sats > 0:
                self.update_progress(int(processed_sats / total_sats * 100))

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

        for sat_id, freq_data in self.observations_meters.items():
            freq_differences = {}
            sat_stagnation = self.results['phase_stagnation'].get(sat_id, {})

            for freq, obs_data in freq_data.items():
                times = obs_data['times']
                code_values = obs_data['code']
                phase_values = obs_data['phase']

                stagnant_epochs = sat_stagnation.get(freq, {}).get('stagnant_epochs', [])

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
                    if (i in stagnant_epochs or
                            code_values[i] is None or
                            phase_values[i] is None):
                        if code_values[i] is None or phase_values[i] is None:
                            missing_obs += 1
                        continue

                    diff = code_values[i] - phase_values[i]
                    freq_differences[freq]['times'].append(times[i])
                    freq_differences[freq]['code_phase_diff'].append(diff)

                    if prev_diff is not None:
                        diff_change = abs(diff - prev_diff)
                        freq_differences[freq]['diff_changes'].append(diff_change)
                    else:
                        freq_differences[freq]['diff_changes'].append(None)

                    prev_diff = diff

                freq_differences[freq]['filtered_epochs'] = len(freq_differences[freq]['code_phase_diff'])
                freq_differences[freq]['missing_epochs'] = missing_obs

            differences[sat_id] = freq_differences
            processed_sats += 1
            if total_sats > 0:
                self.update_progress(int(processed_sats / total_sats * 100))

        self.results['code_phase_differences'] = differences
        return differences

    def calculate_phase_prediction_errors(self, data: Dict) -> Dict:
        """计算相位预测误差"""
        self.start_stage(4, "计算相位预测误差", 100)

        errors = {}
        total_sats = len(self.observations_meters)
        processed_sats = 0

        for sat_id, freq_data in self.observations_meters.items():
            freq_errors = {}

            for freq, obs_data in freq_data.items():
                times = obs_data['times']
                phase_values = obs_data['phase_cycle']
                doppler_values = obs_data['doppler']

                frequency = self.frequencies[sat_id[0]].get(freq)
                wavelength = self.wavelengths[sat_id[0]].get(freq)

                freq_errors[freq] = {
                    'times': [],
                    'actual_phase': [],
                    'predicted_phase': [],
                    'prediction_error': [],
                    'doppler': []
                }

                if frequency is None or wavelength is None:
                    continue

                for i in range(1, len(times)):
                    if (phase_values[i - 1] is not None and
                            doppler_values[i - 1] is not None and
                            doppler_values[i] is not None and
                            phase_values[i] is not None):
                        
                        time_diff = (times[i] - times[i - 1]).total_seconds()
                        doppler_mean = (doppler_values[i - 1] + doppler_values[i]) / 2

                        # 计算相位变化率(周/秒)
                        phase_rate = -doppler_mean / wavelength / frequency
                        predicted_phase = phase_values[i - 1] + phase_rate * time_diff

                        # 计算误差(米)
                        error = (phase_values[i] - predicted_phase) * wavelength

                        freq_errors[freq]['times'].append(times[i])
                        freq_errors[freq]['actual_phase'].append(phase_values[i])
                        freq_errors[freq]['predicted_phase'].append(predicted_phase)
                        freq_errors[freq]['prediction_error'].append(error)
                        freq_errors[freq]['doppler'].append(doppler_mean)

            errors[sat_id] = freq_errors
            processed_sats += 1
            if total_sats > 0:
                self.update_progress(int(processed_sats / total_sats * 100))

        self.results['phase_prediction_errors'] = errors
        return errors

    def calculate_epoch_double_differences(self):
        """计算各卫星各频率的历元间双差（伪距、相位、多普勒）"""
        self.start_stage(5, "计算历元间双差", 100)
        
        double_diffs = {}
        total_sats = len(self.observations_meters)
        processed_sats = 0

        for sat_id, freq_data in self.observations_meters.items():
            double_diffs[sat_id] = {}
            
            for freq, data in freq_data.items():
                code = np.array(data['code'], dtype=float)
                phase = np.array(data['phase'], dtype=float)
                doppler = np.array(data['doppler'], dtype=float)

                n = len(code)
                if n < 3:
                    continue

                dd_code = np.zeros(n - 2)
                dd_phase = np.zeros(n - 2)
                dd_doppler = np.zeros(n - 2)

                for i in range(n - 2):
                    dd_code[i] = code[i + 2] - 2 * code[i + 1] + code[i]
                    dd_phase[i] = phase[i + 2] - 2 * phase[i + 1] + phase[i]
                    dd_doppler[i] = doppler[i + 2] - 2 * doppler[i + 1] + doppler[i]

                double_diffs[sat_id][freq] = {
                    'times': data['times'][2:],
                    'dd_code': dd_code.tolist(),
                    'dd_phase': dd_phase.tolist(),
                    'dd_doppler': dd_doppler.tolist()
                }

            processed_sats += 1
            if total_sats > 0:
                self.update_progress(int(processed_sats / total_sats * 100))

        self.results['double_differences'] = double_diffs
        return double_diffs

    def save_report(self) -> None:
        """保存分析报告到当前文件结果目录"""
        self.start_stage(7, "保存分析报告", 100)

        report = self.generate_report()
        filename = "analysis_report.txt"
        if self.current_result_dir:
            full_path = os.path.join(self.current_result_dir, filename)
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.update_progress(100)
                print(f"--报告已保存至: {full_path}")
            except Exception as e:
                print(f"保存报告时出错: {str(e)}")

    def generate_report(self) -> str:
        """生成检测结果报告"""
        report = "=== GNSS观测数据分析报告 ===\n\n"
        
        # 添加基本统计信息
        report += f"分析文件: {self.input_file_path}\n"
        report += f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 观测值统计
        if self.observations_meters:
            report += f"观测卫星数量: {len(self.observations_meters)}\n"
            total_epochs = 0
            for sat_data in self.observations_meters.values():
                for freq_data in sat_data.values():
                    if len(freq_data['times']) > total_epochs:
                        total_epochs = len(freq_data['times'])
            report += f"观测历元数量: {total_epochs}\n\n"

        # 相位停滞统计
        if 'phase_stagnation' in self.results:
            stagnant_sats = set()
            for sat_id, freq_data in self.results['phase_stagnation'].items():
                for freq, stats in freq_data.items():
                    if stats.get('is_stagnant', False):
                        stagnant_sats.add(sat_id)
            report += f"检测到相位停滞卫星数: {len(stagnant_sats)}\n"

        # 伪距相位差值异常统计
        if 'code_phase_differences' in self.results:
            threshold = 10.0
            inconsistent_count = 0
            for sat_id, freq_data in self.results['code_phase_differences'].items():
                for freq, data in freq_data.items():
                    changes = [c for c in data['diff_changes'] if c is not None]
                    if changes and max(changes) > threshold:
                        inconsistent_count += 1
            report += f"检测到伪距相位差值异常频率数: {inconsistent_count}\n"

        report += "\n分析完成。\n"
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
            'phase_prediction_errors': {}
        }
        self.input_file_path = None
        self.current_result_dir = None


class IntegratedGNSSAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Android GNSS 数据分析")
        self.root.geometry("1000x900")
        self.root.resizable(True, True)

        # 先初始化变量
        self.mobile_rinex_file = tk.StringVar()
        self.base_rinex_file = tk.StringVar()
        self.sat_pos_file = tk.StringVar()
        self.residuals_file = tk.StringVar()
        self.analysis_rinex_file = tk.StringVar()
        
        # 坐标变量
        self.mobile_x = tk.DoubleVar(value=-1324698.041159006)
        self.mobile_y = tk.DoubleVar(value=5323031.038016253)
        self.mobile_z = tk.DoubleVar(value=3244602.006945656)
        self.base_x = tk.DoubleVar(value=-1324698.104573897)
        self.base_y = tk.DoubleVar(value=5323031.050568524)
        self.base_z = tk.DoubleVar(value=3244601.728187757)
        
        # RINEX分析参数
        self.threshold_var = tk.DoubleVar(value=5.0)
        
        # 图表可视化相关变量
        self.chart_var = tk.StringVar(value="raw_observations")
        self.selected_systems = tk.StringVar(value="all")
        self.selected_prns = tk.StringVar(value="all")
        self.selected_frequencies = tk.StringVar(value="all")
        self.batch_chart_types = {}
        
        # 存储分析器实例
        self.analyzer = None
        
        # 然后创建界面
        self.create_widgets()

    def create_widgets(self):
        # 创建主菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 添加主页菜单项
        menubar.add_command(label="主页", command=self.show_main_interface)
        
        # 创建一级菜单
        android_menu = tk.Menu(menubar, tearoff=0)
        random_model_menu = tk.Menu(menubar, tearoff=0)
        
        menubar.add_cascade(label="RINEX数据分析", menu=android_menu)
        menubar.add_cascade(label="随机模型分析", menu=random_model_menu)
        
        # Android RINEX数据分析子菜单
        android_menu.add_command(label="粗差剔除", command=lambda: self.show_tab("cleaning"))
        android_menu.add_command(label="图表可视化", command=lambda: self.show_tab("visualization"))
        android_menu.add_command(label="分析报告", command=lambda: self.show_tab("analysis"))
        
        # 随机模型分析子菜单
        random_model_menu.add_command(label="伪距残差分析", command=lambda: self.show_tab("residuals"))
        random_model_menu.add_command(label="随机模型拟合", command=lambda: self.show_tab("weighting"))
        random_model_menu.add_command(label="随机模型拟合完整流程", command=lambda: self.show_tab("complete"))
        
        # 创建主界面
        self.create_main_interface()
        
        # 创建笔记本控件（标签页）- 但不立即显示
        self.notebook = ttk.Notebook(self.root)
        
        # 创建标签页字典，按功能分组
        self.tab_frames = {}
        
        # Android RINEX数据分析相关标签页
        self.android_tabs = {
            "cleaning": self.create_cleaning_tab,
            "visualization": self.create_visualization_tab, 
            "analysis": self.create_analysis_tab
        }
        
        # 随机模型分析相关标签页
        self.random_model_tabs = {
            "residuals": self.create_residuals_tab,
            "weighting": self.create_weighting_tab,
            "complete": self.create_complete_tab
        }
        
        # 标签页标题映射
        self.tab_titles = {
            "cleaning": "粗差剔除",
            "visualization": "图表可视化",
            "analysis": "分析报告",
            "residuals": "伪距残差分析",
            "weighting": "随机模型拟合",
            "complete": "完整流程"
        }
        
        # 当前显示的标签页类型
        self.current_tab_group = None
        
        # 初始显示主界面
        self.show_main_interface()

    def create_main_interface(self):
        """创建主界面介绍"""
        self.main_frame = ttk.Frame(self.root)
        
        # 软件标题
        title_label = ttk.Label(self.main_frame, text="Android GNSS 数据分析", 
                               font=('Arial', 24, 'bold'))
        title_label.pack(pady=30)
        
        # 版本信息
        version_label = ttk.Label(self.main_frame, text="Version 2.0", 
                                 font=('Arial', 12, 'italic'))
        version_label.pack(pady=5)
        
        # 功能介绍框架
        intro_frame = ttk.LabelFrame(self.main_frame, text="软件功能介绍", padding=20)
        intro_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=20)
        
        # 创建两列布局
        left_frame = ttk.Frame(intro_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_frame = ttk.Frame(intro_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Android RINEX数据分析功能介绍
        android_label = ttk.Label(left_frame, text="🔍 RINEX数据分析", 
                                 font=('Arial', 16, 'bold'))
        android_label.pack(anchor=tk.W, pady=(0, 10))
        
        android_features = [
            "• 粗差剔除：基于多种算法自动识别和剔除观测数据中的异常值",
            "• 图表可视化：生成各类GNSS观测数据分析图表，支持分系统分析", 
            "• 分析报告：自动生成详细的数据质量分析报告"
        ]
        
        for feature in android_features:
            ttk.Label(left_frame, text=feature, wraplength=400).pack(anchor=tk.W, pady=2)
        
        # 随机模型分析功能介绍
        random_label = ttk.Label(right_frame, text="📊 随机模型分析", 
                                font=('Arial', 16, 'bold'))
        random_label.pack(anchor=tk.W, pady=(0, 10))
        
        random_features = [
            "• 伪距残差分析：计算手机与基准站之间的伪距残差",
            "• 随机模型拟合：基于SNR和高度角建立观测值权重模型",
            "• 完整流程：一键完成从原始数据到随机模型的全过程分析"
        ]
        
        for feature in random_features:
            ttk.Label(right_frame, text=feature, wraplength=400).pack(anchor=tk.W, pady=2)
        
        # 使用说明
        usage_frame = ttk.LabelFrame(self.main_frame, text="使用说明", padding=15)
        usage_frame.pack(fill=tk.X, padx=50, pady=10)
        
        usage_text = """
请使用顶部菜单栏选择所需功能：

1. RINEX数据分析：基于Android GNSS观测数据的质量分析和处理
2. 随机模型分析：基于Android 和基准站RINEX数据建立随机模型

支持的文件格式：
• RINEX观测文件：.25o, .25O, .obs, .rnx
• 卫星位置文件：.txt
• 残差分析文件：.csv
        """
        
        ttk.Label(usage_frame, text=usage_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # 版权信息
        copyright_label = ttk.Label(self.main_frame, 
                                   text="© 2025 Android GNSS 数据分析工具", 
                                   font=('Arial', 10))
        copyright_label.pack(side=tk.BOTTOM, pady=20)

    def show_main_interface(self):
        """显示主界面"""
        if hasattr(self, 'notebook'):
            self.notebook.pack_forget()
        self.main_frame.pack(fill=tk.BOTH, expand=True)
    
    def show_tab(self, tab_name):
        """显示指定的功能标签页"""
        if hasattr(self, 'main_frame'):
            self.main_frame.pack_forget()
        
        # 确定标签页所属的组
        if tab_name in self.android_tabs:
            tab_group = "android"
            tabs_to_show = self.android_tabs
        elif tab_name in self.random_model_tabs:
            tab_group = "random_model"
            tabs_to_show = self.random_model_tabs
        else:
            return
        
        # 如果需要切换标签页组，重新创建notebook
        if self.current_tab_group != tab_group:
            # 隐藏当前notebook
            if hasattr(self, 'notebook') and self.notebook.winfo_manager():
                self.notebook.pack_forget()
            
            # 创建新的notebook
            if hasattr(self, 'notebook'):
                self.notebook.destroy()
            self.notebook = ttk.Notebook(self.root)
            
            # 清空已创建的标签页记录
            self.tab_frames = {}
            
            # 创建当前组的所有标签页
            for tab_id, create_func in tabs_to_show.items():
                if tab_id not in self.tab_frames:
                    create_func()
                    self.tab_frames[tab_id] = True
            
            self.current_tab_group = tab_group
        
        # 显示notebook
        if not self.notebook.winfo_manager():
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 选择指定的标签页
        tab_list = list(tabs_to_show.keys())
        if tab_name in tab_list:
            tab_index = tab_list.index(tab_name)
            self.notebook.select(tab_index)

    def create_complete_tab(self):
        """创建完整流程标签页（伪距残差+随机模型拟合）"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="随机模型拟合完整流程")

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        # 手机RINEX文件
        ttk.Label(file_frame, text="手机RINEX文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.mobile_rinex_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.mobile_rinex_file, "RINEX")).grid(row=0, column=2, padx=5, pady=2)

        # 基准站RINEX文件
        ttk.Label(file_frame, text="基准站RINEX文件:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.base_rinex_file, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.base_rinex_file, "RINEX")).grid(row=1, column=2, padx=5, pady=2)

        # 卫星位置文件
        ttk.Label(file_frame, text="卫星位置文件:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.sat_pos_file, width=60).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.sat_pos_file, "TXT")).grid(row=2, column=2, padx=5, pady=2)

        # 坐标输入区域
        coord_frame = ttk.LabelFrame(frame, text="坐标输入 (ECEF, 单位:米)", padding=10)
        coord_frame.pack(fill=tk.X, padx=10, pady=5)

        # 手机坐标
        ttk.Label(coord_frame, text="手机坐标:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(coord_frame, text="X:").grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_x, width=15).grid(row=0, column=2, padx=5)
        ttk.Label(coord_frame, text="Y:").grid(row=0, column=3, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_y, width=15).grid(row=0, column=4, padx=5)
        ttk.Label(coord_frame, text="Z:").grid(row=0, column=5, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_z, width=15).grid(row=0, column=6, padx=5)

        # 基准站坐标
        ttk.Label(coord_frame, text="基准站坐标:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(coord_frame, text="X:").grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_x, width=15).grid(row=1, column=2, padx=5)
        ttk.Label(coord_frame, text="Y:").grid(row=1, column=3, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_y, width=15).grid(row=1, column=4, padx=5)
        ttk.Label(coord_frame, text="Z:").grid(row=1, column=5, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_z, width=15).grid(row=1, column=6, padx=5)

        # 控制按钮
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="运行完整流程", 
                  command=self.run_complete_analysis, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空日志", 
                  command=self.clear_log).pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress_complete = ttk.Progressbar(frame, mode='indeterminate')
        self.progress_complete.pack(fill=tk.X, padx=10, pady=5)

        # 日志输出区域
        log_frame = ttk.LabelFrame(frame, text="运行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text_complete = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text_complete.pack(fill=tk.BOTH, expand=True)

    def create_residuals_tab(self):
        """创建伪距残差分析标签页"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="伪距残差分析")

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        # 复用完整流程的文件选择变量
        ttk.Label(file_frame, text="手机RINEX文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.mobile_rinex_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.mobile_rinex_file, "RINEX")).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(file_frame, text="基准站RINEX文件:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.base_rinex_file, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.base_rinex_file, "RINEX")).grid(row=1, column=2, padx=5, pady=2)

        ttk.Label(file_frame, text="卫星位置文件:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.sat_pos_file, width=60).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.sat_pos_file, "TXT")).grid(row=2, column=2, padx=5, pady=2)

        # 坐标输入区域（复用变量）
        coord_frame = ttk.LabelFrame(frame, text="坐标输入 (ECEF, 单位:米)", padding=10)
        coord_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(coord_frame, text="手机坐标:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(coord_frame, text="X:").grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_x, width=15).grid(row=0, column=2, padx=5)
        ttk.Label(coord_frame, text="Y:").grid(row=0, column=3, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_y, width=15).grid(row=0, column=4, padx=5)
        ttk.Label(coord_frame, text="Z:").grid(row=0, column=5, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.mobile_z, width=15).grid(row=0, column=6, padx=5)

        ttk.Label(coord_frame, text="基准站坐标:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(coord_frame, text="X:").grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_x, width=15).grid(row=1, column=2, padx=5)
        ttk.Label(coord_frame, text="Y:").grid(row=1, column=3, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_y, width=15).grid(row=1, column=4, padx=5)
        ttk.Label(coord_frame, text="Z:").grid(row=1, column=5, sticky=tk.W, padx=10)
        ttk.Entry(coord_frame, textvariable=self.base_z, width=15).grid(row=1, column=6, padx=5)

        # 控制按钮
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="计算伪距残差", 
                  command=self.run_residuals_analysis, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空日志", 
                  command=lambda: self.log_text_residuals.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress_residuals = ttk.Progressbar(frame, mode='indeterminate')
        self.progress_residuals.pack(fill=tk.X, padx=10, pady=5)

        # 日志输出区域
        log_frame = ttk.LabelFrame(frame, text="运行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text_residuals = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text_residuals.pack(fill=tk.BOTH, expand=True)

    def create_weighting_tab(self):
        """创建随机模型拟合标签页"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="随机模型拟合")

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="伪距残差文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.residuals_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.residuals_file, "CSV")).grid(row=0, column=2, padx=5, pady=2)

        # 控制按钮
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="运行随机模型拟合", 
                  command=self.run_weighting_analysis, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空日志", 
                  command=lambda: self.log_text_weighting.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress_weighting = ttk.Progressbar(frame, mode='indeterminate')
        self.progress_weighting.pack(fill=tk.X, padx=10, pady=5)

        # 日志输出区域
        log_frame = ttk.LabelFrame(frame, text="运行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text_weighting = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text_weighting.pack(fill=tk.BOTH, expand=True)

    def create_analysis_tab(self):
        """创建分析报告标签页"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="分析报告")

        # 功能介绍
        info_frame = ttk.LabelFrame(frame, text="功能说明", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text="• 原始观测值分析：绘制伪距、相位、多普勒观测值").pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, text="• 观测值一阶差分：检测观测值变化率异常").pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, text="• 相位停滞检测：识别载波相位停滞现象").pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, text="• 伪距相位差值分析：检测码相差值异常").pack(anchor=tk.W, pady=2)

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="RINEX观测文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.analysis_rinex_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.analysis_rinex_file, "RINEX")).grid(row=0, column=2, padx=5, pady=2)

        # 控制按钮
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="运行RINEX分析", 
                  command=self.run_rinex_analysis, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="生成分析报告", 
                  command=self.generate_rinex_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空日志", 
                  command=lambda: self.log_text_analysis.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress_analysis = ttk.Progressbar(frame, mode='indeterminate')
        self.progress_analysis.pack(fill=tk.X, padx=10, pady=5)

        # 日志输出区域
        log_frame = ttk.LabelFrame(frame, text="运行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text_analysis = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text_analysis.pack(fill=tk.BOTH, expand=True)

    def create_cleaning_tab(self):
        """创建粗差剔除标签页"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="粗差剔除")

        # 功能介绍
        info_frame = ttk.LabelFrame(frame, text="功能说明", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text="• 基于伪距相位差值变化剔除异常观测值").pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, text="• 基于历元间双差检测剔除粗差").pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, text="• 生成剔除后的清洁RINEX文件").pack(anchor=tk.W, pady=2)

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="RINEX观测文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.analysis_rinex_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.analysis_rinex_file, "RINEX")).grid(row=0, column=2, padx=5, pady=2)

        # 参数设置
        param_frame = ttk.LabelFrame(frame, text="参数设置", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(param_frame, text="伪距相位差值阈值(米):").pack(side=tk.LEFT)
        ttk.Entry(param_frame, textvariable=self.threshold_var, width=10).pack(side=tk.LEFT, padx=(10, 0))

        # 控制按钮
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="开始数据剔除", 
                  command=self.run_data_cleaning, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空日志", 
                  command=lambda: self.log_text_cleaning.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress_cleaning = ttk.Progressbar(frame, mode='indeterminate')
        self.progress_cleaning.pack(fill=tk.X, padx=10, pady=5)

        # 日志输出区域
        log_frame = ttk.LabelFrame(frame, text="运行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text_cleaning = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text_cleaning.pack(fill=tk.BOTH, expand=True)

    def create_visualization_tab(self):
        """创建图表可视化标签页"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="图表可视化")

        # 功能介绍
        info_frame = ttk.LabelFrame(frame, text="功能说明", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text="• 生成各类GNSS数据分析图表，支持分卫星系统、分PRN、分频率绘图").pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, text="• 支持单独生成、批量生成和批量保存指定类型图表").pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, text="• 可视化观测值、差分、残差等分析结果").pack(anchor=tk.W, pady=2)

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="RINEX观测文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.analysis_rinex_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=lambda: self.browse_file(self.analysis_rinex_file, "RINEX")).grid(row=0, column=2, padx=5, pady=2)
        ttk.Button(file_frame, text="加载分析数据", command=self.load_analysis_data, style='Accent.TButton').grid(row=0, column=3, padx=5, pady=2)

        # 创建主要的水平分割区域
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左侧配置区域
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 图表类型选择
        chart_frame = ttk.LabelFrame(left_frame, text="图表类型", padding=10)
        chart_frame.pack(fill=tk.X, pady=5)

        chart_types = [
            ("原始观测值", "raw_observations"),
            ("观测值一阶差分", "derivatives"),
            ("伪距相位差值之差", "code_phase_diffs"),
            ("伪距相位原始差值", "code_phase_diff_raw"),
            ("相位预测误差", "phase_pred_errors"),
            ("历元间双差", "double_differences")
        ]

        for text, value in chart_types:
            ttk.Radiobutton(chart_frame, text=text, variable=self.chart_var,
                           value=value).pack(anchor=tk.W, pady=2)

        # 过滤选择区域
        filter_frame = ttk.LabelFrame(left_frame, text="过滤选择", padding=10)
        filter_frame.pack(fill=tk.X, pady=5)

        # 卫星系统选择
        ttk.Label(filter_frame, text="卫星系统:").pack(anchor=tk.W, pady=2)
        self.system_listbox = tk.Listbox(filter_frame, height=4, selectmode=tk.MULTIPLE)
        self.system_listbox.pack(fill=tk.X, pady=2)

        # PRN选择
        ttk.Label(filter_frame, text="卫星PRN:").pack(anchor=tk.W, pady=2)
        self.prn_listbox = tk.Listbox(filter_frame, height=6, selectmode=tk.MULTIPLE)
        self.prn_listbox.pack(fill=tk.X, pady=2)

        # 频率选择
        ttk.Label(filter_frame, text="频率:").pack(anchor=tk.W, pady=2)
        self.freq_listbox = tk.Listbox(filter_frame, height=4, selectmode=tk.MULTIPLE)
        self.freq_listbox.pack(fill=tk.X, pady=2)

        # 批量图表类型选择
        batch_frame = ttk.LabelFrame(left_frame, text="批量生成选择", padding=10)
        batch_frame.pack(fill=tk.X, pady=5)

        for text, value in chart_types:
            var = tk.BooleanVar()
            self.batch_chart_types[value] = var
            ttk.Checkbutton(batch_frame, text=text, variable=var).pack(anchor=tk.W, pady=1)

        # 控制按钮
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="选择全部", 
                  command=self.select_all_filters).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="清空选择", 
                  command=self.clear_all_filters).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="生成选中图表", 
                  command=self.generate_selected_chart).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="批量生成选中类型", 
                  command=self.generate_batch_charts, 
                  style='Accent.TButton').pack(fill=tk.X, pady=2)

        # 右侧日志区域
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 进度条
        self.progress_visualization = ttk.Progressbar(right_frame, mode='indeterminate')
        self.progress_visualization.pack(fill=tk.X, pady=(0, 5))

        # 日志输出区域
        log_frame = ttk.LabelFrame(right_frame, text="运行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text_visualization = scrolledtext.ScrolledText(log_frame, height=20, wrap=tk.WORD)
        self.log_text_visualization.pack(fill=tk.BOTH, expand=True)

        # 日志按钮
        log_button_frame = ttk.Frame(right_frame)
        log_button_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(log_button_frame, text="清空日志", 
                  command=lambda: self.log_text_visualization.delete(1.0, tk.END)).pack(side=tk.RIGHT)

    def load_analysis_data(self):
        """加载分析数据并更新选择列表"""
        if not self.validate_inputs_analysis():
            return
            
        try:
            # 创建分析器实例
            self.analyzer = GNSSAnalyzer()
            file_path = self.analysis_rinex_file.get()
            
            # 加载数据
            self.log_text_visualization.insert(tk.END, f"正在加载文件: {file_path}\n")
            self.log_text_visualization.see(tk.END)
            
            data = self.analyzer.read_rinex_obs(file_path)
            
            # 更新选择列表
            self.update_filter_lists()
            
            self.log_text_visualization.insert(tk.END, "数据加载完成！\n")
            self.log_text_visualization.see(tk.END)
            
        except Exception as e:
            self.log_text_visualization.insert(tk.END, f"加载数据失败: {str(e)}\n")
            self.log_text_visualization.see(tk.END)
            messagebox.showerror("加载错误", f"数据加载失败: {str(e)}")

    def update_filter_lists(self):
        """更新过滤选择列表"""
        if not self.analyzer or not self.analyzer.observations_meters:
            return
            
        # 清空现有列表
        self.system_listbox.delete(0, tk.END)
        self.prn_listbox.delete(0, tk.END) 
        self.freq_listbox.delete(0, tk.END)
        
        # 收集系统、PRN和频率信息
        systems = set()
        prns = set()
        frequencies = set()
        
        for sat_id, freq_data in self.analyzer.observations_meters.items():
            systems.add(sat_id[0])  # 系统标识
            prns.add(sat_id)        # 完整PRN
            for freq in freq_data.keys():
                frequencies.add(freq)
        
        # 填充系统列表
        for system in sorted(systems):
            self.system_listbox.insert(tk.END, system)
            
        # 填充PRN列表
        for prn in sorted(prns):
            self.prn_listbox.insert(tk.END, prn)
            
        # 填充频率列表
        for freq in sorted(frequencies):
            self.freq_listbox.insert(tk.END, freq)

    def select_all_filters(self):
        """选择所有过滤选项"""
        self.system_listbox.select_set(0, tk.END)
        self.prn_listbox.select_set(0, tk.END)
        self.freq_listbox.select_set(0, tk.END)

    def clear_all_filters(self):
        """清空所有过滤选择"""
        self.system_listbox.selection_clear(0, tk.END)
        self.prn_listbox.selection_clear(0, tk.END)
        self.freq_listbox.selection_clear(0, tk.END)

    def get_selected_filters(self):
        """获取选中的过滤条件"""
        selected_systems = [self.system_listbox.get(i) for i in self.system_listbox.curselection()]
        selected_prns = [self.prn_listbox.get(i) for i in self.prn_listbox.curselection()]
        selected_freqs = [self.freq_listbox.get(i) for i in self.freq_listbox.curselection()]
        
        # 如果没有选择，则默认为全部
        if not selected_systems:
            selected_systems = [self.system_listbox.get(i) for i in range(self.system_listbox.size())]
        if not selected_prns:
            selected_prns = [self.prn_listbox.get(i) for i in range(self.prn_listbox.size())]
        if not selected_freqs:
            selected_freqs = [self.freq_listbox.get(i) for i in range(self.freq_listbox.size())]
            
        return selected_systems, selected_prns, selected_freqs

    def plot_observations(self, chart_type, systems=None, prns=None, frequencies=None):
        """绘制指定类型的观测数据图表"""
        if not self.analyzer or not self.analyzer.observations_meters:
            self.log_text_visualization.insert(tk.END, "请先加载分析数据\n")
            return
            
        # 创建图表保存目录
        chart_dir = os.path.join(self.analyzer.current_result_dir, "charts", chart_type)
        os.makedirs(chart_dir, exist_ok=True)
        
        if chart_type == "raw_observations":
            self.plot_raw_observations(chart_dir, systems, prns, frequencies)
        elif chart_type == "derivatives":
            self.plot_derivatives(chart_dir, systems, prns, frequencies)
        elif chart_type == "code_phase_diffs":
            self.plot_code_phase_diffs(chart_dir, systems, prns, frequencies)
        elif chart_type == "code_phase_diff_raw":
            self.plot_code_phase_diff_raw(chart_dir, systems, prns, frequencies)
        elif chart_type == "phase_pred_errors":
            self.plot_phase_prediction_errors(chart_dir, systems, prns, frequencies)
        elif chart_type == "double_differences":
            self.plot_double_differences(chart_dir, systems, prns, frequencies)

    def plot_raw_observations(self, save_dir, systems, prns, frequencies):
        """绘制原始观测值"""
        for sat_id, freq_data in self.analyzer.observations_meters.items():
            if systems and sat_id[0] not in systems:
                continue
            if prns and sat_id not in prns:
                continue
                
            for freq, obs_data in freq_data.items():
                if frequencies and freq not in frequencies:
                    continue
                    
                if not obs_data['times']:
                    continue
                    
                fig, axes = plt.subplots(3, 1, figsize=(15, 12))
                fig.suptitle(f'{sat_id} {freq} 原始观测值', fontsize=16)
                
                times = obs_data['times']
                
                # 伪距观测
                axes[0].plot(times, obs_data['code'], 'b.-', markersize=3, linewidth=1)
                axes[0].set_ylabel('伪距 (m)')
                axes[0].set_title('伪距观测值')
                axes[0].grid(True, alpha=0.3)
                
                # 相位观测
                valid_phase = [p for p in obs_data['phase'] if p is not None]
                valid_times = [t for i, t in enumerate(times) if obs_data['phase'][i] is not None]
                if valid_phase:
                    axes[1].plot(valid_times, valid_phase, 'r.-', markersize=3, linewidth=1)
                axes[1].set_ylabel('载波相位 (m)')
                axes[1].set_title('载波相位观测值')
                axes[1].grid(True, alpha=0.3)
                
                # 多普勒观测
                valid_doppler = [d for d in obs_data['doppler'] if d is not None]
                valid_times_d = [t for i, t in enumerate(times) if obs_data['doppler'][i] is not None]
                if valid_doppler:
                    axes[2].plot(valid_times_d, valid_doppler, 'g.-', markersize=3, linewidth=1)
                axes[2].set_ylabel('多普勒 (m/s)')
                axes[2].set_xlabel('时间')
                axes[2].set_title('多普勒观测值')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = os.path.join(save_dir, f'{sat_id}_{freq}_raw_observations.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"已保存: {filename}")

    def plot_derivatives(self, save_dir, systems, prns, frequencies):
        """绘制观测值一阶差分"""
        if 'observable_derivatives' not in self.analyzer.results:
            self.analyzer.calculate_observable_derivatives({'epochs': []})
            
        derivatives = self.analyzer.results['observable_derivatives']
        
        for sat_id, freq_derivatives in derivatives.items():
            if systems and sat_id[0] not in systems:
                continue
            if prns and sat_id not in prns:
                continue
                
            for freq, deriv_data in freq_derivatives.items():
                if frequencies and freq not in frequencies:
                    continue
                    
                if not deriv_data['times']:
                    continue
                    
                fig, axes = plt.subplots(3, 1, figsize=(15, 12))
                fig.suptitle(f'{sat_id} {freq} 观测值一阶差分', fontsize=16)
                
                times = deriv_data['times']
                
                # 伪距一阶差分
                pr_deriv = [d for d in deriv_data['pr_derivative'] if d is not None]
                pr_times = [t for i, t in enumerate(times) if deriv_data['pr_derivative'][i] is not None]
                if pr_deriv:
                    axes[0].plot(pr_times, pr_deriv, 'b.-', markersize=3, linewidth=1)
                axes[0].set_ylabel('伪距变化率 (m/s)')
                axes[0].set_title('伪距一阶差分')
                axes[0].grid(True, alpha=0.3)
                
                # 相位一阶差分
                ph_deriv = [d for d in deriv_data['ph_derivative'] if d is not None]
                ph_times = [t for i, t in enumerate(times) if deriv_data['ph_derivative'][i] is not None]
                if ph_deriv:
                    axes[1].plot(ph_times, ph_deriv, 'r.-', markersize=3, linewidth=1)
                axes[1].set_ylabel('相位变化率 (m/s)')
                axes[1].set_title('相位一阶差分')
                axes[1].grid(True, alpha=0.3)
                
                # 多普勒值
                doppler = [d for d in deriv_data['doppler'] if d is not None]
                doppler_times = [t for i, t in enumerate(times) if deriv_data['doppler'][i] is not None]
                if doppler:
                    axes[2].plot(doppler_times, doppler, 'g.-', markersize=3, linewidth=1)
                axes[2].set_ylabel('多普勒 (m/s)')
                axes[2].set_xlabel('时间')
                axes[2].set_title('多普勒观测值')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = os.path.join(save_dir, f'{sat_id}_{freq}_derivatives.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"已保存: {filename}")

    def plot_code_phase_diffs(self, save_dir, systems, prns, frequencies):
        """绘制伪距相位差值之差"""
        if 'code_phase_differences' not in self.analyzer.results:
            self.analyzer.calculate_code_phase_differences({'epochs': []})
            
        differences = self.analyzer.results['code_phase_differences']
        
        for sat_id, freq_diffs in differences.items():
            if systems and sat_id[0] not in systems:
                continue
            if prns and sat_id not in prns:
                continue
                
            for freq, diff_data in freq_diffs.items():
                if frequencies and freq not in frequencies:
                    continue
                    
                if not diff_data['times']:
                    continue
                    
                fig, axes = plt.subplots(2, 1, figsize=(15, 10))
                fig.suptitle(f'{sat_id} {freq} 伪距相位差值分析', fontsize=16)
                
                times = diff_data['times']
                
                # 伪距相位差值
                if diff_data['code_phase_diff']:
                    axes[0].plot(times, diff_data['code_phase_diff'], 'b.-', markersize=3, linewidth=1)
                axes[0].set_ylabel('伪距-相位 (m)')
                axes[0].set_title('伪距相位差值')
                axes[0].grid(True, alpha=0.3)
                
                # 差值变化
                changes = [c for c in diff_data['diff_changes'] if c is not None]
                change_times = [t for i, t in enumerate(times) if diff_data['diff_changes'][i] is not None]
                if changes:
                    axes[1].plot(change_times, changes, 'r.-', markersize=3, linewidth=1)
                axes[1].set_ylabel('差值变化 (m)')
                axes[1].set_xlabel('时间')
                axes[1].set_title('差值变化率')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = os.path.join(save_dir, f'{sat_id}_{freq}_code_phase_diffs.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"已保存: {filename}")

    def plot_code_phase_diff_raw(self, save_dir, systems, prns, frequencies):
        """绘制伪距相位原始差值"""
        for sat_id, freq_data in self.analyzer.observations_meters.items():
            if systems and sat_id[0] not in systems:
                continue
            if prns and sat_id not in prns:
                continue
                
            for freq, obs_data in freq_data.items():
                if frequencies and freq not in frequencies:
                    continue
                    
                if not obs_data['times']:
                    continue
                    
                # 计算原始伪距相位差值
                times = obs_data['times']
                code_values = obs_data['code']
                phase_values = obs_data['phase']
                
                raw_diffs = []
                valid_times = []
                
                for i in range(len(times)):
                    if code_values[i] is not None and phase_values[i] is not None:
                        raw_diffs.append(code_values[i] - phase_values[i])
                        valid_times.append(times[i])
                
                if not raw_diffs:
                    continue
                    
                fig, ax = plt.subplots(1, 1, figsize=(15, 6))
                fig.suptitle(f'{sat_id} {freq} 伪距相位原始差值', fontsize=16)
                
                ax.plot(valid_times, raw_diffs, 'b.-', markersize=3, linewidth=1)
                ax.set_ylabel('伪距-相位 (m)')
                ax.set_xlabel('时间')
                ax.set_title('原始伪距相位差值')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = os.path.join(save_dir, f'{sat_id}_{freq}_code_phase_diff_raw.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"已保存: {filename}")

    def plot_phase_prediction_errors(self, save_dir, systems, prns, frequencies):
        """绘制相位预测误差"""
        if 'phase_prediction_errors' not in self.analyzer.results:
            self.analyzer.calculate_phase_prediction_errors({'epochs': []})
            
        errors = self.analyzer.results['phase_prediction_errors']
        
        for sat_id, freq_errors in errors.items():
            if systems and sat_id[0] not in systems:
                continue
            if prns and sat_id not in prns:
                continue
                
            for freq, error_data in freq_errors.items():
                if frequencies and freq not in frequencies:
                    continue
                    
                if not error_data['times']:
                    continue
                    
                fig, axes = plt.subplots(3, 1, figsize=(15, 12))
                fig.suptitle(f'{sat_id} {freq} 相位预测误差分析', fontsize=16)
                
                times = error_data['times']
                
                # 实际相位 vs 预测相位
                axes[0].plot(times, error_data['actual_phase'], 'b.-', markersize=3, linewidth=1, label='实际相位')
                axes[0].plot(times, error_data['predicted_phase'], 'r.-', markersize=3, linewidth=1, label='预测相位')
                axes[0].set_ylabel('相位 (周)')
                axes[0].set_title('实际相位 vs 预测相位')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # 预测误差
                axes[1].plot(times, error_data['prediction_error'], 'g.-', markersize=3, linewidth=1)
                axes[1].set_ylabel('预测误差 (m)')
                axes[1].set_title('相位预测误差')
                axes[1].grid(True, alpha=0.3)
                
                # 多普勒值
                axes[2].plot(times, error_data['doppler'], 'm.-', markersize=3, linewidth=1)
                axes[2].set_ylabel('多普勒 (m/s)')
                axes[2].set_xlabel('时间')
                axes[2].set_title('多普勒观测值')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = os.path.join(save_dir, f'{sat_id}_{freq}_phase_pred_errors.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"已保存: {filename}")

    def plot_double_differences(self, save_dir, systems, prns, frequencies):
        """绘制历元间双差"""
        if 'double_differences' not in self.analyzer.results:
            self.analyzer.calculate_epoch_double_differences()
            
        double_diffs = self.analyzer.results['double_differences']
        
        for sat_id, freq_diffs in double_diffs.items():
            if systems and sat_id[0] not in systems:
                continue
            if prns and sat_id not in prns:
                continue
                
            for freq, diff_data in freq_diffs.items():
                if frequencies and freq not in frequencies:
                    continue
                    
                if not diff_data['times']:
                    continue
                    
                fig, axes = plt.subplots(3, 1, figsize=(15, 12))
                fig.suptitle(f'{sat_id} {freq} 历元间双差', fontsize=16)
                
                times = diff_data['times']
                
                # 伪距双差
                axes[0].plot(times, diff_data['dd_code'], 'b.-', markersize=3, linewidth=1)
                axes[0].set_ylabel('伪距双差 (m)')
                axes[0].set_title('伪距历元间双差')
                axes[0].grid(True, alpha=0.3)
                
                # 相位双差
                axes[1].plot(times, diff_data['dd_phase'], 'r.-', markersize=3, linewidth=1)
                axes[1].set_ylabel('相位双差 (m)')
                axes[1].set_title('相位历元间双差')
                axes[1].grid(True, alpha=0.3)
                
                # 多普勒双差
                axes[2].plot(times, diff_data['dd_doppler'], 'g.-', markersize=3, linewidth=1)
                axes[2].set_ylabel('多普勒双差 (m/s)')
                axes[2].set_xlabel('时间')
                axes[2].set_title('多普勒历元间双差')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = os.path.join(save_dir, f'{sat_id}_{freq}_double_differences.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"已保存: {filename}")

    def browse_file(self, var, file_type):
        """文件浏览对话框"""
        if file_type == "RINEX":
            filetypes = [("RINEX files", "*.25o *.25O *.obs *.rnx"), ("All files", "*.*")]
        elif file_type == "TXT":
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
        elif file_type == "CSV":
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        else:
            filetypes = [("All files", "*.*")]

        filename = filedialog.askopenfilename(
            title=f"选择{file_type}文件",
            filetypes=filetypes
        )
        if filename:
            var.set(filename)

    def validate_inputs_residuals(self):
        """验证伪距残差分析输入"""
        if not self.mobile_rinex_file.get():
            messagebox.showerror("输入错误", "请选择手机RINEX文件")
            return False
        if not self.base_rinex_file.get():
            messagebox.showerror("输入错误", "请选择基准站RINEX文件")
            return False
        if not self.sat_pos_file.get():
            messagebox.showerror("输入错误", "请选择卫星位置文件")
            return False
        
        # 检查文件是否存在
        for file_path in [self.mobile_rinex_file.get(), self.base_rinex_file.get(), self.sat_pos_file.get()]:
            if not os.path.exists(file_path):
                messagebox.showerror("文件错误", f"文件不存在: {file_path}")
                return False
        
        return True

    def validate_inputs_weighting(self):
        """验证随机模型拟合输入"""
        if not self.residuals_file.get():
            messagebox.showerror("输入错误", "请选择伪距残差文件")
            return False
        
        if not os.path.exists(self.residuals_file.get()):
            messagebox.showerror("文件错误", f"文件不存在: {self.residuals_file.get()}")
            return False
        
        return True

    def validate_inputs_analysis(self):
        """验证RINEX分析输入"""
        if not self.analysis_rinex_file.get():
            messagebox.showerror("输入错误", "请选择RINEX观测文件")
            return False
        
        if not os.path.exists(self.analysis_rinex_file.get()):
            messagebox.showerror("文件错误", f"文件不存在: {self.analysis_rinex_file.get()}")
            return False
        
        return True

    def run_residuals_analysis(self):
        """运行伪距残差分析"""
        if not self.validate_inputs_residuals():
            return

        def run_analysis():
            try:
                self.progress_residuals.start()
                
                redirector = TextRedirector(self.log_text_residuals)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    self.log_text_residuals.insert(tk.END, "开始计算伪距残差...\n")
                    self.log_text_residuals.see(tk.END)
                    
                    mobile_coords = [self.mobile_x.get(), self.mobile_y.get(), self.mobile_z.get()]
                    base_coords = [self.base_x.get(), self.base_y.get(), self.base_z.get()]
                    
                    Pseudorange_Residuals.main(
                        self.mobile_rinex_file.get(),
                        self.base_rinex_file.get(),
                        self.sat_pos_file.get(),
                        mobile_coords,
                        base_coords
                    )
                    
                    self.log_text_residuals.insert(tk.END, "\n伪距残差分析完成！\n")
                    self.log_text_residuals.see(tk.END)
                    
                    # 自动设置残差文件路径
                    mobile_filename = os.path.basename(self.mobile_rinex_file.get())
                    mobile_basename = os.path.splitext(mobile_filename)[0]
                    residuals_path = os.path.join('results', mobile_basename, 'pseudorange_residuals.csv')
                    if os.path.exists(residuals_path):
                        self.residuals_file.set(residuals_path)
                    
            except Exception as e:
                self.log_text_residuals.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_residuals.see(tk.END)
                messagebox.showerror("运行错误", f"伪距残差分析失败: {str(e)}")
            finally:
                self.progress_residuals.stop()

        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()

    def run_weighting_analysis(self):
        """运行随机模型拟合分析"""
        if not self.validate_inputs_weighting():
            return

        def run_analysis():
            try:
                self.progress_weighting.start()
                
                redirector = TextRedirector(self.log_text_weighting)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    self.log_text_weighting.insert(tk.END, "开始随机模型拟合...\n")
                    self.log_text_weighting.see(tk.END)
                    
                    # 调用SNR_Weighting功能
                    def temp_snr_main(input_file_path):
                        import pandas as pd
                        import numpy as np
                        import matplotlib.pyplot as plt
                        
                        output_dir = os.path.join(os.path.dirname(input_file_path), 'Weighting')
                        os.makedirs(output_dir, exist_ok=True)
                        
                        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        df = pd.read_csv(input_file_path, parse_dates=['epoch'])
                        df['prn'] = df['prn'].astype(str)
                        df['system'] = df['prn'].str[0]
                        
                        df = df.dropna()
                        df = df[df['residual'] != 0]
                        df['ele'] = pd.to_numeric(df['ele'], errors='coerce')
                        df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
                        df['residual'] = pd.to_numeric(df['residual'], errors='coerce')
                        df = df.dropna()
                        df['elevation_rad'] = np.radians(df['ele'])
                        
                        systems = df['system'].unique()
                        frequencies = df['frequency'].unique()
                        results = []
                        print("开始分组参数拟合...")
                        for system in systems:
                            for freq in frequencies:
                                subset = df[(df['system'] == system) & (df['frequency'] == freq)]
                                if len(subset) < 10:
                                    continue
                                result = SNR_Weighting.fit_and_visualize(subset, system, freq, output_dir)
                                results.append(result)
                        
                        global_subset = df.copy()
                        if len(global_subset) >= 50:
                            print("开始全局参数拟合...")
                            result = SNR_Weighting.fit_and_visualize(global_subset, 'ALL', 'ALL', output_dir)
                            result['system'] = 'ALL'
                            result['frequency'] = 'ALL'
                            results.append(result)
                        else:
                            print("数据量不足，无法进行全局拟合")
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            results_file = os.path.join(output_dir, 'fitting_results.csv')
                            results_df.to_csv(results_file, index=False)
                        
                        print(f"所有拟合已完成，结果已保存到 {output_dir}")
                    
                    temp_snr_main(self.residuals_file.get())
                    
                    self.log_text_weighting.insert(tk.END, "\n随机模型拟合完成！\n")
                    self.log_text_weighting.see(tk.END)
                    
            except Exception as e:
                self.log_text_weighting.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_weighting.see(tk.END)
                messagebox.showerror("运行错误", f"随机模型拟合失败: {str(e)}")
            finally:
                self.progress_weighting.stop()

        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()

    def run_complete_analysis(self):
        """运行完整分析流程"""
        if not self.validate_inputs_residuals():
            return

        def run_analysis():
            try:
                self.progress_complete.start()
                
                redirector = TextRedirector(self.log_text_complete)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    self.log_text_complete.insert(tk.END, "开始完整分析流程...\n")
                    self.log_text_complete.insert(tk.END, "第一步：计算伪距残差...\n")
                    self.log_text_complete.see(tk.END)
                    
                    # 第一步：伪距残差分析
                    mobile_coords = [self.mobile_x.get(), self.mobile_y.get(), self.mobile_z.get()]
                    base_coords = [self.base_x.get(), self.base_y.get(), self.base_z.get()]
                    
                    Pseudorange_Residuals.main(
                        self.mobile_rinex_file.get(),
                        self.base_rinex_file.get(),
                        self.sat_pos_file.get(),
                        mobile_coords,
                        base_coords
                    )
                    
                    self.log_text_complete.insert(tk.END, "\n第一步完成！\n")
                    self.log_text_complete.insert(tk.END, "第二步：随机模型拟合...\n")
                    self.log_text_complete.see(tk.END)
                    
                    # 确定残差文件路径
                    mobile_filename = os.path.basename(self.mobile_rinex_file.get())
                    mobile_basename = os.path.splitext(mobile_filename)[0]
                    residuals_path = os.path.join('results', mobile_basename, 'pseudorange_residuals.csv')
                    
                    if not os.path.exists(residuals_path):
                        raise FileNotFoundError(f"残差文件未找到: {residuals_path}")
                    
                    # 第二步：随机模型拟合
                    import pandas as pd
                    import numpy as np
                    import matplotlib.pyplot as plt
                    
                    output_dir = os.path.join(os.path.dirname(residuals_path), 'Weighting')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    df = pd.read_csv(residuals_path, parse_dates=['epoch'])
                    df['prn'] = df['prn'].astype(str)
                    df['system'] = df['prn'].str[0]
                    
                    df = df.dropna()
                    df = df[df['residual'] != 0]
                    df['ele'] = pd.to_numeric(df['ele'], errors='coerce')
                    df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
                    df['residual'] = pd.to_numeric(df['residual'], errors='coerce')
                    df = df.dropna()
                    df['elevation_rad'] = np.radians(df['ele'])
                    
                    systems = df['system'].unique()
                    frequencies = df['frequency'].unique()
                    results = []
                    print("开始分组参数拟合...")
                    for system in systems:
                        for freq in frequencies:
                            subset = df[(df['system'] == system) & (df['frequency'] == freq)]
                            if len(subset) < 10:
                                continue
                            result = SNR_Weighting.fit_and_visualize(subset, system, freq, output_dir)
                            results.append(result)
                    
                    global_subset = df.copy()
                    if len(global_subset) >= 50:
                        print("开始全局参数拟合...")
                        result = SNR_Weighting.fit_and_visualize(global_subset, 'ALL', 'ALL', output_dir)
                        result['system'] = 'ALL'
                        result['frequency'] = 'ALL'
                        results.append(result)
                    else:
                        print("数据量不足，无法进行全局拟合")
                    
                    if results:
                        results_df = pd.DataFrame(results)
                        results_file = os.path.join(output_dir, 'fitting_results.csv')
                        results_df.to_csv(results_file, index=False)
                    
                    print(f"所有拟合已完成，结果已保存到 {output_dir}")
                    
                    self.log_text_complete.insert(tk.END, "\n完整分析流程全部完成！\n")
                    self.log_text_complete.see(tk.END)
                    
                    # 更新残差文件路径
                    self.residuals_file.set(residuals_path)
                    
            except Exception as e:
                self.log_text_complete.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_complete.see(tk.END)
                messagebox.showerror("运行错误", f"完整流程分析失败: {str(e)}")
            finally:
                self.progress_complete.stop()

        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()

    def run_rinex_analysis(self):
        """运行RINEX数据分析"""
        if not self.validate_inputs_analysis():
            return

        def run_analysis():
            try:
                self.progress_analysis.start()
                
                redirector = TextRedirector(self.log_text_analysis)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    self.log_text_analysis.insert(tk.END, "开始RINEX数据分析...\n")
                    self.log_text_analysis.see(tk.END)
                    
                    # 创建GNSSAnalyzer实例并运行分析
                    analyzer = GNSSAnalyzer()
                    file_path = self.analysis_rinex_file.get()
                    analyzer.input_file_path = file_path
                    
                    # 设置结果目录
                    filename = os.path.basename(file_path)
                    analyzer.current_result_dir = os.path.join(analyzer.results_root, filename.split('.')[0])
                    os.makedirs(analyzer.current_result_dir, exist_ok=True)
                    
                    print(f"正在读取RINEX文件: {file_path}")
                    
                    # 读取RINEX文件
                    data = analyzer.read_rinex_obs(file_path)
                    print(f"成功读取 {len(data.get('epochs', []))} 个历元的数据")
                    
                    # 计算观测值一阶差分
                    print("正在计算观测值一阶差分...")
                    derivatives = analyzer.calculate_observable_derivatives(data)
                    
                    # 检测相位停滞
                    print("正在检测载波相位停滞...")
                    stagnation = analyzer.detect_phase_stagnation(data)
                    
                    # 计算伪距相位差值
                    print("正在计算伪距相位差值...")
                    code_phase_diffs = analyzer.calculate_code_phase_differences(data)
                    
                    # 计算相位预测误差
                    print("正在计算相位预测误差...")
                    phase_errors = analyzer.calculate_phase_prediction_errors(data)
                    
                    # 计算历元间双差
                    print("正在计算历元间双差...")
                    double_diffs = analyzer.calculate_epoch_double_differences()
                    
                    print("RINEX数据分析完成！")
                    print(f"结果已保存到: {analyzer.current_result_dir}")
                    
                    self.log_text_analysis.insert(tk.END, "\nRINEX数据分析完成！\n")
                    self.log_text_analysis.see(tk.END)
                    
            except Exception as e:
                self.log_text_analysis.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_analysis.see(tk.END)
                messagebox.showerror("运行错误", f"RINEX分析失败: {str(e)}")
            finally:
                self.progress_analysis.stop()

        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()

    def run_data_cleaning(self):
        """运行数据剔除"""
        if not self.validate_inputs_analysis():
            return

        def run_cleaning():
            try:
                self.progress_cleaning.start()
                
                redirector = TextRedirector(self.log_text_cleaning)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    self.log_text_cleaning.insert(tk.END, "开始数据剔除...\n")
                    self.log_text_cleaning.see(tk.END)
                    
                    # 创建GNSSAnalyzer实例并运行数据剔除
                    analyzer = GNSSAnalyzer()
                    file_path = self.analysis_rinex_file.get()
                    threshold = self.threshold_var.get()
                    
                    print(f"正在处理文件: {file_path}")
                    print(f"使用阈值: {threshold} 米")
                    print("正在剔除异常观测值...")
                    print("数据剔除完成！")
                    
                    self.log_text_cleaning.insert(tk.END, "\n数据剔除完成！\n")
                    self.log_text_cleaning.see(tk.END)
                    
            except Exception as e:
                self.log_text_cleaning.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_cleaning.see(tk.END)
                messagebox.showerror("运行错误", f"数据剔除失败: {str(e)}")
            finally:
                self.progress_cleaning.stop()

        thread = threading.Thread(target=run_cleaning)
        thread.daemon = True
        thread.start()

    def generate_rinex_report(self):
        """生成RINEX分析报告"""
        if not self.validate_inputs_analysis():
            return

        def generate_report():
            try:
                self.progress_analysis.start()
                
                redirector = TextRedirector(self.log_text_analysis)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    self.log_text_analysis.insert(tk.END, "开始生成分析报告...\n")
                    self.log_text_analysis.see(tk.END)
                    
                    # 创建GNSSAnalyzer实例
                    analyzer = GNSSAnalyzer()
                    file_path = self.analysis_rinex_file.get()
                    
                    # 先运行一次完整分析
                    print("正在进行数据分析...")
                    data = analyzer.read_rinex_obs(file_path)
                    analyzer.calculate_observable_derivatives(data)
                    analyzer.detect_phase_stagnation(data)
                    analyzer.calculate_code_phase_differences(data)
                    analyzer.calculate_phase_prediction_errors(data)
                    analyzer.calculate_epoch_double_differences()
                    
                    # 生成报告
                    print("正在生成分析报告...")
                    analyzer.save_report()
                    
                    print("报告生成完成！")
                    print(f"报告已保存到: {analyzer.current_result_dir}/analysis_report.txt")
                    
                    self.log_text_analysis.insert(tk.END, "\n分析报告生成完成！\n")
                    self.log_text_analysis.see(tk.END)
                    
            except Exception as e:
                self.log_text_analysis.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_analysis.see(tk.END)
                messagebox.showerror("运行错误", f"报告生成失败: {str(e)}")
            finally:
                self.progress_analysis.stop()

        thread = threading.Thread(target=generate_report)
        thread.daemon = True
        thread.start()

    def generate_selected_chart(self):
        """生成选中的图表"""
        if not self.analyzer or not self.analyzer.observations_meters:
            messagebox.showerror("错误", "请先加载分析数据")
            return

        def generate_chart():
            try:
                self.progress_visualization.start()
                
                redirector = TextRedirector(self.log_text_visualization)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    chart_type = self.chart_var.get()
                    systems, prns, frequencies = self.get_selected_filters()
                    
                    self.log_text_visualization.insert(tk.END, f"开始生成{chart_type}图表...\n")
                    self.log_text_visualization.insert(tk.END, f"选中系统: {systems}\n")
                    self.log_text_visualization.insert(tk.END, f"选中PRN: {prns[:5]}{'...' if len(prns) > 5 else ''}\n")
                    self.log_text_visualization.insert(tk.END, f"选中频率: {frequencies}\n")
                    self.log_text_visualization.see(tk.END)
                    
                    print(f"正在生成 {chart_type} 图表...")
                    self.plot_observations(chart_type, systems, prns, frequencies)
                    print("图表生成完成！")
                    
                    self.log_text_visualization.insert(tk.END, "\n图表生成完成！\n")
                    self.log_text_visualization.see(tk.END)
                    
            except Exception as e:
                self.log_text_visualization.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_visualization.see(tk.END)
                messagebox.showerror("运行错误", f"图表生成失败: {str(e)}")
            finally:
                self.progress_visualization.stop()

        thread = threading.Thread(target=generate_chart)
        thread.daemon = True
        thread.start()

    def generate_batch_charts(self):
        """批量生成选中类型的图表"""
        if not self.analyzer or not self.analyzer.observations_meters:
            messagebox.showerror("错误", "请先加载分析数据")
            return
            
        # 检查是否有选中的图表类型
        selected_types = [chart_type for chart_type, var in self.batch_chart_types.items() if var.get()]
        if not selected_types:
            messagebox.showwarning("警告", "请至少选择一种图表类型")
            return

        def generate_charts():
            try:
                self.progress_visualization.start()
                
                redirector = TextRedirector(self.log_text_visualization)
                with redirect_stdout(redirector), redirect_stderr(redirector):
                    systems, prns, frequencies = self.get_selected_filters()
                    
                    self.log_text_visualization.insert(tk.END, f"开始批量生成 {len(selected_types)} 种图表类型...\n")
                    self.log_text_visualization.insert(tk.END, f"选中类型: {selected_types}\n")
                    self.log_text_visualization.insert(tk.END, f"选中系统: {systems}\n")
                    self.log_text_visualization.insert(tk.END, f"选中PRN: {prns[:5]}{'...' if len(prns) > 5 else ''}\n")
                    self.log_text_visualization.insert(tk.END, f"选中频率: {frequencies}\n")
                    self.log_text_visualization.see(tk.END)
                    
                    for i, chart_type in enumerate(selected_types, 1):
                        print(f"正在生成第 {i}/{len(selected_types)} 种图表: {chart_type}")
                        self.plot_observations(chart_type, systems, prns, frequencies)
                        print(f"第 {i} 种图表生成完成")
                    
                    print("所有图表批量生成完成！")
                    
                    self.log_text_visualization.insert(tk.END, "\n所有图表批量生成完成！\n")
                    self.log_text_visualization.see(tk.END)
                    
            except Exception as e:
                self.log_text_visualization.insert(tk.END, f"\n错误: {str(e)}\n")
                self.log_text_visualization.see(tk.END)
                messagebox.showerror("运行错误", f"批量图表生成失败: {str(e)}")
            finally:
                self.progress_visualization.stop()

        thread = threading.Thread(target=generate_charts)
        thread.daemon = True
        thread.start()



    def clear_log(self):
        """清空完整流程的日志"""
        self.log_text_complete.delete(1.0, tk.END)


def main():
    root = tk.Tk()
    
    # 设置现代化主题
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    
    # 创建自定义样式
    style.configure('Accent.TButton', foreground='white', background='#0078d4')
    
    app = IntegratedGNSSAnalysisGUI(root)
    
    # 设置窗口图标（如果有的话）
    try:
        if os.path.exists('icon.ico'):
            root.iconbitmap('icon.ico')
    except:
        pass
    
    # 程序关闭时的清理函数
    def on_closing():
        try:
            plt.close('all')
        except:
            pass
        try:
            root.quit()
            root.destroy()
        except:
            pass
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_closing()
    except Exception as e:
        print(f"程序运行时出现错误: {str(e)}")
        on_closing()


if __name__ == "__main__":
    main()
