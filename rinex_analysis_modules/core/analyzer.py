"""
GNSS数据分析核心模块
包含观测数据处理、分析算法等核心功能
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from collections import defaultdict
import bisect

from .config import GNSSConfig

class GNSSAnalyzer:
    """GNSS观测数据分析与问题检测器"""
    
    def __init__(self):
        # 配置对象
        self.config = GNSSConfig()
        
        # 文件路径和结果目录
        self.input_file_path = None
        self.results_root = "results"  # 结果根目录
        self.current_result_dir = None
        self.receiver_data = None  # 存储接收机观测数据
        
        # 进度回调
        self.progress_callback = None
        self.current_stage = 0
        self.total_stages = 5
        self.stage_progress = 0
        self.stage_max = 100
        
        # 缓存
        self._isb_time_index_cache = {}
        
        # 输出和存储
        self.output_format = "rinex"
        self.cleaned_observations = {}
        self.observations_meters = {}
        
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
    
    def calculate_observable_derivatives(self, data: Dict) -> Dict:
        """计算观测量一阶导数"""
        derivatives = {
            'code_derivatives': {},
            'phase_derivatives': {},
            'doppler_derivatives': {},
            'snr_derivatives': {}
        }
        
        if not data['epochs']:
            return derivatives
            
        # 为每个卫星和频率计算导数
        for epoch in data['epochs']:
            for sat_id in epoch['satellites']:
                if sat_id not in derivatives['code_derivatives']:
                    derivatives['code_derivatives'][sat_id] = {}
                    derivatives['phase_derivatives'][sat_id] = {}
                    derivatives['doppler_derivatives'][sat_id] = {}
                    derivatives['snr_derivatives'][sat_id] = {}
                    
                for freq in epoch['satellites'][sat_id]:
                    if freq not in derivatives['code_derivatives'][sat_id]:
                        derivatives['code_derivatives'][sat_id][freq] = []
                        derivatives['phase_derivatives'][sat_id][freq] = []
                        derivatives['doppler_derivatives'][sat_id][freq] = []
                        derivatives['snr_derivatives'][sat_id][freq] = []
        
        # 收集所有时间序列数据
        time_series = {}
        for epoch in data['epochs']:
            epoch_time = epoch['time']
            for sat_id in epoch['satellites']:
                for freq in epoch['satellites'][sat_id]:
                    key = (sat_id, freq)
                    if key not in time_series:
                        time_series[key] = {
                            'times': [],
                            'code': [],
                            'phase': [],
                            'doppler': [],
                            'snr': []
                        }
                    
                    obs = epoch['satellites'][sat_id][freq]
                    time_series[key]['times'].append(epoch_time)
                    time_series[key]['code'].append(obs.get('code', np.nan))
                    time_series[key]['phase'].append(obs.get('phase', np.nan))
                    time_series[key]['doppler'].append(obs.get('doppler', np.nan))
                    time_series[key]['snr'].append(obs.get('snr', np.nan))
        
        # 计算导数
        for (sat_id, freq), series in time_series.items():
            if len(series['times']) < 2:
                continue
                
            times = np.array(series['times'])
            code = np.array(series['code'])
            phase = np.array(series['phase'])
            doppler = np.array(series['doppler'])
            snr = np.array(series['snr'])
            
            # 计算时间差（秒）
            dt = np.diff([(t - times[0]).total_seconds() for t in times])
            
            # 计算各观测量的导数
            if len(dt) > 0:
                derivatives['code_derivatives'][sat_id][freq] = np.diff(code) / dt
                derivatives['phase_derivatives'][sat_id][freq] = np.diff(phase) / dt
                derivatives['doppler_derivatives'][sat_id][freq] = np.diff(doppler) / dt
                derivatives['snr_derivatives'][sat_id][freq] = np.diff(snr) / dt
        
        return derivatives
    
    def detect_phase_stagnation(self, data: Dict, threshold_cycles: float = 0.1, min_consecutive: int = 5) -> Dict:
        """检测载波相位停滞现象"""
        stagnation_results = {
            'detected': False,
            'details': {},
            'summary': {
                'total_satellites': 0,
                'affected_satellites': 0,
                'stagnation_epochs': 0
            }
        }
        
        if not data['epochs']:
            return stagnation_results
        
        # 为每个卫星-频率组合检测相位停滞
        for epoch in data['epochs']:
            for sat_id in epoch['satellites']:
                if sat_id not in stagnation_results['details']:
                    stagnation_results['details'][sat_id] = {}
                    
                for freq in epoch['satellites'][sat_id]:
                    if freq not in stagnation_results['details'][sat_id]:
                        stagnation_results['details'][sat_id][freq] = {
                            'stagnation_periods': [],
                            'is_stagnant': False
                        }
        
        # 收集相位时间序列
        phase_series = {}
        for epoch in data['epochs']:
            for sat_id in epoch['satellites']:
                for freq in epoch['satellites'][sat_id]:
                    key = (sat_id, freq)
                    if key not in phase_series:
                        phase_series[key] = []
                    
                    phase = epoch['satellites'][sat_id][freq].get('phase', np.nan)
                    phase_series[key].append((epoch['time'], phase))
        
        # 检测停滞
        affected_count = 0
        total_count = len(phase_series)
        
        for (sat_id, freq), series in phase_series.items():
            if len(series) < min_consecutive:
                continue
                
            phases = [p[1] for p in series if not np.isnan(p[1])]
            times = [p[0] for p in series if not np.isnan(p[1])]
            
            if len(phases) < min_consecutive:
                continue
                
            # 检测连续的小变化
            stagnation_periods = []
            current_period_start = None
            consecutive_count = 0
            
            for i in range(1, len(phases)):
                # 计算相位变化（周）
                phase_change = abs(phases[i] - phases[i-1])
                wavelength = self.config.get_wavelength(sat_id[0], freq)
                
                if wavelength and phase_change / wavelength < threshold_cycles:
                    if current_period_start is None:
                        current_period_start = i - 1
                    consecutive_count += 1
                else:
                    if consecutive_count >= min_consecutive:
                        stagnation_periods.append({
                            'start_time': times[current_period_start],
                            'end_time': times[current_period_start + consecutive_count],
                            'duration_epochs': consecutive_count + 1
                        })
                    current_period_start = None
                    consecutive_count = 0
            
            # 检查最后一个周期
            if consecutive_count >= min_consecutive and current_period_start is not None:
                stagnation_periods.append({
                    'start_time': times[current_period_start],
                    'end_time': times[-1],
                    'duration_epochs': consecutive_count + 1
                })
            
            if stagnation_periods:
                stagnation_results['details'][sat_id][freq]['stagnation_periods'] = stagnation_periods
                stagnation_results['details'][sat_id][freq]['is_stagnant'] = True
                stagnation_results['detected'] = True
                affected_count += 1
        
        stagnation_results['summary']['total_satellites'] = total_count
        stagnation_results['summary']['affected_satellites'] = affected_count
        
        return stagnation_results
    
    def calculate_code_phase_differences(self, data: Dict) -> Dict:
        """计算码相差（Code-Phase差值）"""
        cp_differences = {
            'differences': {},
            'statistics': {},
            'outliers': {}
        }
        
        if not data['epochs']:
            return cp_differences
        
        # 收集所有码相观测数据
        observations = {}
        
        for epoch in data['epochs']:
            for sat_id in epoch['satellites']:
                if sat_id not in observations:
                    observations[sat_id] = {}
                    
                for freq in epoch['satellites'][sat_id]:
                    if freq not in observations[sat_id]:
                        observations[sat_id][freq] = {
                            'times': [],
                            'code': [],
                            'phase': [],
                            'cp_diff': []
                        }
                    
                    obs = epoch['satellites'][sat_id][freq]
                    code = obs.get('code', np.nan)
                    phase = obs.get('phase', np.nan)
                    
                    if not (np.isnan(code) or np.isnan(phase)):
                        # 计算码相差
                        cp_diff = code - phase
                        
                        observations[sat_id][freq]['times'].append(epoch['time'])
                        observations[sat_id][freq]['code'].append(code)
                        observations[sat_id][freq]['phase'].append(phase)
                        observations[sat_id][freq]['cp_diff'].append(cp_diff)
        
        # 计算统计信息和检测异常值
        for sat_id in observations:
            cp_differences['differences'][sat_id] = {}
            cp_differences['statistics'][sat_id] = {}
            cp_differences['outliers'][sat_id] = {}
            
            for freq in observations[sat_id]:
                cp_diffs = np.array(observations[sat_id][freq]['cp_diff'])
                
                if len(cp_diffs) == 0:
                    continue
                
                # 存储差值
                cp_differences['differences'][sat_id][freq] = {
                    'times': observations[sat_id][freq]['times'],
                    'values': cp_diffs.tolist()
                }
                
                # 计算统计信息
                cp_differences['statistics'][sat_id][freq] = {
                    'mean': float(np.mean(cp_diffs)),
                    'std': float(np.std(cp_diffs)),
                    'min': float(np.min(cp_diffs)),
                    'max': float(np.max(cp_diffs)),
                    'count': len(cp_diffs)
                }
                
                # 检测异常值（3-sigma准则）
                mean_val = np.mean(cp_diffs)
                std_val = np.std(cp_diffs)
                outlier_threshold = 3 * std_val
                
                outlier_indices = np.where(np.abs(cp_diffs - mean_val) > outlier_threshold)[0]
                
                cp_differences['outliers'][sat_id][freq] = {
                    'indices': outlier_indices.tolist(),
                    'values': cp_diffs[outlier_indices].tolist(),
                    'times': [observations[sat_id][freq]['times'][i] for i in outlier_indices],
                    'count': len(outlier_indices)
                }
        
        return cp_differences
    
    def remove_code_phase_outliers(self, data: Dict, threshold: float = 5.0) -> str:
        """移除码相差异常值"""
        removed_count = 0
        log_messages = []
        
        if not data['epochs']:
            return "没有数据需要处理"
        
        # 首先计算码相差
        cp_results = self.calculate_code_phase_differences(data)
        
        # 收集要移除的观测值
        outliers_to_remove = []
        
        for sat_id in cp_results['outliers']:
            for freq in cp_results['outliers'][sat_id]:
                outliers = cp_results['outliers'][sat_id][freq]
                if outliers['count'] > 0:
                    for i, time_val in enumerate(outliers['times']):
                        outliers_to_remove.append((time_val, sat_id, freq))
                        log_messages.append(
                            f"移除异常观测: {sat_id} {freq} at {time_val}, "
                            f"码相差: {outliers['values'][i]:.3f}m"
                        )
        
        # 从数据中移除异常值
        for epoch in data['epochs']:
            for time_val, sat_id, freq in outliers_to_remove:
                if (epoch['time'] == time_val and 
                    sat_id in epoch['satellites'] and 
                    freq in epoch['satellites'][sat_id]):
                    
                    del epoch['satellites'][sat_id][freq]
                    removed_count += 1
                    
                    # 如果该卫星没有其他频率数据，则完全移除
                    if not epoch['satellites'][sat_id]:
                        del epoch['satellites'][sat_id]
        
        summary = f"码相差异常值移除完成，共移除 {removed_count} 个观测值"
        log_messages.insert(0, summary)
        
        return "\\n".join(log_messages)
    
    def calculate_epoch_double_differences(self):
        """计算历元间双差"""
        if not hasattr(self, 'receiver_data') or self.receiver_data is None:
            return "需要先加载接收机数据进行双差计算"
        
        # 这里应该实现双差计算逻辑
        # 由于代码较复杂，这里提供框架
        double_diff_results = {
            'code_dd': {},
            'phase_dd': {},
            'doppler_dd': {},
            'statistics': {},
            'outliers_removed': 0
        }
        
        return double_diff_results