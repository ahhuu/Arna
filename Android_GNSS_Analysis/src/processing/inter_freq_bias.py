"""伪距频间偏差检测与星间单差(ISD)处理模块

该模块实现：
1. 计算原始伪距频间差值（检测手机硬件偏差）
2. 基于星间单差(Inter-Satellite Single Difference, ISD)消除接收机端硬件偏差
3. 验证ISD方法的有效性
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from src.core.config import GNSS_FREQUENCIES


class InterFrequencyBiasAnalyzer:
    """伪距频间偏差分析器"""

    def __init__(self):
        self.frequencies = GNSS_FREQUENCIES

    def compute_raw_inter_freq_diff(self, 
                                     observations: Dict[str, Any], 
                                     freq1: str, 
                                     freq2: str,
                                     constellation: Optional[str] = None) -> Dict[str, Any]:
        """
        计算原始伪距频间差值
        
        物理意义：
        Diff_raw = P_freq1 - P_freq2
        该差值包含接收机硬件延迟偏差，是我们要检测的异常现象
        
        参数:
            observations: 观测数据字典 {sat_id: {freq: {times, code, phase, ...}}}
            freq1: 第一个频率名称 (如 'L1C')
            freq2: 第二个频率名称 (如 'L5Q')
            constellation: 星座系统代码 (如 'G', 'E', 'C')，None表示处理所有星座
            
        返回:
            {
                sat_id: {
                    'times': [timestamps],
                    'diff': [差值列表，单位：米],
                    'freq1_code': [freq1伪距列表],
                    'freq2_code': [freq2伪距列表],
                    'snr1': [freq1信噪比],
                    'snr2': [freq2信噪比]
                }
            }
        """
        results = {}
        
        for sat_id, sat_obs in observations.items():
            # 星座过滤
            if constellation and not sat_id.startswith(constellation):
                continue
            
            # 检查是否包含两个频率
            if freq1 not in sat_obs or freq2 not in sat_obs:
                continue
            
            freq1_data = sat_obs[freq1]
            freq2_data = sat_obs[freq2]
            
            # 提取时间、伪距和信噪比
            times1 = freq1_data.get('times', [])
            times2 = freq2_data.get('times', [])
            code1 = freq1_data.get('code', [])
            code2 = freq2_data.get('code', [])
            snr1 = freq1_data.get('snr', [])
            snr2 = freq2_data.get('snr', [])
            
            # 找到公共历元（两个频率都有观测值的时刻）
            common_times = []
            diff_list = []
            code1_list = []
            code2_list = []
            snr1_list = []
            snr2_list = []
            
            # 构建时间索引映射
            time2_dict = {t: idx for idx, t in enumerate(times2)}
            
            for i, t in enumerate(times1):
                if t in time2_dict:
                    j = time2_dict[t]
                    c1 = code1[i]
                    c2 = code2[j]
                    
                    # 数据质量检查：过滤无效值
                    if c1 is None or c2 is None:
                        continue
                    if c1 <= 0 or c2 <= 0:  # 伪距不应为0或负数
                        continue
                    if np.isnan(c1) or np.isnan(c2):
                        continue
                    
                    # 计算频间差值
                    diff = c1 - c2
                    
                    common_times.append(t)
                    diff_list.append(diff)
                    code1_list.append(c1)
                    code2_list.append(c2)
                    snr1_list.append(snr1[i] if i < len(snr1) else None)
                    snr2_list.append(snr2[j] if j < len(snr2) else None)
            
            if common_times:
                results[sat_id] = {
                    'times': common_times,
                    'diff': diff_list,
                    'freq1_code': code1_list,
                    'freq2_code': code2_list,
                    'snr1': snr1_list,
                    'snr2': snr2_list
                }
        
        return results

    def select_reference_satellite(self,
                                   raw_diffs: Dict[str, Any],
                                   epoch_idx: int) -> Optional[str]:
        """
        选择参考卫星（每个历元）
        
        策略：选择加权C/N0最高的卫星作为参考星
        加权方法：取两个频率的C/N0平均值（或加权平均）
        
        参数:
            raw_diffs: compute_raw_inter_freq_diff的返回结果
            epoch_idx: 历元索引
            
        返回:
            参考卫星ID，如果没有合适的返回None
        """
        best_sat = None
        best_weighted_cn0 = -np.inf
        
        for sat_id, data in raw_diffs.items():
            if epoch_idx >= len(data['times']):
                continue
            
            snr1 = data['snr1'][epoch_idx]
            snr2 = data['snr2'][epoch_idx]
            
            # 计算加权C/N0（这里使用平均值，可根据需要调整权重）
            if snr1 is not None and snr2 is not None:
                weighted_cn0 = (snr1 + snr2) / 2.0
            elif snr1 is not None:
                weighted_cn0 = snr1
            elif snr2 is not None:
                weighted_cn0 = snr2
            else:
                continue
            
            # 选择最大值
            if weighted_cn0 > best_weighted_cn0:
                best_weighted_cn0 = weighted_cn0
                best_sat = sat_id
        
        return best_sat

    def compute_isd_diff(self,
                        raw_diffs: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算星间单差(ISD)后的频间差值
        
        物理原理：
        1. 对freq1和freq2分别做星间单差：
           ΔP_freq1^{i,ref} = P_freq1^i - P_freq1^ref
           ΔP_freq2^{i,ref} = P_freq2^i - P_freq2^ref
        
        2. 计算单差后的频间差：
           Diff_ISD = ΔP_freq1^{i,ref} - ΔP_freq2^{i,ref}
        
        3. 关键：接收机钟差和硬件延迟在单差过程中被消除
           理论上 Diff_ISD 应接近0（±10m以内）
        
        参数:
            raw_diffs: compute_raw_inter_freq_diff的返回结果
            
        返回:
            {
                sat_id: {
                    'times': [timestamps],
                    'isd_diff': [ISD差值列表],
                    'ref_sat': [每个历元的参考星ID]
                }
            }
        """
        if not raw_diffs:
            return {}
        
        # 获取所有卫星的时间序列（假设对齐）
        sample_sat = next(iter(raw_diffs.keys()))
        num_epochs = len(raw_diffs[sample_sat]['times'])
        
        results = {}
        
        # 初始化每颗卫星的结果
        for sat_id in raw_diffs.keys():
            results[sat_id] = {
                'times': raw_diffs[sat_id]['times'].copy(),
                'isd_diff': [],
                'ref_sat': []
            }
        
        # 逐历元处理
        for epoch_idx in range(num_epochs):
            # 选择参考星
            ref_sat = self.select_reference_satellite(raw_diffs, epoch_idx)
            
            if ref_sat is None:
                # 没有参考星，跳过该历元
                for sat_id in raw_diffs.keys():
                    if epoch_idx < len(raw_diffs[sat_id]['times']):
                        results[sat_id]['isd_diff'].append(None)
                        results[sat_id]['ref_sat'].append(None)
                continue
            
            # 获取参考星在两个频率上的伪距
            ref_code1 = raw_diffs[ref_sat]['freq1_code'][epoch_idx]
            ref_code2 = raw_diffs[ref_sat]['freq2_code'][epoch_idx]
            
            # 对每颗卫星计算ISD差值
            for sat_id in raw_diffs.keys():
                if epoch_idx >= len(raw_diffs[sat_id]['times']):
                    continue
                
                if sat_id == ref_sat:
                    # 参考星与自己的单差为0
                    results[sat_id]['isd_diff'].append(0.0)
                    results[sat_id]['ref_sat'].append(ref_sat)
                else:
                    code1 = raw_diffs[sat_id]['freq1_code'][epoch_idx]
                    code2 = raw_diffs[sat_id]['freq2_code'][epoch_idx]
                    
                    # 计算星间单差
                    # ΔP_freq1 = P_freq1^i - P_freq1^ref
                    delta_p_freq1 = code1 - ref_code1
                    # ΔP_freq2 = P_freq2^i - P_freq2^ref
                    delta_p_freq2 = code2 - ref_code2
                    
                    # ISD差值 = ΔP_freq1 - ΔP_freq2
                    isd_diff = delta_p_freq1 - delta_p_freq2
                    
                    results[sat_id]['isd_diff'].append(isd_diff)
                    results[sat_id]['ref_sat'].append(ref_sat)
        
        return results

    def analyze_inter_freq_bias(self,
                                observations: Dict[str, Any],
                                freq1: str,
                                freq2: str,
                                constellation: Optional[str] = None) -> Dict[str, Any]:
        """
        完整分析流程：计算原始频间差和ISD处理后的频间差
        
        参数:
            observations: 观测数据
            freq1: 第一个频率
            freq2: 第二个频率
            constellation: 星座系统（可选）
            
        返回:
            {
                'raw_diffs': 原始频间差结果,
                'isd_diffs': ISD处理后的频间差结果,
                'freq_pair': (freq1, freq2),
                'constellation': constellation
            }
        """
        # 计算原始频间差
        raw_diffs = self.compute_raw_inter_freq_diff(
            observations, freq1, freq2, constellation
        )
        
        if not raw_diffs:
            return {
                'raw_diffs': {},
                'isd_diffs': {},
                'freq_pair': (freq1, freq2),
                'constellation': constellation,
                'error': '未找到同时包含两个频率的卫星数据'
            }
        
        # 计算ISD差值
        isd_diffs = self.compute_isd_diff(raw_diffs)
        
        return {
            'raw_diffs': raw_diffs,
            'isd_diffs': isd_diffs,
            'freq_pair': (freq1, freq2),
            'constellation': constellation
        }

    def get_statistics(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算统计信息
        
        返回:
            {
                'raw_stats': {
                    'mean': 平均值,
                    'std': 标准差,
                    'min': 最小值,
                    'max': 最大值,
                    'rms': 均方根
                },
                'isd_stats': {...},
                'improvement': 改善率 (%)
            }
        """
        raw_diffs = analysis_result.get('raw_diffs', {})
        isd_diffs = analysis_result.get('isd_diffs', {})
        
        # 收集所有有效的原始差值
        raw_values = []
        for sat_data in raw_diffs.values():
            raw_values.extend([d for d in sat_data['diff'] if d is not None])
        
        # 收集所有有效的ISD差值
        isd_values = []
        for sat_data in isd_diffs.values():
            isd_values.extend([d for d in sat_data['isd_diff'] if d is not None])
        
        def calc_stats(values):
            if not values:
                return None
            arr = np.array(values)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'rms': float(np.sqrt(np.mean(arr**2)))
            }
        
        raw_stats = calc_stats(raw_values)
        isd_stats = calc_stats(isd_values)
        
        improvement = None
        if raw_stats and isd_stats and raw_stats['rms'] > 0:
            improvement = (1 - isd_stats['rms'] / raw_stats['rms']) * 100
        
        return {
            'raw_stats': raw_stats,
            'isd_stats': isd_stats,
            'improvement': improvement
        }
