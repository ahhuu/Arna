"""
周跳探测模块
实现 Melbourne-Wübbena (MW)、Geometry-Free (GF) 组合以及 LLI 标识的周跳探测算法
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from src.core.config import SPEED_OF_LIGHT, GNSS_FREQUENCIES


class CycleSlipDetector:
    """周跳探测器，实现 MW、GF 和 LLI 标识算法"""
    
    def __init__(self, mw_threshold_sigma: float = 4.0, gf_threshold: float = 0.4,
                 gf_k_sigma: float = 4.0, delta_i_max: float = 0.4,
                 use_custom_threshold: bool = False,
                 custom_mw_threshold: Optional[float] = None,
                 custom_gf_threshold: Optional[float] = None):
        """
        初始化周跳探测器
        
        Args:
            mw_threshold_sigma: MW组合的阈值倍数（默认4倍标准差，仅在use_custom_threshold=False时使用）
            gf_threshold: GF组合的基础阈值（米，仅在use_custom_threshold=False时使用）
            gf_k_sigma: GF组合的sigma系数
            delta_i_max: GF组合的最大电离层变化率（米/小时）
            use_custom_threshold: 是否使用自定义阈值（True则忽略动态阈值计算）
            custom_mw_threshold: 自定义MW阈值（米），仅在use_custom_threshold=True时使用
            custom_gf_threshold: 自定义GF阈值（米），仅在use_custom_threshold=True时使用
        """
        self.mw_threshold_sigma = mw_threshold_sigma
        self.gf_threshold = gf_threshold
        self.gf_k_sigma = gf_k_sigma
        self.delta_i_max = delta_i_max
        
        # 自定义阈值配置
        self.use_custom_threshold = use_custom_threshold
        self.custom_mw_threshold = custom_mw_threshold or 10  # 默认10米
        self.custom_gf_threshold = custom_gf_threshold or 0.05  # 默认0.05米
        
    def detect_cycle_slips(self, observations: Dict[str, Any], 
                          frequencies: Dict[str, Dict[str, float]],
                          wavelengths: Dict[str, Dict[str, float]],
                          freq_pair: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        对所有卫星执行周跳探测
        
        Args:
            observations: 观测数据字典 {sat_id: {freq: {'times': [...], 'code': [...], 'phase_cycle': [...]}}}
            frequencies: 频率字典 {system: {freq: Hz}}
            wavelengths: 波长字典 {system: {freq: meters}}
            freq_pair: 要使用的频率对，例如 ('L1C', 'L5Q')，如果为None则自动选择
            
        Returns:
            包含所有卫星周跳探测结果的字典
        """
        results = {}
        
        for sat_id, sat_data in observations.items():
            system = sat_id[0] if sat_id else ''
            
            # 自动选择频率对：优先使用系统推荐组合，其次选择频率差最大的两个可用频率作为回退
            if freq_pair is None:
                # 获取系统推荐组合
                def _recommended_pairs(system_code: str):
                    keys = list(GNSS_FREQUENCIES.get(system_code, {}).keys())
                    opts = []
                    if system_code in ('G', 'R', 'J'):
                        if 'L1C' in keys and 'L5Q' in keys:
                            opts.append(('L1C', 'L5Q'))
                    elif system_code == 'E':
                        for a, b in (('L1C', 'L5Q'), ('L1C', 'L7Q'), ('L5Q', 'L7Q')):
                            if a in keys and b in keys:
                                opts.append((a, b))
                    elif system_code == 'C':
                        for a, b in (('L2I', 'L1P'), ('L2I', 'L5P'), ('L1P', 'L5P')):
                            if a in keys and b in keys:
                                opts.append((a, b))
                    # 回退：若无推荐组合，则返回空列表
                    return opts

                # 尝试推荐组合
                rec_pairs = _recommended_pairs(system)
                chosen = None
                for a, b in rec_pairs:
                    if a in sat_data and b in sat_data:
                        chosen = (a, b)
                        break
                if chosen:
                    freq1, freq2 = chosen
                else:
                    # 回退：选择频率差最大的两个频率名
                    candidates = list(sat_data.keys())
                    best_pair = None
                    best_sep = -1
                    for i in range(len(candidates)):
                        for j in range(i+1, len(candidates)):
                            a = candidates[i]; b = candidates[j]
                            fa = frequencies.get(system, {}).get(a)
                            fb = frequencies.get(system, {}).get(b)
                            if not fa or not fb:
                                continue
                            sep = abs(fa - fb)
                            if sep > best_sep:
                                best_sep = sep
                                best_pair = (a, b)
                    if best_pair and best_sep >= 1e6:
                        freq1, freq2 = best_pair
                    else:
                        results[sat_id] = {'error': 'Insufficient or too-close frequency options for satellite'}
                        continue
            else:
                # 优先使用用户指定的频率对；如果卫星不包含这些频率，尝试按频率值近似匹配可用信号
                req_f1, req_f2 = freq_pair
                freq1, freq2 = None, None
                # 若卫星直接包含请求的频率名，则直接使用
                if req_f1 in sat_data and req_f2 in sat_data:
                    freq1, freq2 = req_f1, req_f2
                else:
                    # 尝试通过频率值进行近似匹配
                    f1_val = frequencies.get(system, {}).get(req_f1)
                    f2_val = frequencies.get(system, {}).get(req_f2)
                    if f1_val and f2_val:
                        candidates = list(sat_data.keys())
                        best_pair = None
                        best_score = float('inf')
                        for a in candidates:
                            for b in candidates:
                                if a == b:
                                    continue
                                fa = frequencies.get(system, {}).get(a)
                                fb = frequencies.get(system, {}).get(b)
                                if not fa or not fb:
                                    continue
                                score = abs(fa - f1_val) + abs(fb - f2_val)
                                if score < best_score:
                                    best_score = score
                                    best_pair = (a, b)
                        if best_pair:
                            freq1, freq2 = best_pair
                    # 若仍未找到匹配，则回退为自动选择该卫星的前两个频率
                    if not freq1 or not freq2:
                        available_freqs = list(sat_data.keys())
                        if len(available_freqs) < 2:
                            results[sat_id] = {'error': 'Insufficient frequencies for satellite'}
                            continue
                        # choose pair with largest separation
                        best_pair = None
                        best_sep = -1
                        for i in range(len(available_freqs)):
                            for j in range(i+1, len(available_freqs)):
                                a = available_freqs[i]; b = available_freqs[j]
                                fa = frequencies.get(system, {}).get(a)
                                fb = frequencies.get(system, {}).get(b)
                                if not fa or not fb:
                                    continue
                                sep = abs(fa - fb)
                                if sep > best_sep:
                                    best_sep = sep
                                    best_pair = (a, b)
                        if best_pair and best_sep >= 1e6:
                            freq1, freq2 = best_pair
                        else:
                            results[sat_id] = {'error': 'Insufficient or too-close frequency options for satellite'}
                            continue
            
            # 获取频率和波长
            f1 = frequencies.get(system, {}).get(freq1)
            f2 = frequencies.get(system, {}).get(freq2)
            lambda1 = wavelengths.get(system, {}).get(freq1)
            lambda2 = wavelengths.get(system, {}).get(freq2)
            
            if not all([f1, f2, lambda1, lambda2]):
                results[sat_id] = {
                    'error': 'Missing frequency or wavelength info',
                    'freq1': freq1, 'freq2': freq2, 'f1': f1, 'f2': f2,
                    'lambda1': lambda1, 'lambda2': lambda2
                }
                continue
            
            # 检查频率差是否足够大（避免除零错误）
            if abs(f1 - f2) < 1e6:  # 频率差小于1MHz，跳过
                results[sat_id] = {'error': 'Frequency separation too small', 'f1': f1, 'f2': f2}
                continue
            
            # 执行MW/GF/LLI探测
            mw_result = self._detect_mw(sat_data, freq1, freq2, f1, f2, lambda1, lambda2)
            gf_result = self._detect_gf(sat_data, freq1, freq2, lambda1, lambda2)
            lli_result = self._detect_lli(sat_data)
            
            results[sat_id] = {
                'freq_pair': (freq1, freq2),
                'mw': mw_result,
                'gf': gf_result,
                'lli': lli_result,
                'frequencies': (f1, f2),
                'wavelengths': (lambda1, lambda2)
            }
        
        return results
    
    def _detect_mw(self, sat_data: Dict[str, Any], freq1: str, freq2: str,
                   f1: float, f2: float, lambda1: float, lambda2: float) -> Dict[str, Any]:
        """
        MW组合周跳探测
        
        Args:
            sat_data: 单颗卫星的观测数据
            freq1, freq2: 频率名称
            f1, f2: 频率值（Hz）
            lambda1, lambda2: 波长（米）
            
        Returns:
            MW探测结果字典
        """
        # 获取观测数据
        data1 = sat_data.get(freq1, {})
        data2 = sat_data.get(freq2, {})
        
        times1 = data1.get('times', [])
        times2 = data2.get('times', [])
        code1 = data1.get('code', [])
        code2 = data2.get('code', [])
        phase1 = data1.get('phase_cycle', []) or data1.get('phase', [])
        phase2 = data2.get('phase_cycle', []) or data2.get('phase', [])
        
        # 确保数据长度一致
        n = min(len(times1), len(times2), len(code1), len(code2), len(phase1), len(phase2))
        if n < 3:
            return {'error': 'Insufficient data for MW detection'}
        
        # 计算宽巷波长
        lambda_w = SPEED_OF_LIGHT / abs(f1 - f2)
        
        # 计算MW观测值 Nw
        nw_series = []
        valid_epochs = []
        for i in range(n):
            if (code1[i] is not None and code2[i] is not None and 
                phase1[i] is not None and phase2[i] is not None):
                # L_i 和 L_j 单位需要是米
                L1_m = phase1[i] * lambda1
                L2_m = phase2[i] * lambda2
                P1_m = code1[i]
                P2_m = code2[i]
                
                # MW组合公式
                L_mw = (f1 * L1_m - f2 * L2_m) / (f1 - f2) - (f1 * P1_m + f2 * P2_m) / (f1 + f2)
                nw = L_mw / lambda_w
                
                nw_series.append(nw)
                valid_epochs.append(i + 1)
        
        if len(nw_series) < 3:
            return {'error': 'Insufficient valid observations for MW'}
        
        # 递归平滑和周跳检测
        delta_mw = []
        mean_history = []
        sigma_history = []
        threshold_history = []
        cycle_slips = []
        outliers = []
        
        # 初始化
        mean_nw = nw_series[0]
        sigma2_nw = 0.0
        k = 1
        
        mean_history.append(mean_nw)
        sigma_history.append(0.0)
        threshold_history.append(0.0)
        delta_mw.append(0.0)
        
        for i in range(1, len(nw_series)):
            # 计算与前一历元均值的差异
            delta = abs(nw_series[i] - mean_nw)
            delta_mw.append(delta)
            
            # 计算阈值：使用自定义或动态阈值
            if self.use_custom_threshold:
                threshold = self.custom_mw_threshold
            else:
                threshold = self.mw_threshold_sigma * np.sqrt(sigma2_nw)
            threshold_history.append(threshold)
            
            # 周跳或粗差判定
            is_anomaly = False
            if k > 1 and delta >= threshold:
                # 检查是否是周跳或粗差
                if i < len(nw_series) - 1:
                    delta_next = abs(nw_series[i] - nw_series[i + 1])
                    if delta_next <= 1.0:
                        cycle_slips.append({
                            'epoch': valid_epochs[i],
                            'index': i,
                            'delta': delta,
                            'threshold': threshold,
                            'type': 'cycle_slip'
                        })
                    else:
                        outliers.append({
                            'epoch': valid_epochs[i],
                            'index': i,
                            'delta': delta,
                            'threshold': threshold,
                            'type': 'outlier'
                        })
                else:
                    # 最后一个历元，保守判定为周跳
                    cycle_slips.append({
                        'epoch': valid_epochs[i],
                        'index': i,
                        'delta': delta,
                        'threshold': threshold,
                        'type': 'cycle_slip'
                    })
                
                is_anomaly = True
                # 重置平滑窗口
                mean_nw = nw_series[i]
                sigma2_nw = 0.0
                k = 1
            else:
                # 递推更新均值和方差
                k += 1
                delta_mean = nw_series[i] - mean_nw
                mean_nw = mean_nw + delta_mean / k
                sigma2_nw = sigma2_nw + (delta_mean ** 2 - sigma2_nw) / k
            
            mean_history.append(mean_nw)
            sigma_history.append(np.sqrt(sigma2_nw))
        
        return {
            'nw_series': nw_series,
            'delta_mw': delta_mw,
            'epochs': valid_epochs,
            'mean_history': mean_history,
            'sigma_history': sigma_history,
            'threshold_history': threshold_history,
            'cycle_slips': cycle_slips,
            'outliers': outliers,
            'lambda_w': lambda_w,
            'threshold_mode': 'custom' if self.use_custom_threshold else 'dynamic',
            'threshold_value': self.custom_mw_threshold if self.use_custom_threshold else None
        }
    
    def _detect_gf(self, sat_data: Dict[str, Any], freq1: str, freq2: str,
                   lambda1: float, lambda2: float) -> Dict[str, Any]:
        """
        GF组合周跳探测
        
        Args:
            sat_data: 单颗卫星的观测数据
            freq1, freq2: 频率名称
            lambda1, lambda2: 波长（米）
            
        Returns:
            GF探测结果字典
        """
        # 获取观测数据
        data1 = sat_data.get(freq1, {})
        data2 = sat_data.get(freq2, {})
        
        times1 = data1.get('times', [])
        times2 = data2.get('times', [])
        phase1 = data1.get('phase_cycle', []) or data1.get('phase', [])
        phase2 = data2.get('phase_cycle', []) or data2.get('phase', [])
        
        # 确保数据长度一致
        n = min(len(times1), len(times2), len(phase1), len(phase2))
        if n < 3:
            return {'error': 'Insufficient data for GF detection'}
        
        # 计算GF观测值
        gf_series = []
        valid_epochs = []
        for i in range(n):
            if phase1[i] is not None and phase2[i] is not None:
                # GF = L1 - L2 (单位：米)
                gf = lambda1 * phase1[i] - lambda2 * phase2[i]
                gf_series.append(gf)
                valid_epochs.append(i + 1)
        
        if len(gf_series) < 2:
            return {'error': 'Insufficient valid observations for GF'}
        
        # 计算历元间差分
        delta_gf = []
        cycle_slips = []
        
        delta_gf.append(0.0)  # 第一个历元没有差分
        
        # 计算GF序列的标准差用于自适应阈值
        if len(gf_series) > 10:
            # 使用前10个历元估计标准差
            gf_diffs = [gf_series[i] - gf_series[i-1] for i in range(1, min(11, len(gf_series)))]
            sigma_gf = np.std(gf_diffs) if gf_diffs else 0.1
        else:
            sigma_gf = 0.1
        
        # 使用自定义或动态阈值
        if self.use_custom_threshold:
            threshold = self.custom_gf_threshold
        else:
            threshold = self.gf_k_sigma * sigma_gf + self.delta_i_max
        
        for i in range(1, len(gf_series)):
            delta = abs(gf_series[i] - gf_series[i - 1])
            delta_gf.append(delta)
            
            if delta > threshold:
                cycle_slips.append({
                    'epoch': valid_epochs[i],
                    'index': i,
                    'delta': delta,
                    'threshold': threshold,
                    'gf_value': gf_series[i],
                    'gf_prev': gf_series[i - 1]
                })
        
        return {
            'gf_series': gf_series,
            'delta_gf': delta_gf,
            'epochs': valid_epochs,
            'threshold': threshold,
            'cycle_slips': cycle_slips,
            'sigma_gf': sigma_gf,
            'threshold_mode': 'custom' if self.use_custom_threshold else 'dynamic',
            'threshold_value': self.custom_gf_threshold if self.use_custom_threshold else None
        }

    def _detect_lli(self, sat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于 LLI 标识检测周跳事件。

        规则（与转换器定义一致）：
        - bit0 (=1): RESET 或 CYCLE_SLIP -> 记为 LLI周跳事件
        - bit1 (=2): HALF_CYCLE_REPORTED 且未 RESOLVED -> 记为半周模糊事件（非周跳）
        """
        lli_freqs = []
        for freq_name, freq_data in sat_data.items():
            lli_list = freq_data.get('phase_lli', []) if isinstance(freq_data, dict) else []
            if lli_list:
                lli_freqs.append(freq_name)

        lli_freqs = sorted(lli_freqs)
        if not lli_freqs:
            return {'error': 'No LLI data available'}

        n = 0
        for fn in lli_freqs:
            fd = sat_data.get(fn, {})
            n = max(n, len(fd.get('phase_lli', []) or []), len(fd.get('times', []) or []))
        if n == 0:
            return {'error': 'No LLI data available'}

        cycle_slip_events = []
        half_cycle_events = []
        epochs = []
        lli_raw_by_freq = {fn: [] for fn in lli_freqs}
        bit0_by_freq = {fn: [] for fn in lli_freqs}
        bit1_by_freq = {fn: [] for fn in lli_freqs}
        lock_loss_union = []
        half_cycle_union = []

        for i in range(n):
            t = None
            lli_by_freq = {}
            lock_loss_any = False
            half_cycle_any = False

            for fn in lli_freqs:
                fd = sat_data.get(fn, {})
                lli_list = fd.get('phase_lli', []) or []
                times = fd.get('times', []) or []
                raw = int(lli_list[i]) if i < len(lli_list) and lli_list[i] is not None else 0
                lli_by_freq[fn] = raw
                lli_raw_by_freq[fn].append(raw)

                b0 = 1 if (raw & 1) != 0 else 0
                b1 = 1 if (raw & 2) != 0 else 0
                bit0_by_freq[fn].append(b0)
                bit1_by_freq[fn].append(b1)
                lock_loss_any = lock_loss_any or (b0 == 1)
                half_cycle_any = half_cycle_any or (b1 == 1)

                if t is None and i < len(times):
                    t = times[i]

            e = i + 1
            epochs.append(e)
            lock_loss_flag = 1 if lock_loss_any else 0
            half_cycle_flag = 1 if half_cycle_any else 0
            lock_loss_union.append(lock_loss_flag)
            half_cycle_union.append(half_cycle_flag)
            lli_values_str = ';'.join([f"{fn}:{lli_by_freq.get(fn, 0)}" for fn in lli_freqs])

            if lock_loss_flag:
                cycle_slip_events.append({
                    'epoch': e,
                    'index': i,
                    'time': t,
                    'lli_by_freq': lli_by_freq.copy(),
                    'lli_values_str': lli_values_str
                })

            if half_cycle_flag:
                half_cycle_events.append({
                    'epoch': e,
                    'index': i,
                    'time': t,
                    'lli_by_freq': lli_by_freq.copy(),
                    'lli_values_str': lli_values_str
                })

        return {
            'epochs': epochs,
            'lli_freqs': lli_freqs,
            'lli_raw_by_freq': lli_raw_by_freq,
            'bit0_by_freq': bit0_by_freq,
            'bit1_by_freq': bit1_by_freq,
            'lock_loss_union': lock_loss_union,
            'half_cycle_union': half_cycle_union,
            'cycle_slips': cycle_slip_events,
            'half_cycle_events': half_cycle_events
        }
    
    def format_detection_summary(self, detection_results: Dict[str, Any]) -> str:
        """
        格式化周跳探测摘要
        
        Args:
            detection_results: detect_cycle_slips的返回结果
            
        Returns:
            格式化的摘要字符串
        """
        total_cycle_slips = 0
        total_outliers = 0
        affected_satellites = []
        
        for sat_id, result in detection_results.items():
            mw = result.get('mw', {})
            gf = result.get('gf', {})
            lli = result.get('lli', {})
            
            mw_slips = len(mw.get('cycle_slips', []))
            mw_outliers = len(mw.get('outliers', []))
            gf_slips = len(gf.get('cycle_slips', []))
            lli_slips = len(lli.get('cycle_slips', []))
            lli_half = len(lli.get('half_cycle_events', []))
            
            if mw_slips > 0 or gf_slips > 0 or lli_slips > 0 or lli_half > 0:
                affected_satellites.append(sat_id)
                total_cycle_slips += mw_slips + gf_slips + lli_slips
                total_outliers += mw_outliers
        
        summary = f"周跳探测摘要:\n"
        summary += f"  处理卫星数: {len(detection_results)}\n"
        summary += f"  发现周跳数: {total_cycle_slips}\n"
        summary += f"  发现粗差数: {total_outliers}\n"
        summary += f"  受影响卫星: {', '.join(affected_satellites) if affected_satellites else '无'}\n"
        
        return summary
