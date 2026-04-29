"""
周跳探测日志记录模块
实现周跳探测结果的详细日志输出
"""
import os
from datetime import datetime
from typing import Dict, Any, Optional


class CycleSlipLogger:
    """周跳探测日志记录器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化日志记录器
        
        Args:
            output_dir: 输出目录，如果为None则使用当前目录
        """
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _format_lli_values(event: Dict[str, Any]) -> str:
        """Format LLI values as 'L1C:3;L5Q:2;L7Q:0'."""
        if isinstance(event.get('lli_values_str'), str) and event.get('lli_values_str'):
            return event['lli_values_str']
        lli_by_freq = event.get('lli_by_freq')
        if isinstance(lli_by_freq, dict) and lli_by_freq:
            parts = [f"{k}:{lli_by_freq[k]}" for k in sorted(lli_by_freq.keys())]
            return ';'.join(parts)
        # backward compatibility
        lli_v1 = event.get('lli_freq1')
        lli_v2 = event.get('lli_freq2')
        return f"{lli_v1 if lli_v1 is not None else 0}/{lli_v2 if lli_v2 is not None else 0}"
    
    def save_cycle_slip_log(self, detection_results: Dict[str, Any], 
                           filename: Optional[str] = None) -> str:
        """
        保存周跳探测详细日志
        
        Args:
            detection_results: 周跳探测结果字典
            filename: 日志文件名，如果为None则自动生成
            
        Returns:
            日志文件的完整路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cycle_slip_{timestamp}.log"
        
        log_path = os.path.join(self.output_dir, filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            # 写入头部信息
            f.write("=" * 80 + "\n")
            f.write("周跳探测详细报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 提取阈值信息
            threshold_info = self._extract_threshold_info(detection_results)
            if threshold_info:
                f.write(f"阈值模式: {threshold_info}\n")
            
            f.write("\n")
            
            # 统计信息
            total_satellites = len(detection_results)
            total_mw_slips = 0
            total_gf_slips = 0
            total_lli_slips = 0
            total_lli_half_cycles = 0
            total_outliers = 0
            affected_satellites = []
            total_observations = 0
            
            for sat_id, result in detection_results.items():
                mw = result.get('mw', {})
                gf = result.get('gf', {})
                lli = result.get('lli', {})
                
                mw_slips = len(mw.get('cycle_slips', []))
                gf_slips = len(gf.get('cycle_slips', []))
                lli_slips = len(lli.get('cycle_slips', []))
                lli_half_cycles = len(lli.get('half_cycle_events', []))
                mw_outliers = len(mw.get('outliers', []))
                
                total_mw_slips += mw_slips
                total_gf_slips += gf_slips
                total_lli_slips += lli_slips
                total_lli_half_cycles += lli_half_cycles
                total_outliers += mw_outliers
                
                if mw_slips > 0 or gf_slips > 0 or lli_slips > 0:
                    affected_satellites.append(sat_id)
                
                # 统计观测值总数
                total_observations += len(mw.get('epochs', []))
            
            total_cycle_slips = total_mw_slips + total_gf_slips + total_lli_slips
            
            # 写入统计摘要
            f.write("-" * 80 + "\n")
            f.write("统计摘要\n")
            f.write("-" * 80 + "\n")
            f.write(f"处理卫星总数: {total_satellites}\n")
            f.write(f"总历元数: {total_observations}\n")
            f.write(f"MW周跳检测数: {total_mw_slips}\n")
            f.write(f"GF周跳检测数: {total_gf_slips}\n")
            f.write(f"LLI周跳检测数(bit0): {total_lli_slips}\n")
            f.write(f"LLI半周模糊事件(bit1): {total_lli_half_cycles}\n")
            f.write(f"周跳总数: {total_cycle_slips}\n")
            f.write(f"粗差检测数: {total_outliers}\n")
            f.write(f"受影响卫星列表: {', '.join(affected_satellites) if affected_satellites else '无'}\n")
            
            # 计算周跳率 (CSR)
            if total_observations > 0:
                csr = (total_cycle_slips / total_observations) * 1000
                f.write(f"周跳率 (CSR): {csr:.3f} ‰\n")
            
            f.write("\n")
            
            # 详细日志表
            f.write("=" * 80 + "\n")
            f.write("详细周跳记录\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'历元':>8} {'卫星':>8} {'方法':>8} {'检测值(m)':>12} {'阈值(m)':>12} {'判定结果':>12}\n")
            f.write("-" * 80 + "\n")
            
            # 按卫星和历元排序输出
            for sat_id in sorted(detection_results.keys()):
                result = detection_results[sat_id]
                mw = result.get('mw', {})
                gf = result.get('gf', {})
                lli = result.get('lli', {})
                
                # MW周跳
                for slip in mw.get('cycle_slips', []):
                    epoch = slip['epoch']
                    delta = slip['delta']
                    threshold = slip['threshold']
                    f.write(f"{epoch:>8d} {sat_id:>8} {'MW':>8} {delta:>12.4f} {threshold:>12.4f} {'周跳':>12}\n")
                
                # MW粗差
                for outlier in mw.get('outliers', []):
                    epoch = outlier['epoch']
                    delta = outlier['delta']
                    threshold = outlier['threshold']
                    f.write(f"{epoch:>8d} {sat_id:>8} {'MW':>8} {delta:>12.4f} {threshold:>12.4f} {'粗差':>12}\n")
                
                # GF周跳
                for slip in gf.get('cycle_slips', []):
                    epoch = slip['epoch']
                    delta = slip['delta']
                    threshold = slip['threshold']
                    f.write(f"{epoch:>8d} {sat_id:>8} {'GF':>8} {delta:>12.4f} {threshold:>12.4f} {'周跳':>12}\n")

                # LLI周跳（bit0）
                for slip in lli.get('cycle_slips', []):
                    epoch = slip['epoch']
                    lli_desc = self._format_lli_values(slip)
                    f.write(f"{epoch:>8d} {sat_id:>8} {'LLI':>8} {lli_desc:>12} {'bit0':>12} {'周跳':>12}\n")

                # LLI半周模糊（bit1）
                for evt in lli.get('half_cycle_events', []):
                    epoch = evt['epoch']
                    lli_desc = self._format_lli_values(evt)
                    f.write(f"{epoch:>8d} {sat_id:>8} {'LLI':>8} {lli_desc:>12} {'bit1':>12} {'半周模糊':>12}\n")
            
            f.write("-" * 80 + "\n")
            f.write("\n")
            
            # 卫星详细信息
            f.write("=" * 80 + "\n")
            f.write("各卫星详细统计\n")
            f.write("=" * 80 + "\n")
            
            for sat_id in sorted(detection_results.keys()):
                result = detection_results[sat_id]
                mw = result.get('mw', {})
                gf = result.get('gf', {})
                lli = result.get('lli', {})
                freq_pair = result.get('freq_pair', ('?', '?'))
                
                f.write(f"\n卫星: {sat_id}\n")
                f.write(f"  频率组合: {freq_pair[0]} / {freq_pair[1]}\n")
                f.write(f"  有效历元数: {len(mw.get('epochs', []))}\n")
                
                # MW统计
                mw_slips = len(mw.get('cycle_slips', []))
                mw_outliers = len(mw.get('outliers', []))
                f.write(f"  MW检测:\n")
                f.write(f"    周跳数: {mw_slips}\n")
                f.write(f"    粗差数: {mw_outliers}\n")
                if 'lambda_w' in mw:
                    f.write(f"    宽巷波长: {mw['lambda_w']:.4f} m\n")
                
                # GF统计
                gf_slips = len(gf.get('cycle_slips', []))
                f.write(f"  GF检测:\n")
                f.write(f"    周跳数: {gf_slips}\n")
                if 'threshold' in gf:
                    f.write(f"    阈值: {gf['threshold']:.4f} m\n")
                if 'sigma_gf' in gf:
                    f.write(f"    GF标准差: {gf['sigma_gf']:.4f} m\n")

                # LLI统计
                lli_slips = len(lli.get('cycle_slips', []))
                lli_half_cycles = len(lli.get('half_cycle_events', []))
                f.write(f"  LLI检测:\n")
                f.write(f"    周跳数(bit0): {lli_slips}\n")
                f.write(f"    半周模糊(bit1): {lli_half_cycles}\n")
                
                f.write("-" * 80 + "\n")
        
        return log_path
    
    def save_cycle_slip_csv(self, detection_results: Dict[str, Any],
                           filename: Optional[str] = None) -> str:
        """
        保存周跳探测结果为CSV格式（便于进一步分析）
        
        Args:
            detection_results: 周跳探测结果字典
            filename: CSV文件名，如果为None则自动生成
            
        Returns:
            CSV文件的完整路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cycle_slip_{timestamp}.csv"
        
        csv_path = os.path.join(self.output_dir, filename)
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            # 写入CSV表头
            f.write("Satellite,Epoch,Method,Delta_m,Threshold_m,Detection_Type\n")
            
            # 按卫星和历元排序输出
            for sat_id in sorted(detection_results.keys()):
                result = detection_results[sat_id]
                mw = result.get('mw', {})
                gf = result.get('gf', {})
                lli = result.get('lli', {})
                
                # MW周跳
                for slip in mw.get('cycle_slips', []):
                    f.write(f"{sat_id},{slip['epoch']},MW,{slip['delta']:.4f},{slip['threshold']:.4f},Cycle_Slip\n")
                
                # MW粗差
                for outlier in mw.get('outliers', []):
                    f.write(f"{sat_id},{outlier['epoch']},MW,{outlier['delta']:.4f},{outlier['threshold']:.4f},Outlier\n")
                
                # GF周跳
                for slip in gf.get('cycle_slips', []):
                    f.write(f"{sat_id},{slip['epoch']},GF,{slip['delta']:.4f},{slip['threshold']:.4f},Cycle_Slip\n")

                # LLI周跳（bit0）
                for slip in lli.get('cycle_slips', []):
                    lli_desc = self._format_lli_values(slip)
                    f.write(f"{sat_id},{slip['epoch']},LLI,{lli_desc},bit0,Cycle_Slip\n")

                # LLI半周模糊（bit1）
                for evt in lli.get('half_cycle_events', []):
                    lli_desc = self._format_lli_values(evt)
                    f.write(f"{sat_id},{evt['epoch']},LLI,{lli_desc},bit1,Half_Cycle_Ambiguity\n")
        
        return csv_path    
    def _extract_threshold_info(self, detection_results: Dict[str, Any]) -> str:
        """
        从探测结果中提取阈值信息
        
        Args:
            detection_results: 周跳探测结果字典
            
        Returns:
            阈值模式信息字符串
        """
        if not detection_results:
            return ""
        
        # 从第一个有结果的卫星中提取阈值信息
        for sat_id, result in detection_results.items():
            mw = result.get('mw', {})
            gf = result.get('gf', {})
            
            # 检查MW阈值信息
            if 'threshold_mode' in mw:
                threshold_mode = mw['threshold_mode']
                if threshold_mode == 'dynamic':
                    return "动态阈值模式"
                elif threshold_mode == 'custom':
                    mw_threshold = mw.get('threshold_value', 'N/A')
                    return f"自定义阈值模式 (MW: {mw_threshold}m)"
        
        return ""
