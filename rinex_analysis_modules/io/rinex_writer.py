"""
RINEX文件写入模块
负责生成修正后的RINEX文件
"""

import os
import datetime
import pandas as pd
import numpy as np
from typing import Dict

class RinexWriter:
    """RINEX观测文件写入器"""
    
    def __init__(self, config=None):
        from ..core.config import GNSSConfig
        self.config = config if config else GNSSConfig()
    
    def generate_corrected_rinex_file(self, input_rinex_path: str, corrections: Dict, 
                                    output_path: str = None) -> str:
        """生成修正后的RINEX文件"""
        
        if output_path is None:
            base_name = os.path.splitext(input_rinex_path)[0]
            output_path = f"{base_name}_corrected.rnx"
        
        # 读取原始文件
        with open(input_rinex_path, 'r', encoding='utf-8') as f:
            input_lines = f.readlines()
        
        # 找到头部结束位置
        header_end = 0
        for i, line in enumerate(input_lines):
            if 'END OF HEADER' in line:
                header_end = i + 1
                break
        
        # 写入修正后的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            # 复制头部
            for i in range(header_end):
                f.write(input_lines[i])
            
            # 处理观测数据部分
            current_epoch = None
            i = header_end
            
            while i < len(input_lines):
                line = input_lines[i]
                
                if line.startswith('>'):
                    # 历元行
                    current_epoch = self._parse_epoch_time(line)
                    f.write(line)
                elif current_epoch and len(line) >= 3:
                    # 卫星观测行
                    sat_system = line[0]
                    sat_prn = line[1:3].strip()
                    sat_id = f"{sat_system}{sat_prn}"
                    
                    # 应用修正
                    corrected_line = self._apply_corrections_to_line(
                        line, sat_id, current_epoch, corrections
                    )
                    f.write(corrected_line)
                else:
                    # 其他行原样复制
                    f.write(line)
                
                i += 1
        
        return output_path
    
    def _parse_epoch_time(self, epoch_line: str) -> datetime.datetime:
        """解析历元时间"""
        try:
            parts = epoch_line[1:].split()
            if len(parts) >= 6:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3])
                minute = int(parts[4])
                second_float = float(parts[5])
                
                return pd.Timestamp(
                    year=year, month=month, day=day,
                    hour=hour, minute=minute, second=int(second_float),
                    microsecond=int((second_float - int(second_float)) * 1000000)
                )
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _apply_corrections_to_line(self, obs_line: str, sat_id: str, epoch: datetime.datetime, 
                                 corrections: Dict) -> str:
        """对观测行应用修正"""
        
        # 如果没有该卫星的修正数据，直接返回原行
        if sat_id not in corrections:
            return obs_line
        
        sat_corrections = corrections[sat_id]
        
        # 解析观测数据
        sat_data = obs_line[3:]  # 跳过卫星标识
        field_width = 16
        
        # 计算字段数量
        num_fields = (len(sat_data) + field_width - 1) // field_width
        
        corrected_fields = []
        
        for field_idx in range(num_fields):
            start_idx = field_idx * field_width
            end_idx = start_idx + field_width
            field = sat_data[start_idx:end_idx]
            
            # 检查是否需要修正这个字段
            # 这里需要根据具体的修正类型来处理
            # 例如：相位修正、伪距修正等
            
            corrected_field = self._apply_field_correction(
                field, field_idx, sat_id, epoch, sat_corrections
            )
            corrected_fields.append(corrected_field)
        
        # 重构观测行
        corrected_line = obs_line[:3]  # 保留卫星标识
        for field in corrected_fields:
            corrected_line += field.ljust(field_width)
        
        corrected_line += '\\n'
        return corrected_line
    
    def _apply_field_correction(self, field: str, field_idx: int, sat_id: str, 
                              epoch: datetime.datetime, corrections: Dict) -> str:
        """对单个字段应用修正"""
        
        # 解析字段值
        value_str = field[:14].strip()
        if not value_str:
            return field  # 空值不处理
        
        try:
            value = float(value_str)
        except ValueError:
            return field  # 无法解析的值不处理
        
        # 根据字段类型和修正数据应用修正
        # 这里需要根据具体的修正算法来实现
        
        # 示例：相位修正
        if 'phase_corrections' in corrections:
            phase_corr = corrections['phase_corrections'].get(epoch)
            if phase_corr is not None:
                # 应用相位修正
                corrected_value = value + phase_corr
                # 格式化回字符串
                corrected_str = f"{corrected_value:14.3f}"
                return corrected_str + field[14:]
        
        # 示例：ISB修正
        if 'isb_correction' in corrections:
            isb_corr = corrections['isb_correction']
            # 根据观测类型应用ISB修正
            # ...
            pass
        
        return field  # 无修正时返回原字段
    
    def save_processing_log(self, log_data: Dict, output_path: str) -> str:
        """保存处理日志"""
        
        log_content = []
        log_content.append("=== RINEX数据处理日志 ===\\n")
        log_content.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        # 添加处理步骤
        if 'processing_steps' in log_data:
            log_content.append("处理步骤:\\n")
            for step in log_data['processing_steps']:
                log_content.append(f"- {step}\\n")
            log_content.append("\\n")
        
        # 添加统计信息
        if 'statistics' in log_data:
            log_content.append("统计信息:\\n")
            stats = log_data['statistics']
            for key, value in stats.items():
                log_content.append(f"- {key}: {value}\\n")
            log_content.append("\\n")
        
        # 添加修正详情
        if 'corrections' in log_data:
            log_content.append("修正详情:\\n")
            corrections = log_data['corrections']
            for sat_id, corr_data in corrections.items():
                log_content.append(f"  卫星 {sat_id}:\\n")
                for corr_type, details in corr_data.items():
                    log_content.append(f"    - {corr_type}: {details}\\n")
            log_content.append("\\n")
        
        # 写入日志文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(log_content)
        
        return output_path
    
    def create_rinex_header(self, original_header: Dict, modifications: Dict = None) -> list:
        """创建RINEX头部"""
        
        header_lines = []
        
        # RINEX版本行
        version = original_header.get('version', 3.0)
        header_lines.append(f"{version:9.2f}           OBSERVATION DATA    M                   RINEX VERSION / TYPE\\n")
        
        # 程序信息
        header_lines.append(f"{'CORRECTED':20}{'Python':20}{'':20}PGM / RUN BY / DATE\\n")
        
        # 标记名
        marker = original_header.get('marker', 'UNKNOWN')
        header_lines.append(f"{marker:60}MARKER NAME\\n")
        
        # 观测类型
        for system in ['G', 'R', 'E', 'C']:
            obs_types_key = f'obs_types_{system}'
            if obs_types_key in original_header:
                obs_types = original_header[obs_types_key]
                num_types = len(obs_types)
                
                # 第一行
                line = f"{system} {num_types:3d} "
                for i, obs_type in enumerate(obs_types[:13]):  # 最多13个观测类型一行
                    line += f"{obs_type:4s}"
                line = line.ljust(60) + "SYS / # / OBS TYPES\\n"
                header_lines.append(line)
                
                # 续行（如果需要）
                remaining_types = obs_types[13:]
                while remaining_types:
                    line = "      "  # 6个空格
                    for obs_type in remaining_types[:13]:
                        line += f"{obs_type:4s}"
                    line = line.ljust(60) + "SYS / # / OBS TYPES\\n"
                    header_lines.append(line)
                    remaining_types = remaining_types[13:]
        
        # 结束标记
        header_lines.append("                                                            END OF HEADER\\n")
        
        return header_lines
    
    def format_observation_value(self, value, field_width=14):
        """格式化观测值"""
        if value is None or np.isnan(value):
            return " " * field_width
        else:
            return f"{value:{field_width}.3f}"