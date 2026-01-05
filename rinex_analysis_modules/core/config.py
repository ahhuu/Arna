"""
GNSS配置模块
包含系统频率、波长、卫星信息等配置参数
"""

import numpy as np

class GNSSConfig:
    """GNSS系统配置类"""
    
    def __init__(self):
        # 光速常数
        self.speed_of_light = 299792458  # m/s
        
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
        
        # 计算对应波长 (m)
        self.wavelengths = {}
        for system, freqs in self.frequencies.items():
            self.wavelengths[system] = {
                freq: self.speed_of_light / f for freq, f in freqs.items()
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
                'MEO': ['C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31',
                        'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53',
                        'C54', 'C55', 'C56', 'C57', 'C58']  # 中地球轨道
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
        
        # 分析阈值配置
        self.r_squared_threshold = 0.5  # R方阈值，默认0.5
        self.cv_threshold = 0.6  # CV值阈值，默认0.5
        
        # 历元间双差最大阈值限制
        self.max_threshold_limits = {
            'code': 10.0,  # 伪距（米）
            'phase': 1.5,  # 相位（米）
            'doppler': 5.0  # 多普勒（米/秒）
        }
        
        # 手机独有卫星分析配置
        self.enable_phone_only_analysis = False  # 是否启用手机独有卫星分析
        self.phone_only_min_data_points = 20  # 手机独有卫星最小数据点数
        
    def get_frequency(self, system, signal):
        """获取指定系统和信号的频率"""
        return self.frequencies.get(system, {}).get(signal, None)
    
    def get_wavelength(self, system, signal):
        """获取指定系统和信号的波长"""
        return self.wavelengths.get(system, {}).get(signal, None)
    
    def get_beidou_type(self, prn):
        """获取北斗卫星类型"""
        for system, orbits in self.beidou_systems.items():
            for orbit_type, sats in orbits.items():
                if prn in sats:
                    return system, orbit_type
        return None, None
    
    def get_glonass_k(self, prn):
        """获取GLONASS卫星的k值"""
        return self.glonass_k_map.get(prn, 0)