#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
广播星历文件解析器和卫星位置计算器
作者: AI Assistant
描述: 读取RINEX格式的广播星历文件，解析卫星轨道参数，并计算卫星位置
"""

import math
import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np


class BroadcastEphemeris:
    """广播星历数据结构"""
    
    def __init__(self):
        # 卫星标识
        self.prn = ""
        
        # 时间信息
        self.toc = None  # 卫星钟时间 (年月日时分秒)
        self.toe = 0.0   # 星历的参考时刻 (TOE)
        self.gps_week = 0  # GPS时间周
        
        # 卫星钟参数
        self.a0 = 0.0    # 卫星钟差 (s)
        self.a1 = 0.0    # 卫星钟偏 (s/s)
        self.a2 = 0.0    # 卫星钟偏移 (s/s²)
        
        # 轨道参数
        self.aode = 0.0      # 数据龄期
        self.crs = 0.0       # 轨道半径改正项 (m)
        self.delta_n = 0.0   # 平均角速度改正项 (rad/s)
        self.m0 = 0.0        # 平近点角 (rad)
        
        self.cuc = 0.0       # 升交点角距改正项 (rad)
        self.e = 0.0         # 轨道偏心率
        self.cus = 0.0       # 升交点角距改正项 (rad)
        self.sqrt_a = 0.0    # 轨道长半轴平方根 (m^0.5)
        
        self.cic = 0.0       # 轨道倾角的改正项 (rad)
        self.omega0 = 0.0    # 升交点经度 (rad)
        self.cis = 0.0       # 轨道倾角改正项 (rad)
        
        self.i0 = 0.0        # 轨道倾角 (rad)
        self.crc = 0.0       # 轨道半径的改正项 (m)
        self.omega = 0.0     # 近地点角距 (rad)
        self.omega_dot = 0.0 # 升交点赤经变化 (rad/s)
        
        self.idot = 0.0      # 轨道倾角的变率 (rad/s)
        self.l2_code = 0.0   # L2频道C/A码标识
        self.l2p_code = 0.0  # L2P码标识
        
        # 质量参数
        self.sva = 0.0       # 卫星精度 (m)
        self.svh = 0.0       # 卫星健康
        self.tgd = 0.0       # 电离层延迟 (s)
        self.iodc = 0.0      # 星钟的数据质量
        
        # 其他参数
        self.transmission_time = 0.0  # 信息发射时间
        self.fit_interval = 0.0       # 拟合区间 (h)
        
        # GLONASS专用参数
        self.tb = 0.0           # GLONASS参考时刻
        self.gamma_n = 0.0      # 频率偏差
        self.tau_n = 0.0        # 时间偏差
        self.x_pos = 0.0        # X位置 (km)
        self.x_vel = 0.0        # X速度 (km/s)
        self.x_acc = 0.0        # X加速度 (km/s²)
        self.y_pos = 0.0        # Y位置 (km)
        self.y_vel = 0.0        # Y速度 (km/s)
        self.y_acc = 0.0        # Y加速度 (km/s²)
        self.z_pos = 0.0        # Z位置 (km)
        self.z_vel = 0.0        # Z速度 (km/s)
        self.z_acc = 0.0        # Z加速度 (km/s²)


class BroadcastEphemerisParser:
    """广播星历解析器"""
    
    def __init__(self):
        self.ephemeris_data = {}  # 存储解析的星历数据
        
    def read_rinex_nav_file(self, file_path: str) -> Dict[str, List[BroadcastEphemeris]]:
        """
        读取RINEX导航文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            字典，键为卫星PRN，值为该卫星的星历数据列表
        """
        ephemeris_data = {}
        
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                
            # 跳过头部信息，找到"END OF HEADER"标识
            data_start_line = 0
            for i, line in enumerate(lines):
                if "END OF HEADER" in line:
                    data_start_line = i + 1
                    break
            
            if data_start_line == 0:
                raise ValueError("未找到'END OF HEADER'标识")
            
            print(f"跳过头部 {data_start_line} 行，开始解析星历数据...")
            
            # 解析星历数据
            i = data_start_line
            while i < len(lines):
                if i >= len(lines):
                    break
                
                # 检查是否是有效的卫星标识行
                line = lines[i].strip()
                if not line or len(line) < 3:
                    i += 1
                    continue
                
                # 获取卫星系统标识
                sat_system = line[0]
                
                # 根据卫星系统确定数据块大小，只处理GRCEJ系统
                if sat_system in ['G', 'E', 'C', 'J']:  # GPS, Galileo, BeiDou, QZSS
                    block_size = 8
                elif sat_system == 'R':  # GLONASS
                    block_size = 4  # GLONASS只有4行数据
                else:
                    i += 1
                    continue
                
                if i + block_size - 1 >= len(lines):
                    break
                    
                ephemeris = self._parse_ephemeris_block(lines[i:i+block_size], sat_system)
                if ephemeris:
                    if ephemeris.prn not in ephemeris_data:
                        ephemeris_data[ephemeris.prn] = []
                    ephemeris_data[ephemeris.prn].append(ephemeris)
                    
                i += block_size
                
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return {}
            
        print(f"成功解析 {len(ephemeris_data)} 个卫星的星历数据")
        return ephemeris_data
    
    def _parse_ephemeris_block(self, lines: List[str], sat_system: str = 'G') -> Optional[BroadcastEphemeris]:
        """
        解析单个星历数据块（8行）
        
        Args:
            lines: 8行星历数据
            
        Returns:
            BroadcastEphemeris对象或None
        """
        try:
            ephemeris = BroadcastEphemeris()
            
            # 第一行：卫星PRN和时间信息
            first_line = lines[0]
            
            # 解析卫星PRN（前3个字符）
            ephemeris.prn = first_line[:3].strip()
            
            # 解析时间：年月日时分秒（固定位置）
            year = int(first_line[4:8])
            month = int(first_line[9:11])
            day = int(first_line[12:14])
            hour = int(first_line[15:17])
            minute = int(first_line[18:20])
            second = int(float(first_line[21:23]))
            
            ephemeris.toc = datetime.datetime(year, month, day, hour, minute, second)
            
            # 卫星钟参数（固定位置解析）
            ephemeris.a0 = self._parse_d_format(first_line[23:42])   # 卫星钟差
            ephemeris.a1 = self._parse_d_format(first_line[42:61])   # 卫星钟偏
            ephemeris.a2 = self._parse_d_format(first_line[61:80])   # 卫星钟偏移
            
            # 解析后续行数据，每行4个参数
            params = []
            if sat_system == 'R':  # GLONASS只有4行
                lines_to_parse = lines[1:4]
            else:  # GPS, Galileo, BeiDou, QZSS有8行
                lines_to_parse = lines[1:8]
            
            for line in lines_to_parse:
                line_params = []
                # 每行按固定格式解析4个科学计数法数值
                for j in range(4):
                    start_pos = j * 19 + 4  # 每个参数占19个字符，前面有4个空格
                    if start_pos + 19 <= len(line):
                        param_str = line[start_pos:start_pos + 19].strip()
                        if param_str:
                            line_params.append(self._parse_d_format(param_str))
                        else:
                            line_params.append(0.0)
                    else:
                        line_params.append(0.0)
                params.extend(line_params)
            
            # 按照参数意义分配
            if sat_system == 'R' and len(params) >= 12:
                # GLONASS系统只有12个参数（3行x4列）
                # 第2行：X坐标相关 - x(km), ẋ(km/s), ẍ(km/s²), 0
                ephemeris.x_pos = params[0]       # X位置 (km)
                ephemeris.x_vel = params[1]       # X速度 (km/s)
                ephemeris.x_acc = params[2]       # X加速度 (km/s²)
                # params[3] = 0 (备用)
                
                # 第3行：Y坐标相关 - y(km), ẏ(km/s), ÿ(km/s²), 0
                ephemeris.y_pos = params[4]       # Y位置 (km)
                ephemeris.y_vel = params[5]       # Y速度 (km/s)
                ephemeris.y_acc = params[6]       # Y加速度 (km/s²)
                # params[7] = 0 (备用)
                
                # 第4行：Z坐标相关 - z(km), ż(km/s), z̈(km/s²), 0
                ephemeris.z_pos = params[8]       # Z位置 (km)
                ephemeris.z_vel = params[9]       # Z速度 (km/s)
                ephemeris.z_acc = params[10]      # Z加速度 (km/s²)
                # params[11] = 0 (备用)
                
                # 从第一行提取的钟差参数已经在前面处理
                ephemeris.tb = ephemeris.toe      # 参考时刻
                ephemeris.gamma_n = ephemeris.a1  # 频率偏差
                ephemeris.tau_n = ephemeris.a0    # 时间偏差
                
            elif sat_system in ['G', 'E', 'C', 'J'] and len(params) >= 28:
                # GPS, Galileo, BeiDou, QZSS系统有28个参数（7行x4列）
                # 第2行
                ephemeris.aode = params[0]        # 数据龄期
                ephemeris.crs = params[1]         # 轨道半径改正项
                ephemeris.delta_n = params[2]     # 平均角速度改正项
                ephemeris.m0 = params[3]          # 平近点角
                
                # 第3行
                ephemeris.cuc = params[4]         # 升交点角距改正项
                ephemeris.e = params[5]           # 轨道偏心率
                ephemeris.cus = params[6]         # 升交点角距改正项
                ephemeris.sqrt_a = params[7]      # 轨道长半轴平方根
                
                # 第4行
                ephemeris.toe = params[8]         # 星历的参考时刻
                ephemeris.cic = params[9]         # 轨道倾角的改正项
                ephemeris.omega0 = params[10]     # 升交点经度
                ephemeris.cis = params[11]        # 轨道倾角改正项
                
                # 第5行
                ephemeris.i0 = params[12]         # 轨道倾角
                ephemeris.crc = params[13]        # 轨道半径的改正项
                ephemeris.omega = params[14]      # 近地点角距
                ephemeris.omega_dot = params[15]  # 升交点赤经变化
                
                # 第6行
                ephemeris.idot = params[16]       # 轨道倾角的变率
                ephemeris.l2_code = params[17]    # L2频道C/A码标识
                ephemeris.gps_week = params[18]   # GPS时间周
                ephemeris.l2p_code = params[19]   # L2P码标识
                
                # 第7行
                ephemeris.sva = params[20]        # 卫星精度
                ephemeris.svh = params[21]        # 卫星健康
                ephemeris.tgd = params[22]        # 电离层延迟
                ephemeris.iodc = params[23]       # 星钟的数据质量
                
                # 第8行
                ephemeris.transmission_time = params[24]  # 信息发射时间
                ephemeris.fit_interval = params[25]       # 拟合区间
                # params[26], params[27] 为空
            
            return ephemeris
            
        except Exception as e:
            print(f"解析星历数据块时出错: {e}")
            return None
    
    def _parse_d_format(self, value_str: str) -> float:
        """
        解析RINEX格式中的D科学计数法格式
        例如：-9.313225746150E-09 或 -9.313225746150D-09
        
        Args:
            value_str: 数值字符串
            
        Returns:
            浮点数值
        """
        if not value_str or value_str.strip() == "":
            return 0.0
        
        # 将D替换为E（RINEX格式有时使用D表示科学计数法）
        value_str = value_str.strip().replace('D', 'E')
        
        try:
            return float(value_str)
        except ValueError:
            print(f"无法解析数值: '{value_str}'")
            return 0.0


class SatellitePositionCalculator:
    """卫星位置计算器"""
    
    # GPS常量
    GM = 3.986005e14        # 地球引力常数 (m³/s²)
    OMEGA_E = 7.2921151467e-5  # 地球自转角速度 (rad/s)
    C = 299792458.0         # 光速 (m/s)
    
    # GLONASS常量
    GM_GLO = 3.9860044e14   # GLONASS地球引力常数 (m³/s²)
    OMEGA_E_GLO = 7.292115e-5  # GLONASS地球自转角速度 (rad/s)
    J20 = 1.08262668e-3     # J2项系数
    RE = 6378136.0          # 地球半径 (m)
    
    def __init__(self):
        pass
    
    def calculate_satellite_position(self, ephemeris: BroadcastEphemeris, 
                                   gps_time: float) -> Tuple[float, float, float]:
        """
        计算指定GPS时间的卫星位置
        
        Args:
            ephemeris: 广播星历数据
            gps_time: GPS时间 (秒)
            
        Returns:
            卫星在ECEF坐标系中的位置 (x, y, z) 单位：米
        """
        try:
            # 1. 计算轨道半长轴
            a = ephemeris.sqrt_a ** 2
            
            # 2. 计算时间差
            dt = gps_time - ephemeris.toe
            
            # 考虑GPS周的边界
            if dt > 302400:
                dt -= 604800
            elif dt < -302400:
                dt += 604800
            
            # 3. 计算平均角速度
            n0 = math.sqrt(self.GM / (a ** 3))
            n = n0 + ephemeris.delta_n
            
            # 4. 计算平近点角
            M = ephemeris.m0 + n * dt
            
            # 5. 求解偏近点角 (开普勒方程)
            E = self._solve_kepler_equation(M, ephemeris.e)
            
            # 6. 计算真近点角
            nu = math.atan2(math.sqrt(1 - ephemeris.e**2) * math.sin(E),
                           math.cos(E) - ephemeris.e)
            
            # 7. 计算升交点角距
            phi = nu + ephemeris.omega
            
            # 8. 计算摄动改正项
            # 升交点角距的改正
            delta_u = ephemeris.cuc * math.cos(2 * phi) + ephemeris.cus * math.sin(2 * phi)
            # 轨道半径的改正
            delta_r = ephemeris.crc * math.cos(2 * phi) + ephemeris.crs * math.sin(2 * phi)
            # 轨道倾角的改正
            delta_i = ephemeris.cic * math.cos(2 * phi) + ephemeris.cis * math.sin(2 * phi)
            
            # 9. 计算改正后的轨道参数
            u = phi + delta_u        # 改正后的升交点角距
            r = a * (1 - ephemeris.e * math.cos(E)) + delta_r  # 改正后的轨道半径
            i = ephemeris.i0 + delta_i + ephemeris.idot * dt   # 改正后的轨道倾角
            
            # 10. 计算升交点经度
            omega = ephemeris.omega0 + (ephemeris.omega_dot - self.OMEGA_E) * dt - self.OMEGA_E * ephemeris.toe
            
            # 11. 计算轨道平面内的坐标
            x_prime = r * math.cos(u)
            y_prime = r * math.sin(u)
            
            # 12. 转换到ECEF坐标系
            x = x_prime * math.cos(omega) - y_prime * math.cos(i) * math.sin(omega)
            y = x_prime * math.sin(omega) + y_prime * math.cos(i) * math.cos(omega)
            z = y_prime * math.sin(i)
            
            return x, y, z
            
        except Exception as e:
            print(f"计算卫星位置时出错: {e}")
            return 0.0, 0.0, 0.0
    
    def _solve_kepler_equation(self, M: float, e: float, tolerance: float = 1e-12) -> float:
        """
        求解开普勒方程 E - e*sin(E) = M
        
        Args:
            M: 平近点角 (rad)
            e: 偏心率
            tolerance: 收敛精度
            
        Returns:
            偏近点角 E (rad)
        """
        E = M  # 初始值
        
        for _ in range(20):  # 最大迭代次数
            E_new = M + e * math.sin(E)
            if abs(E_new - E) < tolerance:
                return E_new
            E = E_new
            
        return E
    
    def calculate_satellite_clock_correction(self, ephemeris: BroadcastEphemeris, 
                                           gps_time: float) -> float:
        """
        计算卫星钟差改正
        
        Args:
            ephemeris: 广播星历数据
            gps_time: GPS时间 (秒)
            
        Returns:
            卫星钟差改正 (秒)
        """
        # 计算时间差（相对于钟差参考时刻）
        toc_seconds = (ephemeris.toc - datetime.datetime(1980, 1, 6)).total_seconds()
        dt = gps_time - toc_seconds
        
        # 卫星钟差改正
        dt_sv = ephemeris.a0 + ephemeris.a1 * dt + ephemeris.a2 * dt**2
        
        # 相对论效应改正
        a = ephemeris.sqrt_a ** 2
        e = ephemeris.e
        E = self._solve_kepler_equation(ephemeris.m0 + math.sqrt(self.GM / (a**3)) * dt, e)
        dt_rel = -2 * math.sqrt(self.GM * a) * e * math.sin(E) / (self.C**2)
        
        return dt_sv + dt_rel
    
    def calculate_glonass_position(self, ephemeris: BroadcastEphemeris, 
                                  gps_time: float) -> Tuple[float, float, float]:
        """
        计算GLONASS卫星位置 - 使用数值积分方法
        
        Args:
            ephemeris: GLONASS广播星历数据
            gps_time: GPS时间 (秒)
            
        Returns:
            卫星在ECEF坐标系中的位置 (x, y, z) 单位：米
            
        Note:
            GLONASS广播星历包含的4行参数含义：
            第1行：tb(参考时刻), γn(频率偏差), τn(时间偏差), 0
            第2行：x(km), ẋ(km/s), ẍ(km/s²), 0
            第3行：y(km), ẏ(km/s), ÿ(km/s²), 0  
            第4行：z(km), ż(km/s), z̈(km/s²), 0
        """
        try:
            # 参考时刻 (TOE in seconds)
            tb = ephemeris.toe
            
            # 检查初始数据是否有效
            if (ephemeris.x_pos == 0.0 and ephemeris.y_pos == 0.0 and ephemeris.z_pos == 0.0):
                # 如果位置数据都为0，可能解析有问题，返回零值
                return 0.0, 0.0, 0.0
            
            # 初始位置 (km -> m)
            x0 = ephemeris.x_pos * 1000    # X位置
            y0 = ephemeris.y_pos * 1000    # Y位置
            z0 = ephemeris.z_pos * 1000    # Z位置
            
            # 初始速度 (km/s -> m/s)
            vx0 = ephemeris.x_vel * 1000   # X速度
            vy0 = ephemeris.y_vel * 1000   # Y速度
            vz0 = ephemeris.z_vel * 1000   # Z速度
            
            # 检查初始距离是否合理（6000-50000 km范围）
            r0 = math.sqrt(x0*x0 + y0*y0 + z0*z0)
            if r0 < 6e6 or r0 > 5e7:  # 不在合理轨道高度范围内
                return 0.0, 0.0, 0.0
            
            # 时间差（限制在合理范围内）
            dt = gps_time - tb
            if abs(dt) > 7200:  # 限制在2小时内
                dt = 7200 if dt > 0 else -7200
            
            # 如果时间差很小，直接返回初始位置
            if abs(dt) < 1.0:
                return x0, y0, z0
            
            # 使用简化的积分方法
            step_size = 60.0  # 增大积分步长到60秒
            num_steps = max(1, int(abs(dt) / step_size))
            actual_step = dt / num_steps
            
            # 初始状态向量 [x, y, z, vx, vy, vz]
            state = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=float)
            
            # 简化的数值积分（只使用引力项）
            for i in range(num_steps):
                # 使用简化的二阶积分方法
                x, y, z, vx, vy, vz = state
                r = math.sqrt(x*x + y*y + z*z)
                
                if r < 1000.0:  # 安全检查
                    break
                
                # 只考虑主要引力
                a = -self.GM_GLO / (r**3)
                ax = a * x
                ay = a * y
                az = a * z
                
                # 更新状态
                state[0] += vx * actual_step + 0.5 * ax * actual_step**2
                state[1] += vy * actual_step + 0.5 * ay * actual_step**2
                state[2] += vz * actual_step + 0.5 * az * actual_step**2
                state[3] += ax * actual_step
                state[4] += ay * actual_step
                state[5] += az * actual_step
            
            return state[0], state[1], state[2]
            
        except Exception as e:
            print(f"计算GLONASS卫星位置时出错: {e}")
            return 0.0, 0.0, 0.0
    
    def _runge_kutta_4(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        4阶龙格-库塔积分方法
        
        Args:
            state: 状态向量 [x, y, z, vx, vy, vz]
            dt: 时间步长
            
        Returns:
            更新后的状态向量
        """
        k1 = dt * self._glonass_derivatives(state)
        k2 = dt * self._glonass_derivatives(state + k1/2)
        k3 = dt * self._glonass_derivatives(state + k2/2)
        k4 = dt * self._glonass_derivatives(state + k3)
        
        return state + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def _glonass_derivatives(self, state: np.ndarray) -> np.ndarray:
        """
        计算GLONASS卫星的状态导数
        
        Args:
            state: [x, y, z, vx, vy, vz]
            
        Returns:
            导数 [vx, vy, vz, ax, ay, az]
        """
        x, y, z, vx, vy, vz = state
        
        # 距离地心的距离
        r = math.sqrt(x*x + y*y + z*z)
        
        # 安全检查：避免除零错误
        if r < 1000.0:  # 如果距离小于1km，使用最小值
            r = 1000.0
        
        # 主要引力加速度
        a_grav = -self.GM_GLO / (r**3)
        ax_grav = a_grav * x
        ay_grav = a_grav * y
        az_grav = a_grav * z
        
        # J2项引力摄动
        r2 = r * r
        z2 = z * z
        
        # 安全检查：确保r2不为零
        if r2 < 1e6:  # 最小值1e6 m²
            r2 = 1e6
        
        temp = 1.5 * self.J20 * (self.RE * self.RE) / (r2 * r2 * r)
        temp1 = 5 * z2 / r2 - 1
        temp2 = 5 * z2 / r2 - 3
        
        ax_j2 = temp * temp1 * x
        ay_j2 = temp * temp1 * y
        az_j2 = temp * temp2 * z
        
        # 地球自转影响
        ax_rot = 2 * self.OMEGA_E_GLO * vy + self.OMEGA_E_GLO**2 * x
        ay_rot = -2 * self.OMEGA_E_GLO * vx + self.OMEGA_E_GLO**2 * y
        az_rot = 0.0
        
        # 总加速度
        ax = ax_grav + ax_j2 + ax_rot
        ay = ay_grav + ay_j2 + ay_rot
        az = az_grav + az_j2 + az_rot
        
        return np.array([vx, vy, vz, ax, ay, az])


def calculate_and_save_satellite_positions(ephemeris_data: Dict[str, List[BroadcastEphemeris]], 
                                          output_file: str = "satellite_positions.txt"):
    """
    计算所有卫星的位置并保存到文件
    
    Args:
        ephemeris_data: 星历数据字典
        output_file: 输出文件名
    """
    calculator = SatellitePositionCalculator()
    
    print(f"\n开始计算所有卫星位置并保存到 {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write("# 卫星位置计算结果\n")
        f.write("# 格式: PRN 历元时间(YYYY-MM-DD HH:MM:SS) GPS时间(秒) X(m) Y(m) Z(m) 距离地心(m) 钟差改正(s)\n")
        f.write("# " + "="*80 + "\n")
        
        total_epochs = 0
        successful_calculations = 0
        
        # 遍历所有卫星
        for prn in sorted(ephemeris_data.keys()):
            eph_list = ephemeris_data[prn]
            
            for i, eph in enumerate(eph_list):
                total_epochs += 1
                
                try:
                    # 使用星历参考时刻计算位置
                    gps_time = eph.toe
                    
                    # 只计算支持的5个卫星系统
                    if eph.prn[0] not in ['G', 'R', 'E', 'C', 'J']:
                        continue
                    
                    # GLONASS卫星使用专用的轨道计算算法（暂时跳过）
                    if eph.prn[0] == 'R':
                        print(f"暂时跳过GLONASS卫星 {eph.prn}（数据解析需要进一步优化）")
                        continue
                    else:
                        x, y, z = calculator.calculate_satellite_position(eph, gps_time)
                    
                    # 计算距离地心的距离
                    distance = math.sqrt(x**2 + y**2 + z**2)
                    
                    # 计算卫星钟差改正
                    clock_corr = calculator.calculate_satellite_clock_correction(eph, gps_time)
                    
                    # 写入结果
                    time_str = eph.toc.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{eph.prn} {time_str} {gps_time:10.1f} "
                           f"{x:15.3f} {y:15.3f} {z:15.3f} {distance:15.3f} {clock_corr:12.9f}\n")
                    
                    successful_calculations += 1
                    
                    # 每处理100个历元显示进度
                    if total_epochs % 100 == 0:
                        print(f"已处理 {total_epochs} 个历元，成功计算 {successful_calculations} 个位置")
                        
                except Exception as e:
                    print(f"计算 {eph.prn} 第 {i+1} 个历元时出错: {e}")
                    continue
    
    print(f"\n计算完成！")
    print(f"总计处理: {total_epochs} 个历元")
    print(f"成功计算: {successful_calculations} 个卫星位置")
    print(f"结果已保存到: {output_file}")


def main():
    """主函数示例"""
    
    # 创建解析器
    parser = BroadcastEphemerisParser()
    calculator = SatellitePositionCalculator()
    
    # 读取广播星历文件
    file_path = "data/brdc1350.25p"
    ephemeris_data = parser.read_rinex_nav_file(file_path)
    
    if not ephemeris_data:
        print("未能读取到星历数据")
        return
    
    # 显示解析结果
    print(f"\n成功解析 {len(ephemeris_data)} 个卫星的星历数据:")
    for prn, eph_list in ephemeris_data.items():
        print(f"  {prn}: {len(eph_list)} 个历元")
        
        # 调试：检查GLONASS数据
        if prn.startswith('R') and eph_list:
            eph = eph_list[0]
            print(f"    GLONASS {prn} 第一个历元数据:")
            print(f"      位置: x={eph.x_pos:.3f} km, y={eph.y_pos:.3f} km, z={eph.z_pos:.3f} km")
            print(f"      速度: vx={eph.x_vel:.6f} km/s, vy={eph.y_vel:.6f} km/s, vz={eph.z_vel:.6f} km/s")
            print(f"      加速度: ax={eph.x_acc:.9f} km/s², ay={eph.y_acc:.9f} km/s², az={eph.z_acc:.9f} km/s²")
    
    # 示例：计算G01卫星在某个时刻的位置
    if "G01" in ephemeris_data and ephemeris_data["G01"]:
        print(f"\n=== G01卫星星历数据示例 ===")
        eph = ephemeris_data["G01"][0]  # 取第一个历元
        
        print(f"卫星PRN: {eph.prn}")
        print(f"星历参考时间: {eph.toc}")
        print(f"GPS周: {eph.gps_week}")
        print(f"轨道半长轴平方根: {eph.sqrt_a:.3f} m^0.5")
        print(f"偏心率: {eph.e:.6f}")
        print(f"轨道倾角: {math.degrees(eph.i0):.3f} °")
        print(f"升交点经度: {math.degrees(eph.omega0):.3f} °")
        
        # 计算卫星位置（使用星历参考时刻）
        gps_time = eph.toe
        x, y, z = calculator.calculate_satellite_position(eph, gps_time)
        
        print(f"\n=== 卫星位置计算结果 ===")
        print(f"GPS时间: {gps_time}")
        print(f"卫星位置 (ECEF):")
        print(f"  X: {x:12.3f} m")
        print(f"  Y: {y:12.3f} m")
        print(f"  Z: {z:12.3f} m")
        
        # 计算距离地心的距离
        distance = math.sqrt(x**2 + y**2 + z**2)
        print(f"距离地心: {distance:12.3f} m ({distance/1000:.1f} km)")
        
        # 计算卫星钟差改正
        clock_corr = calculator.calculate_satellite_clock_correction(eph, gps_time)
        print(f"卫星钟差改正: {clock_corr:.9f} s")
    
    # 计算并保存所有卫星位置
    calculate_and_save_satellite_positions(ephemeris_data, "satellite_positions.txt")


if __name__ == "__main__":
    main()
