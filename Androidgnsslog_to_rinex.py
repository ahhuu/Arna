"""
#   This code is used to convert the GNSS logger output csv file (.txt)
#   format to RINEX format (*.?o)
#           Copyright (C), All rights reserved to
#           Dr. Yang Gao's group.
#   Authors: Farzaneh Zangenehnejad and Yang Jiang from Dr. Yang Gao's group,
#            University of Calgary, Calgary, Canada
#   Contact Email: farzaneh.zangenehnej@ucalgary.ca and yang.jiang1@ucalgary.ca
#   Version : 1  (Dec 2022)
# -----------------------------------------------------------------------------
#   Added : Support visualization and batch processing, QZSS dual-frequency, Beidou and Galileo triple-frequency signal
#   correct some parameter errors, Update the RINEX version to 3.05 and add the approximate XYZ position, model, etc.
#
#   Authors: Zhe Chen from SouthWest Jiaotong University
#   Date: Mar 24, 2025
"""

import math
import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog
from tkinter import ttk, messagebox
import logging  # 新增：日志模块

# -------------------- 日志配置 --------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('gnss_converter.log', mode='a')  # 保存到日志文件
    ]
)
logger = logging.getLogger(__name__)

# Constants
CLIGHT = 299792458.0  # Speed of light (m/s)
LeapSecond = 18  # Leap second for 2021

MAXPRRUNCMPS = 10   # Maximum pseudorange rate (Doppler) uncertainty, 10 m/s, for example
MAXTOWUNCNS = 500   # Maximum Tow uncertainty, 500 ns, for example
MAXADRUNCNS = 1     # Maximum ADR uncertainty, 1 m, for example

# PDF方法：C/N0阈值过滤（暂时禁用）
# CN0_THRESHOLD_GPS_L1 = 23.0    # GPS L1信号C/N0阈值 (dB-Hz)
# CN0_THRESHOLD_GAL_L1 = 15.0    # Galileo L1信号C/N0阈值 (dB-Hz)
# CN0_THRESHOLD_GENERAL = 20.0   # 其他信号的通用C/N0阈值 (dB-Hz)
# CN0_THRESHOLD_SAFE = 30.0      # 安全C/N0阈值，避免低质量测量

# PDF方法：相位调整阈值
PHASE_CODE_DIFF_THRESHOLD = 50.0  # 码相差异阈值 (米)

MAX_SYS = 10
MAX_FRQ = 5

SYS_GPS = 1
SYS_GLO = 3
SYS_GAL = 6
SYS_BDS = 5
SYS_QZS = 4  # Add QZSS system

RNX_VER = "     3.05           OBSERVATION DATA    M: Mixed            RINEX VERSION / TYPE"
RNX_MA1 = "Reference Station                                           MARKER NAME         "
RNX_MA2 = "Unknown                                                     MARKER NUMBER       "
RNX_MA3 = "Unknown                                                     MARKER TYPE         "
RNX_OBS = "SWJTU               SWJTU                                   OBSERVER / AGENCY   "
RNX_AN1 = "unknown             unknown                                 ANT # / TYPE        "
RNX_AN2 = "        0.0000        0.0000        0.0000                  ANTENNA: DELTA H/E/N"
RNX_END = "                                                            END OF HEADER       "

# Measurement states
GPS_MEASUREMENT_STATE_UNKNOWN = 0
STATE_CODE_LOCK = 1  # 2^0
STATE_TOW_KNOWN = 8  # 2^3

STATE_GLO_STRING_SYNC = 64  # 2^6
STATE_GLO_TOD_KNOWN = 128  # 2^7

STATE_GAL_E1C_2ND_CODE_LOCK = 2048  # 2^11
STATE_GAL_E1BC_CODE_LOCK = 1024  # 2^10
STATE_GAL_E1B_PAGE_SYNC = 4096  # 2^12

GPS_ADR_STATE_UNKNOWN = 0
GPS_ADR_STATE_VALID = 1  # 2^0
GPS_ADR_STATE_RESET = 2  # 2^1
GPS_ADR_STATE_CYCLE_SLIP = 4  # 2^2

NEAR_ZERO = 0.0001  # Threshold to judge if a float equals 0


class GnssSat:
    def __init__(self):
        # 基础时间属性
        self.ss = ""  # 原始数据标识
        self.ElapsedRealtimeMillis = 0  # UTC时间戳(毫秒)
        self.time_nano = 0  # 系统时间(纳秒)
        self.leap_second = 0  # 闰秒数
        self.time_uncertainty_nano = 0.0  # 时间不确定度(纳秒)
        self.full_bias_nano = 0  # 时钟偏差(纳秒)
        self.bias_nano = 0.0  # 时钟偏差修正(纳秒)
        self.bias_uncertainty_nano = 0.0  # 时钟偏差不确定度(纳秒)
        self.drift_nano_per_second = 0.0  # 时钟漂移(纳秒/秒)
        self.drift_uncertainty_nano_per_second = 0.0  # 时钟漂移不确定度(纳秒/秒)
        self.hardware_clock_discountinuity_count = 0  # 硬件时钟不连续计数

        # 卫星属性
        self.svid = 0  # 卫星ID
        self.time_offset_nano = 0.0  # 时间偏移(纳秒)
        self.state = 0  # 卫星状态
        self.received_sv_time_nano = 0  # 接收的卫星时间(纳秒)
        self.received_sv_time_uncertainty_nano = 0  # 卫星时间不确定度(纳秒)
        self.cn0_dbhz = 0.0  # 载噪比(dB-Hz)
        self.pseudorange_rate_meter_per_second = 0.0  # 伪距变化率(米/秒)
        self.pseudorange_rate_uncertainty_meter_per_second = 0.0  # 伪距变化率不确定度(米/秒)
        self.accumulated_delta_range_state = 0  # 累积差分距离状态
        self.accumulated_delta_range_meter = 0.0  # 累积差分距离(米)
        self.accumulated_delta_range_uncertainty_meter = 0.0  # 累积差分距离不确定度(米)
        self.carrier_frequency_hz = 0.0  # 载波频率(Hz)
        self.carrier_cycle = 0  # 载波周数
        self.carrier_phase = 0.0  # 载波相位
        self.carrier_phase_uncertainty = 0.0  # 载波相位不确定度
        self.multipath_indicator = 0  # 多径指示
        self.snr_in_db = 0.0  # 信噪比(dB)
        self.constellation_type = 0  # 星座类型(1=GPS,3=GLONASS,6=Galileo,5=北斗)

        # 信号属性
        self.agc_db = 0.0  # 自动增益控制(dB)
        self.baseband_cn0_db_hz = 0.0  # 基带载噪比(dB-Hz)
        self.full_inter_signal_bias_nanos = 0  # 全信号间偏差(纳秒)
        self.full_inter_signal_bias_uncertainty_nanos = 0.0  # 全信号间偏差不确定度(纳秒)
        self.satellite_inter_signal_bias_nanos = 0  # 卫星间信号偏差(纳秒)
        self.satellite_inter_signal_bias_uncertainty_nanos = 0.0  # 卫星间信号偏差不确定度(纳秒)
        self.code_type = ""  # 码类型(C/Q,I等)
        self.chipset_elapsed_realtime_nanos = 0  # 芯片运行时间(纳秒)
        self.is_full_tracking = 0  # 是否完全跟踪

        # 处理后属性
        self.signal_name = ""  # 信号名称(如"L1C","B1I")
        self.sys = 0  # 卫星系统(同constellation_type)

    def parse_from(self, line):
        parts = line.strip().split(',')
        # 处理空字段(用'0'替换)
        parts = [p if p != '' else '0' for p in parts]

        # 基础时间属性
        self.ss = parts[0]
        self.ElapsedRealtimeMillis = int(parts[1])
        self.time_nano = int(parts[2])
        self.leap_second = int(parts[3])
        self.time_uncertainty_nano = float(parts[4])
        self.full_bias_nano = int(parts[5])
        self.bias_nano = float(parts[6])
        self.bias_uncertainty_nano = float(parts[7])
        self.drift_nano_per_second = float(parts[8])
        self.drift_uncertainty_nano_per_second = float(parts[9])
        self.hardware_clock_discountinuity_count = int(parts[10])

        # 卫星属性
        self.svid = int(parts[11])
        self.time_offset_nano = float(parts[12])
        self.state = int(parts[13])
        self.received_sv_time_nano = int(parts[14])
        self.received_sv_time_uncertainty_nano = int(parts[15])
        self.cn0_dbhz = float(parts[16])
        self.pseudorange_rate_meter_per_second = float(parts[17])
        self.pseudorange_rate_uncertainty_meter_per_second = float(parts[18])
        self.accumulated_delta_range_state = int(parts[19])
        self.accumulated_delta_range_meter = float(parts[20])
        self.accumulated_delta_range_uncertainty_meter = float(parts[21])
        self.carrier_frequency_hz = float(parts[22])
        self.carrier_cycle = int(parts[23])
        self.carrier_phase = float(parts[24])
        self.carrier_phase_uncertainty = float(parts[25])
        self.multipath_indicator = int(parts[26])
        self.snr_in_db = float(parts[27])
        self.constellation_type = int(parts[28])

        # 信号属性
        self.agc_db = float(parts[29])
        # 移除carrier_frequency_hz2映射
        # self.carrier_frequency_hz2 = float(parts[30])
        self.baseband_cn0_db_hz = float(parts[30])
        self.full_inter_signal_bias_nanos = float(parts[31])
        self.full_inter_signal_bias_uncertainty_nanos = float(parts[32])
        self.satellite_inter_signal_bias_nanos = float(parts[33])
        self.satellite_inter_signal_bias_uncertainty_nanos = float(parts[34])
        self.code_type = parts[35] if len(parts) > 35 else ""
        self.chipset_elapsed_realtime_nanos = int(parts[36]) if len(parts) > 36 else 0
        self.is_full_tracking = parts[37] == '1' if len(parts) > 37 else False
        self.signal_name = ""

    def print_frequency_and_signal(self):
        """输出载波频率和信号名称，用于调试"""
        print(f"SVID: {self.svid}, Constellation: {self.constellation_type}")
        print(f"Carrier Frequency: {self.carrier_frequency_hz} Hz")
        print(f"Code Type: {self.code_type}")
        print(f"Signal Name: {self.signal_name}")
        print(f"AGC: {self.agc_db} dB, CN0: {self.cn0_dbhz} dB-Hz")
        print("-" * 40)


class GnssEpoch:
    def __init__(self):
        self.full_bias_nano = 0
        self.time_nano = 0
        self.bia_nano = 0.0
        self.nobs = 0
        self.obs = None


class RnxSat:
    def __init__(self):
        self.sys = 0
        self.prn = 0
        self.p = [0.0] * MAX_FRQ
        self.l = [0.0] * MAX_FRQ
        self.d = [0.0] * MAX_FRQ
        self.s = [0.0] * MAX_FRQ
        # PDF方法：添加IPA存储
        self.ipa = [0] * MAX_FRQ  # 初始相位模糊度
        self.first_observation = [True] * MAX_FRQ  # 标记是否为首次观测


class RnxEpoch:
    def __init__(self):
        self.time = [0.0] * 6
        self.sv = 0
        self.sats = []


def sys_code_function(sys):
    if sys == 1: return 0  # GPS
    if sys == 3: return 1  # GLO
    if sys == 6: return 2  # GAL
    if sys == 5: return 3  # BDS
    if sys == 4: return 4  # QZSS
    return -1


def qzss_prn_mapping(svid):
    if svid == 194: return 2
    if svid == 195: return 3
    if svid == 196: return 4
    if svid == 199: return 7
    return svid - 192  # Default mapping


def compare_sats(a, b):
    # Define system priority order: G(1), R(3), E(6), C(5), J(4)
    priority_a = (1 if a.sys == 1 else
                  2 if a.sys == 3 else
                  3 if a.sys == 6 else
                  4 if a.sys == 5 else 5)
    priority_b = (1 if b.sys == 1 else
                  2 if b.sys == 3 else
                  3 if b.sys == 6 else
                  4 if b.sys == 5 else 5)

    # Sort by system first
    if priority_a != priority_b:
        return priority_a < priority_b
    # Then by PRN
    return a.prn < b.prn


def parse_manufacturer_model(file_path):
    """从原始文件中提取Manufacturer和Model字段"""
    with open(file_path, 'r') as f:
        for line in f:
            if '# Version:' in line and 'Manufacturer:' in line and 'Model:' in line:
                parts = line.strip().split()
                manuf_idx = parts.index('Manufacturer:') + 1
                model_idx = parts.index('Model:') + 1
                return parts[manuf_idx], parts[model_idx]
    return '', ''  # 未找到时返回空


def parse_xyz_from_raw(raw_lines):
    """从原始数据中提取第一个Fix行的经纬度和高度，并转换为XYZ"""
    for line in raw_lines:
        if line.startswith('Fix,GPS,'):
            parts = line.strip().split(',')
            if len(parts) >= 5:
                try:
                    lat = float(parts[2])
                    lon = float(parts[3])
                    h = float(parts[4])
                    return latlonh_to_xyz(lat, lon, h)
                except (ValueError, IndexError):
                    continue
    return None  # 未找到有效Fix行


def latlonh_to_xyz(lat_deg, lon_deg, h_m):
    """经纬度（度）转WGS84 XYZ坐标（米）"""
    a = 6378137.0  # WGS84长半轴
    f = 1 / 298.257223563  # 扁率
    e_sq = 2 * f - f ** 2  # 第一偏心率平方

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = a / math.sqrt(1 - e_sq * math.sin(lat) ** 2)

    X = (N + h_m) * math.cos(lat) * math.cos(lon)
    Y = (N + h_m) * math.cos(lat) * math.sin(lon)
    Z = (N * (1 - e_sq) + h_m) * math.sin(lat)

    return X, Y, Z


# def check_cn0_threshold(sat):
#     """PDF方法：检查C/N0阈值（暂时禁用）"""
#     cn0 = sat.cn0_dbhz
#     
#     if sat.sys == SYS_GPS and round(sat.carrier_frequency_hz / 1e4) == 157542:  # GPS L1
#         return cn0 >= CN0_THRESHOLD_GPS_L1
#     elif sat.sys == SYS_GAL and round(sat.carrier_frequency_hz / 1e4) == 157542:  # Galileo L1
#         return cn0 >= CN0_THRESHOLD_GAL_L1
#     else:
#         return cn0 >= CN0_THRESHOLD_GENERAL


def check_multipath_indicator(sat):
    """PDF方法：检查多径指示器"""
    return sat.multipath_indicator == 0  # 0表示未检测到多径


def check_phase_continuity(sat, prev_sat=None):
    """PDF方法：检查相位连续性"""
    if prev_sat is None:
        return True  # 首次测量
    
    # 检查ADR状态
    if sat.accumulated_delta_range_state & GPS_ADR_STATE_CYCLE_SLIP:
        return False  # 检测到周跳
    
    if not (sat.accumulated_delta_range_state & GPS_ADR_STATE_VALID):
        return False  # ADR状态无效
    
    if not (prev_sat.accumulated_delta_range_state & GPS_ADR_STATE_VALID):
        return False  # 上一历元ADR状态无效
    
    return True


def compute_adjusted_adr(sat, current_fbn, current_bn, initial_fbn, initial_bn):
    """PDF方法：计算调整后的ADR"""
    wavelength = CLIGHT / sat.carrier_frequency_hz
    
    # 计算FBN和BN的差异
    fbn_diff = current_fbn - initial_fbn
    bn_diff = current_bn - initial_bn
    
    # 调整ADR：ADR̃ = (1/λ) * ADR - (c/λ) * [(FBN(t) - FBN(t₀)) - (BN(t) - BN(t₀))]
    adjusted_adr = (sat.accumulated_delta_range_meter / wavelength) - \
                   (CLIGHT / wavelength) * (fbn_diff - bn_diff) * 1e-9
    
    return adjusted_adr


def compute_initial_phase_ambiguity(pseudorange, adjusted_adr, wavelength):
    """PDF方法：计算初始相位模糊度"""
    # IPA = int[(1/λ) * (p - ADR̃)]
    ipa = int((pseudorange - adjusted_adr * wavelength) / wavelength)
    return ipa


def compute_phase_range(ipa, adjusted_adr):
    """PDF方法：计算相位范围"""
    # φ = IPA + ADR̃
    return ipa + adjusted_adr


def gpstime2ymdhms(time_nano, full_bias_nano, bias_nano):
    # Compute GPS time: GPS time = time Nano - (fullbiasnano + biasnano)[ns]
    delta_time_nano = time_nano - full_bias_nano  # in ns
    delta_time_sec = delta_time_nano // 1000000000  # full sec in second
    delta_time_frac = (delta_time_nano - delta_time_sec * 1000000000 - bias_nano) / 1e9  # fractional part

    HOURSEC = 3600  # Number of seconds in an hour
    MINSEC = 60  # Number of seconds in a minute
    DAYSEC = 86400  # Number of seconds in a day

    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in each month (not leap year)
    days = delta_time_sec // DAYSEC + 6  # days since 1980/1/1
    years = 1980

    # Calculate the year
    leap = 1  # 1980 was a leap year
    while days > leap + 365:
        days -= (leap + 365)
        years += 1
        leap = (1 if years % 4 == 0 else 0)  # leap = 1 on a leap year, 0 otherwise

    # Calculate the month
    month = 1
    month_days[1] = 29 if years % 4 == 0 else 28  # February days
    while days > month_days[month - 1]:
        days -= month_days[month - 1]
        month += 1

    # Calculate time of day
    since_midnight_seconds = delta_time_sec % DAYSEC
    hour = since_midnight_seconds // HOURSEC
    last_hour_seconds = since_midnight_seconds % HOURSEC
    minute = last_hour_seconds // MINSEC
    second = (last_hour_seconds % MINSEC) + delta_time_frac

    return [years, month, days, hour, minute, second]


def print_rnx_epoch(fp, epoch):
    # Sort satellites
    epoch.sats.sort(key=lambda x: (x.sys, x.prn))

    # Write epoch header
    fp.write("> {:04d} {:02d} {:02d} {:02d} {:02d} {:10.7f}  0 {:2d}\n".format(
        int(epoch.time[0]), int(epoch.time[1]), int(epoch.time[2]),
        int(epoch.time[3]), int(epoch.time[4]), epoch.time[5], epoch.sv))

    # Write satellite observations
    for sat in epoch.sats:
        prn = sat.prn
        if sat.sys == SYS_QZS:
            prn = qzss_prn_mapping(prn)  # Apply QZSS PRN mapping

        sys_n = sys_code_function(sat.sys)
        fp.write("{}{:02d}".format(['G', 'R', 'E', 'C', 'J'][sys_n], prn))

        nsig = nsignals[sys_n]
        for i in range(nsig):
            # Pseudorange (P)
            if sat.p[i]:
                fp.write("{:16.3f}".format(sat.p[i]))
            else:
                fp.write("                ")

            # Carrier phase (L)
            if sat.l[i]:
                fp.write("{:16.3f}".format(sat.l[i]))
            else:
                fp.write("                ")

            # Doppler (D)
            if sat.d[i]:
                fp.write("{:16.3f}".format(sat.d[i]))
            else:
                fp.write("                ")

            # SNR (S)
            if sat.s[i]:
                fp.write("{:16.3f}".format(sat.s[i]))
            else:
                fp.write("                ")

        fp.write("\n")


def print_rnx_header(fp, first_obs_time, manufacturer, model, xyz, current_time):
    fp.write(RNX_VER + "\n")
    run_by = manufacturer if manufacturer else "unknown"
    date_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    fp.write(f"UofC CSV2RINEX      {run_by:20s}{date_str:20s}PGM / RUN BY / DATE \n")
    fp.write(RNX_MA1 + "\n")
    fp.write(RNX_MA2 + "\n")
    fp.write(RNX_MA3 + "\n")
    fp.write(RNX_OBS + "\n")
    model_str = model if model else "unknown"
    fp.write(f"unknown             {run_by:20s}{model_str:20s}REC # / TYPE / VERS \n")
    fp.write(RNX_AN1 + "\n")
    if xyz:
        x, y, z = xyz
        fp.write(f" {x:9.4f}  {y:9.4f}  {z:9.4f}                  APPROX POSITION XYZ \n")
    else:
        fp.write(f"        0.0000        0.0000        0.0000                  APPROX POSITION XYZ " "\n")
    fp.write(RNX_AN2 + "\n")

    # Signal types
    for i in range(5):  # GPS, GLO, GAL, BDS, QZSS
        if nsignals[i] == 0:
            continue

        signal_line = ""
        if nsignals[i] == 1:
            signal_line = "{:1s}   {:2d} C{:<2s} L{:<2s} D{:<2s} S{:<2s}                                      SYS / # / OBS TYPES ".format(
                ['G', 'R', 'E', 'C', 'J'][i], nsignals[i] * 4,
                signals[i][0][1:], signals[i][0][1:],
                signals[i][0][1:], signals[i][0][1:])
        elif nsignals[i] == 2:
            signal_line = "{:1s}   {:2d} C{:<2s} L{:<2s} D{:<2s} S{:<2s} C{:<2s} L{:<2s} D{:<2s} S{:<2s}                      SYS / # / OBS TYPES ".format(
                ['G', 'R', 'E', 'C', 'J'][i], nsignals[i] * 4,
                signals[i][0][1:], signals[i][0][1:], signals[i][0][1:], signals[i][0][1:],
                signals[i][1][1:], signals[i][1][1:], signals[i][1][1:], signals[i][1][1:])
        elif nsignals[i] == 3:
            signal_line = "{:1s}   {:2d} C{:<2s} L{:<2s} D{:<2s} S{:<2s} C{:<2s} L{:<2s} D{:<2s} S{:<2s} C{:<2s} L{:<2s} D{:<2s} S{:<2s}      SYS / # / OBS TYPES ".format(
                ['G', 'R', 'E', 'C', 'J'][i], nsignals[i] * 4,
                signals[i][0][1:], signals[i][0][1:], signals[i][0][1:], signals[i][0][1:],
                signals[i][1][1:], signals[i][1][1:], signals[i][1][1:], signals[i][1][1:],
                signals[i][2][1:], signals[i][2][1:], signals[i][2][1:], signals[i][2][1:])

        fp.write(signal_line + "\n")

    # First observation time
    fp.write("  {:04d}    {:02d}    {:02d}    {:02d}    {:02d}   {:10.6f}     GPS         TIME OF FIRST OBS\n".format(
        int(first_obs_time[0]), int(first_obs_time[1]), int(first_obs_time[2]),
        int(first_obs_time[3]), int(first_obs_time[4]), first_obs_time[5]))

    fp.write(RNX_END + "\n")


def main():
    global signals, nsignals

    root = tk.Tk()
    root.withdraw()

    # 选择多个输入文件
    input_files = filedialog.askopenfilenames(
        title="选择输入文件（支持多选）",
        filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
    )
    if not input_files:
        messagebox.showinfo("提示", "未选择输入文件，程序终止")
        return

    # 选择输出文件目录
    OUTPUT_DIR = filedialog.askdirectory(title="选择输出目录")
    if not OUTPUT_DIR:
        messagebox.showinfo("提示", "未选择输出目录，程序终止")
        return

    # 创建进度窗口
    progress_window = tk.Toplevel()
    progress_window.title("处理进度")
    progress_window.geometry("600x200")
    progress_window.resizable(False, False)

    # 居中显示窗口
    progress_window.update_idletasks()
    width = progress_window.winfo_width()
    height = progress_window.winfo_height()
    x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
    y = (progress_window.winfo_screenheight() // 2) - (height // 2)
    progress_window.geometry(f'{width}x{height}+{x}+{y}')

    # 创建进度条
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_window, variable=progress_var, length=500, mode='determinate')
    progress_bar.pack(pady=20)

    # 创建状态标签
    status_label = tk.Label(progress_window, text="准备处理...", font=("Arial", 10))
    status_label.pack(pady=10)

    # 计算总进度（每个文件为一个进度单位）
    total_files = len(input_files)
    progress_per_file = 100 / total_files

    # 处理每个选择的文件
    for i, input_file in enumerate(input_files):
        try:
            logger.info(f"开始处理...")
            logger.info(f"读取原始文件: {os.path.basename(input_file)}")
            logger.info("使用PDF方法进行GNSS观测处理：")
            logger.info("- 每历元更新FBN和BN")
            logger.info("- 多径检测")
            logger.info("- 完整的相位观测计算（ADR调整+IPA机制）")
            logger.info("- 相位连续性处理")
            logger.info("- 码相差阈值检测")

            # 更新进度条和状态
            status_label.config(text=f"正在处理: {os.path.basename(input_file)}")
            progress_window.update()

            # 提取Manufacturer和Model
            manufacturer, model = parse_manufacturer_model(input_file)

            # 获取当前时间
            current_time = datetime.now()

            # 重置信号和历元数据
            signals = [[[""] * 5 for _ in range(MAX_FRQ)] for _ in range(MAX_SYS)]
            nsignals = [0] * MAX_SYS
            epochs = []

            # 解析当前文件
            with open(input_file, 'r') as fp:
                lines = fp.readlines()
                total_lines = len(lines)
                logger.info(f"读取原始文件完成，共 {total_lines} 行")
                logger.info(f"转换原始CSV格式至RINEX3.05格式...")
                # 提取坐标
                xyz = parse_xyz_from_raw(lines)

                for j, line in enumerate(lines):
                    if "Raw," in line and "#" not in line:
                        # Replace empty fields with '0'
                        line = line.strip()
                        parts = line.split(',')
                        parts = [p if p != '' else '0' for p in parts]
                        line = ','.join(parts)

                        epoch = GnssEpoch()
                        sat = GnssSat()
                        sat.parse_from(line)
                        # sat.print_frequency_and_signal()
                        epoch.obs = sat
                        epochs.append(epoch)

                    # 更新文件内处理进度（每100行更新一次，避免频繁更新）
                    if j % 100 == 0:
                        file_progress = (j / total_lines) * progress_per_file
                        current_progress = i * progress_per_file + file_progress
                        progress_var.set(current_progress)
                        progress_window.update()

            # Record first observation time
            first_obs_time = [0] * 6
            first_obs_set = False

            # Find each constellation and signal type
            for epoch in epochs:
                sat = epoch.obs

                # Record first observation time
                if not first_obs_set:
                    first_obs_time = gpstime2ymdhms(sat.time_nano, epochs[0].obs.full_bias_nano,
                                                    epochs[0].obs.bias_nano)
                    first_obs_set = True

                # Determine system and signal type
                if sat.constellation_type == 1:  # GPS
                    if round(sat.carrier_frequency_hz / 1e4) == 157542:  # GPS L1 1575420000
                        sat.sys = SYS_GPS
                        sat.signal_name = "L1C"
                        add_signal(SYS_GPS, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e4) * 1e4
                    elif round(sat.carrier_frequency_hz / 1e4) == 117645:  # GPS L5 1176450000
                        sat.sys = SYS_GPS
                        sat.signal_name = "L5Q"
                        add_signal(SYS_GPS, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e4) * 1e4
                elif sat.constellation_type == 3:  # GLO
                    if round(sat.carrier_frequency_hz / 1e7) == 160:  # GLO L1 1602000000
                        sat.sys = SYS_GLO
                        sat.signal_name = "L1C"
                        add_signal(SYS_GLO, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e2) * 1e2
                elif sat.constellation_type == 5:  # BDS
                    if round(sat.carrier_frequency_hz / 1e3) == 1561098:  # BDS B1I 1561097984
                        sat.sys = SYS_BDS
                        sat.signal_name = "B2I"
                        add_signal(SYS_BDS, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e3) * 1e3
                    elif round(sat.carrier_frequency_hz / 1e4) == 157542:  # BDS B1C 1575420000
                        sat.sys = SYS_BDS
                        sat.signal_name = "B1P"
                        add_signal(SYS_BDS, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e4) * 1e4
                    elif round(sat.carrier_frequency_hz / 1e4) == 117645:  # BDS B2a 1176450000
                        sat.sys = SYS_BDS
                        sat.signal_name = "B5P"
                        add_signal(SYS_BDS, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e4) * 1e4
                elif sat.constellation_type == 6:  # GAL
                    if round(sat.carrier_frequency_hz / 1e4) == 157542:  # GAL E1 1575420000
                        sat.sys = SYS_GAL
                        sat.signal_name = "E1C"
                        add_signal(SYS_GAL, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e4) * 1e4
                    elif round(sat.carrier_frequency_hz / 1e4) == 117645:  # GAL E5a 1176450000
                        sat.sys = SYS_GAL
                        sat.signal_name = "E5Q"
                        add_signal(SYS_GAL, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e4) * 1e4
                    elif round(sat.carrier_frequency_hz / 1e4) == 120714:  # GAL E5b 1207140000
                        sat.sys = SYS_GAL
                        sat.signal_name = "E7Q"
                        add_signal(SYS_GAL, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e4) * 1e4
                elif sat.constellation_type == 4:  # QZSS
                    if round(sat.carrier_frequency_hz / 1e4) == 157542:  # QZSS L1 1575420000
                        sat.sys = SYS_QZS
                        sat.signal_name = "L1C"
                        add_signal(SYS_QZS, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e4) * 1e4
                    elif round(sat.carrier_frequency_hz / 1e4) == 117645:  # QZSS L5 1176450000
                        sat.sys = SYS_QZS
                        sat.signal_name = "L5Q"
                        add_signal(SYS_QZS, sat.signal_name)
                        sat.carrier_frequency_hz = round(sat.carrier_frequency_hz / 1e4) * 1e4

            # Compute full cycle time of measurement, in milliseconds
            allRxMillis_p = int((epochs[0].obs.time_nano - epochs[0].obs.full_bias_nano) * 1e-6)
            check_clkdiscp = epochs[0].obs.hardware_clock_discountinuity_count
            clkdiscp = True

            epo_bias = 0
            clock_jump_count = 0  # 新增：记录跳变次数
            count = -1
            rnx = []
            repoch = RnxEpoch()
            
            # PDF方法：每历元更新FBN的基准值
            current_fbn = epochs[0].obs.full_bias_nano
            current_bn = epochs[0].obs.bias_nano
            initial_fbn = current_fbn
            initial_bn = current_bn
            
            # 统计信息
            total_observations = 0
            rejected_cn0 = 0
            rejected_multipath = 0
            rejected_uncertainty = 0
            rejected_time_uncertainty = 0
            phase_discontinuities = 0
            ipa_reinitializations = 0

            for epoch in epochs:
                count += 1
                sat = epoch.obs
                allRxMillis = int((sat.time_nano - sat.full_bias_nano) * 1e-6)

                # PDF方法：每历元更新FBN和BN
                current_fbn = sat.full_bias_nano
                current_bn = sat.bias_nano

                # Anything within 1ms is considered same epoch
                if abs(allRxMillis - allRxMillis_p) > NEAR_ZERO:
                    rnx.append(repoch)
                    if repoch.sv <= 4:
                        print("Warning: Number of satellites is less than 4 in this epoch")

                    repoch = RnxEpoch()
                    allRxMillis_p = allRxMillis
                    check_clkdisc = sat.hardware_clock_discountinuity_count
                    if abs(check_clkdisc - check_clkdiscp) > NEAR_ZERO:
                        clock_jump_count += 1
                        time_str = datetime.utcfromtimestamp(sat.time_nano / 1e9).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        logger.debug(
                            f"时钟跳变 #{clock_jump_count} 检测到并处理: 时间={time_str}, 计数器值 {check_clkdiscp} → {check_clkdisc}")
                        check_clkdiscp = check_clkdisc
                        clkdiscp = False
                        epo_bias = count
                        logger.debug(f"更新时钟基准点为历元 #{epo_bias}")
                        # 更新初始FBN和BN基准值
                        initial_fbn = current_fbn
                        initial_bn = current_bn

                # PDF方法：使用当前历元的FBN和BN计算时间
                time = gpstime2ymdhms(sat.time_nano, current_fbn, current_bn)
                repoch.time = time

                # PDF方法：综合质量检查
                total_observations += 1
                
                # 1. 基本状态检查
                available = False
                if sat.sys in [SYS_GPS, SYS_BDS, SYS_QZS]:
                    available = (sat.state & STATE_CODE_LOCK) and (sat.state & STATE_TOW_KNOWN)
                    if round(sat.carrier_frequency_hz / 1e4) == 117645:
                        available = sat.state & STATE_CODE_LOCK
                elif sat.sys == SYS_GLO:
                    available = (sat.state & STATE_GLO_STRING_SYNC) and (sat.state & STATE_GLO_TOD_KNOWN)
                elif sat.sys == SYS_GAL:
                    available = ((sat.state & STATE_GAL_E1C_2ND_CODE_LOCK) or (sat.state & STATE_GAL_E1BC_CODE_LOCK))
                    if round(sat.carrier_frequency_hz / 1e4) == 117645:
                        available = sat.state & STATE_TOW_KNOWN

                if not available:
                    continue  # Reject bad observations with invalid state

                # 2. PDF方法：C/N0阈值检查（暂时禁用）
                # if not check_cn0_threshold(sat):
                #     rejected_cn0 += 1
                #     continue  # Reject low C/N0 measurements

                # 3. PDF方法：多径检测
                if not check_multipath_indicator(sat):
                    rejected_multipath += 1
                    continue  # Reject multipath-affected measurements

                # 4. PDF方法：时间不确定性检查（getReceivedSvTimeUncertaintyNanos != 1 billion）
                # 注意：某些芯片使用10^9纳秒标记不可靠测量（PDF方法）
                if sat.received_sv_time_uncertainty_nano >= 1000000000:  # 1 billion nanoseconds
                    rejected_time_uncertainty += 1
                    continue  # Reject unreliable measurements

                # 5. 不确定性阈值检查（高精度阈值）
                if (sat.pseudorange_rate_uncertainty_meter_per_second > MAXPRRUNCMPS or
                        sat.received_sv_time_uncertainty_nano > MAXTOWUNCNS or
                        sat.accumulated_delta_range_uncertainty_meter > MAXADRUNCNS):
                    rejected_uncertainty += 1
                    continue  # Reject bad observations

                # Find existing satellite or create new one
                existing_sat = None
                for s in repoch.sats:
                    if s.sys == sat.sys and s.prn == sat.svid:
                        existing_sat = s
                        break

                if not existing_sat:
                    existing_sat = RnxSat()
                    existing_sat.sys = sat.sys
                    existing_sat.prn = sat.svid
                    repoch.sats.append(existing_sat)
                    repoch.sv += 1

                frq = find_signal(sat.sys, sat.signal_name)
                if frq == -1:
                    continue

                wavl = CLIGHT / sat.carrier_frequency_hz  # Compute wavelength
                wavl_inv = 1.0 / wavl

                # PDF方法：使用当前历元的FBN和BN计算时间
                time_from_gps_start = sat.time_nano - current_fbn + int(sat.time_offset_nano)

                # Time of reception calculation
                receive_second = 0
                send_second = sat.received_sv_time_nano

                if sat.sys == SYS_GPS:
                    WeekNonano = int((-current_fbn * 1e-9) // 604800)
                    receive_second = time_from_gps_start - WeekNonano * 604800 * 10 ** 9
                elif sat.sys == SYS_GLO:
                    DayNonano = int((-current_fbn) // (86400.00 * 10 ** 9)) * int(86400.00 * 10 ** 9)
                    receive_second = time_from_gps_start - DayNonano + (3 * 3600 - LeapSecond) * 10 ** 9
                elif sat.sys == SYS_BDS:
                    WeekNonano = int((-current_fbn * 1e-9) // 604800)
                    receive_second = time_from_gps_start - WeekNonano * 604800 * 10 ** 9 - 14 * 10 ** 9
                elif sat.sys == SYS_GAL:
                    WeekNonano = int((-current_fbn * 1e-9) // 604800)
                    receive_second = time_from_gps_start - WeekNonano * 604800 * 10 ** 9
                elif sat.sys == SYS_QZS:
                    WeekNonano = int((-current_fbn * 1e-9) // 604800)
                    receive_second = time_from_gps_start - WeekNonano * 604800 * 10 ** 9

                # Time difference between reception and transmission
                pr_second = (receive_second - send_second) * 1e-9 - current_bn * 1e-9

                # Check for week rollover
                if pr_second > 604800 / 2:
                    delS = round(pr_second / 604800) * 604800
                    pr_second -= delS
                    maxBiasSec = 10
                    if pr_second > maxBiasSec:
                        logger.error("周数翻转修正失败")
                    else:
                        logger.debug("检测到周数翻转并已修正")

                if sat.sys in [SYS_GPS, SYS_GAL, SYS_BDS] and pr_second > 604800:
                    pr_second %= 604800
                if sat.sys == SYS_GLO and pr_second > 86400:
                    pr_second %= 86400

                if pr_second > 0.5 or pr_second < 0:
                    continue
                if sat.sys == SYS_GLO and sat.svid > 80:
                    continue  # Delete odd GLONASS numbers > 80

                # PDF方法：计算伪距
                pseudorange = pr_second * CLIGHT

                # PDF方法：计算调整后的ADR
                adjusted_adr = compute_adjusted_adr(sat, current_fbn, current_bn, initial_fbn, initial_bn)

                # PDF方法：相位连续性检查
                if not check_phase_continuity(sat):
                    phase_discontinuities += 1
                    logger.debug(f"相位不连续检测到，卫星 {sat.svid}")
                    existing_sat.first_observation[frq] = True  # 重新初始化

                # PDF方法：计算相位观测
                if existing_sat.first_observation[frq]:
                    # 首次观测：计算IPA
                    existing_sat.ipa[frq] = compute_initial_phase_ambiguity(pseudorange, adjusted_adr, wavl)
                    existing_sat.first_observation[frq] = False
                    logger.debug(f"首次观测卫星 {sat.svid}，IPA = {existing_sat.ipa[frq]}")
                else:
                    # 检查码相差异是否需要重新初始化IPA
                    phase_range = compute_phase_range(existing_sat.ipa[frq], adjusted_adr)
                    phase_meters = phase_range * wavl
                    code_phase_diff = abs(pseudorange - phase_meters)
                    
                    if code_phase_diff > PHASE_CODE_DIFF_THRESHOLD:
                        ipa_reinitializations += 1
                        logger.debug(f"码相差异过大 ({code_phase_diff:.2f}m)，重新初始化IPA")
                        existing_sat.ipa[frq] = compute_initial_phase_ambiguity(pseudorange, adjusted_adr, wavl)

                # PDF方法：计算最终相位观测
                phase_range = compute_phase_range(existing_sat.ipa[frq], adjusted_adr)

                # Store observations
                existing_sat.p[frq] = pseudorange  # Pseudorange
                existing_sat.d[frq] = -sat.pseudorange_rate_meter_per_second * wavl_inv  # Doppler
                existing_sat.l[frq] = phase_range  # Carrier phase in cycles
                existing_sat.s[frq] = sat.cn0_dbhz  # C/N0

                # PDF方法：检查ADR状态
                if sat.accumulated_delta_range_state & GPS_ADR_STATE_UNKNOWN:
                    existing_sat.l[frq] = 0

            # Add the last epoch
            rnx.append(repoch)

            # Build RINEX file
            # 使用用户选择的目录和默认文件名（基于输入文件名）
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            default_output_name = os.path.join(OUTPUT_DIR, input_basename)

            # 替换原有的base_name行
            # 先分离目录和文件名
            dir_name, file_name = os.path.split(default_output_name)
            # 再处理文件名中的扩展名
            file_base = os.path.splitext(file_name)[0]
            # 重新组合路径
            base_name = os.path.join(dir_name, file_base)
            rinex_name = f"{os.path.dirname(base_name)}/new_{os.path.basename(base_name)}.{int(rnx[0].time[0]) - 2000:02d}o"

            with open(rinex_name, 'w') as fpw:
                logger.info(f"开始写入RINEX3.05文件: {rinex_name}")
                print_rnx_header(fpw, first_obs_time, manufacturer, model, xyz, current_time)
                logger.info("RINEX3.05文件头写入完成")

                for epoch in rnx:
                    if epoch.sv > 0:
                        print_rnx_epoch(fpw, epoch)

                logger.info(f"RINEX3.05文件生成完成，共写入 {len(rnx)} 个历元")
                
                # PDF方法：输出统计信息
                logger.info("=== PDF方法处理统计 ===")
                logger.info(f"总观测数: {total_observations}")
                # logger.info(f"因C/N0阈值拒绝: {rejected_cn0} ({rejected_cn0/total_observations*100:.1f}%)")
                logger.info(f"因多径检测拒绝: {rejected_multipath} ({rejected_multipath/total_observations*100:.1f}%)")
                logger.info(f"因不确定性阈值拒绝: {rejected_uncertainty} ({rejected_uncertainty/total_observations*100:.1f}%)")
                logger.info(f"因时间不确定性拒绝: {rejected_time_uncertainty} ({rejected_time_uncertainty/total_observations*100:.1f}%)")
                logger.info(f"相位不连续检测: {phase_discontinuities}")
                logger.info(f"IPA重新初始化: {ipa_reinitializations}")
                logger.info(f"时钟跳变次数: {clock_jump_count}")
                
                logger.info(f"文件 {os.path.basename(input_file)} 处理完成!")
                print("===============================================================================================")

            # 更新总进度
            progress_var.set((i + 1) * progress_per_file)
            progress_window.update()

            # 添加保存成功提示
            status_label.config(text=f"已成功保存: {os.path.basename(rinex_name)}")
            progress_window.update()

        except Exception as e:
            logger.error(f"处理文件 {input_file} 时发生错误: {str(e)}")
            messagebox.showerror("错误", f"处理文件 {input_file} 时出错:\n{str(e)}")
            # 更新状态标签
            status_label.config(text=f"处理失败: {os.path.basename(input_file)}")
            progress_window.update()

    # 所有文件处理完成后
    status_label.config(text="处理完成!")
    progress_window.update()

    # 显示完成提示（使用进度窗口的消息框，避免依赖主窗口）
    messagebox.showinfo("完成", f"已成功处理 {len(input_files)} 个文件", parent=progress_window)

    # 延迟关闭进度窗口，并退出程序
    def close_window():
        progress_window.destroy()
        root.destroy()  # 销毁主窗口，避免残留进程

    progress_window.after(1000, close_window)


def find_signal(sys, sig):
    sys_n = sys_code_function(sys)
    if sys_n == -1:
        return -1

    for i in range(nsignals[sys_n]):
        if signals[sys_n][i] == sig:
            return i
    return -1


def add_signal(sys, sig):
    if find_signal(sys, sig) != -1:
        return

    sys_n = sys_code_function(sys)
    if sys_n == -1:
        return

    signals[sys_n][nsignals[sys_n]] = sig
    nsignals[sys_n] += 1


if __name__ == "__main__":
    main()
