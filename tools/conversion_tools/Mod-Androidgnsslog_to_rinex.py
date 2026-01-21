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
#   Date: Mar 24, 2025:
#       - Support visualization and batch processing, QZSS dual-frequency, Beidou and Galileo triple-frequency signal.
#       - correct some parameter errors.
#       - Update the RINEX version to 3.05.
#
#   Date: Jan 18, 2026:
#       - Added GLONASS SLOT / FRQ # header.
#       - Updated Signal Codes to RINEX 3.05 (e.g. BDS 2I, 1P).
#       - Changed header generation to be dynamically generated.
#       - Fixed LLI bitmask logic.

#   Authors: Zhe Chen from SouthWest Jiaotong University
"""

import math
import os
import tkinter as tk
from datetime import datetime, timezone
from tkinter import filedialog
from tkinter import ttk, messagebox
import logging
import traceback
import re

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gnss_converter.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Constants
CLIGHT = 299792458.0  # Speed of light (m/s)
LeapSecond = 18  # Leap second for 2026

# --- MODIFIED: Relaxed Thresholds ---
MAXPRRUNCMPS = 100.0   # Maximum pseudorange rate (Doppler) uncertainty (Relaxed)
MAXTOWUNCNS = 500      # Maximum Tow uncertainty
MAXADRUNCNS = 10       # Maximum ADR uncertainty (Relaxed)
# ------------------------------------

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
STATE_CODE_LOCK = 1 << 0
STATE_TOW_DECODED = 1 << 3
STATE_MSEC_AMBIGUOUS = 1 << 4

STATE_GLO_STRING_SYNC = 1 << 6
STATE_GLO_TOD_DECODED = 1 << 7

STATE_GAL_E1C_2ND_CODE_LOCK = 1 << 11
STATE_GAL_E1BC_CODE_LOCK = 1 << 10
STATE_GAL_E1B_PAGE_SYNC = 1 << 12

STATE_BDS_D2_BIT_SYNC = 1 << 8
STATE_BDS_D2_SUBFRAME_SYNC = 1 << 9

ADR_STATE_Unknown = 0                   # 0: 载波相位无效或状态未知
ADR_STATE_VALID = 1 << 0                # 1: 载波相位有效
ADR_STATE_RESET = 1 << 1                # 2: 载波相位重置
ADR_STATE_CYCLE_SLIP = 1 << 2           # 4: 检测到周跳
ADR_STATE_HALF_CYCLE_RESOLVED = 1 << 3  # 8: 半周模糊度已解算
ADR_STATE_HALF_CYCLE_REPORTED = 1 << 4  # 16: 存在半周模糊度

NEAR_ZERO = 0.0001  # Threshold to judge if a float equals 0


class GnssSat:
    def __init__(self):
        # 基础时间属性
        self.Raw = ""  # 原始数据标识
        self.utc_time_millis = 0  # UTC时间戳(毫秒)
        self.time_nano = 0  # 系统时间(纳秒)
        self.leap_second = 0  # 闰秒数
        self.time_uncertainty_nano = 0.0  # 时间不确定度(纳秒)
        self.full_bias_nano = 0  # 时钟偏差(纳秒)
        self.bias_nano = 0.0  # 时钟偏差修正(纳秒)
        self.bias_uncertainty_nano = 0.0  # 时钟偏差不确定度(纳秒)
        self.drift_nano_per_second = 0.0  # 时钟漂移(纳秒/秒)
        self.drift_uncertainty_nano_per_second = 0.0  # 时钟漂移不确定度(纳秒/秒)
        self.hardware_clock_discontinuity_count = 0  # 硬件时钟不连续计数

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
        self.Raw = parts[0]
        self.utc_time_millis = int(parts[1])
        self.time_nano = int(parts[2])
        self.leap_second = int(parts[3])
        self.time_uncertainty_nano = float(parts[4])
        self.full_bias_nano = int(parts[5])
        self.bias_nano = float(parts[6])
        self.bias_uncertainty_nano = float(parts[7])
        self.drift_nano_per_second = float(parts[8])
        self.drift_uncertainty_nano_per_second = float(parts[9])
        self.hardware_clock_discontinuity_count = int(parts[10])

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
        """输出载波频率和信号名称--调试"""
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
        # 对应每个频点的 LLI 值（0=无，1=半周，2=周跳，可按位或）
        self.l_lli = [0] * MAX_FRQ


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
    return svid - 192  # Default mapping


def get_smart_signal_code(sys, carrier_freq_hz, android_code_type):
    """
    根据频率确定波段号，直接拼装 Android CodeType
    """
    # 1. 预处理
    freq_mhz = round(carrier_freq_hz / 1e5) / 10.0
    raw_code = android_code_type if android_code_type else ""

    # 2. 定义波段号 (Band ID) 和 默认码 (Default)
    band_id = ""
    default_attr = ""

    # ---  BDS B1I (1561.098 MHz) ---
    # RINEX 3.04+ 规定 B1I 为 B2I
    if sys == SYS_BDS and abs(freq_mhz - 1561.1) < 1.0:
        band_id = "2"
        default_attr = "I"

    # --- L1 / E1 / B1C / G1 ---
    elif abs(freq_mhz - 1575.4) < 1.0 or (sys == SYS_GLO and 1590 < freq_mhz < 1615):
        band_id = "1"
        if sys == SYS_BDS:
            default_attr = "P"
        else:
            default_attr = "C"

    # --- L5 / E5a / B2a ---
    elif abs(freq_mhz - 1176.4) < 1.0:
        band_id = "5"
        if sys == SYS_BDS:
            default_attr = "P"
        else:
            default_attr = "Q"

    # --- L2 / G2 ---
    elif abs(freq_mhz - 1227.6) < 1.0 or (sys == SYS_GLO and 1230 < freq_mhz < 1260):
        band_id = "2"
        default_attr = "C"  # G2 默认为 2C (或者 2P，视接收机而定，消费级多为2C)

    # --- E5b / B2b ---
    elif abs(freq_mhz - 1207.1) < 1.0:
        band_id = "7"
        if sys == SYS_BDS:
            default_attr = "I"
        else:
            default_attr = "Q"

    # --- B3I ---
    elif abs(freq_mhz - 1268.5) < 1.0:
        band_id = "6"
        default_attr = "I"  # 6I

    if not band_id:
        return None

    # 核心逻辑：直接使用code_type
    final_attr = raw_code if raw_code else default_attr

    # 3. BDS B2a 修正
    if sys == SYS_BDS and band_id == "5" and final_attr == "Q":
        final_attr = "P"  # 强制修正为 5P

    return f"{band_id}{final_attr}"


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
    pattern = re.compile(r'Manufacturer:\s*(\S+).*Model:\s*(\S+)', re.IGNORECASE)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return m.group(1), m.group(2)
    return '', ''


def parse_xyz_from_raw(raw_lines):
    """从原始数据中提取第一个Fix行的经纬度和高度，并转换为XYZ坐标"""
    pattern = re.compile(r'^Fix,\s*([A-Z]+),\s*([\d.\-]+),\s*([\d.\-]+),\s*([\d.\-]+)')
    for line in raw_lines:
        m = pattern.match(line.strip())
        if m:
            try:
                lat = float(m.group(2))
                lon = float(m.group(3))
                h = float(m.group(4))
                return latlonh_to_xyz(lat, lon, h)
            except Exception:
                continue
    return None


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

            # Carrier phase (L) with LLI formatting: 14-char value + 1-char LLI + 1-char SSI(blank)
            if sat.l[i]:
                lli_char = str(sat.l_lli[i]) if hasattr(sat, 'l_lli') and sat.l_lli[i] != 0 else ' '
                fp.write(f"{sat.l[i]:14.3f}{lli_char} ")
               #  fp.write("{:16.3f}".format(sat.l[i]))
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


def print_rnx_header(fp, first_obs_time, manufacturer, model, xyz, current_time, glonass_map):
    fp.write(RNX_VER + "\n")
    run_by = manufacturer if manufacturer else "unknown"
    date_str = current_time.strftime("%Y%m%d %H%M%S")
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

    # --- MODIFIED: Dynamic Signal Types Generation (Fixed Alignment) ---
    sys_chars = ['G', 'R', 'E', 'C', 'J']
    for i in range(5):  # GPS, GLO, GAL, BDS, QZSS
        if nsignals[i] == 0:
            continue

        # Build list of "C1C", "L1C", etc.
        descriptors = []
        for j in range(nsignals[i]):
            code = signals[i][j]  # e.g., "1C", "2I"
            descriptors.extend([f"C{code}", f"L{code}", f"D{code}", f"S{code}"])

        num_obs = len(descriptors)
        # RINEX limit: 13 descriptors per line

        # First line
        current_descs = descriptors[:13]
        desc_str = "".join([f"{d:<4s}" for d in current_descs])
        # Format: Sys(1) + 2space + Count(3) + 1space = 7 chars prefix
        fp.write(f"{sys_chars[i]}  {num_obs:3d} {desc_str:<52s} SYS / # / OBS TYPES \n")

        # Subsequent lines
        descriptors = descriptors[13:]
        while descriptors:
            current_descs = descriptors[:13]
            desc_str = "".join([f"{d:<4s}" for d in current_descs])
            # FIX: Changed from 6 spaces to 7 spaces to align with the first line
            fp.write(f"       {desc_str:<52s} SYS / # / OBS TYPES \n")
            descriptors = descriptors[13:]
    # -----------------------------------------------------------------

    # --- MODIFIED: GLONASS SLOT / FRQ # Output (Fixed Alignment) ---
    if glonass_map:
        sorted_prns = sorted(glonass_map.keys())
        num_glo = len(sorted_prns)

        # FIX: Changed to 3d + 1 space (Total 4 chars width)
        fp.write(f"{num_glo:3d} ")

        count = 0
        for prn in sorted_prns:
            frq = glonass_map[prn]
            fp.write(f"R{prn:02d} {frq:2d} ")
            count += 1
            if count == 8:
                fp.write(" " * (60 - fp.tell()) + "GLONASS SLOT / FRQ #\n")
                if prn != sorted_prns[-1]:
                    # FIX: Changed to 4 spaces to align with the first line prefix
                    fp.write("    ")
                count = 0

        if count > 0:
            fp.write(" " * (60 - (count * 7 + 4)) + "GLONASS SLOT / FRQ #\n")
    # ---------------------------------------------------------------

    # First observation time
    fp.write("  {:04d}    {:02d}    {:02d}    {:02d}    {:02d}   {:10.7f}     GPS         TIME OF FIRST OBS\n".format(
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
        filetypes=(("文本文件", "*.txt"), ("所有文件", "*.*"))
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

    # 计算总进度
    total_files = len(input_files)
    progress_per_file = 100 / total_files

    # 处理每个选择的文件
    for i, input_file in enumerate(input_files):
        try:
            logger.info(f"开始处理...")
            logger.info(f"读取原始文件: {os.path.basename(input_file)}")

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

            # --- MODIFIED: GLONASS Freq Map ---
            glonass_freq_map = {}
            # ----------------------------------

            # 解析当前文件
            with open(input_file, 'r', encoding='utf-8') as fp:
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

                    # 更新文件内处理进度
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

                # --- MODIFIED: GLONASS k-value calculation ---
                if sat.constellation_type == 3: # GLO
                    freq = sat.carrier_frequency_hz
                    k = None
                    if freq > 1.5e9: # L1 range
                        k = int(round((freq - 1602.0e6) / 0.5625e6))
                    elif freq > 1.2e9: # L2 range
                        k = int(round((freq - 1246.0e6) / 0.4375e6))
                    if k is not None:
                        glonass_freq_map[sat.svid] = k
                # ---------------------------------------------

                # # --- MODIFIED: Signal Identification Logic (RINEX 3.05) ---
                # # Determine system and signal type
                # if sat.constellation_type == 1:  # GPS
                #     # Use rounded freq logic for robust matching
                #     freq_mhz = round(sat.carrier_frequency_hz / 1e5) / 10.0
                #     if abs(freq_mhz - 1575.4) < 1.0:
                #         sat.sys = SYS_GPS
                #         sat.signal_name = "1C" # was L1C
                #         add_signal(SYS_GPS, sat.signal_name)
                #     elif abs(freq_mhz - 1176.4) < 1.0:
                #         sat.sys = SYS_GPS
                #         sat.signal_name = "5Q" # was L5Q
                #         add_signal(SYS_GPS, sat.signal_name)
                # elif sat.constellation_type == 3:  # GLO
                #     freq_mhz = sat.carrier_frequency_hz / 1e6
                #     if 1590 < freq_mhz < 1615:
                #         sat.sys = SYS_GLO
                #         sat.signal_name = "1C" # was L1C
                #         add_signal(SYS_GLO, sat.signal_name)
                #     # Add G2 if needed:
                #     elif 1230 < freq_mhz < 1260:
                #         sat.sys = SYS_GLO
                #         sat.signal_name = "2C"
                #         add_signal(SYS_GLO, sat.signal_name)
                # elif sat.constellation_type == 5:  # BDS
                #     freq_mhz = round(sat.carrier_frequency_hz / 1e5) / 10.0
                #     if abs(freq_mhz - 1561.1) < 1.0: # B1I (1561.098)
                #         sat.sys = SYS_BDS
                #         sat.signal_name = "2I" # MODIFIED: B2I -> 2I
                #         add_signal(SYS_BDS, sat.signal_name)
                #     elif abs(freq_mhz - 1575.4) < 1.0: # B1C
                #         sat.sys = SYS_BDS
                #         sat.signal_name = "1P" # MODIFIED: B1C -> 1P
                #         add_signal(SYS_BDS, sat.signal_name)
                #     elif abs(freq_mhz - 1176.4) < 1.0: # B2a
                #         sat.sys = SYS_BDS
                #         sat.signal_name = "5P" # MODIFIED: B2a -> 5P
                #         add_signal(SYS_BDS, sat.signal_name)
                #     elif abs(freq_mhz - 1207.1) < 1.0: # B2b
                #         sat.sys = SYS_BDS
                #         sat.signal_name = "7I"
                #         add_signal(SYS_BDS, sat.signal_name)
                #     elif abs(freq_mhz - 1268.5) < 1.0: # B3I
                #         sat.sys = SYS_BDS
                #         sat.signal_name = "6I"
                #         add_signal(SYS_BDS, sat.signal_name)
                # elif sat.constellation_type == 6:  # GAL
                #     freq_mhz = round(sat.carrier_frequency_hz / 1e5) / 10.0
                #     if abs(freq_mhz - 1575.4) < 1.0:
                #         sat.sys = SYS_GAL
                #         sat.signal_name = "1C" # E1C
                #         add_signal(SYS_GAL, sat.signal_name)
                #     elif abs(freq_mhz - 1176.4) < 1.0:
                #         sat.sys = SYS_GAL
                #         sat.signal_name = "5Q" # E5Q
                #         add_signal(SYS_GAL, sat.signal_name)
                #     elif abs(freq_mhz - 1207.1) < 1.0:
                #         sat.sys = SYS_GAL
                #         sat.signal_name = "7Q" # E7Q
                #         add_signal(SYS_GAL, sat.signal_name)
                # elif sat.constellation_type == 4:  # QZSS
                #     freq_mhz = round(sat.carrier_frequency_hz / 1e5) / 10.0
                #     if abs(freq_mhz - 1575.4) < 1.0:
                #         sat.sys = SYS_QZS
                #         sat.signal_name = "1C"
                #         add_signal(SYS_QZS, sat.signal_name)
                #     elif abs(freq_mhz - 1176.4) < 1.0:
                #         sat.sys = SYS_QZS
                #         sat.signal_name = "5Q"
                #         add_signal(SYS_QZS, sat.signal_name)
                # # --------------------------------------------------------

                # ================== 修改开始 ==================
                # 简化版调用：自动拼装频率波段和CodeType
                detected_code = get_smart_signal_code(
                    sat.constellation_type,
                    sat.carrier_frequency_hz,
                    sat.code_type
                )

                if detected_code:
                    sat.sys = sat.constellation_type
                    sat.signal_name = detected_code
                    add_signal(sat.sys, sat.signal_name)
                # ================== 修改结束 ==================

            # Compute full cycle time of measurement, in milliseconds
            allRxMillis_p = int((epochs[0].obs.time_nano - epochs[0].obs.full_bias_nano) * 1e-6)
            check_clkdiscp = epochs[0].obs.hardware_clock_discontinuity_count
            clkdiscp = True

            epo_bias = 0
            clock_jump_count = 0  # 新增：记录跳变次数
            count = -1
            rnx = []
            repoch = RnxEpoch()

            for epoch in epochs:
                count += 1
                sat = epoch.obs
                allRxMillis = int((sat.time_nano - sat.full_bias_nano) * 1e-6)

                # Anything within 1ms is considered same epoch
                if abs(allRxMillis - allRxMillis_p) > NEAR_ZERO:
                    rnx.append(repoch)
                    if repoch.sv <= 4:
                        print("Warning: Number of satellites is less than 4 in this epoch")

                    repoch = RnxEpoch()
                    allRxMillis_p = allRxMillis
                    check_clkdisc = sat.hardware_clock_discontinuity_count
                    if abs(check_clkdisc - check_clkdiscp) > NEAR_ZERO:
                        clock_jump_count += 1
                        time_str = datetime.fromtimestamp(sat.time_nano / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        logger.debug(
                            f"时钟跳变 #{clock_jump_count} 检测到并处理: 时间={time_str}, 计数器值 {check_clkdiscp} → {check_clkdisc}")
                        check_clkdiscp = check_clkdisc
                        clkdiscp = False
                        epo_bias = count
                        logger.debug(f"更新时钟基准点为历元 #{epo_bias}")

                time = gpstime2ymdhms(sat.time_nano, epochs[epo_bias].obs.full_bias_nano,
                                      epochs[epo_bias].obs.bias_nano)
                repoch.time = time

                # Check observation availability
                available = False

                # 1. MSEC AMBIGUOUS
                if (sat.state & STATE_MSEC_AMBIGUOUS) != 0:
                    available = False
                else:
                    # 2. TOW/TOD
                    tow_decoded = False
                    if sat.sys == SYS_GLO:
                        tow_decoded = (sat.state & STATE_GLO_TOD_DECODED) != 0
                    else:
                        tow_decoded = (sat.state & STATE_TOW_DECODED) != 0

                    if not tow_decoded:
                        available = False
                    else:
                        # 3. CODE LOCK
                        if sat.sys in [SYS_GPS, SYS_BDS, SYS_QZS]:
                            available = (sat.state & STATE_CODE_LOCK) != 0
                        elif sat.sys == SYS_GLO:
                            available = (sat.state & STATE_GLO_STRING_SYNC) != 0
                        elif sat.sys == SYS_GAL:
                            available = ((sat.state & STATE_GAL_E1BC_CODE_LOCK) != 0) or \
                                        ((sat.state & STATE_GAL_E1C_2ND_CODE_LOCK) != 0)

                if not available:
                    continue  # Reject bad observations with invalid state

                if (sat.pseudorange_rate_uncertainty_meter_per_second > MAXPRRUNCMPS or
                        sat.received_sv_time_uncertainty_nano > MAXTOWUNCNS or
                        sat.accumulated_delta_range_uncertainty_meter > MAXADRUNCNS):
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

                # 除零保护
                if sat.carrier_frequency_hz == 0:
                    continue

                # --- MODIFIED: Use Nominal Frequency ---
                nominal_freq = sat.carrier_frequency_hz  # 默认回退值

                # 1. GLONASS:
                if sat.sys == SYS_GLO:
                    k = glonass_freq_map.get(sat.svid, 0)
                    if sat.carrier_frequency_hz > 1.5e9:  # G1
                        nominal_freq = 1602.0e6 + k * 0.5625e6
                    else:  # G2
                        nominal_freq = 1246.0e6 + k * 0.4375e6

                # 2. BDS B1I:1561.098 MHz
                elif sat.sys == SYS_BDS and "2I" in sat.signal_name:
                    nominal_freq = 1561.098e6

                # 3. Others
                else:
                    nominal_freq = round(sat.carrier_frequency_hz / 1000.0) * 1000.0

                wavl = CLIGHT / nominal_freq
                wavl_inv = 1.0 / wavl
                # ----------------------------------------

                time_from_gps_start = sat.time_nano - epochs[epo_bias].obs.full_bias_nano + int(sat.time_offset_nano)

                # Time of reception calculation
                receive_second = 0
                send_second = sat.received_sv_time_nano

                if sat.sys == SYS_GPS:
                    WeekNonano = int((-sat.full_bias_nano * 1e-9) // 604800)
                    receive_second = time_from_gps_start - WeekNonano * 604800 * 10 ** 9
                elif sat.sys == SYS_GLO:
                    DayNonano = int((-sat.full_bias_nano) // (86400.00 * 10 ** 9)) * int(86400.00 * 10 ** 9)
                    receive_second = time_from_gps_start - DayNonano + (3 * 3600 - LeapSecond) * 10 ** 9
                elif sat.sys == SYS_BDS:
                    WeekNonano = int((-sat.full_bias_nano * 1e-9) // 604800)
                    receive_second = time_from_gps_start - WeekNonano * 604800 * 10 ** 9 - 14 * 10 ** 9
                elif sat.sys == SYS_GAL:
                    WeekNonano = int((-sat.full_bias_nano * 1e-9) // 604800)
                    receive_second = time_from_gps_start - WeekNonano * 604800 * 10 ** 9
                elif sat.sys == SYS_QZS:
                    WeekNonano = int((-sat.full_bias_nano * 1e-9) // 604800)
                    receive_second = time_from_gps_start - WeekNonano * 604800 * 10 ** 9

                # Time difference between reception and transmission
                pr_second = (receive_second - send_second) * 1e-9 - epochs[epo_bias].obs.bias_nano * 1e-9

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

                # Store observations
                existing_sat.p[frq] = pr_second * CLIGHT  # Pseudorange
                existing_sat.d[frq] = -sat.pseudorange_rate_meter_per_second * wavl_inv  # Doppler

                # --- KEEP ORIGINAL: Phase Calculation ---
                # But wrap in valid check
                adr_state = sat.accumulated_delta_range_state
                if adr_state & ADR_STATE_VALID:
                    existing_sat.l[frq] = sat.accumulated_delta_range_meter * wavl_inv  # Carrier phase
                else:
                    existing_sat.l[frq] = 0.0
                # ----------------------------------------

                # --- MODIFIED: Improved LLI Logic ---
                lli_val = 0
                # Bit 0: Loss of Lock (Reset or Cycle Slip)
                if (adr_state & ADR_STATE_RESET) or (adr_state & ADR_STATE_CYCLE_SLIP):
                    lli_val |= 1
                # Bit 1: Half Cycle Ambiguity reported but not resolved
                if (adr_state & ADR_STATE_HALF_CYCLE_REPORTED) and not (adr_state & ADR_STATE_HALF_CYCLE_RESOLVED):
                    lli_val |= 2

                existing_sat.l_lli[frq] = lli_val
                # ------------------------------------

                # 信噪比
                existing_sat.s[frq] = sat.cn0_dbhz  # C/N0

            # Add the last epoch
            rnx.append(repoch)

            # Build RINEX file
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            default_output_name = os.path.join(OUTPUT_DIR, input_basename)
            dir_name, file_name = os.path.split(default_output_name)
            file_base = os.path.splitext(file_name)[0]
            base_name = os.path.join(dir_name, file_base)
            # rnx非空判断
            if not rnx or not rnx[0].time:
                logger.error(f"RINEX数据为空，跳过文件: {input_file}")
                continue
            rinex_name = os.path.join(os.path.dirname(base_name), f"new_{os.path.basename(base_name)}.{int(rnx[0].time[0]) - 2000:02d}o")
            with open(rinex_name, 'w', encoding='utf-8') as fpw:
                logger.info(f"开始写入RINEX3.05文件: {rinex_name}")
                # --- MODIFIED: Pass glonass_freq_map ---
                print_rnx_header(fpw, first_obs_time, manufacturer, model, xyz, current_time, glonass_freq_map)
                # ---------------------------------------
                logger.info("RINEX3.05文件头写入完成")

                for epoch in rnx:
                    if epoch.sv > 0:
                        print_rnx_epoch(fpw, epoch)

                logger.info(f"RINEX3.05文件生成完成，共写入 {len(rnx)} 个历元")
                logger.info(f"文件 {os.path.basename(input_file)} 处理完成!")


            # 更新总进度
            progress_var.set((i + 1) * progress_per_file)
            progress_window.update()

            # 添加保存成功提示
            status_label.config(text=f"已成功保存: {os.path.basename(rinex_name)}")
            progress_window.update()

        except Exception as e:
            logger.error(f"处理文件 {input_file} 时发生错误: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("错误", f"处理文件 {input_file} 时出错:\n{str(e)}")
            # 更新状态标签
            status_label.config(text=f"处理失败: {os.path.basename(input_file)}")
            progress_window.update()

    # 所有文件处理完成后
    status_label.config(text="处理完成!")
    progress_window.update()

    # 显示完成提示
    messagebox.showinfo("完成", f"已成功处理 {len(input_files)} 个文件", parent=progress_window)

    # 延迟关闭进度窗口，并退出程序
    def close_window():
        progress_window.destroy()
        root.destroy()  # 销毁主窗口，避免残留进程

    progress_window.after(1000, lambda: close_window())


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