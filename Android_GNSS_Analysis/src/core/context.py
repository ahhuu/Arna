from typing import Dict, Any, Optional
from .config import GNSS_FREQUENCIES, GLONASS_K_MAP, SPEED_OF_LIGHT


class AnalysisContext:
    """Simple container for sharing data between modules."""

    def __init__(self):
        self.observations_meters: Dict[str, Any] = {}
        self.receiver_observations: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.input_path: Optional[str] = None
        self.output_dir: Optional[str] = None

        # 全局配置缓存
        self.frequencies = GNSS_FREQUENCIES.copy()
        self.glonass_k_map = GLONASS_K_MAP.copy()
        # 存储计算出的具体卫星具体频率的波长 {system: {freq: wavelength}}
        self.wavelengths: Dict[str, Dict[str, float]] = {} 
        
        # 初始化基础波长
        self._init_wavelengths()

    def _init_wavelengths(self):
        for sys, freqs in self.frequencies.items():
            self.wavelengths[sys] = {}
            for fname, fhz in freqs.items():
                if fhz > 0:
                    self.wavelengths[sys][fname] = SPEED_OF_LIGHT / fhz

    def set_input_path(self, path: str) -> None:
        self.input_path = path

    def set_output_dir(self, path: str) -> None:
        self.output_dir = path

    def clear(self) -> None:
        self.observations_meters.clear()
        self.receiver_observations.clear()
        self.results.clear()
