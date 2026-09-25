from copy import deepcopy
from pathlib import Path
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
        self.nominal_frequencies = deepcopy(GNSS_FREQUENCIES)
        self.frequencies = deepcopy(self.nominal_frequencies)
        self.current_frequencies: Dict[str, Dict[str, float]] = {}
        self.glonass_k_map = GLONASS_K_MAP.copy()
        # 存储计算出的具体卫星具体频率的波长 {system: {freq: wavelength}}
        self.wavelengths: Dict[str, Dict[str, float]] = {} 
        self.current_wavelengths: Dict[str, Dict[str, float]] = {}
        self.phone_source_id: Optional[Dict[str, Any]] = None
        self.receiver_source_id: Optional[Dict[str, Any]] = None
        self.result_sources: Dict[str, Any] = {}
        
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

    @staticmethod
    def build_source_id(path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return a cheap identity for invalidating results when an input changes."""
        if not path:
            return None
        source = Path(path).resolve()
        try:
            stat = source.stat()
            return {
                'path': str(source),
                'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns,
            }
        except OSError:
            return {'path': str(source), 'size': None, 'mtime_ns': None}

    def _invalidate(self, keys, prefixes=()) -> None:
        for key in list(self.results):
            if key in keys or any(key.startswith(prefix) for prefix in prefixes):
                self.results.pop(key, None)
                self.result_sources.pop(key, None)

    def invalidate_phone_dependent_results(self) -> None:
        self._invalidate(
            {
                'observable_derivatives', 'code_phase_differences',
                'phase_prediction_errors', 'epoch_double_diffs',
                'ionofree_cmc', 'isb_analysis', 'inter_freq_bias',
                'cycle_slip_detection',
            },
            prefixes=('pseudorange_multipath_', 'ionofree_cmc_'),
        )

    def invalidate_receiver_dependent_results(self) -> None:
        self._invalidate({'receiver_cmc', 'isb_analysis'})

    def replace_phone_data(self, observations: Dict[str, Any], epochs,
                           source_path: Optional[str] = None,
                           frequencies: Optional[Dict[str, Dict[str, float]]] = None,
                           wavelengths: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self.invalidate_phone_dependent_results()
        self.observations_meters = observations
        self.results['epochs'] = epochs
        self.phone_source_id = self.build_source_id(source_path)
        self.input_path = source_path
        self.current_frequencies = deepcopy(frequencies or {})
        self.current_wavelengths = deepcopy(wavelengths or {})
        self.results['frequencies'] = self.current_frequencies
        self.results['wavelengths'] = self.current_wavelengths

    def replace_receiver_data(self, observations: Dict[str, Any], epochs,
                              source_path: Optional[str] = None,
                              frequencies: Optional[Dict[str, Dict[str, float]]] = None,
                              wavelengths: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self.invalidate_receiver_dependent_results()
        self.receiver_observations = observations
        self.results['receiver_epochs'] = epochs
        self.receiver_source_id = self.build_source_id(source_path)
        self.results['receiver_frequencies'] = deepcopy(frequencies or {})
        self.results['receiver_wavelengths'] = deepcopy(wavelengths or {})

    def cache_result(self, key: str, value: Any, depends_on: str = 'phone') -> Any:
        self.results[key] = value
        self.result_sources[key] = (
            self.receiver_source_id if depends_on == 'receiver'
            else (self.phone_source_id, self.receiver_source_id) if depends_on == 'both'
            else self.phone_source_id
        )
        return value

    def is_result_current(self, key: str, depends_on: str = 'phone') -> bool:
        if key not in self.results:
            return False
        expected = (
            self.receiver_source_id if depends_on == 'receiver'
            else (self.phone_source_id, self.receiver_source_id) if depends_on == 'both'
            else self.phone_source_id
        )
        return self.result_sources.get(key) == expected

    def clear(self) -> None:
        self.observations_meters.clear()
        self.receiver_observations.clear()
        self.results.clear()
        self.result_sources.clear()
        self.current_frequencies.clear()
        self.current_wavelengths.clear()
        self.phone_source_id = None
        self.receiver_source_id = None
