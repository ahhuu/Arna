"""Single source of truth for selected and all-chart visualization exports."""

import json
import os
from itertools import combinations
from pathlib import Path

from src.processing.calculator import MetricCalculator
from src.processing.cycle_slip_detector import CycleSlipDetector
from src.processing.inter_freq_bias import InterFrequencyBiasAnalyzer
from src.reporting.cycle_slip_logger import CycleSlipLogger


class VisualizationBatchExporter:
    FOLDERS = {
        'raw_observations': 'Raw_observations',
        'sat_freq_sequence': 'Satellite_frequency_sequence',
        'satellite_count': 'Satellite_count',
        'cnr_analysis': 'CNR_Analysis',
        'data_integrity': 'Data_Integrity',
        'observation_noise': 'Observation_Noise',
        'doppler_quality': 'Doppler_Quality',
        'pseudorange_multipath_overview': 'Pseudorange_Multipath_Constellation',
        'pseudorange_multipath': 'Pseudorange_Multipath_Satellite',
        'code_phase_diff_raw': 'Code_phase_diffs_raw',
        'code_phase_diffs': 'Code_phase_diffs',
        'derivatives': 'Derivatives',
        'phase_pred_errors': 'Prediction_errors',
        'double_differences': 'Double_differences',
        'cycle_slip_detection': 'Cycle_slips',
        'receiver_cmc': 'Receiver_CMC',
        'ionofree_cmc': 'Ionofree_CMC',
        'isb_analysis': 'ISB_analysis',
        'inter_freq_bias': 'Inter_freq_bias',
    }
    ALL_CHART_TYPES = tuple(FOLDERS)
    MANIFEST = '.arna_export_manifest.json'

    def __init__(self, context, plotter):
        self.context = context
        self.plotter = plotter
        self.calculator = MetricCalculator()

    @property
    def frequencies(self):
        return self.context.current_frequencies or self.context.frequencies

    @property
    def wavelengths(self):
        return self.context.current_wavelengths or self.context.wavelengths

    def _inputs(self):
        return {
            'observations_meters': self.context.observations_meters,
            'epochs': self.context.results.get('epochs', []),
            'frequencies': self.frequencies,
            'wavelengths': self.wavelengths,
        }

    def _cached(self, key, factory, depends_on='phone'):
        if not self.context.is_result_current(key, depends_on=depends_on):
            self.context.cache_result(key, factory(), depends_on=depends_on)
        return self.context.results.get(key, {})

    @staticmethod
    def _snapshot(directory):
        return {
            str(path.relative_to(directory)): (path.stat().st_mtime_ns, path.stat().st_size)
            for path in directory.rglob('*') if path.is_file()
        }

    def _finalize(self, directory, chart_type, before, parameters):
        after = self._snapshot(directory)
        manifest_path = directory / self.MANIFEST
        previous = {}
        if manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                previous = {}
        # Files overwritten in-place are part of this run even if their names existed.
        generated_names = sorted(name for name, signature in after.items()
                                 if name != self.MANIFEST and before.get(name) != signature)
        removed = []
        for name in previous.get('generated_files', []):
            if name in generated_names:
                continue
            target = (directory / name).resolve()
            try:
                target.relative_to(directory.resolve())
            except ValueError:
                continue
            if target.is_file():
                target.unlink()
                removed.append(name)
        manifest = {
            'chart_type': chart_type,
            'phone_source': self.context.phone_source_id,
            'receiver_source': self.context.receiver_source_id,
            'parameters': parameters,
            'generated_files': generated_names,
            'removed_stale_files': sorted(removed),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        return generated_names

    def export_chart(self, chart_type, project_dir, parameters=None):
        parameters = dict(parameters or {})
        target = Path(project_dir) / self.FOLDERS[chart_type]
        target.mkdir(parents=True, exist_ok=True)
        before = self._snapshot(target)
        result = {'chart_type': chart_type, 'count': 0, 'skipped': [], 'errors': [], 'files': []}
        try:
            count = self._run(chart_type, target, parameters, result)
            result['count'] = count
            result['files'] = self._finalize(target, chart_type, before, parameters)
        except Exception as exc:
            result['errors'].append({'error': str(exc)})
        return result

    def export_all(self, project_dir, parameters=None):
        results = [self.export_chart(kind, project_dir, parameters) for kind in self.ALL_CHART_TYPES]
        errors = [item for result in results for item in result['errors']]
        error_path = Path(project_dir) / 'visualization_export_errors.json'
        error_path.write_text(json.dumps({'results': results}, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        return {
            'results': results,
            'success_count': sum(item['count'] for item in results),
            'skipped_count': sum(len(item['skipped']) for item in results),
            'failure_count': len(errors),
            'error_log': str(error_path),
        }

    def _run(self, chart_type, target, p, result):
        obs = self.context.observations_meters
        filters = {
            'system_filters': p.get('system_filters'),
            'sat_filters': p.get('sat_filters'),
            'freq_filters': p.get('freq_filters'),
        }
        if chart_type == 'raw_observations':
            return self._per_sat_freq(target, result, lambda sat, _freq: self.plotter.plot_raw_observations(
                {'observations_meters': obs}, sat, save=True, output_dir=str(target)), per_frequency=False)
        if chart_type == 'sat_freq_sequence':
            self.plotter.plot_satellite_frequency_sequence({'observations_meters': obs}, save=True, output_dir=str(target), **filters); return 1
        if chart_type == 'satellite_count':
            self.plotter.plot_satellite_count({'observations_meters': obs}, save=True, output_dir=str(target), **filters); return 1
        if chart_type == 'cnr_analysis':
            self.plotter.plot_cnr_analysis({'observations_meters': obs}, save=True, output_dir=str(target), **filters); return 1
        if chart_type == 'data_integrity':
            self.plotter.plot_data_integrity({'observations_meters': obs}, save=True, output_dir=str(target), **filters); return 1
        if chart_type == 'observation_noise':
            self.plotter.plot_observation_noise({'observations_meters': obs}, save=True, output_dir=str(target), **filters); return 1
        if chart_type == 'doppler_quality':
            self.plotter.plot_doppler_quality({'observations_meters': obs}, save=True, output_dir=str(target), **filters); return 1
        if chart_type == 'pseudorange_multipath_overview':
            self.plotter.plot_pseudorange_multipath_overview(
                {'observations_meters': obs}, save=True, output_dir=str(target),
                smoothing_window=int(p.get('smoothing_window', 5)), **filters); return 1
        if chart_type == 'pseudorange_multipath':
            count = 0
            for sat, sat_data in sorted(obs.items()):
                for pair in combinations(sat_data, 2):
                    try:
                        value = self.calculator.calculate_pseudorange_multipath(
                            {'observations_meters': {sat: sat_data}}, freq_pair=pair,
                            smoothing_window=int(p.get('smoothing_window', 5)))
                        if sat in value:
                            self.plotter.plot_pseudorange_multipath(value, sat, freq_pair=pair, save=True, output_dir=str(target)); count += 1
                    except Exception as exc:
                        result['errors'].append({'sat_id': sat, 'freq_pair': pair, 'error': str(exc)})
            return count
        if chart_type == 'cycle_slip_detection':
            pair = p.get('freq_pair')
            detector = CycleSlipDetector(
                use_custom_threshold=bool(p.get('use_custom_threshold')),
                custom_mw_threshold=p.get('mw_threshold'),
                custom_gf_threshold=p.get('gf_threshold'))
            values = detector.detect_cycle_slips(obs, self.frequencies, self.wavelengths, freq_pair=pair)
            logger = CycleSlipLogger(output_dir=str(target)); logger.save_cycle_slip_log(values); logger.save_cycle_slip_csv(values)
            count = 0
            for sat, value in values.items():
                try:
                    self.plotter.plot_cycle_slip_analysis(value, sat, save=True, output_dir=str(target)); count += 1
                except Exception as exc:
                    result['errors'].append({'sat_id': sat, 'error': str(exc)})
            return count
        if chart_type == 'receiver_cmc':
            if not self.context.receiver_observations:
                result['skipped'].append('receiver RINEX not loaded'); return 0
            values = self._cached('receiver_cmc', lambda: self.calculator.calculate_receiver_cmc({
                'receiver_observations': self.context.receiver_observations,
                'receiver_frequencies': self.context.results.get('receiver_frequencies', {}),
                'receiver_wavelengths': self.context.results.get('receiver_wavelengths', {}),
            }), depends_on='receiver')
            if not values:
                result['skipped'].append('no valid receiver CMC observations'); return 0
            self.plotter.plot_receiver_cmc(values, save=True, output_dir=str(target)); return 1
        if chart_type == 'isb_analysis':
            if not obs or not self.context.receiver_observations:
                result['skipped'].append('phone and receiver RINEX are both required'); return 0
            values = self._cached('isb_analysis', lambda: self.calculator.calculate_isb({
                'observations_meters': obs, 'receiver_observations': self.context.receiver_observations,
                'epochs': self.context.results.get('epochs', []),
            }), depends_on='both')
            if values.get('error'):
                result['skipped'].append(values['error']); return 0
            self.plotter.plot_isb_analysis(values, save=True, output_dir=str(target)); return 1
        if chart_type == 'inter_freq_bias':
            count = 0
            analyzer = InterFrequencyBiasAnalyzer()
            for system in sorted({sat[0] for sat in obs if sat}):
                for pair in combinations(self.frequencies.get(system, {}), 2):
                    try:
                        value = analyzer.analyze_inter_freq_bias(obs, pair[0], pair[1], constellation=system)
                        if not value.get('error') and value.get('raw_diffs'):
                            self.plotter.plot_inter_freq_bias(value, save=True, output_dir=str(target)); count += 1
                    except Exception as exc:
                        result['errors'].append({'system': system, 'freq_pair': pair, 'error': str(exc)})
            return count

        derived = {
            'derivatives': ('observable_derivatives', self.calculator.calculate_derivatives, self.plotter.plot_derivatives),
            'phase_pred_errors': ('phase_prediction_errors', self.calculator.calculate_phase_prediction_errors, self.plotter.plot_prediction_errors),
            'double_differences': ('epoch_double_diffs', self.calculator.calculate_epoch_double_differences, None),
            'code_phase_diff_raw': ('code_phase_differences', self.calculator.calculate_code_phase_differences, None),
            'code_phase_diffs': ('code_phase_differences', self.calculator.calculate_code_phase_differences, None),
        }
        if chart_type in derived:
            key, calculator, plot = derived[chart_type]
            values = self._cached(key, lambda: calculator(self._inputs()))
            if chart_type == 'double_differences':
                plot = lambda value, sat, freq, **kw: self.plotter.plot_epoch_double_diffs({'epoch_double_diffs': value}, sat, freq, **kw)
            elif chart_type == 'code_phase_diff_raw':
                plot = lambda value, sat, freq, **kw: self.plotter.plot_code_phase_raw_diff({'code_phase_differences': value}, sat, freq, **kw)
            elif chart_type == 'code_phase_diffs':
                plot = lambda value, sat, freq, **kw: self.plotter.plot_code_phase_diff_variation({'code_phase_differences': value}, sat, freq, **kw)
            return self._per_sat_freq(target, result, lambda sat, freq: plot(values, sat, freq, save=True, output_dir=str(target)))
        if chart_type == 'ionofree_cmc':
            code_phase = self._cached('code_phase_differences', lambda: self.calculator.calculate_code_phase_differences(self._inputs()))
            pair = p.get('freq_pair')
            label = '+'.join(pair) if pair else 'auto'
            key = f'ionofree_cmc_{label}'
            values = self._cached(key, lambda: self.calculator.calculate_ionofree_cmc({'code_phase_differences': code_phase}, freq_pair=pair))
            count = 0
            for sat in sorted(values):
                try:
                    self.plotter.plot_ionofree_cmc(values, sat_id=sat, save=True, output_dir=str(target)); count += 1
                except Exception as exc:
                    result['errors'].append({'sat_id': sat, 'error': str(exc)})
            return count
        raise ValueError(f'Unsupported chart type: {chart_type}')

    def _per_sat_freq(self, target, result, callback, per_frequency=True):
        count = 0
        for sat, freq_data in sorted(self.context.observations_meters.items()):
            frequencies = freq_data if per_frequency else (None,)
            for freq in frequencies:
                try:
                    callback(sat, freq); count += 1
                except Exception as exc:
                    result['errors'].append({'sat_id': sat, 'freq': freq, 'error': str(exc)})
        return count
