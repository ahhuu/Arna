from typing import Dict, Any, Optional
import base64
import csv
import datetime
import html
import json
import mimetypes
import os


class ReportGenerator:
    """Create reproducible reports from a compact preprocessing manifest."""

    MANIFEST_NAME = 'processing_manifest.json'
    VISUALIZATION_MANIFEST_NAME = 'visualization_manifest.json'

    def _get_attr(self, context: Any, name: str, default=None):
        return context.get(name, default) if isinstance(context, dict) else getattr(context, name, default)

    @staticmethod
    def _json_value(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple, set)):
            return [ReportGenerator._json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(k): ReportGenerator._json_value(v) for k, v in value.items()}
        return str(value)

    @staticmethod
    def _nested_count(value):
        if isinstance(value, dict):
            return sum(ReportGenerator._nested_count(v) for v in value.values())
        if isinstance(value, (list, tuple, set)):
            return len(value)
        return int(bool(value))

    @staticmethod
    def _inventory(root):
        items = []
        if not root or not os.path.isdir(root):
            return items
        for current, _dirs, files in os.walk(root):
            for name in sorted(files):
                path = os.path.abspath(os.path.join(current, name))
                if name == ReportGenerator.MANIFEST_NAME:
                    continue
                items.append({'name': name, 'path': path,
                              'relative_path': os.path.relpath(path, root),
                              'extension': os.path.splitext(name)[1].lower(),
                              'size_bytes': os.path.getsize(path)})
        return items

    def write_processing_manifest(self, *, preprocessing_dir: str, input_path: str,
                                  receiver_path: Optional[str], final_output_path: str,
                                  parameters: Dict[str, Any], scope: Dict[str, Any],
                                  flags: Dict[str, Any], results: Dict[str, Any],
                                  step_summary=None) -> str:
        doppler = results.get('doppler_prediction') or {}
        roc = results.get('roc_model') or {}
        corrected = results.get('corrected_phase') or {}
        isb = results.get('isb_analysis') or {}
        multipath = results.get('multipath_correction') or {}
        smoothing = results.get('doppler_smoothing') or {}
        multipath_status = {}
        for detail in multipath.get('details', []) if isinstance(multipath, dict) else []:
            status = str(detail.get('status', 'unknown'))
            multipath_status[status] = multipath_status.get(status, 0) + 1
        smoothing_meta = smoothing.get('smoothing_meta', {}) if isinstance(smoothing, dict) else {}
        smoothing_combinations = sum(len(freqs) for freqs in smoothing_meta.values())
        smoothing_resets = sum(len(info.get('resets', [])) for freqs in smoothing_meta.values()
                               for info in freqs.values())
        roc_types, roc_quality = {}, {}
        for model in roc.values():
            model_type = str(model.get('model_type', 'unknown'))
            quality = str(model.get('quality_level', 'unknown'))
            roc_types[model_type] = roc_types.get(model_type, 0) + 1
            roc_quality[quality] = roc_quality.get(quality, 0) + 1
        summary = {
            'doppler_prediction': {
                'missing_phase_count': doppler.get('total_missing', 0),
                'predicted_phase_count': doppler.get('total_predicted', 0),
                'satellite_statistics': doppler.get('sv_missing_stats', {}),
                'artifacts': doppler.get('artifact_paths', {}),
            },
            'pseudorange_multipath': {
                'enabled': bool(flags.get('enable_pseudorange_multipath')),
                'correction_count': self._nested_count(multipath.get('corrections', {})),
                'detail_record_count': len(multipath.get('details', [])),
                'status_counts': multipath_status,
            },
            'doppler_smoothing': {
                'enabled': bool(smoothing),
                'satellite_frequency_count': smoothing_combinations,
                'reset_count': smoothing_resets,
            },
            'cci': {
                'enabled': bool(flags.get('enable_cci')),
                'dcmc_satellites': len(results.get('dcmc') or {}),
                'roc_model_count': len(roc),
                'model_type_counts': roc_types,
                'quality_level_counts': roc_quality,
                'corrected_entry_count': self._nested_count(corrected.get('corrected_results', corrected)),
            },
            'coarse_error': {
                'cmc_flag_count': self._nested_count(results.get('cmc_flags') or {}),
                'double_difference_entry_count': self._nested_count(results.get('triple_errors') or {}),
            },
            'isb': {
                'enabled': bool(flags.get('enable_isb')),
                'mean_m': isb.get('isb_mean'), 'std_m': isb.get('isb_std'),
                'median_m': isb.get('isb_median'),
                'epoch_count': len(isb.get('isb_epochs', isb.get('isb_estimates', []))),
            },
        }
        manifest = {
            'schema_version': 1,
            'generated_at': datetime.datetime.now().astimezone().isoformat(),
            'application': 'Android_GNSS_Analysis',
            'input': {'phone_rinex': os.path.abspath(input_path),
                      'receiver_rinex': os.path.abspath(receiver_path) if receiver_path else None},
            'output': {'preprocessing_dir': os.path.abspath(preprocessing_dir),
                       'final_rinex': os.path.abspath(final_output_path) if final_output_path else None},
            'scope': self._json_value(scope), 'parameters': self._json_value(parameters),
            'flags': self._json_value(flags), 'steps': list(step_summary or []),
            'summary': self._json_value(summary),
        }
        manifest['artifacts'] = self._inventory(preprocessing_dir)
        path = os.path.join(preprocessing_dir, self.MANIFEST_NAME)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        return path

    def find_manifest(self, selected_path: Optional[str]) -> Optional[str]:
        if not selected_path:
            return None
        path = os.path.abspath(selected_path)
        if os.path.isfile(path) and os.path.basename(path) == self.MANIFEST_NAME:
            return path
        start = path if os.path.isdir(path) else os.path.dirname(path)
        candidates = [os.path.join(start, self.MANIFEST_NAME), os.path.join(start, 'preprocessing', self.MANIFEST_NAME)]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        if os.path.isfile(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            candidate = os.path.join(os.path.dirname(path), 'Arna_results', stem, 'preprocessing', self.MANIFEST_NAME)
            if os.path.isfile(candidate):
                return candidate
        if not os.path.isdir(start):
            return None
        for current, dirs, files in os.walk(start):
            dirs[:] = [d for d in dirs if d != '.git']
            if self.MANIFEST_NAME in files:
                return os.path.join(current, self.MANIFEST_NAME)
            if os.path.relpath(current, start).count(os.sep) >= 3:
                dirs[:] = []
        return None

    def load_manifest(self, path: str) -> Dict[str, Any]:
        manifest_path = self.find_manifest(path)
        if not manifest_path:
            raise FileNotFoundError('未找到 processing_manifest.json；请先完成一次预处理，或选择预处理结果目录。')
        with open(manifest_path, encoding='utf-8') as fh:
            data = json.load(fh)
        data['_manifest_path'] = manifest_path
        return data

    def _csv_rows(self, manifest, artifact_key):
        artifacts = manifest.get('summary', {}).get('doppler_prediction', {}).get('artifacts', {})
        path = artifacts.get(artifact_key)
        if not path or not os.path.isfile(path):
            return []
        with open(path, encoding='utf-8-sig', newline='') as fh:
            return list(csv.DictReader(fh))

    def generate_text_report(self, context: Any) -> str:
        manifest = context if isinstance(context, dict) and 'schema_version' in context else None
        if manifest is None:
            selected = self._get_attr(context, 'manifest_path', None) or self._get_attr(context, 'input_path', None)
            found = self.find_manifest(selected)
            if found:
                manifest = self.load_manifest(found)
            else:
                results = self._get_attr(context, 'results', {}) or {}
                return '\n'.join(['GNSS观测数据报告（当前会话，尚无预处理任务清单）',
                                  f'输入文件: {selected or "未选择"}',
                                  '请完成一次预处理以生成包含参数、统计、图表和文件索引的完整报告。',
                                  f'当前内存结果项: {", ".join(sorted(results.keys())) if isinstance(results, dict) else "无"}'])
        d = manifest.get('summary', {}).get('doppler_prediction', {})
        predicted, missing = d.get('predicted_phase_count', 0), d.get('missing_phase_count', 0)
        lines = ['=' * 78, 'GNSS观测数据预处理综合报告', '=' * 78,
                 f"生成时间: {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
                 f"任务清单: {manifest.get('_manifest_path', '当前会话')}",
                 f"手机RINEX: {manifest.get('input', {}).get('phone_rinex', '未记录')}",
                 f"接收机RINEX: {manifest.get('input', {}).get('receiver_rinex') or '未提供'}",
                 f"最终RINEX: {manifest.get('output', {}).get('final_rinex') or '未生成'}", '',
                 '一、任务概览', '-' * 40]
        overview_labels = {'processing_systems_var': '卫星系统',
                           'processing_frequencies_var': '观测频率'}
        for key, label in overview_labels.items():
            if key in manifest.get('parameters', {}):
                lines.append(f"{label}: {manifest['parameters'][key]}")
        lines += ['', '二、处理步骤', '-' * 40]
        lines.extend(f'- {step}' for step in manifest.get('steps', []))
        lines += ['', '三、各项预处理结果概览', '-' * 40,
                  f"1. 多普勒相位预测: 缺失 {missing}，成功预测 {predicted}，修补率: " +
                  (f'{100.0 * predicted / missing:.2f}%' if missing else '无法计算')]
        labels = [('pseudorange_multipath', '伪距多路径改正'), ('doppler_smoothing', '多普勒平滑'),
                  ('cci', '码相不一致性建模与校正'), ('coarse_error', 'CMC变化异常与双差异常剔除'),
                  ('isb', 'BDS-2/BDS-3 ISB')]
        for index, (key, label) in enumerate(labels, start=2):
            value = manifest.get('summary', {}).get(key, {})
            lines.append(f"{index}. {label}: {'启用/已执行' if value.get('enabled', key == 'coarse_error') else '未启用'}")
        lines += ['', '四、说明', '-' * 40,
                  '完整参数、各算法统计表、真实图表和伴随文件索引请查看HTML综合报告。',
                  '本报告评估RINEX观测数据预处理、完整率和基于真实载波相位的预测误差。',
                  '当前流程未接入PPP解算，因此不能据此宣称定位精度或收敛时间得到改善。']
        return '\n'.join(lines)

    @staticmethod
    def _image_uri(path):
        if not path or not os.path.isfile(path):
            return None
        mime = mimetypes.guess_type(path)[0] or 'image/png'
        with open(path, 'rb') as fh:
            return f'data:{mime};base64,{base64.b64encode(fh.read()).decode("ascii")}'

    def generate_html_report(self, manifest: Dict[str, Any]) -> str:
        esc = lambda value: html.escape(str(value))
        d = manifest.get('summary', {}).get('doppler_prediction', {})
        missing, predicted = d.get('missing_phase_count', 0), d.get('predicted_phase_count', 0)
        integrity, errors = self._csv_rows(manifest, 'integrity_summary_rms_csv'), self._csv_rows(manifest, 'prediction_error_rms_csv')
        figures = []
        for key, title in (('phase_integrity_before_png', '预测前相位完整率'), ('phase_integrity_after_png', '预测后相位完整率'),
                           ('observation_integrity_comparison_png', '观测数据完整率对比'), ('prediction_error_rms_png', '多步预测误差RMS')):
            uri = self._image_uri(d.get('artifacts', {}).get(key))
            if uri:
                figures.append(f'<figure><img src="{uri}" alt="{esc(title)}"><figcaption>{esc(title)}</figcaption></figure>')
        integrity_html = ''.join(f"<tr><td>{esc(r.get('system'))}</td><td>{esc(r.get('frequency'))}</td><td>{100*float(r.get('code_rate',0)):.2f}%</td><td>{100*float(r.get('phase_before_rate',0)):.2f}%</td><td>{100*float(r.get('phase_after_rate',0)):.2f}%</td></tr>" for r in integrity)
        params = ''.join(f'<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>' for k, v in manifest.get('parameters', {}).items())
        steps = ''.join(f'<li>{esc(v)}</li>' for v in manifest.get('steps', []))
        files = ''.join(f"<tr><td>{esc(v['relative_path'])}</td><td>{v['size_bytes']}</td></tr>" for v in manifest.get('artifacts', []))
        summary = manifest.get('summary', {})
        mp, smoothing, cci = summary.get('pseudorange_multipath', {}), summary.get('doppler_smoothing', {}), summary.get('cci', {})
        coarse, isb = summary.get('coarse_error', {}), summary.get('isb', {})
        all_params = manifest.get('parameters', {})
        def parameter_table(tokens):
            selected = [(k, v) for k, v in all_params.items() if any(token in k for token in tokens)]
            body = ''.join(f'<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>' for k, v in selected)
            return f'<h3>实际参数</h3><table>{body}</table>' if body else '<p>本步骤没有额外可调参数。</p>'
        def result_table(values):
            body = ''.join(f'<tr><th>{esc(k)}</th><td>{esc(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)}</td></tr>' for k, v in values.items())
            return f'<h3>实际统计</h3><table>{body}</table>'
        def artifact_table(folder_tokens):
            selected = [v for v in manifest.get('artifacts', [])
                        if any(token.lower() in v['relative_path'].lower() for token in folder_tokens)]
            body = ''.join(f"<tr><td>{esc(v['relative_path'])}</td><td>{v['size_bytes']}</td></tr>" for v in selected)
            return f'<h3>输出文件与日志</h3><table><tr><th>文件</th><th>大小(bytes)</th></tr>{body}</table>' if body else '<p>没有生成本步骤的输出文件。</p>'
        validation_samples = sum(int(float(row.get('sample_count', 0) or 0)) for row in errors)
        weighted_rms = None
        weighted_mae = None
        if validation_samples:
            weighted_rms = (sum(int(float(row.get('sample_count', 0) or 0)) * float(row.get('rms_m', 0) or 0) ** 2
                                for row in errors) / validation_samples) ** 0.5
            weighted_mae = sum(int(float(row.get('sample_count', 0) or 0)) * float(row.get('mae_m', 0) or 0)
                               for row in errors) / validation_samples
        doppler_stats = {
            'missing_phase_count': missing,
            'predicted_phase_count': predicted,
            'repair_rate_percent': round(100 * predicted / missing, 4) if missing else None,
            'validation_group_count': len(errors),
            'validation_sample_count': validation_samples,
            'weighted_validation_rms_m': round(weighted_rms, 6) if weighted_rms is not None else None,
            'weighted_validation_mae_m': round(weighted_mae, 6) if weighted_mae is not None else None,
        }
        doppler_section = f'''<h2>3. 多普勒相位预测</h2><p><b>状态：</b>{'已启用' if manifest.get('flags', {}).get('enable_doppler') else '未启用'}。</p><p><b>方法：</b>对相邻多普勒观测形成的相位距离变化率作梯形积分，在连续长度、采样间隔、LLI和多普勒可用性门控下修补短载波相位缺口；预测长度同时映射为供未来PPP使用的建议标准差。</p>{parameter_table(['doppler_prediction'])}{result_table(doppler_stats)}<h3>系统频率完整率</h3><table><tr><th>系统</th><th>频率</th><th>Code完整率</th><th>预测前Phase</th><th>预测后Phase</th></tr>{integrity_html}</table>{''.join(figures)}{artifact_table(['doppler prediction'])}'''
        detail_sections = f'''<h2>4. 伪距多路径改正</h2><p><b>状态：</b>{'已启用' if mp.get('enabled') else '未启用'}。</p><p><b>方法：</b>利用双频码相组合估计伪距多路径；按连续弧段、半周标志、MAD门限和最大改正量进行门控，只改写通过门控的伪距。</p>{parameter_table(['pseudorange_multipath'])}{result_table(mp)}{artifact_table(['pseudorange multipath correction'])}
<h2>5. 多普勒平滑伪距</h2><p><b>状态：</b>{'已执行' if smoothing.get('enabled') else '未启用'}。</p><p><b>方法：</b>使用多普勒积分得到历元间距离变化，通过Hatch形式平滑伪距；采样中断或残差超限时重置平滑弧段。</p>{parameter_table(['doppler_window', 'doppler_threshold_smooth'])}{result_table(smoothing)}{artifact_table(['doppler smoothing'])}
<h2>6. 码相不一致性建模与校正</h2><p><b>状态：</b>{'已启用' if cci.get('enabled') else '未启用'}。</p><p><b>方法：</b>手机与接收机参考数据构造站间单差CMC，筛选线性漂移序列、建立系统级或卫星级ROC模型，再对手机载波相位进行校正。</p>{parameter_table(['r_squared', 'cv_threshold', 'phone_only'])}{result_table(cci)}{artifact_table(['code-carrier inconsistency'])}
<h2>7. CMC变化异常剔除</h2><p><b>状态：</b>始终执行。</p><p><b>方法：</b>计算码相组合变化，使用固定阈值或数据驱动的自适应阈值标记异常观测，并写入第一阶段清理RINEX。</p>{parameter_table(['threshold_mode', 'cmc_threshold'])}{result_table({'cmc_flag_count': coarse.get('cmc_flag_count', 0)})}{artifact_table(['code_phase_cleaning', 'cleaned1'])}
<h2>8. 历元间双差异常剔除</h2><p><b>状态：</b>始终执行。</p><p><b>方法：</b>对伪距、载波相位和多普勒构造历元间双差，使用MAD稳健尺度和配置上限判断异常，生成第二阶段清理RINEX。</p>{parameter_table(['code_threshold', 'phase_threshold', 'doppler_threshold', 'threshold_mode'])}{result_table({'double_difference_entry_count': coarse.get('double_difference_entry_count', 0)})}{artifact_table(['double_diffs_cleaning', 'cleaned2'])}
<h2>9. BDS-2/BDS-3 ISB分析</h2><p><b>状态：</b>{'已启用' if isb.get('enabled') else '未启用'}。</p><p><b>方法：</b>在手机与接收机的共同历元选择稳定参考卫星，以双差估计BDS-2/BDS-3系统间偏差并按配置写回；没有接收机参考文件时不执行。</p>{result_table(isb)}{artifact_table(['BDS23_ISB'])}'''
        return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>GNSS预处理综合报告</title><style>body{{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1200px;margin:24px auto;color:#222;line-height:1.55}}h1,h2{{color:#17365d}}.meta,.warning{{padding:12px;background:#f4f7fb;border-left:4px solid #3f5aa9}}.warning{{background:#fff4df;border-color:#e39800}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #bbb;padding:6px;text-align:left}}th{{background:#edf2f7}}figure{{margin:22px 0}}img{{max-width:100%;height:auto}}figcaption{{text-align:center;color:#555}}@media print{{body{{max-width:none}}figure{{break-inside:avoid}}}}</style></head><body><h1>GNSS观测数据预处理综合报告</h1><div class="meta">生成时间：{esc(datetime.datetime.now().astimezone().isoformat(timespec='seconds'))}<br>手机RINEX：{esc(manifest.get('input',{}).get('phone_rinex'))}<br>接收机RINEX：{esc(manifest.get('input',{}).get('receiver_rinex') or '未提供')}<br>最终RINEX：{esc(manifest.get('output',{}).get('final_rinex') or '未生成')}</div><h2>1. 处理参数</h2><table>{params}</table><h2>2. 处理步骤</h2><ol>{steps}</ol>{doppler_section}{detail_sections}<h2>10. 结论边界</h2><div class="warning">本报告只评估RINEX观测数据预处理、完整率及基于真实载波相位的预测误差。当前未接入PPP解算，不能据此宣称定位精度或收敛时间得到改善。</div><h2>11. 伴随文件</h2><table><tr><th>相对路径</th><th>大小(bytes)</th></tr>{files}</table></body></html>'''

    def write_visualization_report(self, visualization_dir: str, input_path: str,
                                   selection: str = 'all', metadata: Optional[Dict[str, Any]] = None,
                                   generate_reports: bool = True,
                                   report_output_dir: Optional[str] = None):
        """Inventory visualization outputs and optionally create its reports."""
        artifacts = [item for item in self._inventory(visualization_dir)
                     if not item['relative_path'].startswith('report' + os.sep)
                     and item['name'] != self.VISUALIZATION_MANIFEST_NAME]
        manifest = {
            'schema_version': 1, 'report_type': 'visualization',
            'generated_at': datetime.datetime.now().astimezone().isoformat(),
            'input_path': os.path.abspath(input_path),
            'visualization_dir': os.path.abspath(visualization_dir),
            'selection': selection, 'metadata': self._json_value(metadata or {}),
            'artifacts': artifacts,
        }
        manifest_path = os.path.join(visualization_dir, self.VISUALIZATION_MANIFEST_NAME)
        with open(manifest_path, 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        if not generate_reports:
            return {'manifest': manifest_path}
        # Keep both report families under one task-level report directory:
        # report/preprocessing and report/visualization.
        report_dir = report_output_dir or os.path.join(
            os.path.dirname(visualization_dir), 'report', 'visualization')
        os.makedirs(report_dir, exist_ok=True)
        text_path = os.path.join(report_dir, 'visualization-report.txt')
        html_path = os.path.join(report_dir, 'visualization-report.html')
        categories = {}
        for item in artifacts:
            category = item['relative_path'].split(os.sep, 1)[0]
            categories.setdefault(category, []).append(item)
        descriptions = {
            'Raw_observations': '各卫星原始伪距、载波相位、多普勒和载噪比时序。',
            'Pseudorange_Multipath_Satellite': '按卫星和频率组合计算的伪距多路径序列。',
            'Pseudorange_Multipath_Constellation': '星座级伪距多路径统计与对比。',
            'Satellite_frequency_sequence': '真实卫星与频率观测的时间连续性。',
            'Satellite_count': '各历元可用卫星数量。', 'Data_Integrity': '码、相位等观测数据完整率。',
            'CNR_Analysis': '载噪比时序与统计。', 'Observation_Noise': '三阶差分观测噪声评估。',
            'Doppler_Quality': '多普勒完整性、相位一致性和跨频一致性质量。',
            'Cycle_slips': 'MW/GF/LLI等周跳探测结果。', 'Prediction_errors': '载波相位预测误差。',
            'Double_differences': '历元间双差序列及异常。', 'ISB_analysis': 'BDS-2/BDS-3系统间偏差。',
        }
        lines = ['GNSS可视化处理报告（概览）', f'输入文件: {manifest["input_path"]}',
                 f'生成时间: {manifest["generated_at"]}', f'保存范围: {selection}',
                 f'伴随文件总数: {len(artifacts)}', '']
        for category, items in sorted(categories.items()):
            images = sum(1 for item in items if item['extension'] in ('.png', '.jpg', '.jpeg', '.svg'))
            lines.append(f'{category}: {len(items)} 个文件，其中图像 {images} 张')
        with open(text_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines))
        sections = []
        for category, items in sorted(categories.items()):
            cards = []
            for item in items:
                rel_from_report = os.path.relpath(item['path'], report_dir).replace(os.sep, '/')
                if item['extension'] in ('.png', '.jpg', '.jpeg', '.gif', '.svg'):
                    cards.append(f'<figure><a href="{html.escape(rel_from_report)}"><img loading="lazy" src="{html.escape(rel_from_report)}"></a><figcaption>{html.escape(item["name"])}</figcaption></figure>')
                else:
                    cards.append(f'<li><a href="{html.escape(rel_from_report)}">{html.escape(item["name"])}</a> ({item["size_bytes"]} bytes)</li>')
            sections.append(f'<section><h2>{html.escape(category)}</h2><p>{html.escape(descriptions.get(category, "该类别的可视化图表和统计伴随文件。"))}</p><div class="gallery">{"".join(cards)}</div></section>')
        doc = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>GNSS可视化处理报告</title><style>body{{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1400px;margin:24px auto;color:#222}}h1,h2{{color:#17365d}}.meta{{padding:12px;background:#f4f7fb;border-left:4px solid #3f5aa9}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}figure{{margin:0;padding:10px;border:1px solid #ccc}}img{{width:100%;height:auto}}figcaption{{word-break:break-all;color:#555}}@media print{{.gallery{{display:block}}figure{{break-inside:avoid;margin:12px 0}}}}</style></head><body><h1>GNSS可视化处理报告</h1><div class="meta">输入文件：{html.escape(manifest['input_path'])}<br>生成时间：{html.escape(manifest['generated_at'])}<br>保存范围：{html.escape(selection)}<br>文件总数：{len(artifacts)}</div>{''.join(sections)}</body></html>'''
        with open(html_path, 'w', encoding='utf-8') as fh:
            fh.write(doc)
        return {'manifest': manifest_path, 'text': text_path, 'html': html_path}

    def save_report_bundle(self, manifest_or_path: Any, output_dir: str, base_name='gnss-preprocessing-report'):
        manifest = self.load_manifest(manifest_or_path) if isinstance(manifest_or_path, str) else manifest_or_path
        os.makedirs(output_dir, exist_ok=True)
        paths = {'text': os.path.join(output_dir, base_name + '.txt'), 'html': os.path.join(output_dir, base_name + '.html')}
        with open(paths['text'], 'w', encoding='utf-8') as fh:
            fh.write(self.generate_text_report(manifest))
        with open(paths['html'], 'w', encoding='utf-8') as fh:
            fh.write(self.generate_html_report(manifest))
        return paths

    def save_logs(self, log_type: str, content: str, output_dir: str, prefix: Optional[str] = None) -> str:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        path = os.path.join(output_dir, f"{prefix + '-' if prefix else ''}{log_type}-{ts}.log")
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        return path
