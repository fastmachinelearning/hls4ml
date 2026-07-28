import glob
import os
import xml.etree.ElementTree as ET

from hls4ml.report.vivado_report import (
    _parse_csim_results,
    _parse_implementation_report,
    _parse_power_report,
    _parse_rtl_cosim_results,
    _parse_timing_report,
)


def _coerce_value(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return raw
    try:
        return int(raw)
    except (ValueError, TypeError):
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw


def _parse_result_file(path):
    """Parse a single bambu_results XML file produced by Bambu 2026.06+.

    The reference schema (PandA 2026.06) has root tag ``<application>`` with:
    - meta as root attributes: ``args``, ``version``, ``benchmark``, ``timestamp``
    - ``<resources>`` child: flat attributes per metric (LUTS, REGISTERS, DSPS,
      BRAMS, DRAMS, SLICES, FE, IOPINS, POWER, FREQUENCY, SLACK, DELAY, …).
      Attribute names are vendor-specific (Xilinx has SLICES; NanoXplore has FE).
      Absent when P&R fails (e.g. NanoXplore routing errors) — tolerated.
    - ``<evaluation>`` child (top-level): CYCLES, AREA, PERIOD, FREQUENCY, …
    - ``<timing>`` child: ``<simulation>`` or ``<evaluation>`` sub-element each
      containing ``<run>`` text nodes with per-execution cycle counts.
    - ``<hls_results>`` child: per-function scheduling info (not parsed here).
    """
    tree = ET.parse(path)
    root = tree.getroot()

    meta = {
        'Args': root.attrib.get('args'),
        'Version': root.attrib.get('version'),
        'Timestamp': root.attrib.get('timestamp'),
        'Benchmark': root.attrib.get('benchmark'),
        'File': os.path.basename(path),
    }

    metrics = {}

    # Resource metrics — absent when P&R fails, silently skipped.
    resources = root.find('resources')
    if resources is not None:
        for key, val in resources.attrib.items():
            metrics[key] = _coerce_value(val)

    # Top-level <evaluation> carries CYCLES, AREA, PERIOD, FREQUENCY, … in 2026.06.
    # Use setdefault so <resources> values (FREQUENCY, REGISTERS, …) take precedence
    # when both are present (resources come from synthesis, evaluation may repeat them).
    evaluation = root.find('evaluation')
    if evaluation is not None:
        for key, val in evaluation.attrib.items():
            metrics.setdefault(key, _coerce_value(val))

    # Cycle counts from <timing>/<simulation|evaluation>/<run> text nodes.
    # 2026.06 uses <simulation>; older PandA used <evaluation>; try both.
    timing = root.find('timing')
    if timing is not None:
        timing_node = timing.find('simulation') or timing.find('evaluation')
        if timing_node is not None:
            runs = [_coerce_value(r.text) for r in timing_node.findall('run')]
            if runs:
                metrics['Total cycles'] = sum(runs)
                metrics['Number of executions'] = len(runs)
                metrics['Average execution'] = sum(runs) / len(runs)

    return {'meta': meta, 'metrics': metrics}


def parse_bambu_report(hls_dir, part_family):
    """Parse Bambu result files from ``hls_dir``.

    Parses the ``bambu_results*.xml`` file(s) produced by Bambu 2026.06
    (root ``<application>``, ``<resources>`` attrs, ``<evaluation>`` attrs,
    ``<timing>/<simulation|evaluation>/<run>`` cycle counts).  For Xilinx
    targets, also reads the Vivado implementation, timing and power reports.

    Args:
        hls_dir: directory containing ``bambu_results*.xml`` and, for Xilinx
            targets, the Vivado report tree.
        part_family: ``"Xilinx"`` or ``"NanoXplore"`` (or ``None``).  Controls
            whether Vivado reports are parsed.

    Returns:
        dict with zero or more of the following keys:
        - ``'BambuMetrics'``: dict of resource/timing metrics from the XML
          (e.g. LUTS, REGISTERS, DSPS, CYCLES, Total cycles, …).  Absent
          when no ``bambu_results*.xml`` is found or P&R failed and the
          file contains no ``<resources>`` block (NanoXplore routing errors).
        - ``'CSimResults'``, ``'CosimResults'``: parsed C-sim / RTL-cosim logs.
        - ``'ImplementationReport'``, ``'TimingReport'``, ``'PowerReport'``:
          Vivado reports (Xilinx only).
    """
    result = {}

    # Parse CSim and Cosim
    csim_results = _parse_csim_results(hls_dir)
    if csim_results is not None:
        result['CSimResults'] = csim_results

    cosim_results = _parse_rtl_cosim_results(hls_dir)
    if cosim_results is not None:
        result['CosimResults'] = cosim_results

    # Parse metrics reported by Bambu
    pattern = os.path.join(hls_dir, 'bambu_results*.xml')
    matches = sorted(glob.glob(pattern))
    if matches:
        parsed = [_parse_result_file(path) for path in matches]
        result['BambuMetrics'] = parsed[-1]['metrics']

    # Parse Vivado reports if target is from Xilinx
    if part_family == 'Xilinx':
        implementation_report = _parse_implementation_report(hls_dir, is_vivado_accelerator=False, percentage_columns=False)
        if implementation_report is not None:
            result['ImplementationReport'] = implementation_report
        else:
            print('Implementation report not found.')

        timing_report = _parse_timing_report(hls_dir, is_vivado_accelerator=False)
        if timing_report is not None:
            result['TimingReport'] = timing_report
        else:
            print('Timing report not found.')

        power_report = _parse_power_report(hls_dir, is_vivado_accelerator=False)
        if power_report is not None:
            result['PowerReport'] = power_report
        else:
            print('Power report not found.')

    return result
