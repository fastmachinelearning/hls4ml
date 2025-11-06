import json
from pathlib import Path

import pytest


def get_baseline_path(baseline_file_name, backend, version):
    """
    Construct the full path to a baseline synthesis report file.

    Args:
        baseline_file_name (str): The name of the baseline report file.
        backend (str): The backend used (e.g., 'Vivado', 'Vitis').
        version (str): The tool version (e.g., '2020.1').

    Returns:
        Path: A pathlib.Path object pointing to the baseline file location.
    """
    return Path(__file__).parent / 'baselines' / backend / version / baseline_file_name


def save_report(data, filename):
    """
    Save synthesis data to a JSON file in the same directory as this script.

    Args:
        data (dict): The synthesis output data to be saved.
        filename (str): The filename to write to (e.g., 'synthesis_report_test_x.json').

    Raises:
        OSError: If the file cannot be written.
    """
    out_path = Path(__file__).parent / filename
    with open(out_path, 'w') as fp:
        json.dump(data, fp, indent=4)


def compare_dicts(data, baseline, tolerances):
    """
    Compare two flat dictionaries with tolerances.

    Args:
        report (dict): The generated report dictionary.
        baseline (dict): The expected/baseline dictionary.
        tolerances (dict): Dictionary of tolerances per key.

    Raises:
        AssertionError: If values differ outside the allowed tolerance.
    """
    for key, expected in baseline.items():
        actual = data.get(key)
        tolerance = tolerances.get(key, 0)

        try:
            actual = float(actual)
            expected = float(expected)
            assert actual == pytest.approx(expected, rel=tolerance), (
                f'{key}: expected {expected}, got {actual} (tolerance={tolerance * 100}%)'
            )
        except ValueError:
            assert actual == expected, f"{key}: expected '{expected}', got '{actual}'"


def compare_vitis_backend(data, baseline):
    """
    Compare reports from Vivado/Vitis backends.

    Args:
        data (dict): The current synthesis report.
        baseline (dict): The expected synthesis report.
    """

    tolerances = {
        'EstimatedClockPeriod': 0.01,
        'FF': 0.1,
        'LUT': 0.1,
        'BRAM_18K': 0.1,
        'DSP': 0.1,
        'URAM': 0.1,
        'AvailableBRAM_18K': 0.1,
        'AvailableDSP': 0.1,
        'AvailableFF': 0.1,
        'AvailableLUT': 0.1,
        'AvailableURAM': 0.1,
    }

    compare_dicts(data['CSynthesisReport'], baseline['CSynthesisReport'], tolerances)


def compare_oneapi_backend(data, baseline):
    """
    Compare reports from the oneAPI backend.

    Args:
        data (dict): The current synthesis report.
        baseline (dict): The expected synthesis report.
    """

    tolerances = {
        'HLS': {
            'total': {'alut': 0.1, 'reg': 0.1, 'ram': 0.1, 'dsp': 0.1, 'mlab': 0.1},
            'available': {'alut': 0.1, 'reg': 0.1, 'ram': 0.1, 'dsp': 0.1, 'mlab': 0.1},
        },
        'Loop': {'worstFrequency': 0.1, 'worstII': 0.1, 'worstLatency': 0.1},
    }

    data = data['report']
    baseline = baseline['report']

    compare_dicts(data['HLS']['total'], baseline['HLS']['total'], tolerances['HLS']['total'])
    compare_dicts(data['HLS']['available'], baseline['HLS']['available'], tolerances['HLS']['available'])
    compare_dicts(data['Loop'], baseline['Loop'], tolerances['Loop'])


COMPARE_FUNCS = {
    'Vivado': compare_vitis_backend,
    'Vitis': compare_vitis_backend,
    'oneAPI': compare_oneapi_backend,
}


EXPECTED_REPORT_KEYS = {
    'Vivado': {'CSynthesisReport'},
    'Vitis': {'CSynthesisReport'},
    'oneAPI': {'report'},
}


def run_synthesis_test(config, hls_model, baseline_file_name, backend):
    """
    Run HLS synthesis and compare the output with a stored baseline report.

    If synthesis is disabled via the configuration (`run_synthesis=False`),
    no synthesis is executed and the method silently returns.

    Args:
        config (dict): Test-wide synthesis configuration fixture.
        hls_model (object): hls4ml model instance to build and synthesize.
        baseline_file_name (str): The name of the baseline file for comparison.
        backend (str): The synthesis backend used (e.g., 'Vivado', 'Vitis').
    """
    if not config.get('run_synthesis', False):
        return

    # Skip Quartus backend
    if backend == 'Quartus':
        return

    # Run synthesis
    build_args = config['build_args']
    try:
        data = hls_model.build(**build_args.get(backend, {}))
    except Exception as e:
        pytest.fail(f'hls_model.build failed: {e}')

    # Save synthesis report
    save_report(data, f'synthesis_report_{baseline_file_name}')

    # Check synthesis report keys
    expected_keys = EXPECTED_REPORT_KEYS.get(backend, set())
    assert data and expected_keys.issubset(data.keys()), (
        f'Synthesis failed: Missing expected keys in synthesis report: expected {expected_keys}, got {set(data.keys())}'
    )

    # Load baseline report
    version = config['tools_version'].get(backend)
    baseline_path = get_baseline_path(baseline_file_name, backend, version)
    try:
        with open(baseline_path) as fp:
            baseline = json.load(fp)
    except FileNotFoundError:
        pytest.fail(f"Baseline file '{baseline_path}' not found.")

    # Compare report against baseline using backend-specific rules
    compare_func = COMPARE_FUNCS.get(backend)
    if compare_func is None:
        raise AssertionError(f'No comparison function defined for backend: {backend}')

    compare_func(data, baseline)
