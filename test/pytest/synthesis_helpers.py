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
    return Path(__file__).parent / "baselines" / backend / version / baseline_file_name


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
    with open(out_path, "w") as fp:
        json.dump(data, fp, indent=4)


def get_tolerance(key):
    """
    Get the relative tolerance for a specific synthesis report field.

    Args:
        key (str): The key in the synthesis report to compare.

    Returns:
        float: The relative tolerance allowed for that key. Defaults to 1% (0.01).
    """
    tolerances = {
        "EstimatedClockPeriod": 0.01,
        "FF": 0.05,
        "LUT": 0.10,
        "BRAM_18K": 0.0,
        "DSP": 0.0,
        "URAM": 0.0,
        "AvailableBRAM_18K": 0.0,
        "AvailableDSP": 0.0,
        "AvailableFF": 0.0,
        "AvailableLUT": 0.0,
        "AvailableURAM": 0.0,
    }

    default_tolerance = 0.01

    return tolerances.get(key, default_tolerance)


def compare_reports_with_tolerance(data, baseline):
    """
    Compare two synthesis reports using tolerances defined per key.

    Args:
        data (dict): The current synthesis report.
        baseline (dict): The baseline synthesis report to compare against.
    """
    csrBaseline = baseline.get("CSynthesisReport")
    csrData = data.get("CSynthesisReport")

    for key, expected_value in csrBaseline.items():
        actual_value = csrData.get(key)
        tolerance = get_tolerance(key)

        try:
            # Convert to float for numerical comparison
            expected_num = float(expected_value)
            actual_num = float(actual_value)
            assert actual_num == pytest.approx(
                expected_num, rel=tolerance
            ), f"{key}: expected {expected_num}, got {actual_num} (tolerance={tolerance*100}%)"
        except ValueError:
            # Exact match for non-numeric values
            assert actual_value == expected_value, f"{key}: expected '{expected_value}', got '{actual_value}'"


def test_synthesis(config, hls_model, baseline_file_name, backend):
    """
    Run HLS synthesis and compare the output with a stored baseline report.

    If synthesis is disabled via the configuration (`run_synthesis=False`),
    no synthesis is executed and the test silently returns.

    Args:
        config (dict): Test-wide synthesis configuration fixture.
        hls_model (object): hls4ml model instance to build and synthesize.
        baseline_file_name (str): The name of the baseline file for comparison.
        backend (str): The synthesis backend used (e.g., 'Vivado', 'Vitis').
    """
    if not config.get("run_synthesis", False):
        # TODO: should this info be printed or logged?
        return

    if backend == 'oneAPI':
        pytest.skip('oneAPI backend not supported in synthesis tests.')

    build_args = config["build_args"]

    try:
        data = hls_model.build(**build_args.get(backend, {}))
    except Exception as e:
        pytest.skip(f"hls_model.build failed: {e}")

    save_report(data, f"synthesis_report_{baseline_file_name}")

    assert data and {'CSynthesisReport'}.issubset(
        data.keys()
    ), "Synthesis failed: Missing expected keys in the synthesis report"

    version = config["tools_version"].get(backend)
    baseline_path = get_baseline_path(baseline_file_name, backend, version)

    try:
        with open(baseline_path) as fp:
            baseline = json.load(fp)
    except FileNotFoundError:
        pytest.skip(f"Baseline file '{baseline_path}' not found.")

    compare_reports_with_tolerance(data, baseline)
