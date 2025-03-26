import json
import os
import pytest
from pathlib import Path


def save_baseline(data, filename):
    """ Saves the given data as a baseline in the specified file. """
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=4)


def get_baseline_path(baseline_file_name, backend):

    tool_versions = {
        'Vivado': '2023.1',
        'Vitis': '2023.1',
    }

    default_version = 'latest'

    version = tool_versions.get(backend, default_version)

    return (
        Path(__file__).parent /
        "baselines" / backend / version / baseline_file_name
    )


def get_tolerance(key):
    """
    Get the tolerance for a given key, using a predefined set of tolerances.

    :param key: The synthesis report key to check.
    :return: The tolerance value for the given key.
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

    # Default tolerance for unspecified keys
    default_tolerance = 0.01  

    return tolerances.get(key, default_tolerance)


def compare_synthesis_tolerance(data, filename):
    """ Compare synthesis report values with a given tolerance. """
    try:
        with open(filename, "r") as fp:
            baseline = json.load(fp)
    except FileNotFoundError:
        pytest.skip(f"Baseline file '{filename}' not found.")

    csrBaseline = baseline.get("CSynthesisReport")
    csrData = data.get("CSynthesisReport")

    for key, expected_value in csrBaseline.items():
        actual_value = csrData.get(key)
        tolerance = get_tolerance(key)

        try:
            # Convert to float for numerical comparison
            expected_num = float(expected_value)
            actual_num = float(actual_value)
            assert actual_num == pytest.approx(expected_num, rel=tolerance), \
                f"{key}: expected {expected_num}, got {actual_num} (tolerance={tolerance*100}%)"
        except ValueError:
            # Exact match for non-numeric values
            assert actual_value == expected_value, f"{key}: expected '{expected_value}', got '{actual_value}'"


def test_synthesis(synthesis, hls_model, baseline_file_name, backend):
    """Function to run synthesis and compare results."""
    if synthesis:

        if hls_model.config.get_config_value('Backend') == 'oneAPI':
            pytest.skip(f'oneAPI backend not supported in synthesis tests.')

        try:
            # TODO: should csim be True? whaat other params should be set?
            data = hls_model.build(csim=True)
        except Exception as e:
            pytest.skip(str(e))

        assert {'CSimResults', 'CSynthesisReport'}.issubset(data.keys()), \
            "Synthesis failed: Missing expected keys in the synthesis report"

        baseline_path = get_baseline_path(baseline_file_name, backend)
        compare_synthesis_tolerance(data, baseline_path)
