import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from hls4ml.model.profiling import boxplot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def mock_data():
    """Fixture to create mock data for testing."""
    return pd.DataFrame({'x': np.random.lognormal(mean=0, sigma=1, size=100), 'weight': ['layer1'] * 100})


def generate_random_summary_data(num_layers=3, num_weights_per_layer=2):
    data = []
    for layer_idx in range(1, num_layers + 1):
        layer_name = f'dense_{layer_idx}'
        for weight_idx in range(num_weights_per_layer):
            weight_type = 'w' if weight_idx == 0 else 'b'
            weight_label = f'{layer_name}/{weight_type}'
            med = np.random.uniform(0.1, 1.0)
            q1 = med - np.random.uniform(0.05, 0.2)
            q3 = med + np.random.uniform(0.05, 0.2)
            whislo = max(0.0, q1 - np.random.uniform(0.05, 0.1))
            whishi = q3 + np.random.uniform(0.05, 0.1)
            data.append(
                {
                    'med': med,
                    'q1': q1,
                    'q3': q3,
                    'whislo': whislo,
                    'whishi': whishi,
                    'layer': layer_name,
                    'weight': weight_label,
                }
            )
    return data


@pytest.mark.parametrize('fmt', ['longform', 'summary'])
def test_boxplot_vertical_line(mock_data, fmt):
    # Test if the vertical line at x=1 exists in the boxplot.
    if fmt == 'summary':
        mock_data = generate_random_summary_data()
    fig = boxplot(mock_data, fmt=fmt)
    output_path = f'boxplot_vertical_line_{fmt}.png'
    fig.savefig(output_path)
    logger.info(f'Boxplot with vertical line ({fmt}) saved to: {output_path}')
    ax = plt.gca()
    found_vertical_line = any(
        len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1] == 1 for line in ax.get_lines()
    )
    assert found_vertical_line, f'Vertical line at x=1 (2^0) is missing in the boxplot ({fmt}).'
    plt.close(fig)


@pytest.mark.parametrize('fmt', ['longform', 'summary'])
def test_boxplot_output(mock_data, fmt):
    # Test if the boxplot function produces a valid matplotlib figure.
    if fmt == 'summary':
        mock_data = generate_random_summary_data()
    fig = boxplot(mock_data, fmt=fmt)
    output_path = f'boxplot_output_{fmt}.png'
    fig.savefig(output_path)
    logger.info(f'Boxplot output ({fmt}) saved to: {output_path}')
    assert isinstance(fig, plt.Figure), f'The boxplot function did not return a matplotlib Figure ({fmt}).'
    plt.close(fig)
