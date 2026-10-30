"""Source-level guards for Vitis streaming 2D convolution line-buffer declarations.

These checks do not replace Vitis HLS synthesis or RTL co-simulation.
"""

from pathlib import Path

import pytest

VITIS_UTILS = Path(__file__).resolve().parents[2] / 'hls4ml' / 'templates' / 'vitis' / 'nnet_utils'


@pytest.mark.parametrize(
    ('header', 'expected_declarations'),
    [
        ('nnet_conv2d_stream.h', 2),
        ('nnet_sepconv2d_stream.h', 1),
    ],
)
def test_vitis_conv2d_line_buffer_minimum_depth(header, expected_declarations):
    """Prevent reintroducing a one-row buffer for 2x2 streaming convolutions."""
    source = (VITIS_UTILS / header).read_text()
    assert source.count('line_buffer[MAX(CONFIG_T::filt_height - 1, 2)]') == expected_declarations
    assert 'line_buffer[MAX(CONFIG_T::filt_height - 1, 1)]' not in source
    assert 'line_buffer[CONFIG_T::filt_height - 1]' not in source
