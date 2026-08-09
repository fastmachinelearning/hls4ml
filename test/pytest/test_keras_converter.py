from unittest.mock import Mock

import pytest

from hls4ml.converters.keras.convolution import parse_conv1d_layer


def test_conv1d_rejects_unsupported_dilation():
    keras_layer = {
        'class_name': 'Conv1D',
        'config': {
            'name': 'dilated_conv',
            'data_format': 'channels_last',
            'filters': 4,
            'kernel_size': [3],
            'strides': [1],
            'padding': 'same',
            'dilation_rate': [2],
        },
    }

    with pytest.raises(NotImplementedError, match='dilation_rate=2 is not supported'):
        parse_conv1d_layer(keras_layer, ['input'], [[None, 64, 1]], data_reader=Mock())
