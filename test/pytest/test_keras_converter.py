"""Direct tests of the Keras v2 layer handlers.

These call the handlers with synthetic layer configs instead of building models, so
they exercise the v2 parsing code deterministically under any installed Keras version.
Unsupported layer options must raise a clear error instead of being silently dropped;
options the IR can represent must be parsed faithfully.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from hls4ml.converters.keras.convolution import parse_conv1d_layer, parse_conv2d_layer
from hls4ml.converters.keras.core import parse_activation_layer, parse_batchnorm_layer, parse_embedding_layer
from hls4ml.converters.keras.merge import parse_merge_layer
from hls4ml.converters.keras.recurrent import parse_bidirectional_layer, parse_rnn_layer
from hls4ml.converters.keras.reshaping import parse_zeropadding2d_layer


class SyntheticReader:
    """Serves randomly initialized weights of predefined shapes to the v2 handlers.

    Shapes are matched against the end of the requested variable name, so nested
    paths (e.g. 'forward_lstm/lstm_cell/kernel') resolve like plain names.
    """

    def __init__(self, shapes):
        self.shapes = shapes

    def get_weights_data(self, layer_name, var_name):
        for suffix, shape in self.shapes.items():
            if var_name.endswith(suffix):
                return None if shape is None else np.random.rand(*shape).astype('float32')
        return None


def make_conv2d_config(**overrides):
    config = {
        'name': 'conv',
        'filters': 4,
        'kernel_size': [3, 3],
        'strides': [1, 1],
        'padding': 'valid',
        'data_format': 'channels_last',
        'dilation_rate': [1, 1],
        'groups': 1,
    }
    config.update(overrides)
    return {'class_name': 'Conv2D', 'config': config}


def make_rnn_config(class_name='LSTM', **overrides):
    config = {
        'name': f'{class_name.lower()}_test',
        'units': 4,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'use_bias': True,
        'return_sequences': False,
        'return_state': False,
    }
    if class_name == 'GRU':
        config['reset_after'] = True
    config.update(overrides)
    return {'class_name': class_name, 'config': config}


def make_bidirectional_config(sub_layer, merge_mode='concat'):
    return {
        'class_name': 'Bidirectional',
        'config': {
            'name': 'bidir_test',
            'merge_mode': merge_mode,
            'layer': sub_layer,
        },
    }


# ----- Convolution ----- #


def test_conv1d_parses_dilation():
    # No kernel computes dilation yet; the parser stores the attribute (a field the conv
    # config structs already have) and the backends are responsible for rejecting it
    keras_layer = {
        'class_name': 'Conv1D',
        'config': {
            'name': 'dilated_conv',
            'data_format': 'channels_last',
            'filters': 4,
            'kernel_size': [3],
            'strides': [1],
            'padding': 'valid',
            'dilation_rate': [2],
        },
    }
    reader = SyntheticReader({'kernel': (3, 1, 4), 'bias': (4,)})
    layer, output_shape = parse_conv1d_layer(keras_layer, ['input'], [[None, 64, 1]], reader)
    assert layer['dilation'] == 2
    # effective kernel extent is (3-1)*2+1 = 5, so valid padding gives 64-5+1 = 60
    assert output_shape == [None, 60, 4]
    assert layer['filt_width'] == 3


def test_conv2d_parses_dilation():
    reader = SyntheticReader({'kernel': (3, 3, 3, 4), 'bias': (4,)})
    layer, output_shape = parse_conv2d_layer(make_conv2d_config(dilation_rate=[2, 2]), None, [[None, 16, 16, 3]], reader)
    assert layer['dilation'] == 2
    assert output_shape == [None, 12, 12, 4]
    assert (layer['filt_height'], layer['filt_width']) == (3, 3)


def test_conv2d_rejects_asymmetric_dilation():
    # The conv config structs carry a single dilation value
    with pytest.raises(NotImplementedError, match='dilation'):
        parse_conv2d_layer(make_conv2d_config(dilation_rate=[2, 3]), None, [[None, 16, 16, 3]], Mock())


def test_conv2d_rejects_grouped():
    # groups that are not depthwise-shaped (groups == in_channels == filters) have no kernel
    reader = SyntheticReader({'kernel': (3, 3, 2, 8), 'bias': (8,)})
    with pytest.raises(NotImplementedError, match='[Gg]rouped convolution'):
        parse_conv2d_layer(make_conv2d_config(filters=8, groups=2), None, [[None, 16, 16, 4]], reader)


def test_conv2d_routes_depthwise_shaped_groups():
    # groups == in_channels == filters is a depthwise convolution
    reader = SyntheticReader({'kernel': (3, 3, 1, 4), 'bias': (4,)})
    layer, output_shape = parse_conv2d_layer(make_conv2d_config(filters=4, groups=4), None, [[None, 16, 16, 4]], reader)
    assert layer['class_name'] == 'DepthwiseConv2D'
    assert layer['depthwise_data'].shape == (3, 3, 4, 1)
    assert 'weight_data' not in layer
    assert output_shape == [None, 14, 14, 4]


# ----- ReLU layer options ----- #


@pytest.mark.parametrize(
    'config_overrides, expected_class, expected_param',
    [
        ({}, 'Activation', None),
        ({'negative_slope': 0.25}, 'LeakyReLU', 0.25),
        ({'threshold': 0.5}, 'ThresholdedReLU', 0.5),
        ({'max_value': 6.0}, 'ClippedReLU', 6.0),
    ],
)
def test_relu_option_mapping(config_overrides, expected_class, expected_param):
    keras_layer = {'class_name': 'ReLU', 'config': {'name': 'relu_test', **config_overrides}}
    layer, _ = parse_activation_layer(keras_layer, None, [[None, 8]], Mock())
    assert layer['class_name'] == expected_class
    if expected_param is not None:
        assert layer['activ_param'] == expected_param


def test_relu_rejects_max_value_combo():
    keras_layer = {'class_name': 'ReLU', 'config': {'name': 'relu_test', 'max_value': 6.0, 'threshold': 0.5}}
    with pytest.raises(NotImplementedError, match='max_value'):
        parse_activation_layer(keras_layer, None, [[None, 8]], Mock())


# ----- Recurrent ----- #


def test_rnn_no_bias_substitutes_zeros():
    reader = SyntheticReader({'kernel': (8, 16), 'recurrent_kernel': (4, 16), 'bias': None})
    layer, _ = parse_rnn_layer(make_rnn_config('LSTM', use_bias=False), None, [[None, 5, 8]], reader)
    assert layer['bias_data'].shape == (16,)
    assert not layer['bias_data'].any()


def test_gru_reset_after_false_parsed():
    # The kernels implement only the reset_after=True formulation; the parser represents
    # 'before' faithfully and the backends are responsible for rejecting it
    reader = SyntheticReader({'kernel': (8, 12), 'recurrent_kernel': (4, 12), 'bias': (12,)})
    layer, _ = parse_rnn_layer(make_rnn_config('GRU', reset_after=False), None, [[None, 5, 8]], reader)
    assert layer['apply_reset_gate'] == 'before'
    assert layer['bias_data'].shape == (12,)
    assert layer['recurrent_bias_data'].shape == (12,)
    assert not layer['recurrent_bias_data'].any()


def test_rnn_rejects_go_backwards():
    with pytest.raises(NotImplementedError, match='go_backwards'):
        parse_rnn_layer(make_rnn_config('LSTM', go_backwards=True), None, [[None, 5, 8]], Mock())


# ----- Bidirectional ----- #


def test_bidirectional_merge_mode_n_out():
    # Non-concat merge modes keep the sub-layer width; no kernel merges yet and the
    # backends are responsible for rejecting non-concat modes
    reader = SyntheticReader({'kernel': (8, 16), 'recurrent_kernel': (4, 16), 'bias': (16,)})
    layer, output_shape = parse_bidirectional_layer(
        make_bidirectional_config(make_rnn_config('LSTM'), merge_mode='sum'), None, [[None, 5, 8]], reader
    )
    assert layer['merge_mode'] == 'sum'
    assert layer['n_out'] == 4
    assert output_shape == [None, 4]


def test_bidirectional_concat_n_out():
    reader = SyntheticReader({'kernel': (8, 16), 'recurrent_kernel': (4, 16), 'bias': (16,)})
    layer, output_shape = parse_bidirectional_layer(
        make_bidirectional_config(make_rnn_config('LSTM')), None, [[None, 5, 8]], reader
    )
    assert layer['merge_mode'] == 'concat'
    assert layer['n_out'] == 8
    assert output_shape == [None, 8]


def test_bidirectional_rejects_merge_mode_none():
    # merge_mode=None returns two separate output tensors, which the layer cannot represent
    with pytest.raises(NotImplementedError, match='merge_mode'):
        parse_bidirectional_layer(
            make_bidirectional_config(make_rnn_config('LSTM'), merge_mode=None), None, [[None, 5, 8]], Mock()
        )


def test_bidirectional_rejects_simple_rnn():
    # The backend implements only LSTM/GRU cells
    with pytest.raises(NotImplementedError, match='LSTM or GRU'):
        parse_bidirectional_layer(make_bidirectional_config(make_rnn_config('SimpleRNN')), None, [[None, 5, 8]], Mock())


# ----- Normalization, embedding, merge, padding ----- #


def test_batchnorm_rejects_non_channel_axis():
    keras_layer = {'class_name': 'BatchNormalization', 'config': {'name': 'bn_test', 'axis': [1]}}
    with pytest.raises(NotImplementedError, match='axis'):
        parse_batchnorm_layer(keras_layer, None, [[None, 8, 4]], Mock())


def test_embedding_rejects_mask_zero():
    keras_layer = {
        'class_name': 'Embedding',
        'config': {'name': 'embed_test', 'input_dim': 16, 'output_dim': 3, 'mask_zero': True},
    }
    with pytest.raises(NotImplementedError, match='mask_zero'):
        parse_embedding_layer(keras_layer, None, [[None, 4]], Mock())


def test_dot_rejects_normalize():
    # Cosine similarity was silently computed as a plain dot product
    keras_layer = {'class_name': 'Dot', 'config': {'name': 'dot_test', 'axes': 1, 'normalize': True}}
    with pytest.raises(NotImplementedError, match='normalize'):
        parse_merge_layer(keras_layer, None, [[None, 8], [None, 8]], Mock())


def test_zeropadding2d_symmetric_tuple():
    # A typo wrote pad_bottom twice and never set pad_right for the (sym_h, sym_w) form
    keras_layer = {
        'class_name': 'ZeroPadding2D',
        'config': {'name': 'zp_test', 'padding': (1, 2), 'data_format': 'channels_last'},
    }
    layer, output_shape = parse_zeropadding2d_layer(keras_layer, None, [[None, 8, 8, 3]], Mock())
    assert (layer['pad_top'], layer['pad_bottom'], layer['pad_left'], layer['pad_right']) == (1, 1, 2, 2)
    assert output_shape == [None, 10, 12, 3]
