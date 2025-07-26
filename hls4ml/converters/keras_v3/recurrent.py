import typing
from collections.abc import Sequence

import numpy as np

from ._base import KerasV3LayerHandler, register

if typing.TYPE_CHECKING:
    import keras
    from keras import KerasTensor

rnn_layers = ('SimpleRNN', 'LSTM', 'GRU')


@register
class RecurentHandler(KerasV3LayerHandler):
    handles = (
        'keras.src.layers.rnn.simple_rnn.SimpleRNN',
        'keras.src.layers.rnn.lstm.LSTM',
        'keras.src.layers.rnn.gru.GRU',
    )

    def handle(
        self,
        layer: 'keras.layers.SimpleRNN|keras.layers.LSTM|keras.layers.GRU',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        import keras

        if layer.return_state:
            raise Exception(f'return_state=True is not supported for {layer.__class__} layer.')

        config = {
            'direction': 'forward',
            'return_sequences': layer.return_sequences,
            'return_state': layer.return_state,
            'time_major': False,
            'n_timesteps': in_tensors[0].shape[1],
            'n_in': in_tensors[0].shape[2],
            'n_out': layer.units,
        }

        config['weight_data'] = self.load_weight(layer.cell, 'kernel')
        config['recurrent_weight_data'] = self.load_weight(layer.cell, 'recurrent_kernel')

        if layer.use_bias:
            if isinstance(layer, keras.layers.GRU):
                bias = self.load_weight(layer.cell, 'bias')
                config['bias_data'] = bias[0]
                config['recurrent_bias_data'] = bias[1]
            else:
                config['bias_data'] = self.load_weight(layer.cell, 'bias')
        else:
            d_out = config['weight_data'].shape[-1]
            config['bias_data'] = np.zeros((d_out,), dtype=np.float32)
            if isinstance(layer, keras.layers.GRU):
                config['recurrent_bias_data'] = np.zeros((d_out,), dtype=np.float32)

        if isinstance(layer, keras.layers.GRU):
            config['apply_reset_gate'] = 'after' if layer.reset_after else 'before'

        if hasattr(layer, 'activation'):
            config['activation'] = layer.activation.__name__
        if hasattr(layer, 'recurrent_activation'):
            config['recurrent_activation'] = layer.recurrent_activation.__name__

        _config = {}
        _config.update(self.default_config)
        _config.update(config)

        return (_config,)


@register
class BidirectionalHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.rnn.bidirectional.Bidirectional',)

    def handle(
        self,
        layer: 'keras.layers.Bidirectional',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        import keras

        if layer.forward_layer.go_backwards:
            rnn_forward_layer = layer.backward_layer
            rnn_backward_layer = layer.forward_layer
            print(
                f'WARNING: The selected order for forward and backward layers in \"{layer.name}\" '
                f'({layer.__class__.__name__}) is not supported. Switching to forward layer first, backward layer last.'
            )
        else:
            rnn_forward_layer = layer.forward_layer
            rnn_backward_layer = layer.backward_layer

        for rnn_layer in [rnn_forward_layer, rnn_backward_layer]:
            class_name = rnn_layer.__class__.__name__
            assert class_name in rnn_layers or class_name[1:] in rnn_layers

        config = {}

        config['direction'] = 'bidirectional'
        config['return_sequences'] = layer.return_sequences
        config['return_state'] = layer.return_state
        config['time_major'] = False

        if config['return_state']:
            raise Exception(f'return_state=True is not supported for {layer.__class__} layer.')

        config['n_timesteps'] = in_tensors[0].shape[1]
        config['n_in'] = in_tensors[0].shape[2]
        config['merge_mode'] = layer.merge_mode

        for direction, rnn_layer in [('forward', rnn_forward_layer), ('backward', rnn_backward_layer)]:

            config[f'{direction}_name'] = rnn_layer.name
            config[f'{direction}_class_name'] = rnn_layer.__class__.__name__
            if hasattr(rnn_layer, 'activation'):
                config[f'{direction}_activation'] = rnn_layer.activation.__name__
            if hasattr(rnn_layer, 'recurrent_activation'):
                config[f'{direction}_recurrent_activation'] = rnn_layer.recurrent_activation.__name__

            # config[f'{direction}_data_format'] = getattr(rnn_layer, 'data_format', 'channels_last')
            if hasattr(rnn_layer, 'use_bias'):
                config[f'{direction}_use_bias'] = rnn_layer.use_bias

            config[f'{direction}_weight_data'] = self.load_weight(rnn_layer.cell, 'kernel')
            config[f'{direction}_recurrent_weight_data'] = self.load_weight(rnn_layer.cell, 'recurrent_kernel')

            if rnn_layer.use_bias:
                if isinstance(rnn_layer.cell, keras.layers.GRU):
                    bias = self.load_weight(rnn_layer.cell, 'bias')
                    config[f'{direction}_bias_data'] = bias[0]
                    config[f'{direction}_recurrent_bias_data'] = bias[1]
                else:
                    config[f'{direction}_bias_data'] = self.load_weight(rnn_layer.cell, 'bias')
            else:
                d_out = config[f'{direction}_weight_data'].shape[-1]
                config[f'{direction}_bias_data'] = np.zeros(d_out, dtype=np.float32)
                if isinstance(rnn_layer, keras.layers.GRU):
                    config[f'{direction}_recurrent_bias_data'] = np.zeros(d_out, dtype=np.float32)

            if isinstance(rnn_layer, keras.layers.GRU):
                config[f'{direction}_apply_reset_gate'] = 'after' if rnn_layer.reset_after else 'before'

            config[f'{direction}_n_states'] = rnn_layer.units

        if config['merge_mode'] == 'concat':
            config['n_out'] = config['forward_n_states'] + config['backward_n_states']
        else:
            config['n_out'] = config['forward_n_states']

        return config
