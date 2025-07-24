import typing
from collections.abc import Sequence

from hls4ml.converters.keras.recurrent import parse_rnn_layer

from ._base import KerasV3LayerHandler, register

if typing.TYPE_CHECKING:
    import keras
    from keras import KerasTensor

rnn_layers = ['SimpleRNN', 'LSTM', 'GRU']
weight_dict = {'kernel': 'weight', 'recurrent_kernel': 'recurrent_weight', 'bias': 'bias'}


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
        import numpy as np

        layer_config = layer.get_config()
        layer_dict = {'config': layer_config, 'class_name': layer.__class__.__name__}
        module = layer.__module__

        class IsolatedLayerReader:
            def get_weights_data(self, layer_name, var_name):
                assert layer_name == layer.name, f'Processing {layer.name}, but handler tried to read {layer_name}'
                for w in layer.weights:
                    if var_name in w.name:
                        return np.array(w)
                return None

        reader = IsolatedLayerReader()
        input_shapes = [list(t.shape) for t in in_tensors]
        input_names = [t.name for t in in_tensors]
        output_names = [t.name for t in out_tensors]

        config, _ = parse_rnn_layer(layer_dict, input_names, input_shapes, reader)
        config['module'] = module
        config['input_keras_tensor_names'] = input_names
        config['input_shape'] = input_shapes
        config['output_keras_tensor_names'] = output_names

        return (config,)


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
        config['name'] = layer.name
        config['class_name'] = layer.__class__.__name__

        config['direction'] = 'bidirectional'
        config['return_sequences'] = layer.return_sequences
        config['return_state'] = layer.return_state
        config['time_major'] = getattr(layer, 'time_major', False)
        # TODO Should we handle time_major?
        if config['time_major']:
            raise Exception('Time-major format is not supported by hls4ml')

        config['n_timesteps'] = in_tensors[0].shape[1]
        config['n_in'] = in_tensors[0].shape[2]
        config['merge_mode'] = layer.merge_mode

        for direction, rnn_layer in [('forward', rnn_forward_layer), ('backward', rnn_backward_layer)]:

            config[f'{direction}_name'] = rnn_layer.name
            config[f'{direction}_class_name'] = rnn_layer.__class__.__name__
            if hasattr(rnn_layer, 'activation'):
                config[f'{direction}_activation'] = rnn_layer.activation.__name__
            if 'SimpleRNN' not in rnn_layer.__class__.__name__:
                config[f'{direction}_recurrent_activation'] = rnn_layer.recurrent_activation.__name__

            config[f'{direction}_data_format'] = getattr(rnn_layer, 'data_format', 'channels_last')
            if hasattr(rnn_layer, 'epsilon'):
                config[f'{direction}_epsilon'] = rnn_layer.epsilon
            if hasattr(rnn_layer, 'use_bias'):
                config[f'{direction}_use_bias'] = rnn_layer.use_bias

            for w in rnn_layer.weights:
                name = w.name.replace('kernel', 'weight') if 'kernel' in w.name else w.name
                config[f'{direction}_{name}_data'] = keras.ops.convert_to_numpy(w)

            if 'GRU' in rnn_layer.__class__.__name__:
                config[f'{direction}_apply_reset_gate'] = 'after' if rnn_layer.reset_after else 'before'

                # biases array is actually a 2-dim array of arrays (bias + recurrent bias)
                # both arrays have shape: n_units * 3 (z, r, h_cand)
                biases = config[f'{direction}_bias_data']
                config[f'{direction}_bias_data'] = biases[0]
                config[f'{direction}_recurrent_bias_data'] = biases[1]

            config[f'{direction}_n_states'] = rnn_layer.units

        if config['merge_mode'] == 'concat':
            config['n_out'] = config['forward_n_states'] + config['backward_n_states']
        else:
            config['n_out'] = config['forward_n_states']

        if config['return_state']:
            raise Exception('"return_state" of {} layer is not yet supported.')

        return config
