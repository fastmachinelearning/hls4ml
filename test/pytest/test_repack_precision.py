from tensorflow import keras

from hls4ml.converters import convert_from_keras_model


def test_repack_precision():
    inp = keras.Input(shape=(3, 3), name='inp')
    out = keras.layers.Reshape((3, 3), name='reshape')(inp)
    out = keras.layers.Conv1D(2, 2, name='conv')(out)
    model = keras.Model(inp, out)

    layer_conf = {
        'inp': {'Precision': 'fixed<20,10>'},
        'reshape': {'Precision': 'fixed<20,10>'},
        'conv': {'Precision': 'fixed<20,10>'},
    }

    hls_config = {'Model': {'Precision': 'fixed<2,1>', 'ReuseFactor': 1}, 'LayerName': layer_conf}

    # Repack only happens in io_stream
    model_hls = convert_from_keras_model(model, hls_config=hls_config, io_type='io_stream')
    assert 'repack_reshape' in model_hls.graph, 'repack_reshape not found in graph'
    repack_precision = model_hls.graph['repack_reshape'].attributes['result_t'].precision
    assert repack_precision.integer == 10, 'Precision mismatch'
    assert repack_precision.fractional == 10, 'Precision mismatch'
    assert repack_precision.width == 20, 'Precision mismatch'
    assert repack_precision.signed is True, 'Precision mismatch'
