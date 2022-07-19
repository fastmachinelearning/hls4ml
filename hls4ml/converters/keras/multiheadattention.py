from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

@keras_handler('MultiHeadAttention')
def parse_conv1d_layer(keras_layer, input_names, input_shapes, data_reader, config):
    # assume input_shapes is: [[None, seq, dim]]
    assert('MultiHeadAttention' in keras_layer['class_name'])
    assert(input_shapes[0]==keras_layer['config']['query_shape'])
    
    layer = parse_default_keras_layer(keras_layer, input_names)
    
    layer['num_heads'] = keras_layer['config']['num_heads']
    layer['head_dim_key'] = keras_layer['config']['key_dim']
    layer['head_dim_value'] = keras_layer['config']['value_dim']
    layer['query_shape'] = keras_layer['config']['query_shape']
    layer['key_shape'] = keras_layer['config']['key_shape']
    layer['value_shape'] = keras_layer['config']['value_shape']
    layer['feature_dim'] = layer['query_shape'][-1]
    layer['seq_len'] = layer['query_shape'][-2]

    if keras_layer['config']['output_shape']: 
        # output_shape = keras_layer['config']['output_shape']
        # output_shape = (layer['query_shape'][:2]).extend(out_shape)
        raise Exception('hls4ml does not support a defined output shape, the output shape must equal to the query shape')
    else:  # by default output shape in config is False, and thus select the output shape equal query shape
        output_shape = layer['query_shape']
    
    layer['attention_axes'] = keras_layer['config']['attention_axes'] if (keras_layer['config']['attention_axes'][0]==1) else False
    if layer['attention_axes'] is False: 
        raise Exception('assigning the attention_axe is not currently supported by hls4ml')

    if not((len(layer['query_shape'])) == 3 and (len(layer['query_shape'])) == 3 and (len(layer['query_shape'])) == 3):
        raise Exception('muti-dimension of feature dim is not currently supported by hls4ml')

    attn_scores_rank = 4
    layer['softmax_axis'] = list(range(attn_scores_rank - len(layer['attention_axes']), attn_scores_rank ))

    return layer, output_shape