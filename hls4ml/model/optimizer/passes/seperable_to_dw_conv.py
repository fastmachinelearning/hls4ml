"""
This optimizer converts a seperable convolution to a depthwise followed by a regular convolution.
For backends with a custom pointwise implementations the regular convolution will subsequently
be converted to a pointwise convolution by a different optimizer.
"""

import copy

from hls4ml.model.layers import SeparableConv1D, SeparableConv2D
from hls4ml.model.optimizer import OptimizerPass


class SeperableToDepthwiseAndConv(OptimizerPass):
    """Convert Seperable to DepthwiseConv + Conv (potentially later Pointwise)"""

    _dw_attributes = (
        'in_width',
        'out_width',
        'n_chan',
        'depth_multiplier',
        'pad_left',
        'pad_right',
        'filt_width',
        'stride_width',
        'dilation_width',
        'in_height',
        'out_height',
        'pad_top',
        'pad_bottom',
        'filt_height',
        'stride_height',
        'dilation_height',
        'data_format',
        'depthwise_data',
        'depthwise_quantizer',
    )

    _pw_attributes = ('out_width', 'n_filt', 'dilation_width', 'out_height', 'dilation_height', 'data_format', 'use_bias')

    def match(self, node):
        return isinstance(node, (SeparableConv1D, SeparableConv2D))

    def transform(self, model, node):
        dim = node.__class__.__name__[-2:]  # '1D' or '2D'

        # get the layer configuration name
        layer_config = model.config.get_layer_config(node)

        # First do depthwise
        dw_name = f'{node.name}_depthwise'

        # now the layer config (so that set configuration get copied)
        dw_layer_config = copy.deepcopy(layer_config)

        if dw_layer_config:
            dw_precision_cfg = dw_layer_config.setdefault('Precision', {})
            if isinstance(dw_precision_cfg, dict):
                if 'depthwise' in dw_precision_cfg:
                    dw_precision_cfg['weight'] = dw_precision_cfg['depthwise']
                    del dw_precision_cfg['depthwise']
                if 'depthwise_accum' in dw_precision_cfg:
                    dw_precision_cfg['accum'] = dw_precision_cfg['depthwise_accum']
                    del dw_precision_cfg['depthwise_accum']
                if 'depthwise_result' in dw_precision_cfg:
                    dw_precision_cfg['result'] = dw_precision_cfg['depthwise_result']
                    del dw_precision_cfg['depthwise_result']
                dw_precision_cfg.pop('pointwise', None)
                dw_precision_cfg.pop('pointwise_accum', None)
            model.config.set_name_config(dw_name, dw_layer_config)
            model.config.parse_name_config(dw_name, dw_layer_config)

        # creating the attributes
        dw_attributes = {k: node.attributes[k] for k in SeperableToDepthwiseAndConv._dw_attributes if k in node.attributes}
        dw_attributes['n_filt'] = dw_attributes['n_chan'] * dw_attributes['depth_multiplier']
        dw_attributes['use_bias'] = False

        new_dw = model.make_node('DepthwiseConv' + dim, dw_name, dw_attributes, [node.inputs[0]])

        # Then do convolution
        pw_name = f'{node.name}_pointwise'

        # now the layer config (so that set configuration get copied)
        pw_layer_config = copy.deepcopy(layer_config)

        if pw_layer_config:
            pw_precision_cfg = pw_layer_config.setdefault('Precision', {})
            if isinstance(pw_precision_cfg, dict):
                if 'pointwise' in pw_precision_cfg:
                    pw_precision_cfg['weight'] = pw_precision_cfg['pointwise']
                    del pw_precision_cfg['pointwise']
                if 'pointwise_accum' in pw_precision_cfg:
                    pw_precision_cfg['accum'] = pw_precision_cfg['pointwise_accum']
                    del pw_precision_cfg['pointwise_accum']
                if 'pointwise_result' in pw_precision_cfg:
                    pw_precision_cfg['result'] = pw_precision_cfg['pointwise_result']
                    del pw_precision_cfg['pointwise_result']
                pw_precision_cfg.pop('depthwise', None)
                pw_precision_cfg.pop('depthwise_accum', None)
            model.config.set_name_config(pw_name, pw_layer_config)
            model.config.parse_name_config(pw_name, pw_layer_config)

        # creating the attributes
        pw_attributes = {k: node.attributes[k] for k in SeperableToDepthwiseAndConv._pw_attributes if k in node.attributes}
        pw_attributes['filt_width'] = 1
        pw_attributes['filt_height'] = 1
        pw_attributes['stride_width'] = 1
        pw_attributes['stride_height'] = 1
        pw_attributes['pad_left'] = 0
        pw_attributes['pad_right'] = 0
        pw_attributes['pad_top'] = 0
        pw_attributes['pad_bottom'] = 0
        pw_attributes['in_width'] = pw_attributes['out_width']
        pw_attributes['in_height'] = pw_attributes.get('out_height', 1)
        pw_attributes['n_chan'] = node.get_attr('n_chan') * node.get_attr('depth_multiplier')
        pw_attributes['weight_data'] = node.get_attr('pointwise_data')
        pw_attributes['weight_quantizer'] = node.get_attr('pointwise_quantizer')
        pw_attributes['bias_data'] = node.get_attr('bias_data')
        pw_attributes['bias_quantizer'] = node.get_attr('bias_quantizer')

        # note this is just regular convolution. It is replaced by a special pointwise implementation
        # if available by another optimizer
        new_pw = model.make_node('Conv' + dim, pw_name, pw_attributes, [dw_name])

        model.split_node(node, new_dw, new_pw)

        return True
