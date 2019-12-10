from __future__ import print_function
import numpy as np
import h5py
import json
import math
import os
import tensorflow as tf
from tensorflow.python.framework import tensor_util

from hls4ml.model import HLSModel
from hls4ml.model.optimizer import optimize_model

MAXMULT = 4096

class TFDataReader:
    def __init__(self, graph):
        self.graph = graph
        self.const_ops = [c for c in self.graph.get_operations() if c.type == 'Const']

    def get_weights_data(self, layer_name, var_name):
        tf_op = self.graph.get_operation_by_name(layer_name)
        data = None
        if tf_op is not None:
            if tf_op.type == 'MatMul' and var_name == 'kernel': # MatMul is mapped to Dense, but there is no bias
                w_tensor = tf_op.inputs[1]
                data = self._read_variable_data(w_tensor)

            if tf_op.type == 'BiasAdd':
                b_tensor = tf_op.inputs[1]
                data = self._read_variable_data(b_tensor)

        return data

    def _read_variable_data(self, tensor):
        parent_op = tensor.op
        while parent_op.type != 'Const':
            tensor = parent_op.inputs[0]
            parent_op = tensor.op

        data = tensor_util.MakeNdarray(parent_op.node_def.attr['value'].tensor)

        return data


def get_weights_shape(h5filename, layer_name, var_name='kernel'):
    def h5_visitor_func(name):
        if var_name in name:
            return name

    with h5py.File(h5filename, 'r') as h5file:
        found_data = h5file[layer_name].visit(h5_visitor_func)
        if found_data:
            shape = h5file[layer_name][found_data].shape

    return shape

def _find_graph_outputs(graph):
    input_list = []
    output_list = []

    for tf_op in graph.get_operations():
        input_list.extend(tf_op.inputs)
        output_list.extend(tf_op.outputs)

    return [o.name.rsplit(':', 1)[0] for o in list(set(output_list) - set(input_list))]

def _parse_tensor_names(tensors):
    if not isinstance(tensors, list):
        tensors = [tensors]

    tensor_names = []
    for t in tensors:
        tensor_names.append(t.name.rsplit(':', 1)[0])

    return tensor_names

def tf_to_hls(yamlConfig):

    ######################
    ##  Do translation
    ######################

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    if not os.path.exists(yamlConfig['TensorFlowModel']):
        raise Exception('The specified file does not exist: {}'.format(yamlConfig['TensorFlowModel']))

    graph_def = None
    graph = None

    #Extract model architecture from pb
    try:
        #with tf.io.gfile.GFile(yamlConfig['TensorFlowModel'], "rb") as f:
        with tf.gfile.GFile(yamlConfig['TensorFlowModel'], "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
    except BaseException as e:
        raise Exception('Error loading the graph definition: {}'.format(str(e)))

    try:
        assert graph_def is not None
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name='',
                producer_op_list=None
            )
    except BaseException as e:
        raise Exception('Error importing the graph: {}'.format(str(e)))

    #Define supported operations
    supported_ops = [
        'Placeholder', 'Const', 'Identity', 'MatMul', 'BiasAdd', 'Relu', 'Softmax'
    ]

    input_layers = []
    output_layers = _find_graph_outputs(graph)

    # Get input shape and check for unsupported layer type
    current_shape = None
    for tf_op in graph.get_operations():
        if tf_op.type not in supported_ops:
            raise Exception('ERROR: Unsupported layer type: {}'.format(tf_op.type))
        #if 'batch_input_shape' in tf_op['config']:
        #    current_shape = tf_op['config']['batch_input_shape'] # [None, 100, 7]

    print('Topology:')
    for tf_op in graph.get_operations():
        handled = False

        layer = {}
        layer['name'] = tf_op.name

        if tf_op.type == 'Placeholder':
            if len(tf_op.inputs) == 0: # Input
                current_shape = tf_op.outputs[0].shape.as_list()
                layer['class_name'] = 'InputLayer'
                layer['input_shape'] = current_shape[1:]
                #layer['outputs'] = [tf_op.outputs[0].name for o in tf_op.outputs]
                layer['outputs'] = _parse_tensor_names(tf_op.outputs)
                input_layers.append(layer['name'])
                handled = True

        elif tf_op.type == 'Const' or tf_op.type == 'Identity':
            # Nothing to do here, TFDataReader handles these
            handled = True
            continue

        elif tf_op.type == 'MatMul':
            input_shape = tf_op.inputs[0].shape.as_list()
            output_shape = tf_op.outputs[0].shape.as_list()
            layer['class_name'] = 'Dense'
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])
            layer['n_in'] = input_shape[-1]
            layer['n_out'] = output_shape[-1]
            current_shape = output_shape
            handled = True

        elif tf_op.type == 'BiasAdd':
            input_shape = tf_op.inputs[0].shape.as_list()
            output_shape = tf_op.outputs[0].shape.as_list()
            layer['class_name'] = 'BiasAdd'
            layer['op'] = 'Add'
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])
            current_shape = output_shape
            handled = True

        elif tf_op.type == 'Relu' or tf_op.type == 'Softmax':
            output_shape = tf_op.outputs[0].shape.as_list()
            layer['class_name'] = 'Activation'
            layer['activation'] = tf_op.type
            layer['inputs'] = _parse_tensor_names(tf_op.inputs[0])
            layer['outputs'] = _parse_tensor_names(tf_op.outputs[0])
            current_shape = output_shape
            handled = True

        if not handled:
            raise Exception('Unable to parse operation: {} - {}'.format(tf_op.type, tf_op.name))

        print('Layer name: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], current_shape))
        layer_list.append(layer)

    #################
    ## Generate HLS
    #################

    reader = TFDataReader(graph)
    print('Creating HLS model')
    hls_model = HLSModel(yamlConfig, reader, layer_list, input_layers, output_layers)
    optimizers = ['eliminate_linear_activation', 'merge_batch_norm_quantized_tanh', 'quantize_dense_output', 'fuse_biasadd', 'fuse_dense_batch_norm']
    optimize_model(hls_model, optimizers)
    return hls_model
