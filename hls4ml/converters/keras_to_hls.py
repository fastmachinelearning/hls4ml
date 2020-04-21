from __future__ import print_function
import numpy as np
import h5py
import json
import math

from hls4ml.model import HLSModel
from hls4ml.model.optimizer import optimize_model

MAXMULT = 4096

class KerasFileReader:
    def __init__(self, config):
        self.config = config
        self.h5file = h5py.File(config['KerasH5'], mode='r')

    def __del__(self):
        if self.h5file:
            self.h5file.close()

    def _find_data(self, layer_name, var_name):
        def h5_visitor_func(name):
            if var_name in name:
                return name

        if 'model_weights' in list(self.h5file.keys()): # h5 file comes from model.save()
            layer_path = 'model_weights/{}'.format(layer_name)
        else:
            layer_path = layer_name

        data_path = self.h5file[layer_path].visit(h5_visitor_func)
        if data_path:
            return self.h5file['/{}/{}'.format(layer_path, data_path)]
        else:
            return None

    def get_weights_data(self, layer_name, var_name):
        data = self._find_data(layer_name, var_name)
        if data:
            return data[()]
        else:
            return None

    def get_weights_shape(self, layer_name, var_name):
        data = self._find_data(layer_name, var_name)
        if data is not None:
            return data.shape
        else:
            return None

class KerasModelReader:
    def __init__(self, keras_model):
        self.model = keras_model

    def get_weights_data(self, layer_name, var_name):
        layer = self.model.get_layer(layer_name)
        for i, w in enumerate(layer.weights):
            if var_name in w.name:
                try:
                    return w.numpy() # TF 2.x
                except:
                    return layer.get_weights()[i] # TF 1.x

        return None

    def get_weights_shape(self, layer_name, var_name):
        layer = self.model.get_layer(layer_name)
        for w in layer.weights:
            if var_name in w.name:
                return w.shape.as_list()

        return None

def get_qkeras_quantization(layer, keras_layer):
    if not layer['class_name'].startswith('Q'): # Not a QKeras layer, nothing to do
        return
    kernel_quantizer = keras_layer['config']['kernel_quantizer']['class_name']
    bias_quantizer = keras_layer['config']['bias_quantizer']['class_name']

    if kernel_quantizer != bias_quantizer:
        raise Exception('Mixing quantizers within QKeras layers is not supported')
    if kernel_quantizer == 'binary':
        layer['quantize'] = 2
    elif kernel_quantizer == 'ternary':
        layer['quantize'] = 3
    else:
        raise Exception('Unsupported quantizer {} in {} layer {}'.format(kernel_quantizer, layer['class_name'], layer['name']))

def keras_to_hls(yamlConfig):

    ######################
    ##  Do translation
    ######################

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    if 'KerasModel' in yamlConfig:
        # Model instance passed in config from API
        model_arch = json.loads(yamlConfig['KerasModel'].to_json())
        reader = KerasModelReader(yamlConfig['KerasModel'])
    elif 'KerasJson' in yamlConfig:
        # Extract model architecture from json
        with open( yamlConfig['KerasJson'] ) as json_file:
            model_arch = json.load(json_file)
        reader = KerasFileReader(yamlConfig)
    elif 'KerasH5' in yamlConfig:
        # Model arch and weights are in H5 file (from model.save() function)
        with h5py.File(yamlConfig['KerasH5'], mode='r') as h5file:
            # Load the configuration from h5 using json's decode
            model_arch = h5file.attrs.get('model_config')
            if model_arch is None:
                raise ValueError('No model found in config file.')
            else:
                model_arch = json.loads(model_arch.decode('utf-8'))
        reader = KerasFileReader(yamlConfig)
    else:
        raise ValueError('No model found in config file.')

    #print(model_arch)

    #Define supported laers
    core_layers = ['InputLayer', 'Dropout', 'Flatten', 'Dense', 'BinaryDense', 'TernaryDense', 'Reshape']
    conv_layers = ['Conv1D', 'Conv2D', 'BinaryConv2D']
    pooling_layers = ['MaxPooling1D', 'MaxPooling2D', 'AveragePooling1D', 'AveragePooling2D']
    norm_layers = ['BatchNormalization']
    activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU']
    merge_layers = ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate']
    qkeras_layers = ['QDense', 'QActivation', 'QConv1D', 'QConv2D']
    #Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Flatten']
    #All supported layers
    supported_layers = core_layers + conv_layers + pooling_layers + norm_layers + activation_layers + merge_layers + qkeras_layers + skip_layers

    #Map inputs of skipped and split (activation) layers
    inputs_map = {}

    #Loop through layers
    layer_counter = 0

    input_layers = None
    output_layers = None

    layer_config = None
    if model_arch['class_name'] == 'Sequential':
        print('Interpreting Sequential')
        layer_config = model_arch["config"]
        if 'layers' in layer_config: # Newer Keras versions have 'layers' in 'config' key
            layer_config = layer_config['layers']
        # Sequential doesn't have InputLayer
        input_layer = {}
        input_layer['name'] = 'input1'
        input_layer['class_name'] = 'InputLayer'
        input_layer['input_shape'] = layer_config[0]['config']['batch_input_shape'][1:]
        layer_list.append(input_layer)
        print('Input shape:', input_layer['input_shape'])
    elif model_arch['class_name'] == 'Model':
        print('Interpreting Model')
        layer_config = model_arch["config"]["layers"]
        input_layers = [ inp[0] for inp in model_arch["config"]["input_layers"] ]
        output_layers = [ out[0] for out in model_arch["config"]["output_layers"] ]

    # Get input shape and check for unsupported layer type
    for keras_layer in layer_config:
        if keras_layer["class_name"] not in supported_layers:
            raise Exception('ERROR: Unsupported layer type: {}'.format(keras_layer["class_name"]))

    output_shapes = {}
    output_shape = None

    print('Topology:')
    for keras_layer in layer_config:
        if 'batch_input_shape' in keras_layer['config']:
            input_shapes = [keras_layer['config']['batch_input_shape']]
        else:
            if 'inbound_nodes' in keras_layer:
                input_shapes = [output_shapes[inbound_node[0][0]] for inbound_node in keras_layer['inbound_nodes']]
            else:
                # Sequential model, so output_shape from the previous layer is still valid 
                input_shapes = [output_shape]

        if keras_layer["class_name"] in skip_layers:
            if 'inbound_nodes' in keras_layer:
                name = keras_layer['config']['name']
                #Currently supported skipped layers have only one input
                parent_input = keras_layer['inbound_nodes'][0][0][0]
                #Skipped layers can follow each other (e.g., Dropout -> Flatten)
                inputs_map[name] = inputs_map.get(parent_input, parent_input)

            if keras_layer["class_name"] == 'Flatten':
                output_shapes[keras_layer['name']] = [input_shapes[0][0], np.prod(input_shapes[0][1:])]
            else:
                output_shapes[keras_layer['name']] = input_shapes[0]

            continue

        if keras_layer["class_name"] in supported_layers:
            layer_counter = layer_counter + 1

        #Dictionary to fill in and append to layer_list
        layer = {}

        #Extract name for finding weights and biases
        layer['name']=keras_layer['config']['name']
        layer['class_name']=keras_layer['class_name']

        #Extract inbound nodes
        if 'inbound_nodes' in keras_layer and len(keras_layer['inbound_nodes']) > 0:
            layer['inputs'] = [ inputs_map.get(inp[0], inp[0]) for inp in keras_layer['inbound_nodes'][0] ]

        #Extract type of activation and number of nodes
        for config,config_value in keras_layer["config"].items():
            if(config=="activation"):
                layer['activation']=config_value
            if(config=="epsilon"):
                layer['epsilon']=config_value
            #if(config=="units"):
                #print("PARSED NUM OF NODES",config_value)

        # Default one layer call
        if layer['class_name'] == 'InputLayer':
            layer['input_shape'] = keras_layer['config']['batch_input_shape'][1:]
            if keras_layer['config']['dtype'] == 'int32':
                layer['type_name'] = 'integer_input_t'
                layer['precision'] = 'ap_int<32>'
            output_shape = keras_layer['config']['batch_input_shape']
        elif keras_layer["class_name"] == 'Reshape':
            layer['target_shape'] = keras_layer['config']['target_shape']
            output_shape = input_shapes[0][:1] + keras_layer['config']['target_shape']
        elif 'Dense' in layer['class_name']:
            weights_shape = reader.get_weights_shape(layer['name'], 'kernel')
            layer['n_in'] = weights_shape[0]
            layer['n_out'] = weights_shape[1]
            if 'Binary' in layer['class_name']:
                layer['quantize'] = 2
            elif 'Ternary' in layer['class_name']:
                layer['quantize'] = 3
            elif layer['class_name'] == 'QDense':
                get_qkeras_quantization(layer, keras_layer)
            else:
                layer['quantize'] = 0
            output_shape = [input_shapes[0][0], layer['n_out']]
        elif layer['class_name']=='Conv1D':
            # weights_shape = (filter_width, n_channels, n_filters)
            weights_shape = reader.get_weights_shape(layer['name'], 'kernel')
            layer['n_in']=input_shapes[0][1]
            layer['filt_width']=weights_shape[0] # or keras_layer['config']['kernel_size']
            layer['n_chan']=weights_shape[1]
            layer['n_filt']=weights_shape[2] # or keras_layer['config']['filters']
            layer['stride']=keras_layer['config']['strides'][0]
            layer['padding']=keras_layer['config']['padding']
            if layer['padding']=='same':
                in_width = input_shapes[0][1]
                layer['n_out'] = int(math.ceil(float(in_width) / float(layer['stride'])))
                if (in_width % layer['stride'] == 0):
                    pad_along_width = max(layer['filt_width'] - layer['stride'], 0)
                else:
                    pad_along_width = max(layer['filt_width'] - (in_width % layer['stride']), 0)
                layer['pad_left']  = pad_along_width // 2
                layer['pad_right']  = pad_along_width - layer['pad_left']
            elif layer['padding']=='valid':
                in_width = input_shapes[0][1]
                layer['n_out'] = int(math.ceil(float(in_width - layer['filt_width'] + 1) / float(layer['stride'])))
                layer['pad_left'] = 0
                layer['pad_right'] = 0
            layer['data_format'] = keras_layer['config'].get('data_format', 'channels_last')
            get_qkeras_quantization(layer, keras_layer)
            output_shape=[input_shapes[0][0], layer['n_out'], layer['n_filt']]
        elif 'Conv2D' in layer['class_name']:
            layer['data_format'] = keras_layer['config'].get('data_format', 'channels_last')
            # weights_shape = (filter_height, filter_width, n_channels, n_filters)
            weights_shape = reader.get_weights_shape(layer['name'], 'kernel')
            layer['in_height']=input_shapes[0][1]
            layer['in_width']=input_shapes[0][2]
            if layer['data_format'] == 'channels_first':
                layer['in_height']=input_shapes[0][2]
                layer['in_width']=input_shapes[0][3]
            layer['filt_height']=weights_shape[0]
            layer['filt_width']=weights_shape[1]
            layer['n_chan']=weights_shape[2]
            layer['n_filt']=weights_shape[3]
            layer['stride_height']=keras_layer['config']['strides'][0]
            layer['stride_width']=keras_layer['config']['strides'][1]
            layer['padding']=keras_layer['config']['padding']
            if layer['padding']=='same':
                #Height
                in_height = input_shapes[0][1]
                if layer['data_format'] == 'channels_first': in_height = input_shapes[0][2]
                layer['out_height'] = int(math.ceil(float(in_height) / float(layer['stride_height'])))
                if (in_height % layer['stride_height'] == 0):
                    pad_along_height = max(layer['filt_height'] - layer['stride_height'], 0)
                else:
                    pad_along_height = max(layer['filt_height'] - (in_height % layer['stride_height']), 0)
                layer['pad_top']  = pad_along_height // 2
                layer['pad_bottom']  = pad_along_height - layer['pad_top']
                #Width
                in_width = input_shapes[0][2]
                if layer['data_format'] == 'channels_first': in_width = input_shapes[0][3]
                layer['out_width'] = int(math.ceil(float(in_width) / float(layer['stride_width'])))
                if (in_width % layer['stride_width'] == 0):
                    pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
                else:
                    pad_along_width = max(layer['filt_width'] - (in_width % layer['stride_width']), 0)
                layer['pad_left']  = pad_along_width // 2
                layer['pad_right']  = pad_along_width - layer['pad_left']
            elif layer['padding']=='valid':
                in_height = input_shapes[0][1]
                in_width = input_shapes[0][2]
                if layer['data_format'] == 'channels_first':
                    in_height = input_shapes[0][2]
                    in_width = input_shapes[0][3]
                layer['out_width'] = int(math.ceil(float(in_width - layer['filt_width'] + 1) / float(layer['stride_width'])))
                layer['out_height'] = int(math.ceil(float(in_height - layer['filt_height'] + 1) / float(layer['stride_height'])))
                layer['pad_top'] = 0
                layer['pad_bottom'] = 0
                layer['pad_left'] = 0
                layer['pad_right'] = 0
            get_qkeras_quantization(layer, keras_layer)
            if layer['data_format'] == 'channels_first': output_shape=[input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]
            else: output_shape=[input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]
        elif layer['class_name']=='BatchNormalization':
            in_size = 1
            for dim in input_shapes[0][1:]:
                in_size *= dim
            layer['n_in'] = in_size
            layer['n_out'] = layer['n_in']
            if len(input_shapes[0]) == 2:
                layer['n_filt'] = -1
            elif len(input_shapes[0]) == 3:
                layer['n_filt']=input_shapes[0][2]
            elif len(input_shapes[0]) == 4:
                layer['n_filt']=input_shapes[0][3]
        elif 'Pooling' in layer['class_name']:
            if int(layer['class_name'][-2]) == 1:
                layer['n_in']=input_shapes[0][1]
                layer['n_filt']=input_shapes[0][2]
                layer['pool_size']=keras_layer['config']['pool_size'][0]
                layer['stride']=keras_layer['config']['strides'][0]
                layer['padding']=keras_layer['config']['padding']
                if layer['padding']=='same':
                    in_width = input_shapes[0][1]
                    layer['n_out'] = int(math.ceil(float(in_width) / float(layer['stride'])))
                    if (in_width % layer['stride'] == 0):
                        pad_along_width = max(layer['pool_size'] - layer['stride'], 0)
                    else:
                        pad_along_width = max(layer['pool_size'] - (in_width % layer['stride']), 0)
                    layer['pad_left']  = pad_along_width // 2
                    layer['pad_right']  = pad_along_width - layer['pad_left']
                elif layer['padding']=='valid':
                    in_width = input_shapes[0][1]
                    layer['n_out'] = int(math.ceil(float(in_width - layer['pool_size'] + 1) / float(layer['stride'])))
                    layer['pad_left'] = 0
                    layer['pad_right'] = 0
                output_shape=[input_shapes[0][0], layer['n_out'], layer['n_filt']]
            elif int(layer['class_name'][-2]) == 2:
                layer['data_format'] = keras_layer['config'].get('data_format', 'channels_last')
                layer['in_height']=input_shapes[0][1]
                layer['in_width']=input_shapes[0][2]
                layer['n_filt']=input_shapes[0][3]
                if layer['data_format'] == 'channels_first':
                    layer['in_height']=input_shapes[0][2]
                    layer['in_width']=input_shapes[0][3]
                    layer['n_filt']=input_shapes[0][1]
                layer['stride_height']=keras_layer['config']['strides'][0]
                layer['stride_width']=keras_layer['config']['strides'][1]
                layer['pool_height']=keras_layer['config']['pool_size'][0]
                layer['pool_width']=keras_layer['config']['pool_size'][1]
                layer['padding']=keras_layer['config']['padding']
                if layer['padding']=='same':
                    #Height
                    in_height = input_shapes[0][1]
                    if layer['data_format'] == 'channels_first': in_height = input_shapes[0][2]
                    layer['out_height'] = int(math.ceil(float(in_height) / float(layer['stride_height'])))
                    if (in_height % layer['stride_height'] == 0):
                        pad_along_height = max(layer['pool_height'] - layer['stride_height'], 0)
                    else:
                        pad_along_height = max(layer['pool_height'] - (in_height % layer['stride_height']), 0)
                    layer['pad_top']  = pad_along_height // 2
                    layer['pad_bottom']  = pad_along_height - layer['pad_top']
                    #Width
                    in_width = input_shapes[0][2]
                    if layer['data_format'] == 'channels_first': in_height = input_shapes[0][3]
                    layer['out_width'] = int(math.ceil(float(in_width) / float(layer['stride_width'])))
                    if (in_width % layer['stride_width'] == 0):
                        pad_along_width = max(layer['pool_width'] - layer['stride_width'], 0)
                    else:
                        pad_along_width = max(layer['pool_width'] - (in_width % layer['stride_width']), 0)
                    layer['pad_left']  = pad_along_width // 2
                    layer['pad_right']  = pad_along_width - layer['pad_left']
                elif layer['padding']=='valid':
                    in_height = input_shapes[0][1]
                    in_width = input_shapes[0][2]
                    if layer['data_format'] == 'channels_first':
                        in_height = input_shapes[0][2]
                        in_width = input_shapes[0][3]
                    layer['out_width'] = int(math.ceil(float(in_width - layer['pool_width'] + 1) / float(layer['stride_width'])))
                    layer['out_height'] = int(math.ceil(float(in_height - layer['pool_height'] + 1) / float(layer['stride_height'])))
                    layer['pad_top'] = 0
                    layer['pad_bottom'] = 0
                    layer['pad_left'] = 0
                    layer['pad_right'] = 0
                if layer['data_format'] == 'channels_last': output_shape=[input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]
                elif layer['data_format'] == 'channels_first': output_shape=[input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]

        elif layer['class_name']=='LeakyReLU':
            layer['activation'] = layer['class_name']
            layer['activ_param'] = keras_layer["config"].get('alpha', 0.3)
        elif layer['class_name']=='ThresholdedReLU':
            layer['activation'] = layer['class_name']
            layer['activ_param'] = keras_layer["config"].get('theta', 1.)
        elif layer['class_name']=='ELU':
            layer['activation'] = layer['class_name']
            layer['activ_param'] = keras_layer["config"].get('alpha', 1.)
        elif layer['class_name']=='PReLU':
            layer['activation'] = layer['class_name']
        elif layer['class_name']=='QActivation':
            if 'quantized_relu' in layer['activation']:
                layer['activation'] = 'relu'
            elif 'quantized_tanh' in layer['activation']:
                layer['activation'] = 'tanh'
            else:
                raise Exception('Unsupported activation {} in layer {}'.format(layer['activation'], layer['name']))

        elif layer['class_name'] in merge_layers:
            layer['op'] = layer['class_name'].lower()
            if layer['class_name'] == 'Concatenate':
                rank = len(input_shapes[0][1:])
                if rank > 3:
                    raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')
                layer['op'] = layer['class_name'].lower() + '{}d'.format(rank)
                layer['axis'] = keras_layer['config']['axis']
            else:
                layer['class_name'] = 'Merge'
            if len(layer['inputs']) > 2:
                raise Exception('ERROR: Merging more than two tensors is not yet supported.')

        print('Layer name: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
        layer_list.append( layer )
        if 'activation' in layer and layer['class_name'] not in activation_layers + qkeras_layers:
            act_layer = {}
            act_layer['name'] = layer['name'] + '_' + layer['activation']
            act_layer['activation'] = layer['activation']
            if 'activ_param' in layer:
                act_layer['activ_param'] = layer['activ_param']
                act_layer['class_name'] = layer['activation']
            else:
                act_layer['class_name'] = 'Activation'
            inputs_map[layer['name']] = act_layer['name']
            if output_layers is not None and layer['name'] in output_layers:
                output_layers = [act_layer['name'] if name == layer['name'] else name for name in output_layers]
            layer_list.append(act_layer)

        assert(output_shape is not None)
        
        output_shapes[layer['name']] = output_shape

    #################
    ## Generate HLS
    #################

    print('Creating HLS model')
    hls_model = HLSModel(yamlConfig, reader, layer_list, input_layers, output_layers)
    optimizers = ['eliminate_linear_activation', 'merge_batch_norm_quantized_tanh', 'quantize_dense_output', 'fuse_dense_batch_norm']
    optimize_model(hls_model, optimizers)
    return hls_model
