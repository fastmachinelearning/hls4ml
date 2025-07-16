import glob
import os
import numpy as np
from fxpmath import Fxp
from shutil import copyfile, copytree, rmtree
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import List

from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'

@dataclass(frozen=True)
class XLSLayerConfig:
    name:        str
    in_dim_key:  str
    in_dim_val:  int
    out_dim_key: str
    out_dim_val: int
    in_nb:       str
    in_en:       str
    in_bu:       str
    in_type:     str
    out_nb:      str
    out_en:      str
    out_bu:      str
    out_type:    str
    fxp_weights: List[List[int]]
    fxp_bias:    List[int]

    def to_string(self) -> str:
        # build lines of "key: value"
        lines = [f"{k}: {v}" for k, v in asdict(self).items()]
        return "\n".join(lines)
    
    def is_activation(self) -> bool:
        return self.name in ['Activation', 'Softmax']

class XLSLayerConfigBuilder:
    def __init__(self):
        self._kw = {
            "fxp_weights": np.array([]),
            "fxp_bias":    np.array([]),
        }
    def name(self, v: int):
        self._kw["name"] = v; 
        return self
    def in_dim_key(self, v: str):
        self._kw["in_dim_key"] = v; 
        return self
    def in_dim_val(self, v: int):
        self._kw["in_dim_val"] = v; 
        return self
    def out_dim_key(self, v: str):
        self._kw["out_dim_key"] = v; 
        return self
    def out_dim_val(self, v: int):
        self._kw["out_dim_val"] = v; 
        return self
    def fxp_weights(self, fxp_weights, out_dim, in_dim):
        for w in fxp_weights:
            if (len(list(w)) == out_dim*in_dim):
                mat = np.array(list(w)).reshape(in_dim, out_dim)
                mat_T = mat.T   # in Keras the weights are transposed
                fxp_w = Fxp(mat_T, signed=True, n_word=16, n_frac=10).raw()
                self._kw["fxp_weights"] = fxp_w 
                return self
        return self
    def fxp_bias(self, fxp_weights, out_dim):
        for w in fxp_weights:
            if (len(list(w)) == out_dim):
                fxp_b = Fxp(list(w), signed=True, n_word=16, n_frac=10).raw()
                self._kw["fxp_bias"] = fxp_b
        return self
    def in_nb(self, prev_layer_precision): # TODO: right now we only care about the first defined type in the list
        if prev_layer_precision:
            for _, type_var in prev_layer_precision.items():
                self._kw["in_nb"] = f'u32:{type_var.precision.width}'; 
                return self
        else:
            self._kw["in_nb"] = ''
        return self
    def in_en(self):
        self._kw["in_en"] = 'u32:1'
        return self
    def in_bu(self, prev_layer_precision):
        if prev_layer_precision:
            for _, type_var in prev_layer_precision.items():
                self._kw["in_bu"] = f'u32:{type_var.precision.width - type_var.precision.integer}'; 
                return self
        else:
            self._kw["in_bu"] = ''
        return self
    def in_type(self, prev_layer_precision):
        if prev_layer_precision:
            for _, type_var in prev_layer_precision.items():
                self._kw["in_type"] = f'sN[u32:{type_var.precision.width}]'; 
                return self
        else:
            self._kw["in_type"] = ''
        return self
    def out_nb(self, layer_precision):
        if layer_precision:
            for _, type_var in layer_precision.items():
                self._kw["out_nb"] = f'u32:{type_var.precision.width}'; 
                return self
        else:
            self._kw["out_nb"] = ''
        return self
    def out_en(self):
        self._kw["out_en"] = 'u32:1'
        return self
    def out_bu(self, layer_precision):
        if layer_precision:
            for _, type_var in layer_precision.items():
                self._kw["out_bu"] = f'u32:{type_var.precision.width - type_var.precision.integer}'; 
                return self
        else:
            self._kw["out_bu"] = ''
        return self
    def out_type(self, layer_precision):
        if layer_precision:
            for _, type_var in layer_precision.items():
                self._kw["out_type"] = f'sN[u32:{type_var.precision.width}]'; 
                return self
        else:
            self._kw["out_type"] = ''
        return self

    def build(self) -> XLSLayerConfig:
        return XLSLayerConfig(**self._kw)

class XLSWriter(Writer):
    
    def write_project_dir(self, model):
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f"{model.config.get_output_dir()}/firmware"):
            os.makedirs(f"{model.config.get_output_dir()}/firmware")

    def write_project_dslx(self, model, xls_layers: list[XLSLayerConfig]):
        """Write the main architecture source file (myproject.x)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/xls/firmware/myproject.x'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.x', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        # model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning xls layer documentation' in line:
                print("================= HERE\n")
                newline = line + "TESTEST"
                for layer in xls_layers:
                    newline += layer.to_string()
                    newline += '\n\n'

            elif '// hls-fpga-machine-learning insert dimensions' in line:
                newline = line
                for layer in xls_layers:
                    if layer.is_activation() == False:
                        newline += f'const {layer.out_dim_key} = {layer.out_dim_val};\n'

            elif '// hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([i.definition_cpp(as_reference=True) for i in model_inputs])
                outputs_str = ', '.join([o.definition_cpp(as_reference=True) for o in model_outputs])
                # brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str
                # if len(model_brams) > 0:
                #     newline += ',\n' + brams_str
                newline += '\n'

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                if model.config.get_writer_config()['WriteWeightsTxt']:

                    newline += '#ifndef __SYNTHESIS__\n'
                    newline += '    static bool loaded_weights = false;\n'
                    newline += '    if (!loaded_weights) {\n'

                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            if w.weight_class == 'CompressedWeightVariable':
                                newline += (
                                    indent
                                    + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                        w.type.name, w.nonzeros, w.name, w.name
                                    )
                                )
                            elif w.weight_class == 'ExponentWeightVariable':
                                newline += (
                                    indent
                                    + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                        w.type.name, w.data_length, w.name, w.name
                                    )
                                )
                            else:
                                newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                    w.type.name, w.data_length, w.name, w.name
                                )

                    newline += '        loaded_weights = true;'
                    newline += '    }\n'
                    newline += '#endif'

            # Add input/output type
            elif '// hls-fpga-machine-learning insert IO' in line:
                pass

            elif '// hls-fpga-machine-learning architecture type inference' in line:
                indent = '    '
                newline = indent + 'IN_L0: u32, OUT_L0: u32,\n'
                for i, layer in enumerate(model.get_layers()):
                    if i > 0:
                        newline += indent + f'IN_L{i}: u32 = {{OUT_L{i-1}}}, OUT_L{i}: u32,\n'

            # TODO: infer actual defintion of 'Output_T'
            elif '// hls-fpga-machine-learning output ' in line:
                indent = '    '
                newline = indent + f'Output_T[OUT_L{len(model.get_layers())-1}],\n'

            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                for i, layer in enumerate(model.get_layers()):
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in model_inputs and var not in model_outputs:
                            #TODO: might fail for non fixed point types 
                            newline += f'    let z{i+1} = ' + f'multi_dense_fxd::{var.type.name}<{var.type.precision.width}, 1, {var.type.precision.integer}>' + f'(z{i}, w{i}, b{i});\n'
                newline += f'    z{len(model.get_layers()) - 1}\n'

            # Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/xls/firmware/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

        # ap_types
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/xls/firmware/ap_types/')
        dstpath = f'{model.config.get_output_dir()}/firmware/ap_types/'

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

        # TODO: check if you need this
        # # custom source
        # filedir = os.path.dirname(os.path.abspath(__file__))

        # custom_source = model.config.backend.get_custom_source()
        # for dst, srcpath in custom_source.items():
        #     dstpath = f'{model.config.get_output_dir()}/firmware/{dst}'
        #     copyfile(srcpath, dstpath)



    def write_hls(self, model):
        xls_layers = []
        builder = XLSLayerConfigBuilder()

        prev_out_dim_key = ''
        prev_out_dim_val = -1
        prev_layer_precision = None
        for layer in model.get_layers():
            # print('\n========== Layer: ')
            # for name, val in layer.__dict__.items():
            #     print(f"{name}: {val!r}")
            # print('\nMODEL: ')
            # for name, val in layer.model.__dict__.items():
            #     print(f"{name}: {val!r}")
            # print()
            cur_out_dim_key = list(layer.get_output_variable().get_shape())[0][0]
            cur_out_dim_val = list(layer.get_output_variable().get_shape())[0][1]
            new_layer = (
                builder
                .name(layer.class_name)
                .in_dim_key(prev_out_dim_key)
                .in_dim_val(prev_out_dim_val)
                .out_dim_key(cur_out_dim_key) # TODO: investigate if this is always good
                .out_dim_val(cur_out_dim_val)
                .in_nb(prev_layer_precision)
                .in_en()
                .in_bu(prev_layer_precision)
                .in_type(prev_layer_precision)
                .out_type(layer.get_layer_precision())
                .out_nb(layer.get_layer_precision())
                .out_en()
                .out_bu(layer.get_layer_precision())
                .fxp_weights(layer.get_weights(), out_dim=cur_out_dim_val, in_dim=prev_out_dim_val)
                .fxp_bias(layer.get_weights(), out_dim=cur_out_dim_val)
                .build()
            )
            xls_layers.append(new_layer)

            prev_out_dim_key = new_layer.out_dim_key
            prev_out_dim_val = new_layer.out_dim_val
            prev_layer_precision = layer.get_layer_precision()

        print('Writing HLS project')
        self.write_project_dir(model)
        self.write_project_dslx(model, xls_layers)
        self.write_nnet_utils(model)
        print('Done writing')