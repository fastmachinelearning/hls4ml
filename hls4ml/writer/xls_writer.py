# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import List, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph


import glob
import sys, os
import subprocess
import numpy as np
from fxpmath import Fxp
from shutil import copyfile, copytree, rmtree
from collections import OrderedDict
from dataclasses import dataclass, asdict
from hls4ml.writer.writers import Writer


config_filename = 'hls4ml_config.yml'

@dataclass(frozen=True)
class XLSLayerConfig:
    class_name:  str
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
        return self.class_name in ['Activation', 'Softmax']

    def has_weights(self) -> bool:
        return self.class_name in ['Dense']

class XLSLayerConfigBuilder:
    def __init__(self):
        self._kw: dict[str, Any] = {
            "fxp_weights": np.array([]),
            "fxp_bias":    np.array([]),
        }
    def class_name(self, v: int):
        self._kw["class_name"] = v; 
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
    def fxp_weights(self, weights, out_dim, in_dim):
        for w in weights:
            if (len(list(w)) == out_dim*in_dim):
                mat = np.array(list(w)).reshape(in_dim, out_dim)
                mat_T = mat.T   # in Keras the weights are transposed
                fxp_w = Fxp(mat_T, signed=True, n_word=16, n_frac=10).raw()
                self._kw["fxp_weights"] = fxp_w 
                return self
        return self
    def fxp_bias(self, weights, out_dim):
        for w in weights:
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
        for _, type_var in layer_precision.items():
            self._kw["out_bu"] = f'u32:{type_var.precision.width - type_var.precision.integer}'; 
            return self
        else:
            self._kw["out_bu"] = ''
        return self
    def out_type(self, layer_precision):
        for _, type_var in layer_precision.items():
            self._kw["out_type"] = f'sN[u32:{type_var.precision.width}]'; 
            return self
        else:
            self._kw["out_type"] = ''
        return self

    def build(self) -> XLSLayerConfig:
        return XLSLayerConfig(**self._kw)

    def build_xls_layers(self, model: ModelGraph) -> list[XLSLayerConfig]:
        xls_layers: list[XLSLayerConfig] = []
        
        prev_out_dim_key = ''
        prev_out_dim_val = -1
        prev_layer_precision = None
        for layer in model.get_layers():
            cur_out_dim_key = list(layer.get_output_variable().get_shape())[0][0]
            cur_out_dim_val = list(layer.get_output_variable().get_shape())[0][1]
            new_layer = (
                self
                .class_name(layer.class_name)
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

        return xls_layers

class XLSWriter(Writer):
    
    def write_project_dir(self, model: ModelGraph) -> None:
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f"{model.config.get_output_dir()}/firmware"):
            os.makedirs(f"{model.config.get_output_dir()}/firmware")

        if not os.path.isdir(f"{model.config.get_output_dir()}/predictions"):
            os.makedirs(f"{model.config.get_output_dir()}/predictions")

    def write_project_dslx(self, model: ModelGraph, xls_layers: list[XLSLayerConfig]) -> None:
        """Write the main architecture source file (myproject.x)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/xls/firmware/myproject.x'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.x', 'w')

        indent = '    '
        last_layer_dim_key = ''
        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            # elif '// hls-fpga-machine-learning debugging' in line:
            #     print("================= HERE\n")
            #     newline = line
            #     for layer in xls_layers:
            #         newline += layer.to_string()
            #         newline += '\n\n'

            elif '// hls-fpga-machine-learning insert dimensions' in line:
                newline = line
                for layer in xls_layers:
                    if layer.is_activation() == False:
                        newline += f'const {layer.out_dim_key} = u32:{layer.out_dim_val};\n'

            # elif '// hls-fpga-machine-learning architecture type inference' in line:
            #     newline = indent + 'IN_L0: u32, OUT_L0: u32,\n'
            #     for i, layer in enumerate(xls_layers):
            #         if i > 0 and layer.is_activation() == False:
            #             newline += indent + f'IN_L{i}: u32 = {{OUT_L{i-1}}}, OUT_L{i}: u32,\n'
            #             last_layer_dim_key = f'OUT_L{i}'

            elif '// hls-fpga-machine-learning architecture arguments' in line:
                newline = ''
                weighted_layers_count = 0
                for i, layer in enumerate(xls_layers):
                    if layer.class_name == 'Input':
                        newline += indent + f'x: {layer.out_type}[{layer.out_dim_key}],\n'
                    elif layer.is_activation() == False:
                        newline += indent + f'w{i}: {layer.out_type}[{layer.in_dim_key}][{layer.out_dim_key}],\n'
                        newline += indent + f'b{i}: {layer.out_type}[{layer.out_dim_key}]'
                        if weighted_layers_count < len([layer for layer in xls_layers if layer.has_weights()]) - 1:
                            newline += ',\n'
                            weighted_layers_count += 1
                        else:
                            newline += '\n'

            elif '// hls-fpga-machine-learning output ' in line:
                indent = '    '
                last_layer_type = xls_layers[-1].out_type
                last_layer_dim_key = xls_layers[-1].out_dim_key
                newline = indent + f'{last_layer_type}[{last_layer_dim_key}]\n'

            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line
                prev_var = ''
                for i, layer in enumerate(xls_layers):
                    next_layer = xls_layers[i + 1] if i < len(xls_layers) - 1 else None
                    if layer.class_name == 'Dense' and (next_layer is not None and next_layer.class_name == 'Activation'):
                        if prev_var is '':
                            newline += indent + f'let z{i} = multi_dense_fxd::dense_relu<{layer.in_nb}, {layer.in_en}, {layer.in_bu}, {layer.out_nb}, {layer.out_en}, {layer.out_bu}>(x, w{i}, b{i});\n'
                            prev_var = f'z{i}'
                        else:
                            newline += indent + f'let z{i} = multi_dense_fxd::dense_relu<{layer.in_nb}, {layer.in_en}, {layer.in_bu}, {layer.out_nb}, {layer.out_en}, {layer.out_bu}>({prev_var}, w{i}, b{i});\n'
                            prev_var = f'z{i}'
                    if layer.class_name == 'Dense' and (next_layer is not None and next_layer.class_name == 'Softmax'):
                        if prev_var is '':
                            newline += indent + f'let y{i} = multi_dense_fxd::dense<{layer.in_nb}, {layer.in_en}, {layer.in_bu}, {layer.out_nb}, {layer.out_en}, {layer.out_bu}>(x, w{i}, b{i});\n'
                            prev_var = f'y{i}'
                        else:
                            newline += indent + f'let y{i} = multi_dense_fxd::dense<{layer.in_nb}, {layer.in_en}, {layer.in_bu}, {layer.out_nb}, {layer.out_en}, {layer.out_bu}>({prev_var}, w{i}, b{i});\n'
                            prev_var = f'y{i}'
                    if layer.class_name == 'Softmax':
                        newline += indent + f'let z{i} = multi_dense_fxd::argmax<{layer.out_nb}, {layer.out_en}, {layer.out_bu}>({prev_var});\n'
                        prev_var = f'z{i}'

                newline += indent + prev_var + '\n'

            elif '// hls-fpga-machine-learning top function input' in line:
                newline = indent + f'x: {xls_layers[0].out_type}[{xls_layers[0].out_dim_key}]\n'

            elif '// hls-fpga-machine-learning top function output' in line:
                newline = indent + f'{xls_layers[-1].out_type}[{xls_layers[-1].out_dim_key}]\n'

            elif '// hls-fpga-machine-learning load weights' in line:
                newline = line
                for i, layer in enumerate(xls_layers):
                    if layer.has_weights():
                        # Weights
                        newline += indent + f'let w{i} = {layer.out_type}[{layer.in_dim_key}][{layer.out_dim_key}]:[\n'
                        for idx_row, row in enumerate(layer.fxp_weights):
                            newline += indent + indent + '['
                            for idx_col, w in enumerate(row):
                                newline += f'{layer.out_type}:{w}'
                                if idx_col < len(row) - 1:
                                    newline += ','
                            newline += ']'
                            if idx_row < len(layer.fxp_weights) - 1:
                                    newline += ',\n'
                            else:
                                newline += '\n'
                        newline += indent + '];\n'
                        # Bias
                        newline += indent + f'let b{i} = {layer.out_type}[{layer.out_dim_key}]:[\n'
                        newline += indent + indent
                        for idx_b, b in enumerate(layer.fxp_bias):
                            newline += f'{layer.out_type}:{b}'
                            if idx_b < len(layer.fxp_bias) - 1:
                                newline += ','
                        newline += '\n' + indent + '];\n'

            elif '// hls-fpga-machine-learning call inlined weights' in line:
                newline = indent + indent
                weighted_layers_count = 0
                for i, layer in enumerate(xls_layers):
                    if layer.class_name == 'Input':
                        newline += 'x,'
                    elif layer.has_weights():
                        newline += f'w{i}, b{i}'
                        if weighted_layers_count < len([layer for layer in xls_layers if layer.has_weights()]) - 1:
                            newline += ', '
                            weighted_layers_count += 1
                newline += '\n'

            # Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_nnet_utils(self, model: ModelGraph) -> None:
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


    def gen_interpretable_ir(self, model: ModelGraph):

        if 'linux' in sys.platform:
            path = os.path.expandvars(model.config.get_config_value('xls_bazel_bin_path'))
            if os.path.isdir(path) == 0:
                raise Exception('XLS is expected to be installed in your $HOME dir. We are looking for `$HOME/xls/bazel-bin`')

        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/firmware')
        kernel_name = model.config.get_project_name()

        # ## Run interpreter
        # interpreter_cmd = [ 
        #     f'{path}/xls/dslx/interpreter_main',
        #     f'{kernel_name}.x'
        # ]
        # subprocess.run(interpreter_cmd, check=True)

        ## Generate IR
        with open(f'{kernel_name}.ir', 'w') as ir_file:
            gen_cmd = [ 
                f'{path}/xls/dslx/ir_convert/ir_converter_main',
                f'--top={kernel_name}',
                f'{kernel_name}.x'
            ]
            subprocess.run(gen_cmd, check=True, stdout=ir_file)

        ## Optimize IR
        with open(f'{kernel_name}.opt.ir', 'w') as opt_file:
            opt_cmd = [ 
                f'{path}/xls/tools/opt_main',
                f'{kernel_name}.ir'
            ]
            subprocess.run(opt_cmd, check=True, stdout=opt_file)

        os.chdir(curr_dir)


    def write_hls(self, model: ModelGraph) -> None:
        builder = XLSLayerConfigBuilder()
        xls_layers: list[XLSLayerConfig] = builder.build_xls_layers(model)

        print('Writing HLS project')
        self.write_project_dir(model)
        self.write_project_dslx(model, xls_layers)
        self.write_nnet_utils(model)

        print('Done writing')