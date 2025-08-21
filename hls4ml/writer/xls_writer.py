# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import List, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph

import os
from shutil import copyfile, copytree, rmtree
from hls4ml.writer.writers import Writer



class XLSWriter(Writer):
    
    def _write_weights(self, layer, weights):
        """A recursive function to write weights of any number of dimensions. 

        It uses the function call stack to close paranthesis. 
        """
        indent = '    '
        
        if len(weights.shape) == 1:
            newline = indent + indent + '['
            for idx_col, w in enumerate(weights):
                newline += f'{layer.get_attr("in_type")}:{w}'
                if idx_col < len(weights) - 1:
                    newline += ','
            newline += '],\n'
            return newline
        
        newline = indent + '[\n'
        for idx in range(len(weights)):
            newline += self._write_weights(layer, weights[idx])
        newline += indent + '],\n'
        return newline

    def write_project_dir(self, model: ModelGraph) -> None:
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f"{model.config.get_output_dir()}/firmware"):
            os.makedirs(f"{model.config.get_output_dir()}/firmware")

        if not os.path.isdir(f"{model.config.get_output_dir()}/reports"):
            os.makedirs(f"{model.config.get_output_dir()}/reports")


    def write_build_script(self, model: ModelGraph) -> None:
        # build_prj.tcl
        filedir = os.path.dirname(os.path.abspath(__file__))
        srcpath = os.path.join(filedir, '../templates/xls/build_prj.tcl')
        dstpath = f'{model.config.get_output_dir()}/build_prj.tcl'
        copyfile(srcpath, dstpath)


    def write_project_dslx(self, model: ModelGraph) -> None:
        """Write the main architecture source file (myproject.x)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/xls/firmware/myproject.x'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.x', 'w')

        layers = list(model.get_layers())
        indent = '    '
        last_layer_dim_key = ''
        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning imports' in line:
                newline = line
                seen_libs = []
                for layer in layers:
                    lib = layer.get_attr('func_call').split('::', 1)[0]
                    if lib and lib not in seen_libs:
                        seen_libs.append(lib)
                        newline += f'import nnet_utils.{lib};\n'

            elif '// hls-fpga-machine-learning insert dimensions' in line:
                newline = line
                for layer in layers:
                    if layer.get_attr("write_dims"):
                        for dim in list(layer.get_output_variable().get_shape()):
                            newline += f'const {dim[0]} = u32:{dim[1]};\n'

            elif '// hls-fpga-machine-learning architecture arguments' in line:
                newline = ''
                weighted_layers_count = 0
                for i, layer in enumerate(layers):
                    if layer.class_name == 'Input':
                        newline += indent + f'x: {layer.get_attr("out_type")}'
                        for dim in list(layer.get_output_variable().get_shape()):
                            newline += f'[{dim[0]}]'
                        newline += ',\n'
                    elif layer.get_attr("write_weights"):
                        # weights arguments
                        newline += indent + f'w{i}: {layer.get_attr("in_type")}'
                        for w_dim in layer.get_attr("fxp_weights").shape:
                            newline += f'[u32:{w_dim}]'
                        newline += ',\n'
                        # bias argument
                        newline += indent + f'b{i}: {layer.get_attr("in_type")}'
                        for b_dim in layer.get_attr("fxp_bias").shape:
                            newline += f'[u32:{b_dim}]'
                        if weighted_layers_count < len([layer for layer in layers if layer.get_attr("write_weights")]) - 1:
                            newline += ',\n'
                            weighted_layers_count += 1
                        else:
                            newline += '\n'

            elif '// hls-fpga-machine-learning output' in line:
                last_layer_type = layers[-1].get_attr("out_type")
                newline = indent + f'{last_layer_type}'
                for dim in list(layers[-1].get_output_variable().get_shape()):
                    newline += f'[{dim[0]}]'
                newline += '\n'

            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line
                prev_var = 'x'
                for i, layer in enumerate(layers):
                    if layer.get_attr('write_func'):
                        if layer.get_attr('write_weights'):
                            newline += indent + f'let z{i} = {layer.get_attr("func_call")}({prev_var}, w{i}, b{i});\n'
                            prev_var = f'z{i}'
                        else:
                            newline += indent + f'let z{i} = {layer.get_attr("func_call")}({prev_var});\n'
                            prev_var = f'z{i}'

                newline += indent + prev_var + '\n'

            elif '// hls-fpga-machine-learning top function input' in line:
                newline = indent + f'x: {layer.get_attr("out_type")}'
                for dim in list(layers[0].get_output_variable().get_shape()):
                    newline += f'[{dim[0]}]'
                newline += '\n'

            elif '// hls-fpga-machine-learning load weights' in line:
                newline = line
                for i, layer in enumerate(layers):
                    if layer.get_attr("write_weights"):
                        # Weights
                        newline += indent + f'let w{i} = {layer.get_attr("in_type")}'
                        for w_dim in layer.get_attr("fxp_weights").shape:
                            newline += f'[u32:{w_dim}]'
                        newline += ':\n'
                        newline += indent + '[\n'
                        for idx in range(len(layer.get_attr("fxp_weights"))):
                            newline += self._write_weights(layer, layer.get_attr("fxp_weights")[idx])
                        newline += indent + '];\n'
                        # Bias
                        newline += indent + f'let b{i} = {layer.get_attr("in_type")}[u32:{layer.get_attr("fxp_bias").shape[0]}]:[\n'
                        newline += indent + indent
                        for b in layer.get_attr("fxp_bias"):
                            newline += f'{layer.get_attr("in_type")}:{b},'
                        newline += '\n' + indent + '];\n'

            elif '// hls-fpga-machine-learning call inlined weights' in line:
                newline = indent + indent
                weighted_layers_count = 0
                for i, layer in enumerate(layers):
                    if layer.class_name == 'Input':
                        newline += 'x,'
                    elif layer.get_attr("write_weights"):
                        newline += f'w{i}, b{i}'
                        if weighted_layers_count < len([layer for layer in layers if layer.get_attr("write_weights")]) - 1:
                            newline += ', '
                            weighted_layers_count += 1
                newline += '\n'

            # Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    #TODO: modify with actual table types
    def write_lookup_tables(self, model: ModelGraph) -> None:
        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/xls/firmware/nnet_utils/lookup_tables.x'))
        fout = open(f'{model.config.get_output_dir()}/firmware/nnet_utils/lookup_tables.x', 'w')

        layers = list(model.get_layers())
        indent = '    '
        elems_per_line = 8
        for line in f.readlines():

            if '// hls-fpga-machine-learning insert exponent table' in line:
                newline = line
                for layer in layers:
                    if layer.get_attr('write_table'):
                        # Get types
                        exp_width = layer.get_layer_precision()['softmax_exp_table_t'].precision.width

                        newline += f'pub const EXP_TABLE = sN[{exp_width}][u32:{dict(layer.attributes)["table_size"]}]:[\n'
                        newline += indent
                        for i, elem in enumerate(layer.get_attr("exp_table_xls")):
                            newline += f'sN[{exp_width}]:{elem}'
                            if i < len(layer.get_attr("exp_table_xls")) - 1:
                                newline += ','
                            if (i+1) % elems_per_line == 0:
                                newline += '\n'
                                if i < len(layer.get_attr("inv_table_xls")) - 1:
                                    newline += indent
                        newline += '];\n'

            elif '// hls-fpga-machine-learning insert inversion table' in line:
                newline = line
                for layer in layers:
                    if layer.get_attr('write_table'):
                        # Get types
                        inv_width = layer.get_layer_precision()['softmax_inv_table_t'].precision.width

                        newline += f'pub const INV_TABLE = sN[{inv_width}][u32:{dict(layer.attributes)["table_size"]}]:[\n'
                        newline += indent
                        for i, elem in enumerate(layer.get_attr("inv_table_xls")):
                            newline += f'sN[{inv_width}]:{elem}'
                            if i < len(layer.get_attr("inv_table_xls")) - 1:
                                newline += ', '
                            if (i+1) % elems_per_line == 0:
                                newline += '\n' 
                                if i < len(layer.get_attr("inv_table_xls")) - 1:
                                    newline += indent
                        newline += '];\n'
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

    def write_hls(self, model: ModelGraph) -> None:

        self.write_project_dir(model)
        self.write_build_script(model)
        self.write_project_dslx(model)
        self.write_nnet_utils(model)
        self.write_lookup_tables(model)