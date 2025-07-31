# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import List, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph

import os
from shutil import copyfile, copytree, rmtree
from hls4ml.writer.writers import Writer


config_filename = 'hls4ml_config.yml'


class XLSWriter(Writer):
    
    def write_project_dir(self, model: ModelGraph) -> None:
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f"{model.config.get_output_dir()}/firmware"):
            os.makedirs(f"{model.config.get_output_dir()}/firmware")

        # if not os.path.isdir(f"{model.config.get_output_dir()}/predictions"):
        #     os.makedirs(f"{model.config.get_output_dir()}/predictions")

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

            elif '// hls-fpga-machine-learning insert dimensions' in line:
                newline = line
                for layer in layers:
                    if layer.get_attr("write_dims"):
                        newline += f'const {layer.get_attr("out_dim_key")} = u32:{layer.get_attr("out_dim_val")};\n'

            elif '// hls-fpga-machine-learning architecture arguments' in line:
                newline = ''
                weighted_layers_count = 0
                for i, layer in enumerate(layers):
                    if layer.class_name == 'Input':
                        newline += indent + f'x: {layer.get_attr("out_type")}[{layer.get_attr("out_dim_key")}],\n'
                    elif layer.get_attr("write_weights"):
                        newline += indent + f'w{i}: {layer.get_attr("out_type")}[{layer.get_attr("in_dim_key")}][{layer.get_attr("out_dim_key")}],\n'
                        newline += indent + f'b{i}: {layer.get_attr("out_type")}[{layer.get_attr("out_dim_key")}]'
                        if weighted_layers_count < len([layer for layer in layers if layer.get_attr("write_weights")]) - 1:
                            newline += ',\n'
                            weighted_layers_count += 1
                        else:
                            newline += '\n'

            elif '// hls-fpga-machine-learning output ' in line:
                last_layer_type = layers[-1].get_attr("out_type")
                last_layer_dim_key = layers[-1].get_attr("out_dim_key")
                newline = indent + f'{last_layer_type}[{last_layer_dim_key}]\n'

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
                newline = indent + f'x: {layers[0].get_attr("out_type")}[{layers[0].get_attr("out_dim_key")}]\n'

            elif '// hls-fpga-machine-learning top function output' in line:
                newline = indent + f'{layers[-1].get_attr("out_type")}[{layers[-1].get_attr("out_dim_key")}]\n'

            elif '// hls-fpga-machine-learning load weights' in line:
                newline = line
                for i, layer in enumerate(layers):
                    if layer.get_attr("write_weights"):
                        # Weights
                        newline += indent + f'let w{i} = {layer.get_attr("out_type")}[{layer.get_attr("in_dim_key")}][{layer.get_attr("out_dim_key")}]:[\n'
                        for idx_row, row in enumerate(layer.get_attr('fxp_weights')):
                            newline += indent + indent + '['
                            for idx_col, w in enumerate(row):
                                newline += f'{layer.get_attr("out_type")}:{w}'
                                if idx_col < len(row) - 1:
                                    newline += ','
                            newline += ']'
                            if idx_row < len(layer.get_attr("fxp_weights")) - 1:
                                    newline += ',\n'
                            else:
                                newline += '\n'
                        newline += indent + '];\n'
                        # Bias
                        newline += indent + f'let b{i} = {layer.get_attr("out_type")}[{layer.get_attr("out_dim_key")}]:[\n'
                        newline += indent + indent
                        for idx_b, b in enumerate(layer.get_attr("fxp_bias")):
                            newline += f'{layer.get_attr("out_type")}:{b}'
                            if idx_b < len(layer.get_attr("fxp_bias")) - 1:
                                newline += ','
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

    def write_lookup_tables(self, model: ModelGraph) -> None:
        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/xls/firmware/ap_types/lookup_tables.x'))
        fout = open(f'{model.config.get_output_dir()}/firmware/ap_types/lookup_tables.x', 'w')

        layers = list(model.get_layers())
        indent = '    '
        elems_per_line = 8
        for line in f.readlines():

            if '// hls-fpga-machine-learning insert exponent table' in line:
                newline = line
                for layer in layers:
                    if layer.get_attr('write_table'):
                        newline += f'pub const EXP_TABLE = sN[{layer.get_attr("out_nb")}][u32:{dict(layer.attributes)["table_size"]}]:[\n'
                        newline += indent
                        for i, elem in enumerate(layer.get_attr("exp_table_xls")):
                            newline += f'sN[{layer.get_attr("out_nb")}]:{elem}'
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
                        newline += f'pub const INV_TABLE = sN[{layer.get_attr("out_nb")}][u32:{dict(layer.attributes)["table_size"]}]:[\n'
                        newline += indent
                        for i, elem in enumerate(layer.get_attr("inv_table_xls")):
                            newline += f'sN[{layer.get_attr("out_nb")}]:{elem}'
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
        self.write_project_dslx(model)
        self.write_nnet_utils(model)
        self.write_lookup_tables(model)