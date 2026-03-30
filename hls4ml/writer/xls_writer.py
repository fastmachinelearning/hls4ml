# Typing imports
from __future__ import annotations  # makes all annotations into strings

from collections.abc import Iterable
from pathlib import Path
from typing import Any, TYPE_CHECKING

from hls4ml.backends.xls.xls_types import XLSFunctionCall, XLSConst, XLSTypeAlias, XLSImport, XLSFunctionDefinition, \
    XLSTensorVariable
from hls4ml.model.layers import Layer
from hls4ml.model.types import FixedPrecisionType

if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph

import os
from shutil import copyfile, copytree, rmtree
from hls4ml.writer.writers import Writer

XLS_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / 'templates/xls'
INDENT = ' ' * 4


def firmware_dir(model: ModelGraph):
    return Path(model.config.get_output_dir()) / "firmware"


def reports_dir(model: ModelGraph):
    return Path(model.config.get_output_dir()) / "reports"


def append_line(line: str, x: Any) -> str:
    return line + f'{x}\n'


def append_lines(s: str, *xs: Any) -> str:
    # Allow append_lines(s, [1,2,3]) as well as append_lines(s, 1,2,3)
    if len(xs) == 1 and isinstance(xs[0], Iterable) and not isinstance(xs[0], (str, bytes)):
        xs = tuple(xs[0])

    for x in xs:
        s = append_line(s, x)
    return s


class XLSWriter(Writer):

    def _write_weights(self, layer, weights):
        """A recursive function to write weights of any number of dimensions. 

        It uses the function call stack to close paranthesis. 
        """

        if len(weights.shape) == 1:
            line = INDENT + INDENT + '['
            for idx_col, w in enumerate(weights):
                line = f'{layer.get_attr("in_type")}:{w}'
                if idx_col < len(weights) - 1:
                    line += ','
            line += '],\n'
            return line

        line = INDENT + '[\n'
        for idx in range(len(weights)):
            line += self._write_weights(layer, weights[idx])
        line += INDENT + '],\n'
        return line

    def write_project_dir(self, model: ModelGraph) -> None:
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        firmware = firmware_dir(model)
        if not os.path.isdir(firmware):
            os.makedirs(firmware)

        reports = reports_dir(model)
        if not os.path.isdir(reports):
            os.makedirs(reports)

    def write_build_script(self, model: ModelGraph) -> None:
        # build_prj.tcl
        srcpath = XLS_TEMPLATE_DIR / 'build_prj.tcl'
        dstpath = f'{model.config.get_output_dir()}/build_prj.tcl'
        copyfile(srcpath, dstpath)

    def write_project_dslx(self, model: ModelGraph) -> None:
        """Write the main architecture source file (myproject.x)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        output_path = firmware_dir(model) / f'{model.config.get_project_name()}.x'

        layers = list(model.get_layers())
        with open(output_path, 'w') as f:
            for line in open(XLS_TEMPLATE_DIR / 'firmware/myproject.x'):
                # Add headers to weights and biases
                if 'myproject' in line:
                    line = line.replace('myproject', model.config.get_project_name())

                elif '// hls-fpga-machine-learning insert imports' in line:
                    input_type = None
                    output_type = None
                    for layer in layers:
                        layer_module_name = layer.get_attr('xls_module_name')
                        input_var = layer.get_attr('xls_input_variable')
                        output_var = layer.get_attr('xls_output_variable')

                        input_type = input_type or f'{layer_module_name}::{input_var.type_alias.name}'
                        output_type = f'{layer_module_name}::{output_var.type_alias.name}'
                        line = append_line(line, XLSImport(layer_module_name))
                    line = append_lines(line,
                                        XLSTypeAlias(name='InputType', type=input_type),
                                        XLSTypeAlias(name='OutputType', type=output_type))

                elif '// hls-fpga-machine-learning insert layers' in line:
                    prev_var = 'x'
                    for layer in layers:
                        layer_module_name = layer.get_attr('xls_module_name')
                        var = f'z{layer.index}'
                        line = append_line(line, INDENT + f'let {var} = {layer_module_name}::transform({prev_var});')
                        prev_var = var

                    line = append_line(line, INDENT + prev_var + '\n')

                else:
                    pass

                f.write(line)

    def write_layers(self, model: ModelGraph):
        prev_layer = None
        for layer in model.get_layers():
            self.write_layer(model, layer, prev_layer)
            prev_layer = layer

    def write_layer(self, model: ModelGraph, layer: Layer, prev_layer: Layer | None):
        layer_module_name = layer.get_attr('xls_module_name')
        input_var: XLSTensorVariable = layer.get_attr('xls_input_variable')
        output_var: XLSTensorVariable = layer.get_attr('xls_output_variable')
        with open(firmware_dir(model) / f'{layer_module_name}.x', 'w') as f:
            for line in open(XLS_TEMPLATE_DIR / 'firmware/layer.x'):
                if '// hls-fpga-machine-learning insert imports' in line:
                    imports = []
                    func_call = layer.get_attr('xls_func_call')
                    if isinstance(func_call, XLSFunctionCall) and func_call.namespace is not None:
                        imports.append(XLSImport(name=f'nnet_utils.{func_call.namespace}'))
                    if layer.get_attr('lookup_tables'):
                        imports.append(XLSImport(name='nnet_utils.lookup_table'))
                    if prev_layer is not None:
                        imports.append(XLSImport(name=prev_layer.get_attr('xls_module_name')))
                    line = append_lines(line, imports)

                elif '// hls-fpga-machine-learning insert types' in line:
                    line = append_lines(line, input_var.definitions())
                    line += '\n'
                    line = append_lines(line, output_var.definitions())
                    line += '\n'
                    precision = layer.get_output_variable().type.precision
                    assert isinstance(precision, FixedPrecisionType)
                    line = append_line(line, XLSConst(
                        name='ROUNDING_MODE',
                        value=f'RoundingMode::{precision.rounding_mode}'))
                    line = append_line(line, XLSConst(
                        name='OVERFLOW_MODE',
                        value=f'OverflowMode::{precision.saturation_mode}'))
                    if layer.get_attr('class_name') == 'Conv2D':
                        line = append_lines(
                            line,
                            XLSConst(name='STRIDE_HEIGHT', value=layer.get_attr('stride_height'), type='u32'),
                            XLSConst(name='STRIDE_WIDTH', value=layer.get_attr('stride_width'), type='u32'),
                            XLSConst(name='PAD_TOP', value=layer.get_attr('pad_top'), type='u32'),
                            XLSConst(name='PAD_BOTTOM', value=layer.get_attr('pad_bottom'), type='u32'),
                            XLSConst(name='PAD_LEFT', value=layer.get_attr('pad_left'), type='u32'),
                            XLSConst(name='PAD_RIGHT', value=layer.get_attr('pad_right'), type='u32'),
                            XLSConst(name='DATA_FORMAT',
                                     value=f"conv2d::DataFormat::{layer.get_attr('data_format').upper()}")
                        )

                elif '// hls-fpga-machine-learning insert weights' in line:
                    weights = layer.get_attr('xls_weights')
                    if weights:
                        line = append_line(line, weights)
                    bias = layer.get_attr('xls_bias')
                    if bias:
                        line = append_lines(line, '\n', bias)

                elif '// hls-fpga-machine-learning insert lookup tables' in line:
                    for table in layer.get_attr('lookup_tables', []):
                        line = append_line(line, table)
                        line += '\n'

                elif '// hls-fpga-machine-learning insert helpers for different input ranks' in line:
                    """
                    Generate helper functions for the case of higher-rank input data, for example:
                        transform_1d(x) -> softmax(x)
                        transform_2d(x) -> map(transform_1d, x)
                        transform_3d(x) -> map(transform_2d, x)
                        // top-level function:
                        transform(x) -> transform_3d(x)
                    """
                    min_input_rank = layer.get_attr('xls_min_input_rank')
                    input_rank = len(input_var.shape)
                    for rank in range(min_input_rank, input_rank + 1):
                        input_type = input_var.type_alias.type
                        output_type = output_var.type_alias.type
                        # Get inner type
                        for k in range(input_rank - rank):
                            input_type = input_type.element_type
                            output_type = output_type.element_type
                        assert input_type.rank == rank, f'Input rank mismatch: expected {rank}, got {input_type.rank}'

                        name = f'transform_{rank}d'
                        params = []
                        args = f'x: {input_type}'
                        if rank == min_input_rank:
                            body = layer.get_attr('xls_func_call')
                        else:
                            body = XLSFunctionCall(name='map', params=[], args=['x', f'transform_{rank - 1}d'])

                        line = append_line(line, XLSFunctionDefinition(
                            name=name, params=params, args=args, output_type=output_type, body=body
                        ))

                elif '// hls-fpga-machine-learning insert top-level function call' in line:
                    input_rank = len(input_var.shape)
                    line = append_line(line, INDENT + str(
                        XLSFunctionCall(name=f'transform_{input_rank}d', params=[], args=['x'])))
                else:
                    pass
                f.write(line)

    def write_nnet_utils(self, model: ModelGraph) -> None:
        """Copy the nnet_utils, AP types headers to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        for dirname in 'nnet_utils', 'ap_types':
            srcpath = XLS_TEMPLATE_DIR / 'firmware' / dirname
            dstpath = firmware_dir(model) / dirname
            if os.path.exists(dstpath):
                rmtree(dstpath)
            copytree(srcpath, dstpath)

    def write_hls(self, model: ModelGraph) -> None:
        self.write_project_dir(model)
        self.write_build_script(model)
        self.write_project_dslx(model)
        self.write_layers(model)
        self.write_nnet_utils(model)
