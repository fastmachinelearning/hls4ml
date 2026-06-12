# Typing imports
from __future__ import annotations  # makes all annotations into strings

import tarfile
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hls4ml.backends.xls.xls_types import (
    XLSConst,
    XLSFunctionCall,
    XLSFunctionDefinition,
    XLSImport,
    XLSTensorVariable,
    XLSTypeAlias,
    XLSVariableDefinition,
)
from hls4ml.model.layers import Layer

if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph

import os
from shutil import copyfile, copytree, rmtree

from hls4ml.writer.writers import Writer

XLS_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / 'templates/xls'
INDENT = ' ' * 4


def firmware_dir(model: ModelGraph):
    return Path(model.config.get_output_dir()) / 'firmware'


def reports_dir(model: ModelGraph):
    return Path(model.config.get_output_dir()) / 'reports'


def append_line(line: str, x: Any, indent=None) -> str:
    if indent is None:
        indent = ''
    if isinstance(indent, int):
        indent = INDENT * indent
    return line + f'{indent}{x}\n'


def append_lines(s: str, *xs: Any, indent=None) -> str:
    # Allow append_lines(s, [1,2,3]) as well as append_lines(s, 1,2,3)
    if len(xs) == 1 and isinstance(xs[0], Iterable) and not isinstance(xs[0], (str, bytes)):
        xs = tuple(xs[0])

    for x in xs:
        s = append_line(s, x, indent=indent)
    return s


def to_tuple_or_singleton_str(xs: Iterable[Any], sep: str = ', ') -> str:
    xs = tuple(xs)
    assert len(xs) >= 1
    if len(xs) == 1:
        return str(xs[0])
    return '(' + sep.join(str(x) for x in xs) + ')'


class XLSWriter(Writer):
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
        for name in ('build_prj.tcl', 'constraints.xdc'):
            srcpath = XLS_TEMPLATE_DIR / name
            dstpath = Path(model.config.get_output_dir()) / name
            copyfile(srcpath, dstpath)

    def write_project_dslx(self, model: ModelGraph) -> None:
        """Write the main architecture source file (myproject.x)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        output_path = firmware_dir(model) / f'{model.config.get_project_name()}.x'

        layers = list(model.get_layers())

        output_vars = OrderedDict(
            (model.graph[output].get_attr('xls_module_name'), model.graph[output].get_attr('xls_output_variables')[0])
            for output in model.outputs
        )

        with open(output_path, 'w') as f:
            for line in open(XLS_TEMPLATE_DIR / 'firmware/myproject.x'):
                if 'myproject' in line:
                    line = line.replace('myproject', model.config.get_project_name())
                elif '// hls-fpga-machine-learning insert imports' in line:
                    line = append_lines(line, (XLSImport(layer.get_attr('xls_module_name')) for layer in layers))

                    for name in model.inputs:
                        i = model.graph[name].index
                        input_module = model.graph[name].get_attr('xls_module_name')
                        input_var = model.graph[name].get_attr('xls_input_variables')[0]
                        line = append_lines(
                            line,
                            XLSConst(
                                name=f'INPUT_{i}_BINARY_EXPONENT',
                                value=f'{input_module}::{input_var.binary_exponent.name}',
                                type='s32',
                            ),
                            XLSTypeAlias(name=f'Input_{i}_Type', type=f'{input_module}::{input_var.type_alias.name}'),
                            XLSTypeAlias(
                                name=f'Input_{i}_Type_Bits', type=f'{input_module}::{input_var.type_alias_bits.name}'
                            ),
                        )
                    for name in model.outputs:
                        i = model.graph[name].index
                        output_module = model.graph[name].get_attr('xls_module_name')
                        output_var = model.graph[name].get_attr('xls_output_variables')[0]
                        line = append_lines(
                            line,
                            XLSConst(
                                name=f'OUTPUT_{i}_NUM_BITS', value=f'{output_module}::{output_var.num_bits.name}', type='u32'
                            ),
                            XLSConst(
                                name=f'OUTPUT_{i}_BINARY_EXPONENT',
                                value=f'{output_module}::{output_var.binary_exponent.name}',
                                type='s32',
                            ),
                            XLSTypeAlias(name=f'Output_{i}_Type', type=f'{output_module}::{output_var.type_alias.name}'),
                            XLSTypeAlias(
                                name=f'Output_{i}_Type_Bits', type=f'{output_module}::{output_var.type_alias_bits.name}'
                            ),
                        )
                elif '// hls-fpga-machine-learning insert architecture input' in line:
                    for name in model.inputs:
                        i = model.graph[name].index
                        line = append_line(line, f'input_{i}: Input_{i}_Type,', indent=1)
                elif '// hls-fpga-machine-learning insert architecture output' in line:
                    output_types = [f'Output_{model.graph[name].index}_Type' for name in model.outputs]
                    line = append_line(line, to_tuple_or_singleton_str(output_types))

                elif '// hls-fpga-machine-learning insert layers' in line:
                    output_var_names = []
                    for layer in layers:
                        layer_module_name = layer.get_attr('xls_module_name')
                        layer_input_vars = layer.get_attr('xls_input_variables')
                        layer_output_vars = layer.get_attr('xls_output_variables')

                        if layer.class_name == 'Input':
                            assert len(layer.inputs) == 1, (
                                f'Input layer {layer.name} should have a single input, but got {len(layer.inputs)}.'
                            )
                            input_var_names = [f'input_{layer.index}']
                        else:
                            input_var_names = [var.name for var in layer_input_vars]
                        layer_output_var_names = [var.name for var in layer_output_vars]
                        if layer.name in model.outputs:
                            output_var_names += layer_output_var_names
                        line = append_line(
                            line,
                            XLSVariableDefinition(
                                name=to_tuple_or_singleton_str(layer_output_var_names),
                                value=XLSFunctionCall(name=f'{layer_module_name}::transform', args=input_var_names),
                            ),
                            indent=1,
                        )
                    line = append_line(line, to_tuple_or_singleton_str(output_var_names), indent=1)

                elif '// hls-fpga-machine-learning insert bits input' in line:
                    for name in model.inputs:
                        i = model.graph[name].index
                        line = append_line(line, f'input_bits_{i}: Input_{i}_Type_Bits,', indent=1)

                elif '// hls-fpga-machine-learning insert bits output' in line:
                    out_types = [f'Output_{model.graph[name].index}_Type_Bits' for name in model.outputs]
                    line = append_line(line, to_tuple_or_singleton_str(out_types))

                elif '// hls-fpga-machine-learning insert convert from bits' in line:
                    fixed_point_input_names = []
                    xls_statements: list[XLSVariableDefinition | str] = []
                    for name in model.inputs:
                        i = model.graph[name].index
                        bits_name = f'input_bits_{i}'
                        fixed_point_name = f'input_fixed_point_{i}'
                        input_var = model.graph[name].get_attr('xls_input_variables')[0]
                        rank = len(input_var.shape)
                        fixed_point_input_names.append(fixed_point_name)
                        xls_statements.append(
                            XLSVariableDefinition(
                                name=fixed_point_name,
                                value=XLSFunctionCall(
                                    name=f'fixed_point_util::make_fixed_points_{rank}d',
                                    params=[f'INPUT_{i}_BINARY_EXPONENT'],
                                    args=bits_name,
                                ),
                            )
                        )
                    output_fixed_point_names = tuple(
                        f'output_fixed_point_{output_var.name}' for output_var in output_vars.values()
                    )
                    xls_statements.append(
                        XLSVariableDefinition(
                            name=to_tuple_or_singleton_str(output_fixed_point_names),
                            value=XLSFunctionCall(
                                name=f'{model.config.get_project_name()}_fixed_point', args=fixed_point_input_names
                            ),
                        )
                    )

                    output_bits_names = []
                    for name in model.outputs:
                        output_layer = model.graph[name]
                        i = output_layer.index
                        output_var = output_layer.get_attr('xls_output_variables')[0]
                        bits_name = f'output_bits_{i}'
                        output_bits_names.append(bits_name)
                        fixed_point_name = f'output_fixed_point_{output_var.name}'
                        rank = len(output_var.shape)
                        xls_statements.append(
                            XLSVariableDefinition(
                                name=bits_name,
                                value=XLSFunctionCall(
                                    name=f'fixed_point_util::to_significand_{rank}d',
                                    params=[],
                                    args=fixed_point_name,
                                ),
                            )
                        )
                    xls_statements.append(to_tuple_or_singleton_str(output_bits_names))

                    line = append_lines(line, [f'{x}' for x in xls_statements], indent=1)

                elif '// hls-fpga-machine-learning insert top-level function call' in line:
                    line = append_line(
                        line,
                        XLSFunctionCall(
                            name='myproject_bits',
                            params=[],
                            args=[f'input_bits_{model.graph[name].index}' for name in model.inputs],
                        ),
                        indent=1,
                    )

                else:
                    pass

                f.write(line)

    def write_layers(self, model: ModelGraph):
        for layer in model.get_layers():
            self.write_layer(model, layer)

    def write_layer(self, model: ModelGraph, layer: Layer):
        layer_module_name = layer.get_attr('xls_module_name')
        input_vars: list[XLSTensorVariable] = layer.get_attr('xls_input_variables')
        output_vars: list[XLSTensorVariable] = layer.get_attr('xls_output_variables')
        with open(firmware_dir(model) / f'{layer_module_name}.x', 'w') as f:
            for line in open(XLS_TEMPLATE_DIR / 'firmware/layer.x'):
                if '// hls-fpga-machine-learning insert imports' in line:
                    imports = []
                    func_namespace = layer.get_attr('xls_func_call').name.module_name
                    if func_namespace is not None and func_namespace != 'fixed_point_util':
                        imports.append(XLSImport(name=f'nnet_utils.{func_namespace}'))
                    if layer.get_attr('lookup_tables'):
                        imports.append(XLSImport(name='nnet_utils.lookup_table'))
                    if layer.get_attr('data_format'):
                        imports.append(XLSImport(name='nnet_utils.data_format'))
                    line = append_lines(line, imports)

                elif '// hls-fpga-machine-learning insert types' in line:
                    for in_out_vars in (input_vars, output_vars):
                        for var in in_out_vars:
                            line = append_lines(line, var.definitions())
                            line += '\n'

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

                elif '// hls-fpga-machine-learning insert other constants' in line:
                    # NB: sometimes constant is already defined, e.g. output dimensions for Reshape layer
                    # In that case, we don't write it again.
                    existing_names = {
                        x.name
                        for in_out_vars in (input_vars, output_vars)
                        for var in in_out_vars
                        for x in var.definitions()
                        if isinstance(x, XLSConst)
                    }
                    extra_consts = (
                        x
                        for key in ('xls_extra_func_params', 'xls_extra_func_args')
                        for x in layer.get_attr(key)
                        if x.name not in existing_names
                    )
                    line = append_lines(line, extra_consts)

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
                    input_rank = len(input_vars[0].shape)
                    for rank in range(min_input_rank, input_rank + 1):
                        input_types = [input_var.type_alias.type for input_var in input_vars]
                        output_types = [output_var.type_alias.type for output_var in output_vars]
                        # Get inner type
                        for _ in range(input_rank - rank):
                            input_types = [input_type.element_type for input_type in input_types]
                            output_types = [output_type.element_type for output_type in output_types]
                        assert input_types[0].rank == rank, (
                            f'Input rank mismatch: expected {rank}, got {input_types[0].rank}'
                        )

                        name = f'transform_{rank}d'
                        params = []
                        args = [f'x_{i}: {input_type}' for i, input_type in enumerate(input_types)]

                        output_type = to_tuple_or_singleton_str(output_types)

                        if rank == min_input_rank:
                            body = layer.get_attr('xls_func_call')
                        else:
                            dim_0 = input_types[0].shape[0]
                            acc_vars = tuple(f'acc_{i}' for i in range(len(output_types)))
                            out_var_i = tuple(f'out_{i}' for i in range(len(output_types)))
                            in_vars_i = [f'x_{i}[i]' for i, input_type in enumerate(input_types)]
                            transform_i = XLSVariableDefinition(
                                name=to_tuple_or_singleton_str(out_var_i),
                                value=XLSFunctionCall(name=f'transform_{rank - 1}d', args=in_vars_i),
                            )
                            update_i = to_tuple_or_singleton_str(
                                [f'update({acc}, i, out_{i})' for i, acc in enumerate(acc_vars)]
                            )
                            body = f"""{INDENT}for (i, {to_tuple_or_singleton_str(acc_vars)}) in 0..{dim_0} {{
{INDENT}{INDENT}{transform_i}
{INDENT}{INDENT}{update_i}
{INDENT}}}(zero!<{output_type}>())
                            """
                        line = append_line(
                            line,
                            XLSFunctionDefinition(name=name, params=params, args=args, output_type=output_type, body=body),
                        )
                elif '// hls-fpga-machine-learning insert layer input' in line:
                    input_args = [f'{INDENT}x_{i}: {input_var.type_alias.name}' for i, input_var in enumerate(input_vars)]
                    line = append_line(line, ',\n'.join(input_args))
                elif '// hls-fpga-machine-learning insert layer output' in line:
                    output_types = to_tuple_or_singleton_str(output_var.type_alias.name for output_var in output_vars)
                    line = append_line(line, f'{output_types}')

                elif '// hls-fpga-machine-learning insert top-level function call' in line:
                    input_rank = len(input_vars[0].shape)
                    line = append_line(
                        line,
                        XLSFunctionCall(
                            name=f'transform_{input_rank}d', params=[], args=[f'x_{i}' for i in range(len(input_vars))]
                        ),
                        indent=1,
                    )
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

    @staticmethod
    def write_tar(model):
        """Write the generated project as a .tar.gz archive

        Args:
            model (ModelGraph): the hls4ml model.
        """

        write_tar = model.config.get_writer_config().get('WriteTar', False)
        if write_tar:
            tar_path = Path(model.config.get_output_dir() + '.tar.gz')
            tar_path.unlink(missing_ok=True)
            with tarfile.open(tar_path, mode='w:gz') as archive:
                archive.add(model.config.get_output_dir(), recursive=True, arcname='')

    def write_hls(self, model: ModelGraph) -> None:
        self.write_project_dir(model)
        self.write_build_script(model)
        self.write_project_dslx(model)
        self.write_layers(model)
        self.write_nnet_utils(model)
        self.write_tar(model)
