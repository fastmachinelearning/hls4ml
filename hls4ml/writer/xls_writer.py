# Typing imports
from __future__ import annotations  # makes all annotations into strings

import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hls4ml.backends.xls.xls_layer_util import (
    xls_extra_func_args,
    xls_extra_func_params,
    xls_func_call,
    xls_min_input_rank,
    xls_module_name,
    xls_weights_definitions,
)
from hls4ml.backends.xls.xls_types import (
    XLSConstDefinition,
    XLSFunctionCallDefinition,
    XLSTensorVariableDefinition,
    XLSTypeAliasDefinition,
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


def xls_import_const_definition(const_def: XLSConstDefinition, module_name, new_name=None) -> XLSConstDefinition:
    """Returns: const new_name = module_name::const_name"""
    return XLSConstDefinition(name=new_name or const_def.name, value=f'{module_name}::{const_def.name}', type=const_def.type)


def xls_import_type_alias(type_alias_def: XLSTypeAliasDefinition, module_name, new_name=None) -> XLSTypeAliasDefinition:
    """Returns: type new_name = module_name::type_name"""
    return XLSTypeAliasDefinition(name=new_name or type_alias_def.name, type=f'{module_name}::{type_alias_def.name}')


def xls_import_module(name: str, alias: str | None = None) -> str:
    as_alias = f' as {alias}' if alias else ''
    return f'import {name}{as_alias};'


def xls_function_definition(name, params, args, output_type, body) -> str:
    params = params or []
    args = args or []
    output_type = output_type or '()'
    body = body or ''

    if not isinstance(params, str):
        params = ', '.join(map(str, params))
    if params:
        params = f'<{params}>'
    if not isinstance(args, str):
        args = ', '.join(map(str, args))
    return f"""pub fn {name}{params}({args})
    -> {output_type} {{
    {body}
}}"""


def xls_variable_definition(name, value, type=None):
    type = f': {type}' if type else ''
    return f'let {name}{type} = {value};'


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

        with open(output_path, 'w') as f:
            for line in open(XLS_TEMPLATE_DIR / 'firmware/myproject.x'):
                if 'myproject' in line:
                    line = line.replace('myproject', model.config.get_project_name())
                elif '// hls-fpga-machine-learning insert imports' in line:
                    line = append_lines(line, [xls_import_module(xls_module_name(layer)) for layer in model.get_layers()])

                    for inp in model.inputs:
                        input_layer = model.graph[inp]
                        input_module = xls_module_name(input_layer)
                        input_var = input_layer.get_output_variable()
                        line = append_lines(
                            line,
                            xls_import_const_definition(input_var.xls_binary_exponent, input_module),
                            xls_import_type_alias(input_var.xls_type_alias(), input_module),
                            xls_import_type_alias(input_var.xls_type_alias_bits, input_module),
                        )
                    for out in model.outputs:
                        output_layer = model.graph[out]
                        output_module = xls_module_name(output_layer)
                        output_var = output_layer.get_output_variable()
                        line = append_lines(
                            line,
                            xls_import_const_definition(output_var.xls_num_bits, output_module),
                            xls_import_const_definition(output_var.xls_binary_exponent, output_module),
                            xls_import_type_alias(output_var.xls_type_alias(), output_module),
                            xls_import_type_alias(output_var.xls_type_alias_bits, output_module),
                        )
                elif '// hls-fpga-machine-learning insert architecture input' in line:
                    for inp in model.inputs:
                        input_layer = model.graph[inp]
                        input_var = input_layer.get_output_variable()
                        line = append_line(line, f'x_{input_layer.index}: {input_var.xls_type_alias().name},', indent=1)
                elif '// hls-fpga-machine-learning insert architecture output' in line:
                    output_types = [model.graph[out].get_output_variable().xls_type_alias().name for out in model.outputs]
                    line = append_line(line, to_tuple_or_singleton_str(output_types))

                elif '// hls-fpga-machine-learning insert layers' in line:
                    output_var_names = []
                    for layer in list(model.get_layers()):
                        layer_module_name = xls_module_name(layer)
                        if layer.class_name == 'Input':
                            layer_input_var_names = [f'x_{layer.index}']
                        else:
                            layer_input_var_names = [layer.get_input_variable(inp).xls_name for inp in layer.inputs]
                        layer_output_var_names = [var.xls_name for var in (layer.get_variables())]
                        if layer.name in model.outputs:
                            output_var_names += layer_output_var_names
                        line = append_line(
                            line,
                            xls_variable_definition(
                                name=to_tuple_or_singleton_str(layer_output_var_names),
                                value=XLSFunctionCallDefinition(
                                    name=f'{layer_module_name}::transform', args=layer_input_var_names
                                ),
                            ),
                            indent=1,
                        )
                    line = append_line(line, to_tuple_or_singleton_str(output_var_names), indent=1)

                elif '// hls-fpga-machine-learning insert bits input' in line:
                    for input_var in model.get_input_variables():
                        line = append_line(
                            line, f'{input_var.xls_name}_bits: {input_var.xls_type_alias_bits.name},', indent=1
                        )

                elif '// hls-fpga-machine-learning insert bits output' in line:
                    out_types = [f'{output_var.xls_type_alias_bits.name}' for output_var in model.get_output_variables()]
                    line = append_line(line, to_tuple_or_singleton_str(out_types))

                elif '// hls-fpga-machine-learning insert convert from bits' in line:
                    input_names = []
                    xls_statements: list[xls_variable_definition | str] = []
                    for input_var in model.get_input_variables():
                        input_name = input_var.xls_name
                        input_name_bits = f'{input_name}_bits'
                        rank = len(input_var.shape)
                        input_names.append(input_name)
                        xls_statements.append(
                            xls_variable_definition(
                                name=input_name,
                                value=XLSFunctionCallDefinition(
                                    name=f'fixed_point_util::make_fixed_points_{rank}d',
                                    params=[input_var.xls_binary_exponent.name],
                                    args=input_name_bits,
                                ),
                            )
                        )
                    output_names = tuple(f'{output_var.xls_name}' for output_var in model.get_output_variables())
                    xls_statements.append(
                        xls_variable_definition(
                            name=to_tuple_or_singleton_str(output_names),
                            value=XLSFunctionCallDefinition(
                                name=f'{model.config.get_project_name()}_fixed_point', args=input_names
                            ),
                        )
                    )

                    output_names_bits = []
                    for output_var in model.get_output_variables():
                        output_name = output_var.xls_name
                        output_name_bits = f'{output_name}_bits'
                        output_names_bits.append(output_name_bits)
                        rank = len(output_var.shape)
                        xls_statements.append(
                            xls_variable_definition(
                                name=output_name_bits,
                                value=XLSFunctionCallDefinition(
                                    name=f'fixed_point_util::to_significand_{rank}d',
                                    params=[],
                                    args=output_name,
                                ),
                            )
                        )
                    xls_statements.append(to_tuple_or_singleton_str(output_names_bits))

                    line = append_lines(line, [f'{x}' for x in xls_statements], indent=1)

                elif '// hls-fpga-machine-learning insert top-level function call' in line:
                    line = append_line(
                        line,
                        XLSFunctionCallDefinition(
                            name='myproject_bits',
                            params=[],
                            args=[f'{input_var.xls_name}_bits' for input_var in model.get_input_variables()],
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
        output_vars: list[XLSTensorVariableDefinition] = list(layer.get_variables())
        input_vars: list[XLSTensorVariableDefinition] = (
            list(layer.get_variables())
            if layer.class_name == 'Input'
            else [layer.get_input_variable(inp) for inp in layer.inputs]
        )
        func_call = xls_func_call(layer)
        with open(firmware_dir(model) / f'{xls_module_name(layer)}.x', 'w') as f:
            for line in open(XLS_TEMPLATE_DIR / 'firmware/layer.x'):
                if '// hls-fpga-machine-learning insert imports' in line:
                    imports = []
                    if layer.class_name != 'Input':
                        imports += [xls_module_name(model.graph[inp]) for inp in layer.inputs]
                    func_namespace = func_call.name.module_name
                    if func_namespace is not None and func_namespace != 'fixed_point_util':
                        imports.append(f'nnet_utils.{func_namespace}')
                    if layer.get_attr('lookup_tables'):
                        imports.append('nnet_utils.lookup_table')
                    if layer.get_attr('data_format'):
                        imports.append('nnet_utils.data_format')
                    line = append_lines(line, [xls_import_module(x) for x in imports])
                    if layer.class_name != 'Input':
                        for inp in layer.inputs:
                            line += '\n'
                            input_module = xls_module_name(model.graph[inp])
                            input_var = layer.get_input_variable(inp)
                            for dim in input_var.xls_dims:
                                line = append_line(line, xls_import_const_definition(dim, module_name=input_module))
                            for rank in list(range(1, len(input_var.shape) + 1)) + [None]:
                                alias = xls_import_type_alias(input_var.xls_type_alias(rank), module_name=input_module)
                                line = append_line(line, alias)
                    line += '\n'

                elif '// hls-fpga-machine-learning insert types' in line:
                    for var in layer.get_variables():
                        line = append_line(line, var.xls_definition())
                        line += '\n'

                elif '// hls-fpga-machine-learning insert weights' in line:
                    line = append_lines(line, xls_weights_definitions(layer))

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
                        for x in var.xls_definitions()
                        if isinstance(x, XLSConstDefinition)
                    }
                    extra_consts = (
                        x
                        for consts in (xls_extra_func_params(layer), xls_extra_func_args(layer))
                        for x in consts
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
                    min_input_rank = xls_min_input_rank(layer)
                    input_rank = len(input_vars[0].shape)
                    max_extra_rank = input_rank - min_input_rank

                    min_input_ranks = [len(input_var.shape) - max_extra_rank for input_var in input_vars]
                    min_output_ranks = [len(output_var.shape) - max_extra_rank for output_var in output_vars]

                    for extra_rank in range(max_extra_rank + 1):
                        rank = min_input_rank + extra_rank
                        input_types = [
                            input_var.xls_type_alias(rank=min_rank + extra_rank)
                            for input_var, min_rank in zip(input_vars, min_input_ranks)
                        ]
                        output_types = [
                            output_var.xls_type_alias(rank=min_rank + extra_rank)
                            for output_var, min_rank in zip(output_vars, min_output_ranks)
                        ]
                        name = f'transform_{rank}d'
                        params = []
                        args = [f'x_{i}: {input_type.name}' for i, input_type in enumerate(input_types)]

                        output_type = to_tuple_or_singleton_str(x.name for x in output_types)
                        if extra_rank == 0:
                            body = func_call
                        else:
                            dim_0 = input_types[0].type.shape[0]
                            acc_vars = tuple(f'acc_{i}' for i in range(len(output_types)))
                            out_var_i = tuple(f'out_{i}' for i in range(len(output_types)))
                            in_vars_i = [f'x_{i}[i]' for i, input_type in enumerate(input_types)]
                            transform_i = xls_variable_definition(
                                name=to_tuple_or_singleton_str(out_var_i),
                                value=XLSFunctionCallDefinition(name=f'transform_{rank - 1}d', args=in_vars_i),
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
                            xls_function_definition(name=name, params=params, args=args, output_type=output_type, body=body),
                        )
                elif '// hls-fpga-machine-learning insert layer input' in line:
                    input_args = [
                        f'{INDENT}{input_var.xls_name}: {input_var.xls_type_alias().name}'
                        for i, input_var in enumerate(input_vars)
                    ]
                    line = append_line(line, ',\n'.join(input_args))
                elif '// hls-fpga-machine-learning insert layer output' in line:
                    output_types = to_tuple_or_singleton_str(output_var.xls_type_alias().name for output_var in output_vars)
                    line = append_line(line, f'{output_types}')

                elif '// hls-fpga-machine-learning insert top-level function call' in line:
                    input_rank = len(input_vars[0].shape)
                    line = append_line(
                        line,
                        XLSFunctionCallDefinition(
                            name=f'transform_{input_rank}d',
                            params=[],
                            args=[f'{input_var.xls_name}' for input_var in input_vars],
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
