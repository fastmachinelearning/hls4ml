import os
import stat
from pathlib import Path

from hls4ml.backends.vitis_unified.vitis_unified_config import VitisUnifiedConfig
from hls4ml.writer.vitis_writer import VitisWriter


class VitisUnifiedWriter(VitisWriter):
    """Writer for Vitis Unified backend.

    This class follows the same pattern as Vivado/Vitis writers:
    all generation logic is implemented as writer instance methods instead of
    split helper generator classes.
    """

    def __init__(self):
        super().__init__()
        self.vitis_unified_config = None

    def write_tar(self, model):
        super().write_tar(model)

    # ===== Public helpers used by backend/passes =====
    def get_vitis_unified_working_directory(self, model):
        return os.path.join(model.config.get_output_dir(), 'vitis_workspace')

    def get_vitis_hls_dir(self, model):
        return os.path.join(self.get_vitis_unified_working_directory(model), model.config.get_project_name())

    def get_vitis_hls_exec_dir(self, model):
        return os.path.join(self.get_vitis_hls_dir(model), 'vitis_unified_project')

    def get_vitis_linker_dir(self, model):
        return os.path.join(self.get_vitis_unified_working_directory(model), 'system_link')

    # ===== Internal helpers =====
    def _set_unified_config(self, model):
        self.vitis_unified_config = VitisUnifiedConfig(
            model.config, model.get_input_variables(), model.get_output_variables()
        )

    def _is_axi_stream(self):
        return self.vitis_unified_config.get_axi_mode() == 'axis'

    def _is_axi_master(self):
        return self.vitis_unified_config.get_axi_mode() == 'axim'

    def _get_project_name(self, model):
        return model.config.get_project_name()

    def _get_wrapper_file_name(self, model, is_axi_master):
        suffix = 'axim' if is_axi_master else 'axis'
        return f'{self._get_project_name(model)}_{suffix}'

    def _get_sim_file_name(self):
        return 'myproject_test'

    def _get_top_wrap_func_name(self, model, is_axi_master):
        return self._get_wrapper_file_name(model, is_axi_master)

    def _get_xo_file_path(self, model):
        xo_name = f'{self._get_top_wrap_func_name(model, self._is_axi_master())}.xo'
        return os.path.join(self.get_vitis_hls_exec_dir(model), xo_name)

    def _get_io_port_name(self, tensor_var, is_input, idx):
        direction = 'in' if is_input else 'out'
        return f'gmem_{direction}{idx}_ptr_{tensor_var.name}'

    def _get_local_stream_name(self, tensor_var, is_input, idx):
        direction = 'in' if is_input else 'out'
        return f'stream_{direction}{idx}_{tensor_var.name}'

    def _get_dma_type_name(self):
        return 'dma_data_packet'

    @staticmethod
    def _gen_hex_addr_list(start_addr, stride, size, indent):
        return [f'{indent}{hex(start_addr + index * stride)}' for index in range(size)]

    def _gen_io_signature(self, indent, input_type, output_type, inputs, outputs):
        input_ptrs = [f'{indent} {input_type}* {self._get_io_port_name(inp, True, idx)}' for idx, inp in enumerate(inputs)]
        output_ptrs = [
            f'{indent} {output_type}* {self._get_io_port_name(out, False, idx)}' for idx, out in enumerate(outputs)
        ]
        return ', '.join(input_ptrs) + ',\n' + ', '.join(output_ptrs) + '\n'

    def _ensure_export_path(self, model):
        export_path = Path(model.config.get_output_dir()) / 'export'
        export_path.mkdir(parents=True, exist_ok=True)

    # ===== Build/config generation =====
    def write_build_script(self, model):
        self._write_bridge_build_script(model)
        self._build_unified_project_skeleton(model)
        self._write_hls_kernel_config(model, is_csim=True)
        self._write_hls_kernel_config(model, is_csim=False)
        self._write_linker_dir(model)
        self._write_linker_launcher(model)
        self._write_linker_config(model)

    def _write_bridge_build_script(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/vitis_unified/build_lib.sh')) as fin,
            open(f'{model.config.get_output_dir()}/build_lib.sh', 'w') as fout,
        ):
            for line in fin.readlines():
                if 'myprojectBaseName' in line:
                    line = line.replace('myprojectBaseName', self._get_project_name(model))
                if 'myprojectWrapName' in line:
                    line = line.replace('myprojectWrapName', self._get_wrapper_file_name(model, self._is_axi_master()))
                if 'mystamp' in line:
                    line = line.replace('mystamp', model.config.get_config_value('Stamp'))
                fout.write(line)

        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

    def _write_hls_kernel_config(self, model, is_csim=False):
        filedir = os.path.dirname(os.path.abspath(__file__))
        suffix = 'csim' if is_csim else 'cosim'
        with (
            open(os.path.join(filedir, '../templates/vitis_unified/hls_kernel_config.cfg')) as fin,
            open(f'{model.config.get_output_dir()}/hls_kernel_config_{suffix}.cfg', 'w') as fout,
        ):
            for line in fin.readlines():
                if '{PART}' in line:
                    line = line.replace('{PART}', model.config.get_config_value('Part'))
                if '{CLK}' in line:
                    line = line.replace('{CLK}', model.config.get_config_value('ClockPeriod'))
                if '{CLK_UC}' in line:
                    line = line.replace('{CLK_UC}', model.config.get_config_value('ClockUncertainty'))
                if '{OUTDIR}' in line:
                    line = line.replace('{OUTDIR}', model.config.get_output_dir())
                if '{TOP_NAME}' in line:
                    line = line.replace('{TOP_NAME}', self._get_top_wrap_func_name(model, self._is_axi_master()))
                if '{FILE_NAME_WRAP}' in line:
                    line = line.replace('{FILE_NAME_WRAP}', self._get_wrapper_file_name(model, self._is_axi_master()))
                if '{SIM_FILE_NAME}' in line:
                    line = line.replace('{SIM_FILE_NAME}', self._get_sim_file_name())
                if '{FILE_NAME_BASE}' in line:
                    line = line.replace('{FILE_NAME_BASE}', self._get_project_name(model))
                if '{OUTPUT_KERNEL_TYPE}' in line:
                    line = line.replace('{OUTPUT_KERNEL_TYPE}', 'xo')
                if is_csim and (('enable_fifo_sizing' in line) or ('-DRTL_SIM' in line)):
                    line = '#' + line
                fout.write(line)

    def _build_unified_project_skeleton(self, model):
        workspace_dir = self.get_vitis_unified_working_directory(model)
        hls_dir = self.get_vitis_hls_dir(model)
        exec_dir = self.get_vitis_hls_dir(model)
        vitis_comp = os.path.join(str(hls_dir), 'vitis-comp.json')

        os.makedirs(workspace_dir, exist_ok=True)
        os.makedirs(hls_dir, exist_ok=True)
        os.makedirs(exec_dir, exist_ok=True)

        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/vitis_unified/vitis_workspace/kernel_project/vitis-comp.json')) as fin,
            open(vitis_comp, 'w') as fout,
        ):
            for line in fin.readlines():
                if '{HLS_NAME}' in line:
                    line = line.replace('{HLS_NAME}', self._get_project_name(model))
                if '{CONFIG_FILE}' in line:
                    line = line.replace('{CONFIG_FILE}', f'{model.config.get_output_dir()}/hls_kernel_config.cfg')
                fout.write(line)

    def _write_linker_dir(self, model):
        os.makedirs(self.get_vitis_linker_dir(model), exist_ok=True)

    def _write_linker_launcher(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/vitis_unified/vitis_workspace/system_link/link_system.sh')) as fin,
            open(f'{self.get_vitis_linker_dir(model)}/link_system.sh', 'w') as fout,
        ):
            for line in fin.readlines():
                if '{PLATFORM_XPFM}' in line:
                    line = line.replace('{PLATFORM_XPFM}', self.vitis_unified_config.get_XPFMPath())
                if '{KERNEL_XO}' in line:
                    line = line.replace('{KERNEL_XO}', self._get_xo_file_path(model))
                if '{PROJECT_NAME}' in line:
                    line = line.replace('{PROJECT_NAME}', self._get_project_name(model))
                fout.write(line)

        link_lib_dst = Path(f'{self.get_vitis_linker_dir(model)}/link_system.sh').resolve()
        link_lib_dst.chmod(link_lib_dst.stat().st_mode | stat.S_IEXEC)

    def _write_linker_config(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/vitis_unified/vitis_workspace/system_link/link_system.cfg')) as fin,
            open(f'{self.get_vitis_linker_dir(model)}/link_system.cfg', 'w') as fout,
        ):
            for line in fin.readlines():
                if '{CLK}' in line:
                    line = line.replace('{CLK}', str(100_000_000))
                if '{KERNEL_NAME}' in line:
                    line = line.replace('{KERNEL_NAME}', self._get_top_wrap_func_name(model, self._is_axi_master()))
                if '{GUI_STATUS}' in line:
                    line = line.replace('{GUI_STATUS}', 'true')
                if '# hls-fpga-machine-learning insert custom connection' in line and self._is_axi_stream():
                    top_module_name = self._get_top_wrap_func_name(model, False)
                    top_mod_inst_name = top_module_name + '_1'
                    line += '\n'
                    line += '[connectivity]\n'
                    line += f'nk={top_module_name}:1:{top_mod_inst_name}\n'
                    line += f'stream_connect=DMA_MM2S:{top_mod_inst_name}.in\n'
                    line += f'stream_connect={top_mod_inst_name}.out:DMA_S2MM\n'
                fout.write(line)

    # ===== Bridge generation =====
    def write_bridge(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/vitis_unified/myproject_bridge.cpp')) as fin,
            open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp', 'w') as fout,
        ):
            model_inputs = model.get_input_variables()
            model_outputs = model.get_output_variables()
            model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']
            indent = '    '

            for line in fin.readlines():
                newline = ''
                if 'MYPROJECT' in line:
                    newline = line.replace('MYPROJECT', self._get_project_name(model).upper())
                elif 'myproject' in line:
                    newline = line.replace('myproject', self._get_project_name(model))
                elif 'PROJECT_FILE_NAME' in line:
                    newline = line.replace('PROJECT_FILE_NAME', self._get_wrapper_file_name(model, self._is_axi_master()))
                elif '// hls-fpga-machine-learning insert bram' in line:
                    newline = line
                    for bram in model_brams:
                        newline += f'#include "firmware/weights/{bram.name}.h"\n'
                elif '// hls-fpga-machine-learning insert header' in line:
                    dtype = line.split('#', 1)[1].strip()
                    input_ios = [
                        f'{dtype} {self._get_io_port_name(inp, True, idx)}[{inp.size_cpp()}]'
                        for idx, inp in enumerate(model_inputs)
                    ]
                    output_ios = [
                        f'{dtype} {self._get_io_port_name(out, False, idx)}[{out.size_cpp()}]'
                        for idx, out in enumerate(model_outputs)
                    ]
                    newline = indent + ', '.join(input_ios) + ',\n'
                    newline += indent + ', '.join(output_ios) + '\n'
                elif '// hls-fpga-machine-learning insert wrapper' in line:
                    dtype = line.split('#', 1)[1].strip()
                    newline = ''
                    if dtype == self.vitis_unified_config.get_input_type():
                        if self._is_axi_master():
                            input_vars = [self._get_io_port_name(inp, True, idx) for idx, inp in enumerate(model_inputs)]
                            output_vars = [self._get_io_port_name(out, False, idx) for idx, out in enumerate(model_outputs)]
                            newline += indent + self._get_top_wrap_func_name(model, True) + '(\n'
                            newline += indent + ', '.join(input_vars) + ',\n'
                            newline += indent + ', '.join(output_vars) + ',\n'
                            newline += indent + '1);\n'
                        else:
                            assert len(model_inputs) == 1
                            assert len(model_outputs) == 1
                            inp = model_inputs[0]
                            out = model_outputs[0]
                            inp_func = self._get_io_port_name(inp, True, 0)
                            inp_stream = inp_func + '_ap'
                            out_func = self._get_io_port_name(out, False, 0)
                            out_stream = out_func + '_ap'
                            newline = indent + f'hls::stream<{self._get_dma_type_name()}> {inp_stream};\n'
                            newline += (
                                indent + f'nnet::convert_data_axis<{dtype},{dtype}, N_IN>({inp_func}, {inp_stream});\n'
                            )
                            newline += indent + f'hls::stream<{self._get_dma_type_name()}> {out_stream};\n'
                            newline += indent + self._get_top_wrap_func_name(model, False) + '('
                            newline += inp_stream + ', ' + out_stream + ');\n'
                            newline += (
                                indent + f'nnet::convert_data_axis<{dtype},{dtype}, N_OUT>({out_stream}, {out_func});\n'
                            )
                elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                    newline = ''
                    for layer in model.get_layers():
                        func = layer.get_attr('function_cpp', None)
                        if func and model.config.trace_output and layer.get_attr('trace', False):
                            for var in layer.get_variables():
                                newline += (
                                    indent
                                    + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                    + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                                )
                elif '// hls-fpga-machine-learning insert namespace' in line:
                    newline = ''
                    namespace = model.config.get_writer_config().get('Namespace', None)
                    if namespace is not None:
                        newline += indent + f'using namespace {namespace};\n'
                else:
                    newline = line
                fout.write(newline)

    # ===== Wrapper generation =====
    def write_wrapper(self, model):
        if self._is_axi_master():
            self._write_wrapper_axim(model)
        else:
            self._write_wrapper_axis(model)

    def _write_wrapper_axis(self, model):
        inp_gmem_t, out_gmem_t, inputs, outputs = self.vitis_unified_config.get_corrected_types()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                'AXIS wrapper requires exactly 1 input and 1 output port. '
                f'Found {len(inputs)} inputs and {len(outputs)} outputs.'
            )

        inp, out = inputs[0], outputs[0]
        indent = '      '
        filedir = os.path.dirname(os.path.abspath(__file__))

        with (
            open(os.path.join(filedir, '../templates/vitis_unified/myproject_axis.cpp')) as fin,
            open(f'{model.config.get_output_dir()}/firmware/{self._get_project_name(model)}_axis.cpp', 'w') as fout,
        ):
            for line in fin.readlines():
                if 'MY_PROJECT_TOP_FUNC' in line:
                    newline = line.replace('MY_PROJECT_TOP_FUNC', self._get_top_wrap_func_name(model, False))
                elif '// hls-fpga-machine-learning insert include' in line:
                    newline = f'#include "{self._get_project_name(model)}_axis.h"\n'
                elif '// hls-fpga-machine-learning insert interface' in line:
                    newline = (
                        indent
                        + '#pragma HLS INTERFACE axis port=in\n'
                        + indent
                        + '#pragma HLS INTERFACE axis port=out\n'
                        + indent
                        + '#pragma HLS INTERFACE ap_ctrl_none port=return\n'
                        + indent
                        + '#pragma HLS DATAFLOW\n'
                    )
                elif '// hls-fpga-machine-learning insert local vars' in line:
                    newline = ''
                    newline += indent + 'bool is_last = false;\n'
                    newline += indent + f'hls::stream<{inp.type.name}> in_local("input_1");\n'
                    newline += indent + f'hls::stream<{out.type.name}> out_local("output_1");\n\n'
                    newline += indent + '#pragma HLS STREAM variable=in_local depth={}\n'.format(
                        model.get_input_variables()[0].pragma[1]
                    )
                    newline += indent + '#pragma HLS STREAM variable=out_local depth={}\n'.format(
                        model.get_output_variables()[0].pragma[1]
                    )
                elif '// hls-fpga-machine-learning insert enqueue' in line:
                    newline = ''
                    newline += indent + '/// enqueue input data\n'
                    newline += indent + f'{self._get_dma_type_name()} tmp;\n'
                    newline += indent + 'for(unsigned i = 0; i < N_IN / {input_t}::size; ++i) {{\n'
                    newline += indent + indent + '{input_t} ctype;\n'
                    newline += indent + indent + 'for(unsigned j = 0; j < {input_t}::size; j++) {{\n'
                    newline += indent + indent + indent + 'in.read(tmp);\n'
                    newline += indent + indent + indent + 'ctype[j] = tmp.data;\n'
                    newline += indent + indent + indent + 'is_last = tmp.last;\n'
                    newline += indent + indent + '}\n'
                    newline += indent + indent + 'in_local.write(ctype);\n'
                    newline += indent + '}\n'
                    newline += indent + 'tmp.last = 0;\n'
                    newline = newline.format(input_t=inp.type.name)
                elif '// hls-fpga-machine-learning insert call' in line:
                    newline = indent + f'{self._get_project_name(model)}(in_local, out_local);\n'
                elif '// hls-fpga-machine-learning insert dequeue' in line:
                    newline = ''
                    newline += indent + 'for(unsigned i = 0; i < N_OUT / {result_t}::size; ++i) {{\n'
                    newline += indent + indent + '{result_t} ctype = out_local.read();\n'
                    newline += indent + indent + 'for(unsigned j = 0; j < {result_t}::size; j++) {{\n'
                    newline += indent + indent + indent + f'tmp.data = ({inp_gmem_t}) (ctype[j]);\n'
                    newline += indent + indent + indent + 'if(is_last) {tmp.last = (((i+1)*(j+1))==N_OUT);}\n'
                    newline += indent + indent + indent + 'out.write(tmp);\n'
                    newline += indent + indent + '}\n'
                    newline += indent + '}\n'
                    newline = newline.format(result_t=out.type.name)
                else:
                    newline = line
                fout.write(newline)

        with (
            open(os.path.join(filedir, '../templates/vitis_unified/myproject_axis.h')) as fin,
            open(f'{model.config.get_output_dir()}/firmware/{self._get_project_name(model)}_axis.h', 'w') as fout,
        ):
            for line in fin.readlines():
                if 'MYPROJECT' in line:
                    newline = line.replace('MYPROJECT', self._get_project_name(model).upper())
                elif '// hls-fpga-machine-learning insert include' in line:
                    newline = f'#include "{self._get_project_name(model)}.h"\n#include "ap_axi_sdata.h"\n'
                elif 'MY_PROJECT_TOP_FUNC' in line:
                    newline = line.replace('MY_PROJECT_TOP_FUNC', self._get_top_wrap_func_name(model, False))
                elif '// hls-fpga-machine-learning insert definitions' in line:
                    newline = ''
                    newline += f'static const unsigned N_IN = {inp.size()};\n'
                    newline += f'static const unsigned N_OUT = {out.size()};\n'
                    newline += f'typedef hls::axis<{inp_gmem_t}, 0, 0, 0> {self._get_dma_type_name()};\n'
                else:
                    newline = line
                fout.write(newline)

    def _write_wrapper_axim(self, model):
        inp_gmem_t, out_gmem_t, inputs, outputs = self.vitis_unified_config.get_corrected_types()
        indent = '      '
        filedir = os.path.dirname(os.path.abspath(__file__))

        with (
            open(os.path.join(filedir, f'../templates/vitis_unified/{self._get_project_name(model)}_axim.cpp')) as fin,
            open(f'{model.config.get_output_dir()}/firmware/{self._get_wrapper_file_name(model, True)}.cpp', 'w') as fout,
        ):
            for line in fin.readlines():
                if 'MY_PROJECT_DM_INC' in line:
                    line = line.replace('MY_PROJECT_DM_INC', self._get_wrapper_file_name(model, True))
                elif 'MY_PROJECT_TOP_FUNC' in line:
                    line = line.replace('MY_PROJECT_TOP_FUNC', self._get_top_wrap_func_name(model, True))
                elif 'STREAM_BUF_IN_SZ' in line:
                    line = line.replace('VAL', str(self.vitis_unified_config.get_in_stream_buf_size()))
                elif 'STREAM_BUF_OUT_SZ' in line:
                    line = line.replace('VAL', str(self.vitis_unified_config.get_out_stream_buf_size()))
                elif '// vitis-unified-wrapper-io' in line:
                    line = self._gen_io_signature(indent, inp_gmem_t, out_gmem_t, inputs, outputs) + '\n'
                elif '// vitis-unified-wrapper-interface' in line:
                    for input_idx, inp in enumerate(inputs):
                        line += (
                            f'#pragma HLS INTERFACE m_axi     port={self._get_io_port_name(inp, True, input_idx)} '
                            f'bundle = gmem_in{input_idx} depth={inp.size()}\n'
                        )
                    for output_idx, out in enumerate(outputs):
                        line += (
                            f'#pragma HLS INTERFACE m_axi     port={self._get_io_port_name(out, False, output_idx)} '
                            f'bundle = gmem_out{output_idx} depth={out.size()}\n'
                        )
                elif '// vitis-unified-wrapper-stream-dec' in line:
                    for input_idx, inp in enumerate(inputs):
                        line += (
                            f'{indent} static hls::stream<{inp.type.name}> '
                            f'{self._get_local_stream_name(inp, True, input_idx)};\n'
                        )
                    for output_idx, out in enumerate(outputs):
                        line += (
                            f'{indent} static hls::stream<{out.type.name}> '
                            f'{self._get_local_stream_name(out, False, output_idx)};\n'
                        )
                elif '// vitis-unified-wrapper-stream-config' in line:
                    for input_idx, inp in enumerate(inputs):
                        line += (
                            f'#pragma HLS STREAM variable={self._get_local_stream_name(inp, True, input_idx)} '
                            f'depth=STREAM_BUF_IN_SZ\n'
                        )
                    for output_idx, out in enumerate(outputs):
                        line += (
                            f'#pragma HLS STREAM variable={self._get_local_stream_name(out, False, output_idx)} '
                            f'depth=STREAM_BUF_OUT_SZ\n'
                        )
                elif '// vitis-unified-wrapper-load' in line:
                    for input_idx, inp in enumerate(inputs):
                        line += (
                            f'load_input({self._get_io_port_name(inp, True, input_idx)}, '
                            f'{self._get_local_stream_name(inp, True, input_idx)}, amtQuery, {inp.size()});\n'
                        )
                elif '// vitis-unified-wrapper-compute' in line:
                    pool_list = [self._get_local_stream_name(inp, True, idx) for idx, inp in enumerate(inputs)]
                    pool_list.extend(self._get_local_stream_name(out, False, idx) for idx, out in enumerate(outputs))
                    joined_io = f',\n{indent}{indent}{indent}'.join(pool_list)
                    line += f'{indent} {self._get_project_name(model)}({joined_io});\n'
                elif '// vitis-unified-wrapper-store' in line:
                    for output_idx, out in enumerate(outputs):
                        line += (
                            f'store_result({self._get_io_port_name(out, False, output_idx)}, '
                            f'{self._get_local_stream_name(out, False, output_idx)}, amtQuery, {out.size()});\n'
                        )
                fout.write(line)

        with (
            open(os.path.join(filedir, '../templates/vitis_unified/myproject_axim.h')) as fin,
            open(f'{model.config.get_output_dir()}/firmware/{self._get_wrapper_file_name(model, True)}.h', 'w') as fout,
        ):
            for line in fin.readlines():
                if 'FILENAME' in line:
                    line = line.replace('FILENAME', self._get_wrapper_file_name(model, True).upper())
                elif 'MY_PROJECT_INC.h' in line:
                    line = line.replace('MY_PROJECT_INC', self._get_project_name(model))
                elif 'MY_PROJECT_TOP_FUNC' in line:
                    line = line.replace('MY_PROJECT_TOP_FUNC', self._get_top_wrap_func_name(model, True))
                elif '// vitis-unified-wrapper-io' in line:
                    line += self._gen_io_signature(indent, inp_gmem_t, out_gmem_t, inputs, outputs) + '\n'
                fout.write(line)

    # ===== Driver generation =====
    def write_driver(self, model):
        if self._is_axi_master():
            self._write_driver_axim(model)
        else:
            self._write_driver_axis(model)

    def _write_driver_axis(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/vitis_unified/driver/pynq/pynq_driver_axis.py.hls4ml')) as fin,
            open(f'{model.config.get_output_dir()}/export/pynq_driver.py', 'w') as fout,
        ):
            fout.write(fin.read())

    def _write_driver_axim(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/vitis_unified/driver/pynq/pynq_driver_axim.py.hls4ml')) as fin,
            open(f'{model.config.get_output_dir()}/export/pynq_driver.py', 'w') as fout,
        ):
            _, _, inputs, outputs = self.vitis_unified_config.get_corrected_types()
            stride_in_ptr_addr = 4 * 3
            stride_out_ptr_addr = 4 * 3
            start_in_ptr_addr = 0x10
            start_out_ptr_addr = start_in_ptr_addr + stride_in_ptr_addr * len(inputs)
            start_amt_query_addr = start_out_ptr_addr + stride_out_ptr_addr * len(outputs)
            indent = ' ' * 12

            for line in fin.readlines():
                if 'REG_ADDR_AMT_QUERY' in line:
                    line = line.replace('VAL', str(hex(start_amt_query_addr)))
                if '# hls-driver-input-dbg-name' in line:
                    names = [f'{indent}"{self._get_io_port_name(inp, True, idx)}"' for idx, inp in enumerate(inputs)]
                    line += ',\n'.join(names) + '\n'
                if '# hls-driver-input-ptr' in line:
                    line += (
                        ',\n'.join(self._gen_hex_addr_list(start_in_ptr_addr, stride_in_ptr_addr, len(inputs), indent))
                        + '\n'
                    )
                if '# hls-driver-output-dbg-name' in line:
                    names = [f'{indent}"{self._get_io_port_name(out, False, idx)}"' for idx, out in enumerate(outputs)]
                    line += ',\n'.join(names) + '\n'
                if '# hls-driver-output-ptr' in line:
                    line += (
                        ',\n'.join(self._gen_hex_addr_list(start_out_ptr_addr, stride_out_ptr_addr, len(outputs), indent))
                        + '\n'
                    )
                if '<TOP_NAME>' in line:
                    line = line.replace('<TOP_NAME>', self._get_top_wrap_func_name(model, self._is_axi_master()))
                fout.write(line)

    # ===== Test generation =====
    def write_wrapper_test(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/vitis_unified/myproject_test.cpp')) as fin,
            open(f'{model.config.get_output_dir()}/{self._get_sim_file_name()}.cpp', 'w') as fout,
        ):
            _, _, _, _ = self.vitis_unified_config.get_corrected_types()
            model_inputs = model.get_input_variables()
            model_outputs = model.get_output_variables()
            model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

            fout.write('//// generated by Vitis Unified Backend\n')

            for line in fin.readlines():
                indent = ' ' * (len(line) - len(line.lstrip(' ')))

                if 'myproject' in line:
                    newline = line.replace('myproject', self._get_project_name(model))
                elif '// hls-fpga-machine-learning insert include' in line:
                    newline = line + f'#include "firmware/{self._get_wrapper_file_name(model, self._is_axi_master())}.h"\n'
                elif '// hls-fpga-machine-learning insert bram' in line:
                    newline = line
                    for bram in model_brams:
                        newline += f'#include "firmware/weights/{bram.name}.h"\n'
                elif '// hls-fpga-machine-learning insert data' in line:
                    newline = line
                    if self._is_axi_master():
                        offset = 0
                        for input_idx, inp in enumerate(model_inputs):
                            newline += indent + 'float* {input_port_name} = &in[{start_idx}];\n'.format(
                                input_port_name=self._get_io_port_name(inp, True, input_idx), start_idx=str(offset)
                            )
                            offset += inp.size()
                        for output_idx, out in enumerate(model_outputs):
                            newline += indent + f'float {self._get_io_port_name(out, False, output_idx)}[{out.size()}];\n'
                    else:
                        assert len(model_inputs) == 1, 'Only support one input for axi stream'
                        assert len(model_outputs) == 1, 'Only support one output for axi stream'
                        newline += 3 * indent + f'hls::stream<{self._get_dma_type_name()}> inputs;\n'
                        newline += 3 * indent + 'nnet::convert_data_axis<float,float, N_IN>(in, inputs);\n'
                        newline += 3 * indent + 'std::cout << "input size inputs: " << inputs.size() << std::endl;\n'
                        newline += 3 * indent + f'hls::stream<{self._get_dma_type_name()}> outputs;\n\n'
                elif '// hls-fpga-machine-learning insert top-level-function' in line:
                    newline = line
                    input_ios = []
                    output_ios = []
                    bram_ios = [b.name for b in model_brams]
                    constant_ios = []

                    if self._is_axi_master():
                        input_ios.extend(self._get_io_port_name(inp, True, idx) for idx, inp in enumerate(model_inputs))
                        output_ios.extend(self._get_io_port_name(out, False, idx) for idx, out in enumerate(model_outputs))
                        constant_ios.append('1')
                    else:
                        input_ios.append('inputs')
                        output_ios.append('outputs')

                    all_vars = ' ,'.join(filter(None, [*input_ios, *output_ios, *bram_ios, *constant_ios]))
                    top_level = indent + f'{self._get_top_wrap_func_name(model, self._is_axi_master())}({all_vars});\n'
                    newline += top_level
                elif '// hls-fpga-machine-learning insert predictions' in line:
                    newline = line
                    for out in model_outputs:
                        newline += indent + f'for(int i = 0; i < {out.size()}; i++) {{\n'
                        newline += indent + '  std::cout << pr[i] << " ";\n'
                        newline += indent + '}\n'
                        newline += indent + 'std::cout << std::endl;\n'
                elif '// hls-fpga-machine-learning insert zero' in line:
                    newline = line
                    if self._is_axi_master():
                        for input_idx, inp in enumerate(model_inputs):
                            newline += (
                                indent + f'float {self._get_io_port_name(inp, True, input_idx)}[{inp.size()}] = {{}};\n'
                            )
                        for output_idx, out in enumerate(model_outputs):
                            newline += (
                                indent + f'float {self._get_io_port_name(out, False, output_idx)}[{out.size()}] = {{}};\n'
                            )
                    else:
                        newline += 3 * indent + f'hls::stream<{self._get_dma_type_name()}> inputs;\n'
                        newline += 3 * indent + f'nnet::fill_zero_axi<{self._get_dma_type_name()}, N_IN>(inputs, false);\n'
                        newline += 3 * indent + 'std::cout << "input size inputs: " << inputs.size() << std::endl;\n'
                        newline += 3 * indent + f'hls::stream<{self._get_dma_type_name()}> outputs;\n\n'
                elif '// hls-fpga-machine-learning insert tb-output' in line:
                    newline = line
                    tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                    if tb_stream != 'stdout':
                        if self._is_axi_master():
                            for output_idx, out in enumerate(model_outputs):
                                newline += (
                                    indent
                                    + (
                                        'nnet::print_result<{actual_type}, {copy_size}>('
                                        '{port_name}, {dest}, {keep_output});\n'
                                    ).format(
                                        actual_type='float',
                                        copy_size=out.size(),
                                        port_name=self._get_io_port_name(out, False, output_idx),
                                        dest='fout',
                                        keep_output='false',
                                    )
                                )
                        else:
                            newline += (
                                indent
                                + f'nnet::print_result_axis<{self._get_dma_type_name()}, N_OUT>(outputs, fout, false);\n'
                            )
                elif ('// hls-fpga-machine-learning insert output' in line) or (
                    '// hls-fpga-machine-learning insert quantized' in line
                ):
                    newline = line
                    tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                    keep_output = str(tb_stream != 'stdout').lower()
                    if tb_stream != 'file':
                        if self._is_axi_master():
                            for output_idx, out in enumerate(model_outputs):
                                newline += (
                                    indent
                                    + (
                                        'nnet::print_result<{actual_type}, {copy_size}>('
                                        '{port_name}, {dest}, {keep_output});\n'
                                    ).format(
                                        actual_type='float',
                                        copy_size=out.size(),
                                        port_name=self._get_io_port_name(out, False, output_idx),
                                        dest='std::cout',
                                        keep_output=keep_output,
                                    )
                                )
                        else:
                            newline += (
                                indent
                                + (
                                    f'nnet::print_result_axis<{self._get_dma_type_name()}, N_OUT>('
                                    f'outputs, std::cout, {keep_output});\n'
                                )
                            )
                elif '// hls-fpga-machine-learning insert namespace' in line:
                    newline = ''
                    namespace = model.config.get_writer_config().get('Namespace', None)
                    if namespace is not None:
                        newline += indent + f'using namespace {namespace};\n'
                else:
                    newline = line

                fout.write(newline)

    # ===== Main entrypoint =====
    def write_hls(self, model, is_multigraph=False):
        if is_multigraph:
            raise Exception('Vitis Unified does not support multigraphs.')

        self._set_unified_config(model)
        super().write_hls(model, is_multigraph=False)
        self.write_wrapper(model)
        self._ensure_export_path(model)
        self.write_driver(model)
        self.write_wrapper_test(model)
        self.write_tar(model)
