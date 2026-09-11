import os
import stat
from pathlib import Path
from shutil import copytree

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
        """No-op for the parent write steps: the tar is written once at the end of write_hls, after all files exist."""
        pass

    def write_nnet_utils_unified_overrides(self, model):
        """Copy VitisUnified-specific headers (AXI stream helpers and types) to the project."""
        filedir = os.path.dirname(os.path.abspath(__file__))

        for subdir in ['nnet_utils', 'ap_types']:
            srcpath = os.path.join(filedir, f'../templates/vitis_unified/{subdir}/')
            dstpath = f'{model.config.get_output_dir()}/firmware/{subdir}/'

            copytree(srcpath, dstpath, dirs_exist_ok=True)

    def write_board_script_override(self, model):
        """No-op: Vitis Unified uses vitis-comp.json and hls_kernel_config, not project.tcl."""
        pass

    def write_build_prj_override(self, model):
        """No-op: Vitis Unified builds with v++ and the cfg files, not build_prj.tcl."""
        pass

    def write_build_opts(self, model):
        """No-op: Vitis Unified builds with v++ and the cfg files, not build_opt.tcl."""
        pass

    # ===== sanity check function =====
    def sanity_check(self, model):
        decl = self._get_kernel_declaration(model)
        if len(decl) > 64:
            name = self._get_project_name(model)
            max_name = len(name) - (len(decl) - 64 + 1) // 2
            raise ValueError(
                f'The kernel declaration "{decl}" is {len(decl)} characters, Vitis allows 64. '
                f'Use a project name of at most {max_name} characters (now {len(name)}).'
            )

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
        return self.vitis_unified_config.get_axi_mode() == 'axi_stream'

    def _is_axi_master(self):
        return self.vitis_unified_config.get_axi_mode() == 'axi_master'

    def _get_project_name(self, model):
        return model.config.get_project_name()

    def _get_wrapper_file_name(self, model, is_axi_master):
        suffix = 'axi_master' if is_axi_master else 'axi_stream'
        return f'{self._get_project_name(model)}_{suffix}'

    def _get_sim_file_name(self, model):
        return f'{self._get_project_name(model)}_test'

    def _get_ip_version(self, model):
        return model.config.get_config_value('Version', '1.0.0')

    def _get_ip_vlnv_version(self, model):
        return '.'.join(self._get_ip_version(model).split('.')[:2])

    def _get_kernel_declaration(self, model):
        top_module_name = self._get_top_wrap_func_name(model, self._is_axi_master())
        return f'{top_module_name}:1:{top_module_name}_1'

    def _get_top_wrap_func_name(self, model, is_axi_master):
        return self._get_wrapper_file_name(model, is_axi_master)

    def _get_wrap_ip_name(self, model, is_axi_master):
        return f'{self._get_top_wrap_func_name(model, is_axi_master)}_1'

    def _get_interrupt_pin_name(self, model, is_axi_master):
        if is_axi_master:
            return f'{self._get_wrap_ip_name(model, True)}/interrupt'
        else:
            return f'{self._get_wrap_ip_name(model, False)}/s2mm_introut'

    def _get_xo_file_path(self, model):
        """Path to .xo relative to system_link (for link_system.sh)."""
        xo_name = f'{self._get_top_wrap_func_name(model, self._is_axi_master())}.xo'
        return os.path.join('..', model.config.get_project_name(), 'vitis_unified_project', xo_name)

    def _get_io_port_name(self, tensor_var, is_input, idx):
        direction = 'in' if is_input else 'out'
        return f'gmem_{direction}{idx}_ptr_{tensor_var.name}'

    def _get_local_stream_name(self, tensor_var, is_input, idx):
        direction = 'in' if is_input else 'out'
        return f'stream_{direction}{idx}_{tensor_var.name}'

    def _get_dma_type_name(self):
        return 'dma_data_packet'

    def _get_using_namespace(self, model, indent=''):
        namespace = model.config.get_writer_config().get('Namespace', None)
        return f'{indent}using namespace {namespace};\n' if namespace is not None else ''

    @staticmethod
    def _get_clock_period_ns(model):
        clock_period_ns = float(model.config.get_config_value('ClockPeriod'))
        if clock_period_ns <= 0:
            raise ValueError('ClockPeriod must be positive')
        return clock_period_ns

    def _gen_io_signature(self, indent, input_type, output_type, inputs, outputs):
        input_ptrs = [f'{indent}{input_type}* {self._get_io_port_name(inp, True, idx)}' for idx, inp in enumerate(inputs)]
        output_ptrs = [
            f'{indent}{output_type}* {self._get_io_port_name(out, False, idx)}' for idx, out in enumerate(outputs)
        ]
        return ', '.join(input_ptrs) + ',\n' + ', '.join(output_ptrs) + '\n'

    def _ensure_export_path(self, model):
        export_path = Path(model.config.get_output_dir()) / 'export'
        export_path.mkdir(parents=True, exist_ok=True)

    def _fill_template(self, template, output_path, replacements=None, blocks=None):
        """Copy a template to output_path. Placeholders in `replacements` are substituted on every template line.
        A line that contains a marker of `blocks` is replaced by the text the block returns for the line's indent.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(template):
            template = os.path.join(filedir, '../templates/vitis_unified', template)
        with open(template) as fin, open(output_path, 'w') as fout:
            for line in fin.readlines():
                indent = line[: len(line) - len(line.lstrip(' '))]
                block = next((block for marker, block in (blocks or {}).items() if marker in line), None)
                if block is not None:
                    fout.write(block(indent))
                    continue
                for token, value in (replacements or {}).items():
                    line = line.replace(token, value)
                fout.write(line)

    # ===== Build/config generation =====
    def write_build_script(self, model):
        self._write_bridge_build_script(model)
        self._build_unified_project_skeleton(model)
        rtl_sim = 'tb.file_cflags={OUTDIR}/{SIM_FILE_NAME}.cpp,-DRTL_SIM'
        self._write_hls_kernel_config(model, 'csim', [])
        self._write_hls_kernel_config(model, 'cosim', [rtl_sim, 'cosim.enable_fifo_sizing=false'])
        self._write_hls_kernel_config(model, 'cosim_fifo_sizing', [rtl_sim, 'cosim.enable_fifo_sizing=true'])
        self._write_linker_dir(model)
        self._write_linker_launcher(model)
        self._write_linker_config(model)

    def _write_bridge_build_script(self, model):
        output_path = f'{model.config.get_output_dir()}/build_lib.sh'
        self._fill_template(
            'build_lib.sh',
            output_path,
            replacements={
                'myprojectBaseName': self._get_project_name(model),
                'myprojectWrapName': self._get_wrapper_file_name(model, self._is_axi_master()),
                'mystamp': model.config.get_config_value('Stamp'),
            },
        )
        build_lib_dst = Path(output_path).resolve()
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

    def _write_hls_kernel_config(self, model, suffix, cosim_options):
        hls_dir = self.get_vitis_hls_dir(model)
        replacements = {
            '{PART}': model.config.get_config_value('Part'),
            '{CLK}': f'{self._get_clock_period_ns(model):g}ns',
            '{CLK_UC}': model.config.get_config_value('ClockUncertainty'),
            '{OUTDIR}': os.path.relpath(model.config.get_output_dir(), hls_dir),
            '{TOP_NAME}': self._get_top_wrap_func_name(model, self._is_axi_master()),
            '{FILE_NAME_WRAP}': self._get_wrapper_file_name(model, self._is_axi_master()),
            '{SIM_FILE_NAME}': self._get_sim_file_name(model),
            '{FILE_NAME_BASE}': self._get_project_name(model),
            '{IP_VERSION}': self._get_ip_version(model),
            '{OUTPUT_KERNEL_TYPE}': 'xo',
        }

        def cosim_block(indent):
            lines = ''
            for option in cosim_options:
                for token, value in replacements.items():
                    option = option.replace(token, value)
                lines += option + '\n'
            return lines

        self._fill_template(
            'hls_kernel_config.cfg',
            os.path.join(hls_dir, f'hls_kernel_config_{suffix}.cfg'),
            replacements=replacements,
            blocks={'# hls-fpga-machine-learning insert cosim options': cosim_block},
        )

    def _build_unified_project_skeleton(self, model):
        hls_dir = self.get_vitis_hls_dir(model)
        os.makedirs(hls_dir, exist_ok=True)
        self._fill_template(
            'vitis_workspace/kernel_project/vitis-comp.json',
            os.path.join(hls_dir, 'vitis-comp.json'),
            replacements={'{HLS_NAME}': self._get_project_name(model), '{CONFIG_FILE}': 'hls_kernel_config_csim.cfg'},
        )

    def _write_linker_dir(self, model):
        os.makedirs(self.get_vitis_linker_dir(model), exist_ok=True)
        # Copy board platform files (tcl_scripts, etc.) when using platform generator
        platform_generator_tcl = self.vitis_unified_config.get_platform_generator_tcl()
        if platform_generator_tcl:
            self._copy_board_platform_files(model)

    def _copy_board_platform_files(self, model):
        """Copy board folder (tcl_scripts) to vitis_workspace for local use."""
        filedir = os.path.dirname(os.path.abspath(__file__))
        board = self.vitis_unified_config.get_board()
        src = os.path.join(filedir, '../templates/vitis_unified', board)
        dst = os.path.join(model.config.get_output_dir(), 'vitis_workspace', board)
        if os.path.isdir(src):
            copytree(src, dst, dirs_exist_ok=True)

    def _write_linker_launcher(self, model):
        platform_generator_tcl = self.vitis_unified_config.get_platform_generator_tcl()
        if platform_generator_tcl:
            # Use local copy in vitis_workspace (relative to system_link)
            board = self.vitis_unified_config.get_board()
            tcl_name = os.path.basename(platform_generator_tcl)
            xsa_basename = os.path.basename(self.vitis_unified_config.get_platform_output_path())
            local_tcl_dir = os.path.join('..', board, 'tcl_scripts')
            local_xsa_path = os.path.join('..', board, 'tcl_scripts', 'output', xsa_basename)
            platform_generator_block = f'''if [ ! -f "{local_xsa_path}" ] || \\
   [ "{local_tcl_dir}/{tcl_name}" -nt "{local_xsa_path}" ]; then
  echo "Generating platform from {tcl_name}..."
  (cd "{local_tcl_dir}" && vivado -mode batch -source "{tcl_name}") || \\
    {{ echo "ERROR: Platform generation failed"; exit 1; }}
fi
'''
            platform_path_for_vpp = local_xsa_path
        else:
            platform_path_for_vpp = self.vitis_unified_config.get_platform_path()
            platform_generator_block = ''
            if '${XILINX_VITIS}' in platform_path_for_vpp:
                platform_generator_block = (
                    ': "${XILINX_VITIS:?XILINX_VITIS is not set. Source the Vitis settings64.sh first.}"\n'
                )

        output_path = f'{self.get_vitis_linker_dir(model)}/link_system.sh'
        self._fill_template(
            'vitis_workspace/system_link/link_system.sh',
            output_path,
            replacements={
                '{XSA_GENERATOR_BLOCK}': platform_generator_block,
                '{PLATFORM_PATH}': platform_path_for_vpp,
                '{KERNEL_XO}': self._get_xo_file_path(model),
                '{PROJECT_NAME}': self._get_project_name(model),
            },
        )
        link_lib_dst = Path(output_path).resolve()
        link_lib_dst.chmod(link_lib_dst.stat().st_mode | stat.S_IEXEC)

    def _write_linker_config(self, model):
        def connectivity(indent):
            if not self._is_axi_stream():
                return ''
            top_mod_inst_name = self._get_wrap_ip_name(model, False)
            return (
                '\n[connectivity]\n'
                f'nk={self._get_kernel_declaration(model)}\n'
                f'stream_connect=DMA_MM2S:{top_mod_inst_name}.axi_input_stream\n'
                f'stream_connect={top_mod_inst_name}.axi_output_stream:DMA_S2MM\n'
            )

        self._fill_template(
            'vitis_workspace/system_link/link_system.cfg',
            f'{self.get_vitis_linker_dir(model)}/link_system.cfg',
            replacements={
                '{CLK}': str(round(1_000_000_000 / self._get_clock_period_ns(model))),
                '{KERNEL_NAME}': self._get_top_wrap_func_name(model, self._is_axi_master()),
                '{GUI_STATUS}': 'true',
            },
            blocks={'# hls-fpga-machine-learning insert custom connection': connectivity},
        )

    # ===== Bridge generation =====
    def _gen_bridge_body(self, model, dtype, indent):
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        in_type = self.vitis_unified_config.get_input_type()
        out_type = self.vitis_unified_config.get_output_type()
        newline = ''

        if self._is_axi_master():
            in_bufs = []
            for idx, inp in enumerate(model_inputs):
                port = self._get_io_port_name(inp, True, idx)
                buf = port + '_ap'
                newline += indent + f'{in_type} {buf}[{inp.size_cpp()}];\n'
                newline += indent + f'nnet::convert_data<{dtype}, {in_type}, {inp.size_cpp()}>({port}, {buf});\n'
                in_bufs.append(buf)
            out_bufs = []
            for idx, out in enumerate(model_outputs):
                port = self._get_io_port_name(out, False, idx)
                buf = port + '_ap'
                newline += indent + f'{out_type} {buf}[{out.size_cpp()}];\n'
                out_bufs.append(buf)
            newline += '\n'
            newline += indent + self._get_top_wrap_func_name(model, True) + '(\n'
            newline += indent + ', '.join(in_bufs) + ',\n'
            newline += indent + ', '.join(out_bufs) + ',\n'
            newline += indent + '1);\n'
            newline += '\n'
            for idx, out in enumerate(model_outputs):
                port = self._get_io_port_name(out, False, idx)
                buf = port + '_ap'
                newline += indent + f'nnet::convert_data<{out_type}, {dtype}, {out.size_cpp()}>({buf}, {port});\n'
        else:
            inp = model_inputs[0]
            out = model_outputs[0]
            inp_func = self._get_io_port_name(inp, True, 0)
            inp_stream = inp_func + '_ap'
            out_func = self._get_io_port_name(out, False, 0)
            out_stream = out_func + '_ap'
            dma = self._get_dma_type_name()
            newline += indent + f'hls::stream<{dma}> {inp_stream};\n'
            newline += indent + f'nnet::convert_data_axis<{dma}, {dtype}, N_IN>({inp_func}, {inp_stream});\n'
            newline += indent + f'hls::stream<{dma}> {out_stream};\n'
            newline += indent + self._get_top_wrap_func_name(model, False) + '('
            newline += inp_stream + ', ' + out_stream + ', 1);\n'
            newline += indent + f'nnet::convert_data_axis<{dma}, {dtype}, N_OUT>({out_stream}, {out_func});\n'

        return newline

    def write_bridge(self, model):
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        def header(dtype):
            input_ios = [
                f'{dtype} {self._get_io_port_name(inp, True, idx)}[{inp.size_cpp()}]' for idx, inp in enumerate(model_inputs)
            ]
            output_ios = [
                f'{dtype} {self._get_io_port_name(out, False, idx)}[{out.size_cpp()}]'
                for idx, out in enumerate(model_outputs)
            ]
            return '    ' + ', '.join(input_ios) + ',\n' + '    ' + ', '.join(output_ios) + '\n'

        def trace_outputs(indent):
            lines = ''
            for layer in model.get_layers():
                func = layer.get_attr('function_cpp', None)
                if func and model.config.trace_output and layer.get_attr('trace', False):
                    for var in layer.get_variables():
                        lines += (
                            indent
                            + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                            + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                        )
            return lines

        self._fill_template(
            'myproject_bridge.cpp',
            f'{model.config.get_output_dir()}/{self._get_project_name(model)}_bridge.cpp',
            replacements={
                'MYPROJECT': self._get_project_name(model).upper(),
                'myproject': self._get_project_name(model),
                'PROJECT_FILE_NAME': self._get_wrapper_file_name(model, self._is_axi_master()),
            },
            blocks={
                '// hls-fpga-machine-learning insert bram': lambda indent: ''.join(
                    f'#include "firmware/weights/{bram.name}.h"\n' for bram in model_brams
                ),
                '// hls-fpga-machine-learning insert header #float': lambda indent: header('float'),
                '// hls-fpga-machine-learning insert header #double': lambda indent: header('double'),
                '// hls-fpga-machine-learning insert wrapper #float': lambda indent: self._gen_bridge_body(
                    model, 'float', indent
                ),
                '// hls-fpga-machine-learning insert wrapper #double': lambda indent: self._gen_bridge_body(
                    model, 'double', indent
                ),
                '// hls-fpga-machine-learning insert trace_outputs': trace_outputs,
                '// hls-fpga-machine-learning insert namespace': lambda indent: self._get_using_namespace(model, indent),
            },
        )

    # ===== Wrapper generation =====
    def write_wrapper(self, model):
        if self._is_axi_master():
            self._write_wrapper_axim(model)
        else:
            self._write_wrapper_axis(model)

    def _write_wrapper_axis(self, model):
        inp_gmem_t, out_gmem_t, inputs, outputs = self.vitis_unified_config.get_corrected_types()
        inp, out = inputs[0], outputs[0]
        name = self._get_project_name(model)
        wrapper = self._get_wrapper_file_name(model, False)
        firmware_dir = f'{model.config.get_output_dir()}/firmware'

        def interface(indent):
            return (
                f'{indent}#pragma HLS INTERFACE axis port=axi_input_stream\n'
                f'{indent}#pragma HLS INTERFACE axis port=axi_output_stream\n'
                f'{indent}#pragma HLS INTERFACE s_axilite port=return bundle=control\n'
                f'{indent}#pragma HLS INTERFACE s_axilite port=batch_size bundle=control\n'
            )

        def stream_decl(indent):
            in_depth = self.vitis_unified_config.get_in_stream_buf_size()
            out_depth = self.vitis_unified_config.get_out_stream_buf_size()
            return (
                f'{indent}static hls::stream<{inp.type.name}> model_input_stream("model_input");\n'
                f'{indent}static hls::stream<{out.type.name}> model_output_stream("model_output");\n\n'
                f'{indent}#pragma HLS STREAM variable=model_input_stream depth={in_depth}\n'
                f'{indent}#pragma HLS STREAM variable=model_output_stream depth={out_depth}\n'
            )

        self._fill_template(
            'myproject_axi_stream.cpp',
            f'{firmware_dir}/{wrapper}.cpp',
            replacements={
                'MY_PROJECT_TOP_FUNC': self._get_top_wrap_func_name(model, False),
                'MY_PROJECT': name,
                '// hls-fpga-machine-learning insert stream parameter': (
                    f'hls::stream<{inp.type.name}> &model_input_stream, hls::stream<{out.type.name}> &model_output_stream'
                ),
                'INPUT_LAYER_TYPE': inp.type.name,
                'OUTPUT_LAYER_TYPE': out.type.name,
                'OUTPUT_GMEM_TYPE': out_gmem_t,
            },
            blocks={
                '// hls-fpga-machine-learning insert include': lambda indent: f'#include "{wrapper}.h"\n',
                '// hls-fpga-machine-learning insert interface': interface,
                '// hls-fpga-machine-learning insert stream decl': stream_decl,
            },
        )

        def definitions(indent):
            return (
                f'static const unsigned N_IN = {inp.size()};\n'
                f'static const unsigned N_OUT = {out.size()};\n'
                f'typedef hls::axis_data<{inp_gmem_t}, AXIS_ENABLE_LAST | AXIS_ENABLE_KEEP> {self._get_dma_type_name()};\n'
            )

        self._fill_template(
            'myproject_axi_stream.h',
            f'{firmware_dir}/{wrapper}.h',
            replacements={'MYPROJECT': name.upper(), 'MY_PROJECT_TOP_FUNC': self._get_top_wrap_func_name(model, False)},
            blocks={
                '// hls-fpga-machine-learning insert include': lambda indent: (
                    f'#include "{name}.h"\n#include "ap_axi_sdata.h"\n' + self._get_using_namespace(model)
                ),
                '// hls-fpga-machine-learning insert definitions': definitions,
            },
        )

    def _write_wrapper_axim(self, model):
        inp_gmem_t, out_gmem_t, inputs, outputs = self.vitis_unified_config.get_corrected_types()
        name = self._get_project_name(model)
        wrapper = self._get_wrapper_file_name(model, True)
        firmware_dir = f'{model.config.get_output_dir()}/firmware'
        in_ports = [self._get_io_port_name(inp, True, idx) for idx, inp in enumerate(inputs)]
        out_ports = [self._get_io_port_name(out, False, idx) for idx, out in enumerate(outputs)]
        in_streams = [self._get_local_stream_name(inp, True, idx) for idx, inp in enumerate(inputs)]
        out_streams = [self._get_local_stream_name(out, False, idx) for idx, out in enumerate(outputs)]

        def io_signature(indent):
            return self._gen_io_signature(indent, inp_gmem_t, out_gmem_t, inputs, outputs) + '\n'

        def interface(indent):
            lines = f'{indent}// depth is for simulation only: the test bench sends one sample per kernel start\n'
            for idx, (port, inp) in enumerate(zip(in_ports, inputs)):
                lines += f'{indent}#pragma HLS INTERFACE m_axi port={port} bundle=gmem_in{idx} depth={inp.size()}\n'
            for idx, (port, out) in enumerate(zip(out_ports, outputs)):
                lines += f'{indent}#pragma HLS INTERFACE m_axi port={port} bundle=gmem_out{idx} depth={out.size()}\n'
            lines += f'{indent}#pragma HLS INTERFACE s_axilite port=batch_size bundle=control\n'
            lines += f'{indent}#pragma HLS INTERFACE s_axilite port=return bundle=control\n'
            return lines

        def stream_decl(indent):
            lines = ''.join(f'{indent}static hls::stream<{inp.type.name}> {s};\n' for s, inp in zip(in_streams, inputs))
            lines += ''.join(f'{indent}static hls::stream<{out.type.name}> {s};\n' for s, out in zip(out_streams, outputs))
            return lines

        def stream_config(indent):
            lines = ''.join(f'{indent}#pragma HLS STREAM variable={s} depth=STREAM_BUF_IN_SZ\n' for s in in_streams)
            lines += ''.join(f'{indent}#pragma HLS STREAM variable={s} depth=STREAM_BUF_OUT_SZ\n' for s in out_streams)
            return lines

        def load(indent):
            return ''.join(
                f'{indent}load_input({port}, {s}, batch_size, {inp.size()});\n'
                for port, s, inp in zip(in_ports, in_streams, inputs)
            )

        def store(indent):
            return ''.join(
                f'{indent}store_result({port}, {s}, batch_size, {out.size()});\n'
                for port, s, out in zip(out_ports, out_streams, outputs)
            )

        signature = [f'hls::stream<{inp.type.name}>& {s}' for s, inp in zip(in_streams, inputs)]
        signature += [f'hls::stream<{out.type.name}>& {s}' for s, out in zip(out_streams, outputs)]
        signature.append('int batch_size')

        self._fill_template(
            'myproject_axi_master.cpp',
            f'{firmware_dir}/{wrapper}.cpp',
            replacements={
                'MY_PROJECT_DM_INC': wrapper,
                'MY_PROJECT_TOP_FUNC': self._get_top_wrap_func_name(model, True),
                'HLS4ML_STREAM_BUF_IN_SZ': str(self.vitis_unified_config.get_in_stream_buf_size()),
                'HLS4ML_STREAM_BUF_OUT_SZ': str(self.vitis_unified_config.get_out_stream_buf_size()),
                '// vitis-unified-wrapper-compute-signature': ', '.join(signature),
                '// vitis-unified-wrapper-compute-body': f'{name}({", ".join(in_streams + out_streams)});',
                '// vitis-unified-wrapper-compute-call-args': ', '.join(in_streams + out_streams + ['batch_size']),
            },
            blocks={
                '// vitis-unified-wrapper-io': io_signature,
                '// vitis-unified-wrapper-interface': interface,
                '// vitis-unified-wrapper-stream-dec': stream_decl,
                '// vitis-unified-wrapper-stream-config': stream_config,
                '// vitis-unified-wrapper-load': load,
                '// vitis-unified-wrapper-store': store,
            },
        )

        self._fill_template(
            'myproject_axi_master.h',
            f'{firmware_dir}/{wrapper}.h',
            replacements={'FILENAME': wrapper.upper(), 'MY_PROJECT_TOP_FUNC': self._get_top_wrap_func_name(model, True)},
            blocks={
                '// hls-fpga-machine-learning insert include': lambda indent: (
                    f'#include "{name}.h"\n' + self._get_using_namespace(model)
                ),
                '// vitis-unified-wrapper-io': io_signature,
            },
        )

    # ===== Driver generation =====
    def write_driver(self, model):
        is_axi_master = self._is_axi_master()
        inputs = model.get_input_variables()
        outputs = model.get_output_variables()

        def port_names(variables, is_input):
            return lambda indent: ''.join(
                f"{indent}'{self._get_io_port_name(var, is_input, idx)}',\n" for idx, var in enumerate(variables)
            )

        self._fill_template(
            self.vitis_unified_config.get_driver_template_path(),
            f'{model.config.get_output_dir()}/export/{self.vitis_unified_config.get_driver_file()}',
            replacements={
                '<TOP_WRAPPER_NAME>': self._get_wrap_ip_name(model, is_axi_master),
                '<TOP_NAME>': self._get_top_wrap_func_name(model, is_axi_master),
                '<IP_VERSION>': self._get_ip_vlnv_version(model),
            },
            blocks={
                '# hls-driver-input-names': port_names(inputs, True),
                '# hls-driver-output-names': port_names(outputs, False),
            },
        )

    # ===== Test generation =====
    def write_test_bench(self, model):
        self.write_tb_data(model)
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']
        is_axi_master = self._is_axi_master()
        dma = self._get_dma_type_name()
        top = self._get_top_wrap_func_name(model, is_axi_master)
        tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
        in_ports = [self._get_io_port_name(inp, True, idx) for idx, inp in enumerate(model_inputs)]
        out_ports = [self._get_io_port_name(out, False, idx) for idx, out in enumerate(model_outputs)]

        def data(indent):
            if is_axi_master:
                lines, offset = '', 0
                for port, inp in zip(in_ports, model_inputs):
                    lines += f'{indent}float* {port} = &in[{offset}];\n'
                    offset += inp.size()
                lines += ''.join(f'{indent}float {port}[{out.size()}];\n' for port, out in zip(out_ports, model_outputs))
                return lines
            return (
                f'{indent}hls::stream<{dma}> inputs;\n'
                f'{indent}nnet::convert_data_axis<{dma}, float, N_IN>(in, inputs);\n'
                f'{indent}hls::stream<{dma}> outputs;\n'
            )

        def zero(indent):
            if is_axi_master:
                pairs = list(zip(in_ports, model_inputs)) + list(zip(out_ports, model_outputs))
                return ''.join(f'{indent}float {port}[{var.size()}] = {{}};\n' for port, var in pairs)
            return (
                f'{indent}hls::stream<{dma}> inputs;\n'
                f'{indent}nnet::fill_zero_axi<{dma}, N_IN>(inputs, false);\n'
                f'{indent}hls::stream<{dma}> outputs;\n'
            )

        def top_level(indent):
            args = in_ports + out_ports if is_axi_master else ['inputs', 'outputs']
            args = args + [bram.name for bram in model_brams] + ['1']
            return f'{indent}{top}({", ".join(args)});\n'

        def predictions(indent):
            return ''.join(
                f'{indent}for(int i = 0; i < {out.size()}; i++) {{\n'
                f'{indent}  std::cout << pr[i] << " ";\n'
                f'{indent}}}\n'
                f'{indent}std::cout << std::endl;\n'
                for out in model_outputs
            )

        def print_results(indent, dest, keep_output):
            if is_axi_master:
                return ''.join(
                    f'{indent}nnet::print_result<float, {out.size()}>({port}, {dest}, {keep_output});\n'
                    for port, out in zip(out_ports, model_outputs)
                )
            return f'{indent}nnet::print_result_axis<{dma}, N_OUT>(outputs, {dest}, {keep_output});\n'

        def tb_output(indent):
            return print_results(indent, 'fout', 'false') if tb_stream != 'stdout' else ''

        def output(indent):
            keep_output = str(tb_stream != 'stdout').lower()
            return print_results(indent, 'std::cout', keep_output) if tb_stream != 'file' else ''

        self._fill_template(
            'myproject_test.cpp',
            f'{model.config.get_output_dir()}/{self._get_sim_file_name(model)}.cpp',
            blocks={
                '// hls-fpga-machine-learning insert include': lambda indent: (
                    f'#include "firmware/{self._get_wrapper_file_name(model, is_axi_master)}.h"\n'
                ),
                '// hls-fpga-machine-learning insert bram': lambda indent: ''.join(
                    f'#include "firmware/weights/{bram.name}.h"\n' for bram in model_brams
                ),
                '// hls-fpga-machine-learning insert namespace': lambda indent: self._get_using_namespace(model, indent),
                '// hls-fpga-machine-learning insert data': data,
                '// hls-fpga-machine-learning insert zero': zero,
                '// hls-fpga-machine-learning insert top-level-function': top_level,
                '// hls-fpga-machine-learning insert predictions': predictions,
                '// hls-fpga-machine-learning insert quantized': output,
                '// hls-fpga-machine-learning insert output': output,
                '// hls-fpga-machine-learning insert tb-output': tb_output,
            },
        )

    # ===== Main entrypoint =====
    def write_hls(self, model, is_multigraph=False):
        if is_multigraph:
            raise Exception('Vitis Unified does not support multigraphs.')
        self._set_unified_config(model)
        self.sanity_check(model)
        super().write_hls(model)
        self.write_nnet_utils_unified_overrides(model)
        self.write_wrapper(model)
        self._ensure_export_path(model)
        self.write_driver(model)
        super().write_tar(model)
