import os
import stat
from pathlib import Path

from hls4ml.writer.bambu_writer import BambuWriter


class BambuAcceleratorWriter(BambuWriter):
    """Extends BambuWriter with an integer-container wrapper around the ap_fixed HLS core.

    Each input/output element is packed into the smallest standard unsigned integer
    (8/16/32/64-bit) that fits the fixed-point width, keeping the AXI bus aligned
    to 128 bits without introducing any IEEE 754 hardware.

    Supports both io_parallel (flat C-array interface) and io_stream (AXIS of
    scalars, one unsigned-container beat per ap_fixed scalar, packed into
    nnet::array structs to feed the core IP's hls::stream interface).

    The wrapper INLINES the layer pipeline directly — it does NOT call
    `myproject()` from the core. Inlining is what lets the io_stream path
    work: Bambu's DATAFLOW scheduler binds sub-function array-pointer
    parameters to BRAMs that read zero at runtime, so all layer calls must
    sit inside one DATAFLOW function. We use the same wrapper layout for
    io_parallel for symmetry.
    """

    # The wrapper myproject_float owns the top-level interface. The inner
    # myproject becomes a sub-function (its definition is still emitted but
    # unused on the BambuAccelerator path) and must NOT emit its own
    # `#pragma HLS interface` lines — Bambu's InterfaceInfer rejects two
    # competing declarations on the same port.
    _emit_core_interface_pragmas = False

    @staticmethod
    def _container_width(precision_width):
        """Smallest power-of-2 width (8/16/32/64) that fits precision_width bits."""
        if precision_width <= 8:
            return 8
        if precision_width <= 16:
            return 16
        if precision_width <= 32:
            return 32
        return 64

    @staticmethod
    def _var_precision(v):
        """Return (total_bits, frac_bits) for a model variable.

        Works for both io_parallel (NamedType) and io_stream (PackedType)
        because PackedType inherits from NamedType and stores the scalar
        ap_fixed precision in the same .type.precision attribute.
        """
        try:
            total = v.type.precision.width
            integer = v.type.precision.integer
            return total, total - integer
        except AttributeError:
            return 32, 0

    def _max_container_width(self, variables):
        """Return the container width needed for the widest variable in the list."""
        max_w = max((self._var_precision(v)[0] for v in variables), default=32)
        return self._container_width(max_w)

    def _compute_io_sizes(self, model):
        """Return total flattened input and output sizes (Python ints)."""
        n_in = sum(v.size() for v in model.get_input_variables())
        n_out = sum(v.size() for v in model.get_output_variables())
        return n_in, n_out

    def _n_elem(self, v):
        """Number of scalars packed per stream beat (= shape[-1] for io_stream, else 1)."""
        try:
            return v.type.n_elem
        except AttributeError:
            return 1

    def write_float_header(self, model):
        """Write firmware/<proj>_float.h"""
        filedir = os.path.dirname(os.path.abspath(__file__))
        proj = model.config.get_project_name()
        io_type = model.config.get_config_value('IOType')

        n_in, n_out = self._compute_io_sizes(model)
        in_cw = self._max_container_width(model.get_input_variables())
        out_cw = self._max_container_width(model.get_output_variables())

        tmpl = open(os.path.join(filedir, '../templates/bambu/firmware/myproject_float.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{proj}_float.h', 'w')

        for line in tmpl.readlines():
            if 'MYPROJECT' in line:
                line = line.replace('MYPROJECT', proj.upper())
            if 'myproject' in line:
                line = line.replace('myproject', proj)

            if '// hls-fpga-machine-learning insert float-includes' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += '#include "hls_stream.h"\n'

            elif '// hls-fpga-machine-learning insert definitions' in line:
                newline = line
                newline += f'#define IN_CONTAINER_WIDTH  {in_cw}\n'
                newline += f'#define OUT_CONTAINER_WIDTH {out_cw}\n'
                newline += 'typedef ac_int<IN_CONTAINER_WIDTH,  false> in_container_t;\n'
                newline += 'typedef ac_int<OUT_CONTAINER_WIDTH, false> out_container_t;\n'
                newline += f'static const unsigned N_IN  = {n_in};\n'
                newline += f'static const unsigned N_OUT = {n_out};\n'

            elif '// hls-fpga-machine-learning insert float-signature' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += f'void {proj}_float(hls::stream<in_container_t> &input_stream, hls::stream<out_container_t> &output_stream);\n'  # noqa: E501
                else:
                    newline += f'void {proj}_float(in_container_t input[N_IN], out_container_t output[N_OUT]);\n'

            else:
                newline = line

            fout.write(newline)

        tmpl.close()
        fout.close()

    # Wrapper-side emission helpers. The accelerator owns the
    # container-packing layer and composes with the layer-pipeline helpers
    # inherited from BambuWriter (`_emit_core_*`, `_emit_load_weights_block`,
    # `_emit_internal_stream_decls`, `_emit_layer_calls`).

    def _emit_ingest_helper(self, inp, io_type):
        """Static decode function: container -> core input variable."""
        total, _ = self._var_precision(inp)
        if io_type == 'io_stream':
            n_elem = self._n_elem(inp)
            n_beats = inp.size() // n_elem
            return (
                f'static void ingest_{inp.name}('
                f'hls::stream<in_container_t> &input_stream, '
                f'hls::stream<{inp.type.name}> &{inp.name}) {{\n'
                f'    for (int i = 0; i < {n_beats}; i++) {{\n'
                f'        {inp.type.name} pack;\n'
                f'        #pragma clang loop unroll(full)\n'
                f'        for (int j = 0; j < {inp.type.name}::size; j++) {{\n'
                f'            in_container_t raw = input_stream.read();\n'
                f'            pack[j].set_slc(0, raw.slc<{total}>(0));\n'
                f'        }}\n'
                f'        {inp.name}.write(pack);\n'
                f'    }}\n'
                f'}}\n\n'
            )
        return (
            f'static void ingest_{inp.name}('
            f'in_container_t input[N_IN], '
            f'{inp.type.name} {inp.name}[{inp.size()}]) {{\n'
            f'    #pragma clang loop unroll(full)\n'
            f'    for (int i = 0; i < {inp.size()}; i++)\n'
            f'        {inp.name}[i].set_slc(0, input[i].slc<{total}>(0));\n'
            f'}}\n\n'
        )

    def _emit_egress_helper(self, out_var, io_type):
        """Static encode function: core output variable -> container."""
        total, _ = self._var_precision(out_var)
        if io_type == 'io_stream':
            n_elem = self._n_elem(out_var)
            n_beats = out_var.size() // n_elem
            return (
                f'static void egress_{out_var.name}('
                f'hls::stream<{out_var.type.name}> &{out_var.name}, '
                f'hls::stream<out_container_t> &output_stream) {{\n'
                f'    for (int i = 0; i < {n_beats}; i++) {{\n'
                f'        {out_var.type.name} pack = {out_var.name}.read();\n'
                f'        #pragma clang loop unroll(full)\n'
                f'        for (int j = 0; j < {out_var.type.name}::size; j++) {{\n'
                f'            out_container_t raw = 0;\n'
                f'            raw.set_slc(0, pack[j].slc<{total}>(0));\n'
                f'            output_stream.write(raw);\n'
                f'        }}\n'
                f'    }}\n'
                f'}}\n\n'
            )
        return (
            f'static void egress_{out_var.name}('
            f'{out_var.type.name} {out_var.name}[{out_var.size()}], '
            f'out_container_t output[N_OUT]) {{\n'
            f'    #pragma clang loop unroll(full)\n'
            f'    for (int i = 0; i < {out_var.size()}; i++) {{\n'
            f'        output[i] = 0;\n'
            f'        output[i].set_slc(0, {out_var.name}[i].slc<{total}>(0));\n'
            f'    }}\n'
            f'}}\n\n'
        )

    def _emit_ingest_helpers(self, model, io_type):
        return ''.join(self._emit_ingest_helper(inp, io_type) for inp in model.get_input_variables())

    def _emit_egress_helpers(self, model, io_type):
        return ''.join(self._emit_egress_helper(o, io_type) for o in model.get_output_variables())

    def _emit_ingest_calls(self, model, io_type, indent='    '):
        port = 'input_stream' if io_type == 'io_stream' else 'input'
        return ''.join(f'{indent}ingest_{inp.name}({port}, {inp.name});\n' for inp in model.get_input_variables())

    def _emit_egress_calls(self, model, io_type, indent='    '):
        port = 'output_stream' if io_type == 'io_stream' else 'output'
        return ''.join(f'{indent}egress_{o.name}({o.name}, {port});\n' for o in model.get_output_variables())

    def _emit_wrapper_interface_pragmas(self, io_type):
        """File-scope AXIS pragmas for the wrapper top-level (io_stream only).

        io_parallel emits no interface pragma — Bambu's
        `--generate-interface=INFER` derives it from the typed array
        parameters in the wrapper signature.
        """
        if io_type != 'io_stream':
            return ''
        return '#pragma HLS interface mode=axis port=input_stream\n#pragma HLS interface mode=axis port=output_stream\n'

    def _emit_wrapper_signature(self, proj, io_type):
        if io_type == 'io_stream':
            return (
                f'void {proj}_float('
                f'hls::stream<in_container_t> &input_stream, '
                f'hls::stream<out_container_t> &output_stream)\n'
            )
        return f'void {proj}_float(in_container_t input[N_IN], out_container_t output[N_OUT])\n'

    def write_float_wrapper(self, model):
        """Write firmware/<proj>_float.cpp — ingest/egress helpers wrapping
        the inlined layer pipeline."""
        proj = model.config.get_project_name()
        io_type = model.config.get_config_value('IOType')
        indent = '    '

        # Bambu DATAFLOW (active for io_stream) requires every local stream
        # to be declared before any sub-function call; declarations first,
        # then calls in pipeline order. The same layout works for io_parallel.
        parts = [
            f'#include "{proj}_float.h"\n',
            f'#include "{proj}.h"\n',
            '#include "parameters.h"\n',
            '\n',
            self._emit_ingest_helpers(model, io_type),
            self._emit_egress_helpers(model, io_type),
            self._emit_wrapper_interface_pragmas(io_type),
            self._emit_wrapper_signature(proj, io_type),
            '{\n',
            self._emit_core_io_pragma(model, indent=indent),
            self._emit_core_input_declarations(model, indent=indent),
            self._emit_core_output_declarations(model, indent=indent),
            self._emit_internal_stream_decls(model, indent=indent),
            self._emit_ingest_calls(model, io_type, indent=indent),
            self._emit_load_weights_block(model, indent=indent),
            # `_emit_load_weights_block` ends with `#endif` (no trailing
            # newline — kept byte-faithful to BambuBackend's existing
            # emission, where the next template line is blank and provides
            # the separator). Composing strings here, we add the newline
            # explicitly so the `nnet::dense(...)` call below isn't gobbled
            # as garbage at end of #endif.
            '\n' if model.config.get_writer_config()['WriteWeightsTxt'] else '',
            self._emit_layer_calls(model, indent=indent),
            self._emit_egress_calls(model, io_type, indent=indent),
            '}\n',
        ]

        with open(f'{model.config.get_output_dir()}/firmware/{proj}_float.cpp', 'w') as fout:
            fout.write(''.join(parts))

    def write_float_test_bench(self, model):
        """Write <proj>_float_test.cpp"""
        filedir = os.path.dirname(os.path.abspath(__file__))
        proj = model.config.get_project_name()
        io_type = model.config.get_config_value('IOType')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        tmpl = open(os.path.join(filedir, '../templates/bambu/myproject_float_test.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{proj}_float_test.cpp', 'w')

        for line in tmpl.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            if 'myproject' in line and '// hls-fpga-machine-learning' not in line:
                newline = line.replace('myproject', proj)

            elif '// hls-fpga-machine-learning insert float-data' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += f'{indent}hls::stream<in_container_t> input_stream("input_stream");\n'
                    in_offset = 0
                    for inp in model_inputs:
                        total, frac = self._var_precision(inp)
                        scale = 1 << frac
                        max_val = (1 << (total - 1)) - 1
                        min_val = -(1 << (total - 1))
                        newline += f'{indent}for(int i = 0; i < {inp.size()}; i++) {{\n'
                        newline += f'{indent}    long long s = (long long)floor((double)in[{in_offset} + i] * {scale}LL);\n'
                        newline += f'{indent}    if (s >  {max_val}LL) s =  {max_val}LL;\n'
                        newline += f'{indent}    if (s < {min_val}LL) s = {min_val}LL;\n'
                        newline += f'{indent}    input_stream.write((in_container_t)(long long)s);\n'
                        newline += f'{indent}}}\n'
                        in_offset += inp.size()
                    newline += f'{indent}hls::stream<out_container_t> output_stream("output_stream");\n'
                else:
                    newline += f'{indent}in_container_t input[N_IN];\n'
                    in_offset = 0
                    for inp in model_inputs:
                        total, frac = self._var_precision(inp)
                        scale = 1 << frac
                        max_val = (1 << (total - 1)) - 1
                        min_val = -(1 << (total - 1))
                        newline += f'{indent}for(int i = 0; i < {inp.size()}; i++) {{\n'
                        newline += f'{indent}    long long s = (long long)floor((double)in[{in_offset} + i] * {scale}LL);\n'
                        newline += f'{indent}    if (s >  {max_val}LL) s =  {max_val}LL;\n'
                        newline += f'{indent}    if (s < {min_val}LL) s = {min_val}LL;\n'
                        newline += f'{indent}    input[{in_offset} + i] = (in_container_t)(long long)s;\n'
                        newline += f'{indent}}}\n'
                        in_offset += inp.size()
                    newline += f'{indent}out_container_t output[N_OUT];\n'

            elif '// hls-fpga-machine-learning insert float-zero' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += f'{indent}hls::stream<in_container_t> input_stream("input_stream");\n'
                    newline += f'{indent}for(int i = 0; i < N_IN; i++) input_stream.write((in_container_t)0);\n'
                    newline += f'{indent}hls::stream<out_container_t> output_stream("output_stream");\n'
                else:
                    # `in_container_t input[N_IN] = {};` trips ac_int's explicit
                    # default constructor under clang; zero the array explicitly.
                    newline += f'{indent}in_container_t input[N_IN];\n'
                    newline += f'{indent}for(int i = 0; i < N_IN; i++) input[i] = (in_container_t)0;\n'
                    newline += f'{indent}out_container_t output[N_OUT];\n'

            elif '// hls-fpga-machine-learning insert float-top-level-function' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += '#ifdef  __BAMBU__\n'
                    newline += f'{indent}m_param_alloc(0, sizeof(input_stream));\n'
                    newline += f'{indent}m_param_alloc(1, sizeof(output_stream));\n'
                    newline += '#endif\n'
                    newline += f'{indent}{proj}_float(input_stream, output_stream);\n'
                else:
                    newline += '#ifdef  __BAMBU__\n'
                    newline += f'{indent}m_param_alloc(0, sizeof(input));\n'
                    newline += f'{indent}m_param_alloc(1, sizeof(output));\n'
                    newline += '#endif\n'
                    newline += f'{indent}{proj}_float(input, output);\n'

            elif '// hls-fpga-machine-learning insert float-tb-output' in line:
                newline = line
                out_offset = 0
                for out in model_outputs:
                    total, frac = self._var_precision(out)
                    scale = 1 << frac
                    if io_type == 'io_stream':
                        newline += f'{indent}for(int i = 0; i < {out.size()}; i++) {{\n'
                        newline += f'{indent}    out_container_t raw_i = output_stream.read();\n'
                        newline += f'{indent}    long long raw = ((long long)raw_i.to_uint64() << (64 - {total})) >> (64 - {total});\n'  # noqa: E501
                        newline += f'{indent}    fout << (double)raw / {scale}.0 << " ";\n'
                        newline += f'{indent}}}\n'
                    else:
                        newline += f'{indent}for(int i = 0; i < {out.size()}; i++) {{\n'
                        newline += f'{indent}    long long raw = ((long long)output[{out_offset} + i].to_uint64() << (64 - {total})) >> (64 - {total});\n'  # noqa: E501
                        newline += f'{indent}    fout << (double)raw / {scale}.0 << " ";\n'
                        newline += f'{indent}}}\n'
                    out_offset += out.size()
                newline += f'{indent}fout << "\\n";\n'

            elif '// hls-fpga-machine-learning insert float-output' in line:
                newline = line
                out_offset = 0
                for out in model_outputs:
                    total, frac = self._var_precision(out)
                    scale = 1 << frac
                    if io_type == 'io_stream':
                        newline += f'{indent}for(int i = 0; i < {out.size()}; i++) {{\n'
                        newline += f'{indent}    out_container_t raw_i = output_stream.read();\n'
                        newline += f'{indent}    long long raw = ((long long)raw_i.to_uint64() << (64 - {total})) >> (64 - {total});\n'  # noqa: E501
                        newline += f'{indent}    std::cout << (double)raw / {scale}.0 << " ";\n'
                        # Write the value straight back so the immediately
                        # following '// hls-fpga-machine-learning insert
                        # float-tb-output' block (which reads output_stream
                        # again for the results file) doesn't drain an
                        # already-empty FIFO. Mirrors the `keep` idiom
                        # nnet::print_result already uses for the same
                        # read-stream-twice situation (nnet_helpers.h).
                        newline += f'{indent}    output_stream.write(raw_i);\n'
                        newline += f'{indent}}}\n'
                    else:
                        newline += f'{indent}for(int i = 0; i < {out.size()}; i++) {{\n'
                        newline += f'{indent}    long long raw = ((long long)output[{out_offset} + i].to_uint64() << (64 - {total})) >> (64 - {total});\n'  # noqa: E501
                        newline += f'{indent}    std::cout << (double)raw / {scale}.0 << " ";\n'
                        newline += f'{indent}}}\n'
                    out_offset += out.size()
                newline += f'{indent}std::cout << std::endl;\n'

            else:
                newline = line

            fout.write(newline)

        tmpl.close()
        fout.close()

    def write_float_build_scripts(self, model):
        """Write build_tb_float_exe.sh and overwrite build_lib.sh with float-aware version."""
        filedir = Path(__file__).parent

        # build_tb_float_exe.sh
        tb_float_src = (filedir / '../templates/bambu/build_tb_float_exe.sh').resolve()
        tb_float_dst = Path(f'{model.config.get_output_dir()}/build_tb_float_exe.sh').resolve()
        with open(tb_float_src) as src, open(tb_float_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))
                dst.write(line)
        tb_float_dst.chmod(tb_float_dst.stat().st_mode | stat.S_IEXEC)

        # Overwrite build_lib.sh with float-aware version that links both
        # myproject.o and myproject_float.o into the bridge .so.
        build_lib_float_src = (filedir / '../templates/bambu/build_lib_float.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        with open(build_lib_float_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))
                dst.write(line)
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

    def write_hls(self, model, is_multigraph=False):
        super().write_hls(model, is_multigraph=is_multigraph)
        if not is_multigraph:
            self.write_float_header(model)
            self.write_float_wrapper(model)
            self.write_float_test_bench(model)
            self.write_float_build_scripts(model)
