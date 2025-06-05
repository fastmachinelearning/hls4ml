import os
import subprocess
import sys

from hls4ml.backends import FPGABackend
from hls4ml.model.attributes import ChoiceAttribute
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import Dense, Layer
from hls4ml.model.optimizer import layer_optimizer
from hls4ml.report import parse_libero_report


class LiberoBackend(FPGABackend):
    def __init__(self):
        super().__init__(name='Libero')
        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self):
        strategy_layers = [
            Dense,
        ]

        for layer in strategy_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(
                ChoiceAttribute(
                    'strategy',
                    choices=['latency', 'resource'],
                    default='latency',
                )
            )
            self.attribute_map[layer] = attrs

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        libero_types = [
            'libero:transform_types',
            'libero:set_pipeline_style',
        ]
        libero_types_flow = register_flow('specific_types', libero_types, requires=[init_flow], backend=self.name)

        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', 'libero:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['libero:ip'], backend=self.name)

        ip_flow_requirements = [
            'optimize',
            init_flow,
            libero_types_flow,
            template_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def create_initial_config(
        self,
        fpga_family='PolarFire',
        part='MPF300',
        board='hw_only',
        clock_period=5,
        clock_uncertainty='27%',
        io_type='io_parallel',
        namespace=None,
        write_weights_txt=True,
        write_tar=False,
        **_,
    ):
        """Create initial configuration of the Libero backend.

        Args:
            part (str, optional): The FPGA part to be used. Defaults to 'MPF300'.
            clock_period (int, optional): The clock period. Defaults to 5.
            clock_uncertainty (str, optional): The clock uncertainty. Defaults to 27%.
            io_type (str, optional): Type of implementation used. One of
                'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
            namespace (str, optional): If defined, place all generated code within a namespace. Defaults to None.
            write_weights_txt (bool, optional): If True, writes weights to .txt files which speeds up compilation.
                Defaults to True.
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file. Defaults to False.

        Returns:
            dict: initial configuration.
        """
        config = {}

        config['FPGAFamily'] = fpga_family if fpga_family is not None else 'PolarFire'
        config['Part'] = part if part is not None else 'MPF300'
        config['Board'] = board if board is not None else 'hw_only'
        config['ClockPeriod'] = clock_period if clock_period is not None else 5
        config['IOType'] = io_type if io_type is not None else 'io_parallel'
        config['HLSConfig'] = {}
        config['WriterConfig'] = {
            'Namespace': namespace,
            'WriteWeightsTxt': write_weights_txt,
            'WriteTar': write_tar,
        }

        return config

    def build(
        self,
        model,
        reset=False,
        skip_preqs=False,
        sw_compile=True,
        hw=True,
        cosim=False,
        rtl_synth=False,
        fpga=False,
        **kwargs,
    ):
        """Build the model using Libero suite and SmartHLS compiler. Additional arguments passed to the function in form of
        `<arg>=True` will be passed as an argument to the `shls` command. See SmartHLS user guide for list of possible
        command line options.

        Args:
            model (ModelGraph): Model to build
            reset (bool, optional): Clean up any existing files. Defaults to False.
            skip_preqs(bool, optional): Skip any prerequisite step that is outdated. Defaults to False.
            sw_compile (bool, optional): Compile the generated HLS in software. Defaults to True.
            hw (bool, optional): Compile the software to hardware, producing a set of Verilog HDL files. Defaults to True.
            cosim (bool, optional): Run co-simulation. Defaults to False.
            rtl_synth (bool, optional): Run RTL synthesis for resource results. This will take less time than `fpga`.
                Defaults to False.
            fpga (bool, optional): Synthesize the generated hardware to target FPGA. This runs RTL synthesis and
                place-and-route for resource and timing results. Defaults to False.

        Raises:
            Exception: Raised if the `shls` command has not been found
            CalledProcessError: Raised if SmartHLS returns non-zero code for any of the commands executed

        Returns:
            dict: Detailed report produced by SmartHLS.
        """
        if 'linux' in sys.platform:
            found = os.system('command -v shls > /dev/null')
            if found != 0:
                raise Exception('Libero/SmartHLS installation not found. Make sure "shls" is on PATH.')

        def run_shls_cmd(cmd_name):
            subprocess.run(
                ['shls', '-s', cmd_name],
                shell=False,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                cwd=model.config.get_output_dir(),
            )

        if reset:
            run_shls_cmd('clean')
        if sw_compile:
            run_shls_cmd('sw_compile')
        if hw:
            run_shls_cmd('hw')
        if cosim:
            run_shls_cmd('cosim')
        if rtl_synth:
            run_shls_cmd('rtl_synth')
        if fpga:
            run_shls_cmd('fpga')

        for arg_name, arg_val in kwargs.items():
            if arg_val:
                run_shls_cmd(arg_name)

        return parse_libero_report(model.config.get_output_dir())

    @layer_optimizer(Layer)
    def init_base_layer(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('reuse_factor', reuse_factor)

    @layer_optimizer(Dense)
    def init_dense(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')
