import importlib.util
import json
import os
import shutil
import subprocess
import sys

from hls4ml.backends import VivadoBackend
from hls4ml.model.flow import get_flow, register_flow
from hls4ml.report import aggregate_graph_reports, parse_vivado_report
from hls4ml.utils.simulation_utils import (
    annotate_axis_stream_widths,
    prepare_tb_inputs,
    read_testbench_log,
    write_verilog_testbench,
)


class VitisBackend(VivadoBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='Vitis')
        self._register_layer_attributes()
        self._register_flows()

    def _register_flows(self):
        validation_passes = [
            'vitis:validate_conv_implementation',
            'vitis:validate_resource_strategy',
            'vitis:validate_resource_unrolled_strategy',
        ]
        validation_flow = register_flow('validation', validation_passes, requires=['vivado:init_layers'], backend=self.name)

        # Any potential templates registered specifically for Vitis backend
        template_flow = register_flow(
            'apply_templates', self._get_layer_templates, requires=['vivado:init_layers'], backend=self.name
        )

        writer_passes = ['make_stamp', 'vitis:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['vitis:ip'], backend=self.name)

        ip_flow_requirements = get_flow('vivado:ip').requires.copy()
        ip_flow_requirements.insert(ip_flow_requirements.index('vivado:init_layers'), validation_flow)
        ip_flow_requirements.insert(ip_flow_requirements.index('vivado:apply_templates'), template_flow)

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

        # Register the fifo depth optimization flow which is different from the one for vivado
        fifo_depth_opt_passes = [
            'vitis:fifo_depth_optimization'
        ] + writer_passes  # After optimization, a new project will be written

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=['vitis:ip'], backend=self.name)

    def create_initial_config(
        self,
        part='xcvu13p-flga2577-2-e',
        clock_period=5,
        clock_uncertainty='27%',
        io_type='io_parallel',
        namespace=None,
        write_weights_txt=True,
        write_tar=False,
        tb_output_stream='both',
        **_,
    ):
        """Create initial configuration of the Vitis backend.

        Args:
            part (str, optional): The FPGA part to be used. Defaults to 'xcvu13p-flga2577-2-e'.
            clock_period (int, optional): The clock period. Defaults to 5.
            clock_uncertainty (str, optional): The clock uncertainty. Defaults to 27%.
            io_type (str, optional): Type of implementation used. One of
                'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
            namespace (str, optional): If defined, place all generated code within a namespace. Defaults to None.
            write_weights_txt (bool, optional): If True, writes weights to .txt files which speeds up compilation.
                Defaults to True.
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file. Defaults to False.
            tb_output_stream (str, optional): Controls where to write the output. Options are 'stdout', 'file' and 'both'.
                Defaults to 'both'.

        Returns:
            dict: initial configuration.
        """
        config = {}

        config['Part'] = part if part is not None else 'xcvu13p-flga2577-2-e'
        config['ClockPeriod'] = clock_period if clock_period is not None else 5
        config['ClockUncertainty'] = clock_uncertainty if clock_uncertainty is not None else '27%'
        config['IOType'] = io_type if io_type is not None else 'io_parallel'
        config['HLSConfig'] = {}
        config['WriterConfig'] = {
            'Namespace': namespace,
            'WriteWeightsTxt': write_weights_txt,
            'WriteTar': write_tar,
            'TBOutputStream': tb_output_stream,
        }

        return config

    def build(
        self,
        model,
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
        log_to_stdout=True,
    ):
        if 'linux' in sys.platform:
            found = os.system('command -v vitis_hls > /dev/null')
            if found != 0:
                raise Exception('Vitis HLS installation not found. Make sure "vitis_hls" is on PATH.')

        build_command = (
            'vitis_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} '
            'validation={validation} export={export} vsynth={vsynth} fifo_opt={fifo_opt}"'
        ).format(
            reset=reset,
            csim=csim,
            synth=synth,
            cosim=cosim,
            validation=validation,
            export=export,
            vsynth=vsynth,
            fifo_opt=fifo_opt,
        )

        output_dir = model.config.get_output_dir()
        stdout_log = os.path.join(output_dir, 'build_stdout.log')
        stderr_log = os.path.join(output_dir, 'build_stderr.log')

        stdout_target = None if log_to_stdout else open(stdout_log, 'w')
        stderr_target = None if log_to_stdout else open(stderr_log, 'w')

        try:
            process = subprocess.Popen(
                build_command, shell=True, cwd=output_dir, stdout=stdout_target, stderr=stderr_target, text=True
            )
            process.communicate()

            if process.returncode != 0:
                raise Exception(f'Build failed for {model.config.get_project_name()}. See logs for details.')
        finally:
            if not log_to_stdout:
                stdout_target.close()
                stderr_target.close()

        return parse_vivado_report(output_dir)

    def build_stitched_design(
        self,
        model,
        stitch_design=True,
        sim_stitched_design=False,
        export_stitched_design=False,
        graph_reports=None,
        simulation_input_data=None,
    ):

        nn_config = model.nn_config
        os.makedirs(nn_config['OutputDir'], exist_ok=True)
        stitched_design_dir = os.path.join(nn_config['OutputDir'], nn_config['StitchedProjectName'])
        if stitch_design:
            if os.path.exists(stitched_design_dir):
                shutil.rmtree(stitched_design_dir)
            os.makedirs(stitched_design_dir)

        spec = importlib.util.find_spec('hls4ml')
        hls4ml_path = os.path.dirname(spec.origin)
        ip_stitcher_path = os.path.join(hls4ml_path, 'templates/vivado/ip_stitcher.tcl')
        stdout_log = os.path.join(stitched_design_dir, 'stitcher_stdout.log')
        stderr_log = os.path.join(stitched_design_dir, 'stitcher_stderr.log')
        nn_config_path = os.path.join(stitched_design_dir, 'nn_config.json')
        testbench_path = os.path.join(stitched_design_dir, 'testbench.v')
        testbench_log_path = os.path.join(stitched_design_dir, 'testbench_log.csv')

        try:
            shutil.copy(ip_stitcher_path, stitched_design_dir)
        except Exception as e:
            print(f"Error: {e}. Cannot copy 'ip_stitcher.tcl' to {nn_config['StitchedProjectName']} folder.")

        # Verilog output bitwidths are rounded up and may differ from HLS output bitwidths
        if nn_config['outputs'][0]['pragma'] == 'stream':
            last_graph_project_path = os.path.join(
                model.graphs[-1].config.get_output_dir(), model.graphs[-1].config.get_project_dir()
            )
            annotate_axis_stream_widths(nn_config, last_graph_project_path)
        with open(nn_config_path, "w") as file:
            json.dump(nn_config, file, indent=4)

        if sim_stitched_design:
            write_verilog_testbench(nn_config, testbench_path)
            tb_inputs = prepare_tb_inputs(simulation_input_data, nn_config['inputs'])
            model.write_tb_inputs(tb_inputs, stitched_design_dir)
            print('Verilog testbench and its input data were generated.')

        print('Running build process of stitched IP...\n')
        stitch_command = [
            'vivado',
            '-mode',
            'batch',
            '-nojournal',
            '-nolog',
            '-notrace',
            '-source',
            ip_stitcher_path,
            '-tclargs',
            f'stitch_design={int(stitch_design)}',
            f'sim_design={int(sim_stitched_design)}',
            f'export_design={int(export_stitched_design)}',
            f"stitch_project_name={nn_config['StitchedProjectName']}",
            f"original_project_name={nn_config['OriginalProjectName']}",
            'sim_verilog_file=testbench.v',
        ]

        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            process = subprocess.Popen(
                stitch_command, cwd=stitched_design_dir, stdout=stdout_file, stderr=stderr_file, text=True, shell=False
            )
            process.communicate()
            if process.returncode != 0:
                raise Exception(f"Stitching failed for {nn_config['StitchedProjectName']}. See logs for details.")

        stitched_report = {'StitchedDesignReport': {}}
        if stitch_design:
            stitched_report = aggregate_graph_reports(graph_reports)

        if sim_stitched_design:
            testbench_output = read_testbench_log(testbench_log_path, nn_config['outputs'])
            stitched_report['BehavSimResults'] = testbench_output['BehavSimResults']
            stitched_report['StitchedDesignReport']['BestLatency'] = testbench_output['BestLatency']
            stitched_report['StitchedDesignReport']['WorstLatency'] = testbench_output['WorstLatency']

        return stitched_report
