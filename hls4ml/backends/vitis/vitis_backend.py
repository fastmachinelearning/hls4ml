import os
import sys
import subprocess
import importlib.util
import json
import shutil

from hls4ml.backends import VivadoBackend
from hls4ml.model.flow import get_flow, register_flow
from hls4ml.report import parse_vivado_report
from hls4ml.utils.simulation_utils import generate_verilog_testbench, read_testbench_log


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

    def create_initial_config(
        self,
        part='xcvu13p-flga2577-2-e',
        clock_period=5,
        clock_uncertainty='27%',
        io_type='io_parallel',
        namespace=None,
        write_weights_txt=True,
        write_tar=False,
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
        }

        return config

    def build(self, model, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, vsynth=False):
        if 'linux' in sys.platform:
            found = os.system('command -v vitis_hls > /dev/null')
            if found != 0:
                raise Exception('Vitis HLS installation not found. Make sure "vitis_hls" is on PATH.')

        build_command = (
            'vitis_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} '
            'validation={validation} export={export} vsynth={vsynth}"'
        ).format(reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, vsynth=vsynth)

        output_dir = model.config.get_output_dir()
        # Define log file paths
        # NOTE - 'build_stdout.log' is the same as 'vitis_hls.log'
        stdout_log = os.path.join(output_dir, 'build_stdout.log')
        stderr_log = os.path.join(output_dir, 'build_stderr.log')
        
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            # Use subprocess.Popen to capture output
            process = subprocess.Popen(
                build_command,
                shell=True,
                cwd=output_dir,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True
            )
            process.communicate()
            if process.returncode != 0:
                raise Exception(f'Build failed for {model.config.get_project_name()}. See logs for details.')

        return parse_vivado_report(output_dir)
    
    def stitch_design(self, output_dir, project_name, sim_stitched_design=False, export_stitched_design=False, nn_config=None, build_results=None):
        
        os.makedirs(output_dir, exist_ok=True)
        stitched_design_dir = os.path.join(output_dir, 'vivado_stitched_design')
        os.makedirs(stitched_design_dir, exist_ok=True)
        spec = importlib.util.find_spec("hls4ml")
        hls4ml_path = os.path.dirname(spec.origin)
        ip_stitcher_path = os.path.join(hls4ml_path, 'templates/vivado/ip_stitcher.tcl')
        
        try:
            shutil.copy(ip_stitcher_path, stitched_design_dir)
        except Exception as e:
            print(f"Error: {e}. Cannot copy 'ip_stitcher.tcl' to 'vivado_stitched_design' folder.")

        nn_config_path = os.path.join(stitched_design_dir, "nn_config.json")
        if nn_config:
            with open(nn_config_path, "w") as file:
                json.dump(nn_config, file, indent=4)
        
        if(sim_stitched_design):
            testbench_path =  os.path.join(stitched_design_dir, "testbench.v")
            generate_verilog_testbench(nn_config, testbench_path)
            print('Verilog testbench generated.')

        print('Running build process of stitched IP...\n')
        stitch_command = [
            'vivado', '-mode', 'batch', '-nojournal', '-nolog', '-notrace',
            '-source', ip_stitcher_path,
            '-tclargs',
            f'sim_design={int(sim_stitched_design)}',
            f'export_design={int(export_stitched_design)}',
            f'sim_verilog_file=vivado_stitched_design/testbench.v'
        ]
                
        stdout_log = os.path.join(stitched_design_dir, 'stitcher_stdout.log')
        stderr_log = os.path.join(stitched_design_dir, 'stitcher_stderr.log')
        
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            process = subprocess.Popen(
                stitch_command,
                cwd=output_dir,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                shell=False
            )
            process.communicate()
            if process.returncode != 0:
                raise Exception(f'Stitching failed for {project_name}. See logs for details.')
        
        stitched_report = self._aggregate_build_results(build_results)

        if(sim_stitched_design):
            testbench_log_path = os.path.join(stitched_design_dir, 'vivado_stitched_design.sim/sim_1/behav/xsim/testbench_log.csv')
            sim_data = read_testbench_log(testbench_log_path)
            csim_results = []
            for name, arr in sim_data['outputs'].items():
                # Convert floats to strings
                arr_str = [f"{val:.6f}" for val in arr]
                csim_results.append(arr_str)

            # Add simulation data to stitched report
            stitched_report['CSimResults'] = csim_results
            stitched_report['CSynthesisReport']['Stiched_Design_Latency'] = sim_data['latency_cycles']

        return stitched_report
    
    def _aggregate_build_results(self, build_results):
        """
        Aggregate the resources of each subgraph into a single dictionary.
        For resources like BRAM_18K, DSP, FF, LUT, URAM we sum them.
        For timing/latency we picked we sum them.
        Here we:
        - Take TargetClockPeriod from the first subgraph.
        - Take the maximum EstimatedClockPeriod among subgraphs.
        - Take maximum BestLatency, WorstLatency, IntervalMin, IntervalMax among subgraphs.
        - Sum the resource fields.
        """

        if build_results is None or len(build_results) == 0:
            return {}

        keys_to_sum = ['BRAM_18K', 'DSP', 'FF', 'LUT', 'URAM', 'WorstLatency']
        # Non-resource fields we might want to handle
        # We'll initialize them from the first subgraph
        first_subgraph = next(iter(build_results))
        base_report = build_results[first_subgraph]['CSynthesisReport']

        final_report = {
            'TargetClockPeriod': base_report.get('TargetClockPeriod', '5.00'),
            'EstimatedClockPeriod': float(base_report.get('EstimatedClockPeriod', '5.00')),
            'WorstLatency': int(base_report.get('WorstLatency', '0')),
        }

        # Initialize resources
        for k in keys_to_sum:
            final_report[k] = int(base_report.get(k, '0'))

        # Also include availability fields from the first subgraph 
        # TODO match actual device resources
        final_report['AvailableBRAM_18K'] = base_report.get('AvailableBRAM_18K', '5376')
        final_report['AvailableDSP'] = base_report.get('AvailableDSP', '12288')
        final_report['AvailableFF'] = base_report.get('AvailableFF', '3456000')
        final_report['AvailableLUT'] = base_report.get('AvailableLUT', '1728000')
        final_report['AvailableURAM'] = base_report.get('AvailableURAM', '1280')

        # Aggregate from other subgraphs
        for subgraph, data in build_results.items():
            if subgraph == first_subgraph:
                continue
            report = data.get('CSynthesisReport', {})
            # Update non-resource fields
            est_cp = float(report.get('EstimatedClockPeriod', '5.00'))
            if est_cp > final_report['EstimatedClockPeriod']:
                final_report['EstimatedClockPeriod'] = est_cp

            # Take max of these latency fields
            final_report['WorstLatency'] = max(final_report['WorstLatency'], int(report.get('WorstLatency', '0')))

            # Sum resource fields
            for k in keys_to_sum:
                final_report[k] += int(report.get(k, '0'))

        # Convert numbers back to strings
        final_report['EstimatedClockPeriod'] = f"{final_report['EstimatedClockPeriod']:.3f}"
        final_report['WorstLatency'] = str(final_report['WorstLatency'])

        for k in keys_to_sum:
            final_report[k] = str(final_report[k])

        # Return in the desired structure
        return {'CSynthesisReport': final_report}
