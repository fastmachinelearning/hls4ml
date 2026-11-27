import importlib.util
import json
import os
import time

from hls4ml.backends.vitis.passes.fifo_depth_optimization import (
    generate_depths_file,
    set_optimized_fifo_depths,
)
from hls4ml.model.optimizer.optimizer import ConfigurableOptimizerPass, ModelOptimizerPass

_KNOWN_SOLVERS = ('heuristic', 'sa', 'group-sa', 'random', 'group-random')


class FifoDepthOptimizationAdvisor(ConfigurableOptimizerPass, ModelOptimizerPass):
    """Size the inter-layer FIFOs of an ``io_stream`` design from a LightningSim trace, without co-simulation."""

    def __init__(self):
        self.solvers = _KNOWN_SOLVERS
        self.selection = 'min_bram'
        self.search_scope = 'compute'
        self.write_report = True
        self.skip_synthesis = False

    def transform(self, model):
        """Synthesise at the default depths, search FIFO depths with FIFO-Advisor and apply the selected ones.

        Args:
            model (ModelGraph): The model to which FIFO depth optimization is applied.

        Raises:
            RuntimeError: If the IO type is not "io_stream", if ``axi_mode`` is not ``'axi_master'``, if
                ``lightningsim`` or ``fifo_advisor`` is not installed, if ``skip_synthesis`` is set but the project
                has not been synthesised, or if the reconstructed ``hls.app`` does not match the backend ``.cfg``.
            ValueError: If ``search_scope``, ``selection`` or ``solvers`` is invalid.

        Returns:
            bool: The execution state of the Optimizer Pass
        """
        if model.config.get_config_value('IOType') != 'io_stream':
            raise RuntimeError('To use this optimization you have to set `IOType` field to `io_stream` in the HLS config.')
        axi_mode = model.config.get_config_value('VitisUnifiedConfig')['axi_mode']
        if axi_mode != 'axi_master':
            raise RuntimeError(f"FIFO-Advisor depth optimization only supports axi_mode='axi_master', got {axi_mode!r}.")
        if self.search_scope not in ('all', 'compute'):
            raise ValueError(f"search_scope must be 'all' or 'compute', got {self.search_scope!r}")
        if not callable(self.selection) and self.selection not in ('min_bram', 'min_latency'):
            raise ValueError(f"selection must be 'min_bram', 'min_latency' or a callable, got {self.selection!r}")
        unknown_solvers = [s for s in self.solvers if s not in _KNOWN_SOLVERS]
        if unknown_solvers:
            raise ValueError(f'unknown solver(s) {unknown_solvers}; known: {list(_KNOWN_SOLVERS)}')

        # The helper modules import these lazily, so importing them would not detect their absence.
        missing = [name for name in ('lightningsim', 'fifo_advisor') if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError(
                f'FIFO-Advisor depth optimization requires the {" and ".join(missing)} '
                f'package{"s" if len(missing) > 1 else ""}, which {"are" if len(missing) > 1 else "is"} '
                'not installed. See docs/advanced/fifo_depth.rst for installation.'
            )

        output_dir = model.config.get_output_dir()
        project_name = model.config.get_project_name()
        exec_dir = model.config.backend.writer.get_vitis_hls_exec_dir(model)
        solution_dir = os.path.join(exec_dir, 'hls')

        # LightningSim waits indefinitely for a synthesis database that is never written.
        if self.skip_synthesis and not os.path.exists(os.path.join(solution_dir, '.autopilot', 'db', 'dut.hcp')):
            raise RuntimeError(f'skip_synthesis is set, but {solution_dir} has not been synthesised')

        try:
            from .fifo_advisor import hls_app, ls_compat
            from .fifo_advisor.sweep import analyze_solution
        except ImportError as exc:  # pragma: no cover - depends on the user's env
            raise RuntimeError(
                f'FIFO-Advisor depth optimization could not import its helper modules. Original error: {exc}'
            ) from exc

        # Must be installed before anything imports lightningsim.trace_file.
        ls_compat.install()

        synthesis_wall = 0.0
        if not self.skip_synthesis:
            t_synth = time.time()
            model.write()
            model.build(reset=False, csim=False, synth=True, cosim=False, bitfile=False, log_to_stdout=False)
            synthesis_wall = round(time.time() - t_synth, 3)

        # The csim variant, so the traced testbench is built without -DRTL_SIM.
        cfg_path = os.path.join(model.config.backend.writer.get_vitis_hls_dir(model), 'hls_kernel_config_csim.cfg')
        hls_app_path = os.path.join(exec_dir, 'hls.app')

        # LightningSim needs hls.app, which the v++ flow does not write.
        hls_app.generate(cfg_path, hls_app_path, project_name)
        ok_top, ok_files, _ = hls_app.verify(cfg_path, hls_app_path)
        if not (ok_top and ok_files):
            raise RuntimeError('reconstructed hls.app does not match the backend .cfg it was derived from')

        # Only inter-layer FIFOs can be written back; wrapper channels have no hls4ml variable.
        known = {v.name for v in model.output_vars.values() if 'StreamVariable' in str(type(v)) and v.pragma}

        result = analyze_solution(
            solution_dir,
            solvers=self.solvers,
            selection=self.selection,
            restrict_names=known if self.search_scope == 'compute' else None,
        )

        depths = {name: depth for name, depth in result.depths.items() if name in known}

        initial_depths = {
            v.name: int(v.pragma[1])
            for v in model.output_vars.values()
            if 'StreamVariable' in str(type(v)) and isinstance(v.pragma, tuple) and v.name in depths
        }

        # FIFO-Advisor's minimum depth is 2, so when the defaults are already 1 the winner can tie the
        # baseline while using more slots. Keep the defaults unless the winner is actually better.
        baseline = result.per_solver.get('baseline', {})
        base_cost = (baseline.get('best_bram'), baseline.get('best_latency'))
        win_cost = (result.winner['bram'], result.winner['latency'])
        strictly_better = None not in base_cost and win_cost < base_cost
        fewer_slots = sum(depths.values()) < sum(initial_depths.values())
        if strictly_better or fewer_slots:
            set_optimized_fifo_depths(model, depths)
        else:
            print(
                'FIFO-Advisor: default depths are already optimal '
                f'(winner {win_cost} does not beat baseline {base_cost} and uses '
                f'{sum(depths.values())} slots vs {sum(initial_depths.values())}); leaving them unchanged'
            )
            depths = dict(initial_depths)
        generate_depths_file(model, initial_depths, depths)

        if self.write_report:
            report = result.to_dict()
            report['search_scope'] = self.search_scope
            report['timings_s']['synthesis'] = synthesis_wall
            report['applied_depths'] = depths
            with open(os.path.join(output_dir, 'fifo_depths_advisor.json'), 'w') as report_file:
                json.dump(report, report_file, indent=2)

        print(
            f'FIFO optimization completed (FIFO-Advisor): applied {len(depths)} channel depths, '
            f'estimated latency {result.winner["latency"]} / BRAM {result.winner["bram"]}'
        )

        return False
