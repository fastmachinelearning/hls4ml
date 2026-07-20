import json
import pathlib
import subprocess

from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import BambuAcceleratorBackend


class NanoXploreAcceleratorBackend(BambuAcceleratorBackend):
    """Concrete BambuAccelerator backend targeting NanoXplore NG-ULTRA devices."""

    _default_device: str | None = 'nx2h540tsc'

    def __init__(self):
        super().__init__()
        # Flows, passes and the writer are registered under 'BambuAccelerator'
        # by super().__init__() (their ids are stored on self, so lookups keep
        # working). The instance name must be the *registered* alias though:
        # hls4ml writes backend.name into the model config and round-trips it
        # through get_backend(), and 'BambuAccelerator' is abstract/unregistered.
        self.name = 'NanoXploreAccelerator'

    # Pass lookups that run lazily (after the rename above) must keep using
    # the name the passes were registered under. Without this, apply_templates
    # resolves zero templates, no layer gets a function_cpp, and the generated
    # <proj>_float.cpp body contains no layer calls — Bambu then dead-codes the
    # entire datapath (no output write ports on the HLS top).
    _passes_name = 'BambuAccelerator'

    def _get_layer_templates(self):
        from hls4ml.backends.template import Template
        from hls4ml.model.optimizer import get_backend_passes, get_optimizer

        return [name for name in get_backend_passes(self._passes_name) if isinstance(get_optimizer(name), Template)]

    def _get_layer_initializers(self):
        real_name = self.name
        self.name = self._passes_name
        try:
            return super()._get_layer_initializers()
        finally:
            self.name = real_name

    def create_initial_config(self, part='nx2h540tsc', clock_period=20, **kwargs):
        """NG-ULTRA defaults: nx2h540tsc (mapped in partname_to_bambu) and 20 ns,
        matching the DevKit's 50 MHz oscillator so the P&R constraint equals the
        physical clock without a PLL. The inherited Bambu defaults (Xilinx part,
        5 ns) would silently mis-target both HLS scheduling and the manifest."""
        return super().create_initial_config(part=part, clock_period=clock_period, **kwargs)

    def _generate_bitstream(self, model, project_dir: str, manifest: dict) -> dict:
        """Shell out to hls4ml-nanoxplore-bitstream and return parsed metrics."""
        cmd = self._resolve_bitstream_command(model)
        try:
            # stdout/stderr inherited: P&R runs for a long time and the CLI
            # streams live progress; capturing here would silence the chain.
            ret = subprocess.run(
                [cmd, project_dir],
                check=False,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f'NanoXplore bitstream driver not installed '
                f'(command not found: {cmd!r}). '
                f'Build produced the manifest at {project_dir}/manifest.json.'
            )
        if ret.returncode != 0:
            raise RuntimeError(
                f'hls4ml-nanoxplore-bitstream failed (rc={ret.returncode}); '
                f'see its output above and the logs in {project_dir}'
            )
        report_path = pathlib.Path(project_dir) / 'report.json'
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        return {}

    def _resolve_bitstream_command(self, model) -> str:
        if hasattr(self, '_bitstream_command') and self._bitstream_command:
            return self._bitstream_command
        try:
            cmd = model.config.get_config_value('BitStreamCommand')
            if cmd:
                return cmd
        except Exception:
            pass
        import shutil

        found = shutil.which('hls4ml-nanoxplore-bitstream')
        if found:
            return found
        raise RuntimeError(
            'NanoXplore bitstream driver not installed. '
            'Set BitStreamCommand in hls4ml config or install '
            'hls4ml-nanoxplore-bitstream on PATH.'
        )
