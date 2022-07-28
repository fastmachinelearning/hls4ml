import os
import sys

from hls4ml.backends import VivadoBackend
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_vivado_report


class VitisBackend(VivadoBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='Vitis')
        self._register_flows()

    def _register_flows(self):
        vivado_ip = 'vivado:ip'
        writer_passes = ['make_stamp', 'vitis:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=[vivado_ip], backend=self.name)
        self._default_flow = vivado_ip

    def build(self, model, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, vsynth=False):
        if 'linux' in sys.platform:
            found = os.system('command -v vitis_hls > /dev/null')
            if found != 0:
                raise Exception('Vitis HLS installation not found. Make sure "vitis_hls" is on PATH.')
        
        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        os.system('vitis_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} validation={validation} export={export} vsynth={vsynth}"'
            .format(reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, vsynth=vsynth))
        os.chdir(curr_dir)

        return parse_vivado_report(model.config.get_output_dir())