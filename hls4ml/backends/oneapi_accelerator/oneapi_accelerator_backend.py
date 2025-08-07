from hls4ml.backends import OneAPIBackend
from hls4ml.model.flow import get_flow, register_flow


class OneAPIAcceleratorBackend(OneAPIBackend):
    """
    This is the backend to run oneAPI code on an accelerator using the oneAPI framework.
    """

    def __init__(self):
        super().__init__(name='OneAPIAccelerator')

    def _register_flows(self):
        writer_passes = ['make_stamp', 'oneapiaccelerator:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['oneapi:ip'], backend=self.name)

        ip_flow_requirements = get_flow('oneapi:ip').requires.copy()
        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)
