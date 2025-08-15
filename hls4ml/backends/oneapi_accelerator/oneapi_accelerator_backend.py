from hls4ml.backends import OneAPIBackend
from hls4ml.model.flow import get_flow, register_flow


class OneAPIAcceleratorBackend(OneAPIBackend):
    """
    This is the backend to run oneAPI code on an accelerator using the oneAPI framework.
    """

    def __init__(self):
        super().__init__(name='oneAPIAccelerator')

    def _register_flows(self):
        writer_passes = ['make_stamp', 'oneapiaccelerator:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['oneapi:ip'], backend=self.name)

        oneapi_types = [
            'oneapiaccelerator:transform_types',
            'oneapi:register_bram_weights',
            'oneapi:apply_resource_strategy',
            'oneapi:apply_winograd_kernel_transformation',
        ]
        oneapi_types_flow = register_flow('specific_types', oneapi_types, requires=['oneapi:init_layers'], backend=self.name)

        ip_flow_requirements = [
            oneapi_types_flow if opt == 'oneapi:specific_types' else opt for opt in get_flow('oneapi:ip').requires
        ]
        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def create_initial_config(
        self, part, clock_period=5, hyperopt_handshake=False, io_type='io_parallel', write_tar=False, **_
    ):
        """Create initial configuration of the oneAPI backend.

        Args:
            part (str): The path to the board support package to be used. Can add :<board-variant>
            clock_period (int, optional): The clock period in ns. Defaults to 5.
            hyperopt_handshake (bool, optional): Should hyper-optimized handshaking be used? Defaults to False
            io_type (str, optional): Type of implementation used. One of
                'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file. Defaults to False.

        Returns:
            dict: initial configuration.
        """
        config = super().create_initial_config(part, clock_period, hyperopt_handshake, io_type, write_tar, **_)
        config['UseOneAPIBSP'] = True
        return config
