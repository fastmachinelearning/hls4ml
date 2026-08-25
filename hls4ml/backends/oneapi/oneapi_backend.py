from hls4ml.backends.altera.altera_backend import AlteraBackend


class OneAPIBackend(AlteraBackend):
    compiler_executable = 'icpx'

    def __init__(self):
        super().__init__('oneAPI')
