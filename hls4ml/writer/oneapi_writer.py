from hls4ml.writer.altera_writer import AlteraWriter


class OneAPIWriter(AlteraWriter):
    compiler_executable = 'icpx'
    compiler_definitions = ('HLS4ML_ONEAPI',)
