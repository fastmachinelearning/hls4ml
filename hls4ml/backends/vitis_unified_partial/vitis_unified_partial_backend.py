import os
import sys
import subprocess
from shutil import copy2


from hls4ml.backends import VitisUnifiedBackend, VivadoBackend
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_vivado_report

from hls4ml.writer.vitis_unified_partial_writer.meta_gen import VitisUnifiedPartial_MetaGen  as mg


class VitisUnifiedPartialBackend(VitisUnifiedBackend):

    def __init__(self):
        super(VivadoBackend, self).__init__(name='VitisUnifiedPartial')
        self._register_layer_attributes()
        self._register_flows()


    def build(
        self,
        model,
        reset=False,
        csim=False,
        synth=False,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
        bitfile=False,
        log_to_stdout=True
    ):

        ##### do magic streamer generation


        pass

        # super().build(
        # model,
        # reset,
        # csim,
        # synth,
        # cosim,
        # validation,
        # export,
        # vsynth,
        # fifo_opt,
        # bitfile,
        # log_to_stdout
        # )

    def create_initial_config(
        self,
        board='pynq-z2',
        part=None,
        clock_period=5,
        clock_uncertainty='12.5%',
        io_type='io_parallel',
        interface='axi_stream',
        driver='python',
        input_type='float',
        output_type='float',
        gmemBuf_in_size=12,
        gmemBuf_out_size=12,
        xpfmPath='/tools/Xilinx/Vitis/2023.2/base_platforms/'
                 'xilinx_zcu102_base_202320_1/xilinx_zcu102_base_202320_1.xpfm',
        input_interim_type='io_stream',  #### it should be io_stream or io_free_stream/ io_stream
        output_interim_type='io_stream',
        init_mgs_meta=None,
        **_
    ):

        if init_mgs_meta is None:
            init_mgs_meta = list()
        if init_mgs_meta is None:
            init_mgs_meta = []
        config = super().create_initial_config(
            board=board,
            part=part,
            clock_period=clock_period,
            clock_uncertainty=clock_uncertainty,
            io_type=io_type,
            interface=interface,
            driver=driver,
            input_type=input_type,
            output_type=output_type,
            gmemBuf_in_size=1,
            gmemBuf_out_size=1,
            xpfmPath=xpfmPath
        )

        config['MultiGraphConfig'] = {}
        config['MultiGraphConfig']['amtGraph'] = -1 # it should be set by the multigraph system
        config['MultiGraphConfig']['graphIdx'] = -1 # -1 means unset yet or it is multigraph stitcher
        print(f"mgs initial is set to {init_mgs_meta}")
        config['MultiGraphConfig']['MgsMeta']  = init_mgs_meta if init_mgs_meta is not None else [] #### it should be only used for stitcher


        config['MultiGraphConfig']['IOInterimType'] = {}
        config['MultiGraphConfig']['IOInterimType']['Input'] = input_interim_type
        config['MultiGraphConfig']['IOInterimType']['Output'] = output_interim_type

        return config



    def _register_flows(self):
        vitis_ip = 'vitis:ip'
        writer_passes = ['make_stamp', 'vitisunifiedpartial:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['vitis:ip'], backend=self.name)
        self._default_flow = vitis_ip