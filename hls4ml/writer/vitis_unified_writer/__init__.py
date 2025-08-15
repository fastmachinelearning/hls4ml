import os
from pathlib import Path
import stat
from shutil import copyfile

from hls4ml.writer.vitis_writer import VitisWriter

from . import build_gen       as bg
from . import meta_gen        as mtg
from . import mm_gen          as mmg
from . import stream_gen      as sg
from . import test_bridge_gen as tbg
from . import test_cosim_gen  as tcg

from .meta import VitisUnifiedWriterMeta



class VitisUnifiedWriter(VitisWriter):

    def __init__(self):
        super().__init__()
        self.writer_meta = VitisUnifiedWriterMeta()

    def write_hls(self, model, is_multigraph=False):

        from hls4ml.backends import VitisUnifiedConfig

        self.writer_meta.vitis_unified_config = VitisUnifiedConfig(
            model.config, model.get_input_variables(), model.get_output_variables()
        )
        super().write_hls(model)
        mmg.write_gmem_wrapper(self.writer_meta, model)
        sg.write_axi_wrapper   (self.writer_meta, model)
        bg.write_hls_kernel_cfg(self.writer_meta, model)
        bg .write_driver       (self.writer_meta, model)
        tcg.write_wrapper_test (self.writer_meta, model)


        #self.write_new_tar(model)
        #if not is_multigraph:

        #else:
        #    self.write_bridge_multigraph(model)
            # self.modify_write_build_script_multigraph(model)