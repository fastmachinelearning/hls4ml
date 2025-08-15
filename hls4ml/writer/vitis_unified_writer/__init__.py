import os
from pathlib import Path
import stat
from shutil import copyfile

from hls4ml.writer.vitis_writer import VitisWriter



class VitisUnifiedWriter(VitisWriter):

    def __init__(self):
        super().__init__()
        self.vitis_unified_config = None

    def write_hls(self, model, is_multigraph=False):

        from hls4ml.backends import VitisUnifiedConfig

        self.vitis_unified_config = VitisUnifiedConfig(
            model.config, model.get_input_variables(), model.get_output_variables()
        )
        super().write_hls(model)
        if not is_multigraph:
            self.write_board_script(model)
            self.write_driver(model)
            self.write_wrapper_test(model)
            self.write_axi_wrapper(model)
            self.modify_build_script(model)
            self.write_new_tar(model)
        else:
            self.write_bridge_multigraph(model)
            # self.modify_write_build_script_multigraph(model)