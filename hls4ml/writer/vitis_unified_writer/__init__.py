import os
from pathlib import Path
import stat
from shutil import copyfile

from hls4ml.writer.vitis_writer import VitisWriter
from .meta import VitisUnifiedWriterMeta



class VitisUnifiedWriter(VitisWriter):

    def __init__(self):
        super().__init__()
        self.writer_meta = VitisUnifiedWriterMeta()

        from .build_gen       import VitisUnified_BuildGen
        from .driver_gen      import VitisUnified_DriverGen
        from .meta_gen        import VitisUnified_MetaGen
        from .test_bridge_gen import VitisUnified_BridgeGen
        from .test_cosim_gen  import VitisUnified_TestGen
        from .wrap_gen        import VitisUnified_WrapperGen

        self.bg   = VitisUnified_BuildGen
        self.dg   = VitisUnified_DriverGen
        self.mg   = VitisUnified_MetaGen
        self.tbg  = VitisUnified_BridgeGen
        self.tcg  = VitisUnified_TestGen
        self.wg   = VitisUnified_WrapperGen




    def write_board_script_override(self, model):
        pass
    def write_build_prj_override(self, model):
        pass
    def write_build_opts(self, model):
        pass
    def write_tar(self, model):
        pass

    def write_bridge(self, model): ### test bench gen
        self.tbg.write_bridge(self.writer_meta, model, self.mg)

    def write_build_script(self, model):
        #### for bridge simulation
        self.bg.write_bridge_build_script(self.writer_meta, model, self.mg)
        #### for hls kernel generation
        self.bg.build_unified_project_ske(self.writer_meta, model, self.mg)
        self.bg.write_hls_kernel_cfg(self.writer_meta, model, self.mg)
        #### for v++ to link hls to the system
        self.bg.write_launch_vitis_linker_dir(self.writer_meta, model, self.mg)
        self.bg.write_launch_vitis_linker_launcher(self.writer_meta, model, self.mg)
        self.bg.write_launch_vitis_linker_cfg(self.writer_meta, model, self.mg)

    def generate_config(self, model):
        from hls4ml.backends import VitisUnifiedConfig
        self.writer_meta.vitis_unified_config = VitisUnifiedConfig(
            model.config, model.get_input_variables(), model.get_output_variables()
        )

    def make_export_path(self, model):
        export_path = f'{model.config.get_output_dir()}/export'
        if not os.path.exists(export_path):
            os.makedirs(export_path)

    def write_hls(self, model, is_multigraph=False):


        if is_multigraph:
            super().write_hls(model, is_multigraph = True)
            return

        self.generate_config(model)


        super().write_hls(model, is_multigraph = False)
        self.wg.write_wrapper(self.writer_meta, model, self.mg)


        #########
        self.make_export_path(model)
        self.dg .write_driver            (self.writer_meta, model, self.mg)
        self.tcg.write_wrapper_test      (self.writer_meta, model, self.mg)


        #self.write_new_tar(model)
        #if not is_multigraph:

        #else:
        #    self.write_bridge_multigraph(model)
            # self.modify_write_build_script_multigraph(model)