
from hls4ml.writer.vitis_unified_writer import VitisUnifiedWriter


class VitisUnifiedPartialWriter(VitisUnifiedWriter):

    def __init__(self):
        super().__init__()

        #from .build_gen       import VitisUnifiedPartial_BuildGen
        from .driver_gen      import VitisUnifiedPartial_DriverGen
        from .meta_gen        import VitisUnifiedPartial_MetaGen
        from .test_bridge_gen import VitisUnifiedPartial_BridgeGen
        from .test_cosim_gen  import VitisUnifiedPartial_TestGen
        from .wrap_gen        import VitisUnifiedPartial_WrapperGen

        from .mgs_gen         import VitisUnifiedPartial_MagicArchGen



        #################################################
        ######### override the vitisUnified Writer ######
        #################################################

        self.dg  = VitisUnifiedPartial_DriverGen
        self.mg  = VitisUnifiedPartial_MetaGen
        self.tbg = VitisUnifiedPartial_BridgeGen
        self.tcg = VitisUnifiedPartial_TestGen
        self.wg  = VitisUnifiedPartial_WrapperGen

        #################################################
        ######### override the vitisUnified Writer ######
        #################################################

        self.magic_gen = VitisUnifiedPartial_MagicArchGen


        #################################################
        ######### magic streamer controller variable ####
        #################################################
        self.mgs_mng = None

    def set_mgs_mng(self, mgs_mng):
        self.mgs_mng = mgs_mng

    def generate_config(self, model):
        from hls4ml.backends.vitis_unified_partial.vitis_unified_partial_config import VitisUnifiedPartialConfig
        self.writer_meta.vitis_unified_config = VitisUnifiedPartialConfig(
            model.config, model.get_input_variables(), model.get_output_variables(), self.mgs_mng

        )


    def write_hls(self, model, is_multigraph=False):

        super().write_hls(model, is_multigraph)

        if is_multigraph:
            self.magic_gen.copyMagicArchIp(self.writer_meta, model)
            self.magic_gen.write_mgs(self.writer_meta, model)
            self.magic_gen.gen_vivado_project(self.writer_meta, model, self.mg)