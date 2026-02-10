import os

from hls4ml.writer.vitis_unified_writer import VitisUnifiedWriterMeta

# file and directory


class VitisUnified_MetaGen:

    @classmethod
    def is_axi_stream(self, meta: VitisUnifiedWriterMeta):
        axi_mode = meta.vitis_unified_config.get_axi_mode()
        return axi_mode == "axis"

    @classmethod
    def is_axi_master(self, meta: VitisUnifiedWriterMeta):
        axi_mode = meta.vitis_unified_config.get_axi_mode()
        return axi_mode == "axim"

    @classmethod
    def get_project_name(self, model):
        return model.config.get_project_name()

    @classmethod
    def get_wrapper_file_name(self, model, is_axi_master):
        if is_axi_master:
            return f"{model.config.get_project_name()}_dm"
        else:
            return f"{model.config.get_project_name()}_axis"

    @classmethod
    def get_sim_file_name(cls):
        return "myproject_test"

    @classmethod
    def get_main_file_name(self, model):
        return model.config.get_project_name()

    @classmethod
    def get_vitis_unified_working_directory_dir(self, model):
        return os.path.join(model.config.get_output_dir(), "unifiedWorkspace")

    @classmethod
    def get_vitis_hls_dir(self, model):
        vitisWorkingDir = self.get_vitis_unified_working_directory_dir(model)
        return os.path.join(vitisWorkingDir, model.config.get_project_name())

    @classmethod
    def get_vitis_hls_exec_dir(self, model):
        hlsDir = self.get_vitis_hls_dir(model)
        return os.path.join(hlsDir, "unifiedPrj")

    @classmethod
    def get_vitis_linker_dir(self, model):
        vitisWorkingDir = self.get_vitis_unified_working_directory_dir(model)
        return os.path.join(vitisWorkingDir, "linker")

    @classmethod
    def get_xo_file_name(self, model):
        return f"{self.get_top_wrap_func_name(model, True)}.xo"
        # todo fix it

    @classmethod
    def get_xo_file_path(self, model):
        return os.path.join(self.get_vitis_hls_exec_dir(model), self.get_xo_file_name(model))

    # naming of variable function helper

    # FOR GMEM WRAPPER

    @classmethod
    def get_io_port_name(self, tensorVar, isInput: bool, idx: int):
        ioDirect = "in" if isInput else "out"
        return f"gmem_{ioDirect}{str(idx)}_ptr_{tensorVar.name}"

    @classmethod
    def get_io_port_size_name(self, tensorVar, isInput: bool, idx: int):
        ioDirect = "in" if isInput else "out"
        return f"gmem_{ioDirect}{str(idx)}_size_{tensorVar.name}"

    @classmethod
    def get_local_stream_name(self, tensorVar, isInput: bool, idx: int):
        ioDirect = "in" if isInput else "out"
        return f"stream_{ioDirect}{str(idx)}_{tensorVar.name}"

    @classmethod
    def get_dma_type_name(self):
        return "dma_data_packet"

    @classmethod
    def get_wrapper_port_name(self, tensorVar, isInput: bool):
        ioStr = "in" if isInput else "out"
        return f"par_{ioStr}_{tensorVar.name}"

    @classmethod
    def get_top_model_name(self, model):
        return f"{model.config.get_project_name()}"

    @classmethod
    def get_top_wrap_func_name(self, model, is_axi_master):
        if is_axi_master:
            return f"{model.config.get_project_name()}_gem"
        else:
            return f"{model.config.get_project_name()}_axi"

    # it is renamed for stitch layer
    @classmethod
    def rename_type(self, tensorVar, layerIdx: int, isInput: bool):
        return "result_" + tensorVar.type.name + f"_at_layer_{str(layerIdx)}"

    @classmethod
    def get_output_kernel_type(cls):
        return "xo"
