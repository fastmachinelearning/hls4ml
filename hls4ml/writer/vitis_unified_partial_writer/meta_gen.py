from hls4ml.writer.vitis_unified_writer.meta_gen import VitisUnified_MetaGen

class VitisUnifiedPartial_MetaGen(VitisUnified_MetaGen):

    ##################################################
    ## file and directory ############################
    ##################################################

    @classmethod
    def get_wrapper_file_name(self, model):
        return f"{model.config.get_project_name()}_axi"

    ##################################################
    ## naming function and variable    ###############
    ##################################################

    @classmethod
    def get_io_port_name(self, tensorVar, isInput: bool, idx: int):
        ioDirect = "in" if isInput else "out"
        return f"streamIo_{ioDirect}{str(idx)}_{tensorVar.name}"

    @classmethod
    def get_top_wrap_func_name(self, model):
        return f"{model.config.get_project_name()}_axis"

    @classmethod
    def get_input_size_arr_name(self, model):
        return "N_IN_" + model.config.get_project_name()

    @classmethod
    def get_output_size_arr_name(self, model):
        return "N_OUT_" + model.config.get_project_name()

    @classmethod
    def get_axi_wrapper_type(self, tensorVar):
        return f"{tensorVar.type.name}_packet"

    @classmethod
    def get_axi_wrapper_dec(self, tensorVar):
        return f"typedef hls::axis<{tensorVar.type.name}, 0,0,0, AXIS_ENABLE_LAST> {self.get_axi_wrapper_type(tensorVar)};"

    @classmethod
    def get_is_last_var(self, idx):
        return f"isLast_{str(idx)}"

    @classmethod
    def get_all_last_logic(self, amt):
        isLastList = [self.get_is_last_var(idx) for idx in range(amt)]
        return " & ".join(isLastList)

    ##################################################
    ## generation function call        ###############
    ##################################################

    @classmethod
    def get_enqueue_func_atom2stream(self, tensorVar, idx: int):
        result = "enqueue_atom2layer<{INPUT_LAYER_ARR}, {SIZE}>({SRC_STREAM}, {RAW_STREAM}, {IS_LAST});".format(
            INPUT_LAYER_ARR = tensorVar.type.name,
            SIZE            = str(tensorVar.size()),
            SRC_STREAM      = self.get_io_port_name(tensorVar, True, idx),
            RAW_STREAM      = self.get_local_stream_name(tensorVar, True, idx),
            IS_LAST         = self.get_is_last_var(idx),
        )
        return result

    @classmethod   ## rStream == raw stream (no tlast)
    def get_enqueue_func_stream2rstream(self, tensorVar, idx: int):
        result = "enqueue_layerStream2layer<{INPUT_LAYER_STREAM}, {INPUT_LAYER_ARR}, {SIZE}>({SRC_STREAM}, {RAW_STREAM}, {IS_LAST});".format(
            INPUT_LAYER_STREAM = self.get_axi_wrapper_type(tensorVar),
            INPUT_LAYER_ARR    = tensorVar.type.name,
            SIZE               = str(tensorVar.size()),
            SRC_STREAM         = self.get_io_port_name(tensorVar, True, idx),
            RAW_STREAM         = self.get_local_stream_name(tensorVar, True, idx),
            IS_LAST            = self.get_is_last_var(idx),
        )
        return result

    @classmethod
    def get_dequeue_func_rstream2atom(self, tensorVar, idx: int, lastcheck: str,out_dma_type: str = "float"):
        result = "dequeue_layer2atom<{ATOMIC_TYPE}, {OUTPUT_LAYER_ARR}, {SIZE}>({DES_STREAM}, {RAW_STREAM}, {IS_LAST_CHECK});".format(
            ATOMIC_TYPE       = out_dma_type,
            OUTPUT_LAYER_ARR  = tensorVar.type.name,
            SIZE              = str(tensorVar.size()),
            DES_STREAM        = self.get_io_port_name(tensorVar, False, idx),
            RAW_STREAM        = self.get_local_stream_name(tensorVar, False, idx),
            IS_LAST_CHECK      = lastcheck
        )
        return result

    @classmethod
    def get_dequeue_func_rstream2stream(self, tensorVar, idx: int, lastcheck: str):
        result = "dequeue_layer2layer><{OUTPUT_LAYER_STREAM}, {OUTPUT_LAYER_ARR}, {SIZE}>({DES_STREAM}, {RAW_STREAM}, {IS_LAST_CHECK})".format(
            OUTPUT_LAYER_STREAM = self.get_axi_wrapper_type(tensorVar),
            OUTPUT_LAYER_ARR    = tensorVar.type.name,
            SIZE                = str(tensorVar.size()),
            DES_STREAM          = self.get_io_port_name(tensorVar, False, idx),
            RAW_STREAM          = self.get_local_stream_name(tensorVar, False, idx),
            IS_LAST_CHECK       = lastcheck
        )
        return result


