import os
from pathlib import Path
import stat



from shutil import copyfile

#######################################################
## file and directory #################################
#######################################################

class MetaGen:

    def getGmemWrapperFileName(self, model):
        return f"{model.config.get_project_name()}_dm"

    def getAxiWrapperFileName(self, model):
        return f"{model.config.get_project_name()}_axi"

    def getMainWrapperFileName(self, model):
        return model.config.get_project_name()

    def getMainFileName(self, model):
        return f"{model.config.get_project_name()}"

    def getVitisUnifiedWorkingDirectoryDir(self, model):
        return os.path.join(model.config.get_output_dir(), "unifiedWorkspace")

    def getVitisHlsDir(self, model):
        vitisWorkingDir = self.getVitisUnifiedWorkingDirectoryDir(model)
        return os.path.join(vitisWorkingDir, model.config.get_project_name())

    def getVitisHlsExecDir(self, model):
        hlsDir = self.getVitisHlsDir(model)
        return os.path.join(hlsDir, "unifiedPrj")

    def getVitisLinkerDir(self, model):
        vitisWorkingDir = self.getVitisUnifiedWorkingDirectoryDir(model)
        return os.path.join(vitisWorkingDir, "linker")

    def getXOfileName(self, model):
        return f"{self.getGemTopFuncName(model)}.xo"

    def getXOfilePath(self, model):
        return os.path.join(self.getVitisHlsExecDir(model), self.getXOfileName(model))

    #######################################################
    ## naming of variable function helper #################
    #######################################################

    ####### FOR GMEM WRAPPER

    def getGmemIOPortName(self, tensorVar, isInput: bool, idx: int):
        ioDirect = "in" if isInput else "out"
        return f"gmem_{ioDirect}{str(idx)}_ptr_{tensorVar.name}"
    def getGmemIOPortSizeName(self, tensorVar, isInput: bool, idx: int):
        ioDirect = "in" if isInput else "out"
        return f"gmem_{ioDirect}{str(idx)}_size_{tensorVar.name}"
    def getGmemLocalStreamName(self, tensorVar, isInput: bool, idx: int):
        ioDirect = "in" if isInput else "out"
        return f"stream_{ioDirect}{str(idx)}_{tensorVar.name}"

    def getDmaTypeName(self):
        return "dma_data_packet"

    def getWrapperPortName(self, tensorVar, isInput: bool):
        ioStr = "in" if isInput else "out"
        return f"par_{ioStr}_{tensorVar.name}"

    def getTopModelName(self, model):
        return f"{model.config.get_project_name()}"

    def getGemTopFuncName(self, model):
        return f"{model.config.get_project_name()}_gem"

    def getAxisTopFuncName(self, model):
        return f"{model.config.get_project_name()}_axi"

    ### it is renamed for stitch layer
    def renameType(self, tensorVar, layerIdx:int, isInput: bool):
        return "result_" + tensorVar.type.name + f"_at_layer_{str(layerIdx)}"

    def get_inputSizeArrName(self, model):
        return "N_IN_" + model.config.get_project_name()

    def get_outputSizeArrName(self, model):
        return "N_OUT_" + model.config.get_project_name()

    def get_axi_wrapper_type(self, tensorVar):
        return f"{tensorVar.type.name}_packet"

    def get_axi_wrapper_dec(self, tensorVar):
        return f"typedef hls::axis<{tensorVar.type.name}, 0,0,0, AXIS_ENABLE_LAST> {self.get_axi_wrapper_type(tensorVar)};"


    ########################################################
    ## axi_wrapper.h & axi_wrapper.cpp  function helper ####
    ########################################################
    ##### variable
    def getWrapperPortNameLocal(self, tensorVar, isInput: bool):
        ioStr = "in" if isInput else "out"
        return f"par_{ioStr}_{tensorVar.name}_local"

    def getWrapperTmpName(self, tensorVar, isInput: bool):
        ioStr = "in" if isInput else "out"
        return f"par_{ioStr}_{tensorVar.name}_tmp"

    def getWrapperIsLastCnt(self, idx):
        return f"isLastCnt_{str(idx)}"