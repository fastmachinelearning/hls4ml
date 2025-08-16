import os
from pathlib import Path
import stat



from shutil import copyfile

#######################################################
## file and directory #################################
#######################################################

def getGmemWrapperFileName(model):
    return f"{model.config.get_project_name()}_dm"

def getAxiWrapperFileName(model):
    return f"{model.config.get_project_name()}_axi"

def getMainWrapperFileName(model):
    return model.config.get_project_name()

def getMainFileName(model):
    return f"{model.config.get_project_name()}"

def getVitisUnifiedWorkingDirectoryDir(model):
    return os.path.join(model.config.get_output_dir(), "unifiedWorkspace")

def getVitisHlsDir(model):
    vitisWorkingDir = getVitisUnifiedWorkingDirectoryDir(model)
    return os.path.join(vitisWorkingDir, model.config.get_project_name())

def getVitisHlsExecDir(model):
    hlsDir = getVitisHlsDir(model)
    return os.path.join(hlsDir, "unifiedPrj")

def getVitisLinkerDir(model):
    vitisWorkingDir = getVitisUnifiedWorkingDirectoryDir(model)
    return os.path.join(vitisWorkingDir, "linker")

def getXOfileName(model):
    return f"{getGemTopFuncName(model)}.xo"

def getXOfilePath(model):
    return os.path.join(getVitisHlsExecDir(model), getXOfileName(model))

#######################################################
## naming of variable function helper #################
#######################################################

####### FOR GMEM WRAPPER

def getGmemIOPortName(tensorVar, isInput: bool, idx: int):
    ioDirect = "in" if isInput else "out"
    return f"gmem_{ioDirect}{str(idx)}_ptr_{tensorVar.name}"
def getGmemIOPortSizeName(tensorVar, isInput: bool, idx: int):
    ioDirect = "in" if isInput else "out"
    return f"gmem_{ioDirect}{str(idx)}_size_{tensorVar.name}"
def getGmemLocalStreamName(tensorVar, isInput: bool, idx: int):
    ioDirect = "in" if isInput else "out"
    return f"stream_{ioDirect}{str(idx)}_{tensorVar.name}"

def getDmaTypeName():
    return "dma_data_packet"

def getWrapperPortName(tensorVar, isInput: bool):
    ioStr = "in" if isInput else "out"
    return f"par_{ioStr}_{tensorVar.name}"

def getTopModelName(model):
    return f"{model.config.get_project_name()}"

def getGemTopFuncName(model):
    return f"{model.config.get_project_name()}_gem"

def getAxisTopFuncName(model):
    return f"{model.config.get_project_name()}_axi"

### it is renamed for stitch layer
def renameType(tensorVar, layerIdx:int, isInput: bool):
    return "result_" + tensorVar.type.name + f"_at_layer_{str(layerIdx)}"

def get_inputSizeArrName(model):
    return "N_IN_" + model.config.get_project_name()

def get_outputSizeArrName(model):
    return "N_OUT_" + model.config.get_project_name()

def get_axi_wrapper_type(tensorVar):
    return f"{tensorVar.type.name}_packet"

def get_axi_wrapper_dec(tensorVar):
    return f"typedef hls::axis<{tensorVar.type.name}, 0,0,0, AXIS_ENABLE_LAST> {get_axi_wrapper_type(tensorVar)};"


########################################################
## axi_wrapper.h & axi_wrapper.cpp  function helper ####
########################################################
##### variable
def getWrapperPortNameLocal(tensorVar, isInput: bool):
    ioStr = "in" if isInput else "out"
    return f"par_{ioStr}_{tensorVar.name}_local"

def getWrapperTmpName(tensorVar, isInput: bool):
    ioStr = "in" if isInput else "out"
    return f"par_{ioStr}_{tensorVar.name}_tmp"

def getWrapperIsLastCnt(idx):
    return f"isLastCnt_{str(idx)}"