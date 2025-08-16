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

#######################################################
## naming of variable function helper #################
#######################################################

def getGmemTypeName(atomicType: str):
    if atomicType not in  ["float", "double"]:
        raise Exception(f"Unsupported atomic type {atomicType}")
    return f"{atomicType}*"

def getGmemPortName(isInput: bool):
    return "in" if isInput else "out"

def getDmaTypeName():
    return "dma_data_packet"

def getWrapperPortName(tensorVar, isInput: bool):
    ioStr = "in" if isInput else "out"
    return f"par_{ioStr}_{tensorVar.name}"

def getTopModelName(model):
    return f"{model.config.get_project_name()}_axi"

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