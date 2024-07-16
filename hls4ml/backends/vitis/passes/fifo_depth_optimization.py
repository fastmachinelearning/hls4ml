import json
import os
from hls4ml.model.optimizer.optimizer import (
    ConfigurableOptimizerPass,
    ModelOptimizerPass,
)

def set_big_fifos(model, profiling_fifo_depth):
    # initialize all the fifos to `profiling_fifo_depth` so that they will be automatically implemented in BRAMs and so they will be profiled
    # alternatively, "config_dataflow -override_user_fifo_depth profiling_fifo_depth" can be used inside build_prj.tcl to override all FIFO depths with the specified value 
    if profiling_fifo_depth:
        vars_to_profile = {
            k: v
            for k, v in model.output_vars.items()
            if v != model.get_output_variables()[0] and v != model.get_input_variables()[0]
        }
        for v in vars_to_profile.values():
            if v.pragma:
                v.pragma = (v.pragma[0], profiling_fifo_depth)

def execute_cosim_to_profile_fifos(model):
    model.write()
    model.build(
        reset=False,
        csim=True,
        synth=True,
        cosim=True,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=True,
    )

def get_vitis_optimized_fifo_depths(model):
    
    # channel.zip is generated after the cosimulation and contains the chan_status*.csv files
    # in the chan_status*.csv files the max depth achieved during cosimulation can be found at the last (4th) line
    path_to_zip_file = (
        model.config.get_output_dir()
        + "/"
        + model.config.get_project_name()
        + "_prj"
        + "/solution1/.autopilot/db/channel_depth_info/"
    )
    os.system(f"unzip -q {path_to_zip_file}channel.zip -d {path_to_zip_file}")
    
    names_file_path = (
        model.config.get_output_dir()
        + "/"
        + model.config.get_project_name()
        + "_prj"
        + "/solution1/.autopilot/db/channel_info.csv"
    )

    # the channel_info.csv file contains the mapping of the fifo names with the respective chan_status*.csv file
    csv_fifo_depth_files = {}
    with open(names_file_path) as names_file:
        for line in names_file:
            # if "layer" in line:
            layer_name = line.split(",")[1]
            csv_file_name = line.split(",")[3][:-1]
            csv_fifo_depth_files[layer_name] = csv_file_name
    
    optmized_fifo_depths = {}
    for layer_name, file_name in csv_fifo_depth_files.items():
        with open(path_to_zip_file+file_name) as chan_status_file:
            lines = chan_status_file.readlines()
            optmized_fifo_depths[layer_name] = int(lines[-1])
            
    return optmized_fifo_depths
        
def generate_max_depth_file(model, maxs):
    with open(model.config.get_output_dir() + "/max_depth.json", "w") as f:
        json.dump(maxs, f, indent=4)

def set_optimized_fifo_depths(model, optmized_fifo_depths):
    for v in model.output_vars.values():
        if v.pragma:
            if v.name in optmized_fifo_depths.keys():
                filtered_depth = optmized_fifo_depths[v.name]
            else:
                continue
            # if len(filtered_max) == 0:
            #     continue
            # if len(filtered_max) > 1:
            #     print("WARNING! Check names of FIFOs")
            v.pragma = (v.pragma[0], filtered_depth)

class FifoDepthOptimization(ConfigurableOptimizerPass, ModelOptimizerPass):
    def __init__(self):
        self.profiled_fifo_data = []

    def transform(self, model):
        # use `large_fifo_depth = 0` to keep the default fifo depth
        profiling_fifo_depth = getattr(
            self, "profiling_fifo_depth", 100_000
        )  # consider changing 100_000 either with a very very large value > of any total bram storage space or via vitis 2023.2 c-simulation

        # check axi-stream or io-stream
        if not (model.config.get_config_value("IOType") == "io_stream"):
            raise RuntimeError(
                "To use this optimization you have to set `IOType` field to `io_stream` in the HLS config"
            )

        set_big_fifos(model, profiling_fifo_depth)

        execute_cosim_to_profile_fifos(model)
        optmized_fifo_depths = get_vitis_optimized_fifo_depths(model)

        # maxs = [
        #     {"name": i["name"], "max": i["max"], "depth": i["depth"]}
        #     for i in self.values
        # ]

        # generate_max_depth_file(model, optmized_fifo_depths)

        set_optimized_fifo_depths(model, optmized_fifo_depths)

        print("[hls4ml] - FIFO optimization completed")
        return False
