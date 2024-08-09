import json
import argparse
import warnings
data = {}
filelist = ["io_stream-Vitis-activation_function3.json","io_stream-Vitis-valid.json","Vitis-channels_last-valid-AveragePooling1D.json","io_parallel-Vitis-activation_function1.json","Vitis-channels_last-valid-MaxPooling1D.json","io_parallel-Vitis-same.json","io_parallel-Vitis-activation_function4.json","io_parallel-Vitis-activation_function0.json","Vitis-channels_last-valid-MaxPooling2D.json","Vitis-channels_last-same-MaxPooling1D.json","io_stream-Vitis-activation_function1.json","io_stream-Vitis-valid-channels_last.json","Vitis-channels_last-same-AveragePooling1D.json","io_parallel-Vitis-valid-channels_last.json","io_stream-Vitis-same.json","io_stream-Vitis.json","io_parallel-Vivado-activation_function2.json","io_parallel-Vitis.json","io_parallel-Vitis-same-channels_last.json","io_parallel-Vitis-valid.json","Vitis-channels_last-valid-AveragePooling2D.json","io_stream-Vitis-same-channels_last.json","io_stream-Vitis-activation_function4.json","io_parallel-Vitis-activation_function2.json","io_parallel-Vitis-activation_function3.json","io_stream-Vitis.json","io_stream-Vitis-activation_function0.json","io_stream-Vitis-activation_function2.json","Vitis-channels_last-same-MaxPooling2D.json","Vitis-channels_last-same-AveragePooling2D.json","io_stream-Vitis.json"]
def main():
    parser = argparse.ArgumentParser()
    parser.add_arguement("filepath",type=str)
    args = parser.parse_args()
    for filename in filelist:
        data = 'results/' + filename
        with open(args.filepath + filename, "w") as fp:
            baseline = json.dump(data,fp)
        if data == baseline:
            return True
        else: 
            warnings.warn("Results don't match baseline")

