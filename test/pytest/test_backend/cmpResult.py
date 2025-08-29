import os
from pathlib import Path

import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model, load_model

import hls4ml
import hls4ml.model

test_root_path = Path(__file__).parent
os.environ['XILINX_VITIS'] = "/tools/Xilinx/Vitis/2023.2"
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']

def checkEqual(a, b):
    equal = np.array_equal(a, b)
    if equal:
        print("Test pass both are equal \U0001F642")
    else:
        print("Test Fail both are not equal \U0001F62C")



bridge_result = np.load(test_root_path / "output_file/outputGenbit.npy")
zcu_result    = np.load(test_root_path / "output_file/zcu102_mandriver.npy")
zcu_flat      = zcu_result.reshape(zcu_result.shape[0], -1)

print(bridge_result.shape)
print(zcu_result.shape)

checkEqual(bridge_result, zcu_flat)
