#include <iostream>

#include "myproject.h"
#include "parameters.h"

// hls-fpga-machine-learning insert namespace-start

void myproject(
    // hls-fpga-machine-learning insert header
) {

    // hls-fpga-machine-learning insert IO

#ifndef HLS4ML_EXTERNAL_WEIGHT_LOAD
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        loaded_weights = true;
    }
#endif
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers
}

// hls-fpga-machine-learning insert namespace-end
