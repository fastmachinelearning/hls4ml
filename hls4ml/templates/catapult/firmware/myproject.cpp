#include <iostream>

#include "myproject.h"
#include "parameters.h"

#include <mc_scverify.h>

#pragma hls_design top
// hls-fpga-machine-learning insert IFSynPragmas
void CCS_BLOCK(myproject)(
    // hls-fpga-machine-learning insert header
) {

    // hls-fpga-machine-learning insert IO

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers
}
