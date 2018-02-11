#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_activation.h"
#include "nnet_recursive.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
#define N_LOOP 5
#define N_INPUTS 2
#define N_OUTPUTS 2
#define N_STATE 3

//hls-fpga-machine-learning insert layer-precision

//hls-fpga-machine-learning insert layer-config

struct config1 : nnet::rnn_config {
    typedef float state_t;
    typedef float U_t;  // State x Input
    typedef float W_t;  // State x State
    typedef float V_t;  // Output x State
    static const unsigned n_in = N_INPUTS;
    static const unsigned n_out = N_OUTPUTS;
    static const unsigned n_state = N_STATE;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};

#endif 
