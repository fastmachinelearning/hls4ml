#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_activation.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<18,8> accum_default_t;
typedef ap_fixed<18,8> weight_default_t;
typedef ap_fixed<18,8> bias_default_t;
typedef ap_fixed<18,8> input_t;
typedef ap_fixed<18,8> result_t;
#define Y_INPUTS 10
#define N_CHAN 1
#define Y_FILT 3


//hls-fpga-machine-learning insert layer-precision
//typedef ap_fixed<18,8> conv1_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::conv_config {
        static const unsigned y_in = Y_INPUTS;
        static const unsigned n_chan = N_CHAN;
        static const unsigned y_filt = Y_FILT;

        static const bool fully_unrolled = true;
        static const unsigned roll_factor_in = 1;
        static const unsigned roll_factor_out = 1;
        static const bool store_weights_in_bram = false;

        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
};

#endif 
