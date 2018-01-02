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
#define Y_INPUTS 32
#define N_CHAN 4
#define Y_FILT 5
#define N_FILT 3
#define STRIDE 1
#define PAD_LEFT 2
#define PAD_RIGHT 2
#define Y_OUTPUTS 32

//hls-fpga-machine-learning insert layer-precision
//typedef ap_fixed<18,8> conv1_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::conv_config {
	static const unsigned pad_left = PAD_LEFT;
	static const unsigned pad_right = PAD_RIGHT;
	static const unsigned y_in = Y_INPUTS;
	static const unsigned n_chan = N_CHAN;
	static const unsigned y_filt = Y_FILT;
	static const unsigned n_filt = N_FILT;
	static const unsigned stride = STRIDE;
	static const unsigned y_out = Y_OUTPUTS;

	static const unsigned reuse_factor = 1;
        static const bool store_weights_in_bram = false;

        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
};

#endif 
