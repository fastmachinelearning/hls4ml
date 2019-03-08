#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_dense.h"
#include "nnet_conv.h"
#include "nnet_activation.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<32,8> accum_default_t;
typedef ap_fixed<32,8> weight_default_t;
typedef ap_fixed<32,8> bias_default_t;
typedef ap_fixed<32,8> input_t;
typedef ap_fixed<32,8> result_t;
#define Y_INPUTS 4
#define N_CHAN 6
#define Y_FILT 2
#define N_FILT 3
#define STRIDE 1
#define PAD_LEFT 0
#define PAD_RIGHT 1
#define Y_OUTPUTS 4
#define N_OUTPUTS 6

//hls-fpga-machine-learning insert layer-precision
//typedef ap_fixed<18,8> conv1_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::conv1d_config {
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
struct relu_config1 : nnet::activ_config {
    static const unsigned n_in = Y_OUTPUTS*N_FILT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config2 : nnet::dense_config {
    static const unsigned n_in = Y_OUTPUTS*N_FILT;
    static const unsigned n_out = N_OUTPUTS;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
};
struct softmax_config2 : nnet::activ_config {
    static const unsigned n_in = N_OUTPUTS;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

#endif 
