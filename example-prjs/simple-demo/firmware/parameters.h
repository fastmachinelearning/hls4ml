#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_activation.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUTS 10
#define N_LAYER_1 32
#define N_OUTPUTS 1

typedef ap_fixed<18,10> accum_t;
typedef ap_fixed<18,8> weight_t;
typedef ap_fixed<18,8> bias_t;
typedef ap_fixed<18,8> input_t;
typedef ap_fixed<18,8> result_t;
//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<18,8> layer1_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::layer_config {
        static const unsigned n_in = N_INPUTS;
        static const unsigned n_out = N_LAYER_1;
        static const bool full_parallel = false;
        static const unsigned roll_factor_in = 1;
        static const unsigned roll_factor_out = 1;
        static const bool store_weights_in_bram = false;
        typedef ap_fixed<18,10> acc_t;
        typedef ap_fixed<18,8> bias_t;
        typedef ap_fixed<18,8> weight_t;
        };
struct relu_config : nnet::activ_config {
        static const unsigned n_in = N_LAYER_1;
        static const bool full_parallel = true;
        };
struct config2 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned n_out = N_OUTPUTS;
        static const bool full_parallel = false;
        static const unsigned roll_factor_in = 1;
        static const unsigned roll_factor_out = 1;
        static const bool store_weights_in_bram = false;
        typedef ap_fixed<18,10> acc_t;
        typedef ap_fixed<18,8> bias_t;
        typedef ap_fixed<18,8> weight_t;
        };
struct sigmoid_config : nnet::activ_config {
        static const unsigned n_in = N_OUTPUTS;
        static const unsigned table_size = 1024;
        static const bool full_parallel = true;
        };

#endif 
