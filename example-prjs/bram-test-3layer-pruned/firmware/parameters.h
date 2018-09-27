#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_sublayer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_activation.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<18,8> accum_default_t;
typedef ap_fixed<18,8> weight_default_t;
typedef ap_fixed<18,8> bias_default_t;
typedef ap_fixed<18,8> input_t;
typedef ap_fixed<18,8> result_t;
#define N_INPUTS 16
#define N_LAYER_1 64
#define N_LAYER_2 32
#define N_LAYER_3 32
#define N_OUTPUTS 5

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<18,8> layer1_t;
typedef ap_fixed<18,8> layer2_t;
typedef ap_fixed<18,8> layer3_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::layer_config {
        static const unsigned n_in = N_INPUTS;
        static const unsigned n_out = N_LAYER_1;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 16;
        static const unsigned n_zeros = 1021;
        static const bool store_weights_in_bram = true;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct relu_config1 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config2 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned n_out = N_LAYER_2;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 16;
        static const unsigned n_zeros = 2043;
        static const bool store_weights_in_bram = true;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct relu_config2 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_2;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config3 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_2;
        static const unsigned n_out = N_LAYER_3;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 16;
        static const unsigned n_zeros = 1009;
        static const bool store_weights_in_bram = true;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct relu_config3 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_3;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config4 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_3;
        static const unsigned n_out = N_OUTPUTS;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 16;
        static const unsigned n_zeros = 128;
        static const bool store_weights_in_bram = true;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct softmax_config4 : nnet::activ_config {
        static const unsigned n_in = N_OUTPUTS;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };

#endif 
