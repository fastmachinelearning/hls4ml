#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_conv1d_stream.h"
#include "nnet_utils/nnet_recurrent.h"
#include "nnet_utils/nnet_recurrent_stream.h"

// hls-fpga-machine-learning insert layer-config
struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 9;
    static const unsigned n_out = 16;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef b2_t bias_t;
    typedef w2_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2 : nnet::conv1d_config {
    static const unsigned in_width = 32;
    static const unsigned n_chan = 3;

    static const unsigned filt_width = 3;
    static const unsigned impl_filt_width = 3;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 16;
    static const unsigned out_width = 32;

    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef model_default_t accum_t;
    typedef b2_t bias_t;
    typedef w2_t weight_t;
    typedef config2_mult mult_config;
};

struct relu_config4 : nnet::activ_config {
    static constexpr unsigned n_in = 512;
    static constexpr unsigned table_size = 1024;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef relu1_table_t table_t;
};

struct config5_x_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16 * 3;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;
    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;
    typedef model_default_t accum_t;
    typedef b5_t bias_t;
    typedef w5_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config5_h_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16 * 3;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;
    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;
    typedef model_default_t accum_t;
    typedef br5_t bias_t;
    typedef wr5_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct sigmoid_config5_rec_act : nnet::activ_config {
    static const unsigned n_in = 16 * 2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef gru_table_t table_t;
};

struct tanh_config5_act : nnet::activ_config {
    static const unsigned n_in = 16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef gru_table_t table_t;
};

struct config5 : nnet::gru_config {
    static const unsigned n_in  = 16;
    static const unsigned n_out = 16;
    static const unsigned n_units = 16;
    static const unsigned n_timesteps = 32;
    static const unsigned n_outputs = 1;
    static const bool return_sequences = false;

    typedef model_default_t accum_t;
    typedef w5_t weight_t;
    typedef b5_t bias_t;
    typedef wr5_t recurrent_weight_t;
    typedef br5_t recurrent_bias_t;

    typedef config5_x_mult mult_config_x;
    typedef config5_h_mult mult_config_h;

    typedef tanh_config5_act ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::tanh<x_T, y_T, config_T>;

    typedef sigmoid_config5_rec_act ACT_CONFIG_RECURRENT_T;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::sigmoid<x_T, y_T, config_T>;

    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};


#endif
