#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_stream.h"

// hls-fpga-machine-learning insert layer-config
struct config2 : nnet::dense_config {
    static constexpr unsigned n_in = 8;
    static constexpr unsigned n_out = 4;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned n_zeros = 0;
    static constexpr unsigned n_nonzeros = 32;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef b2_t bias_t;
    typedef w2_t weight_t;
    typedef layer2_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct relu_config4 : nnet::activ_config {
    static constexpr unsigned n_in = 4;
    static constexpr unsigned table_size = 1024;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 1;
    typedef relu1_table_t table_t;
};

struct config5 : nnet::dense_config {
    static constexpr unsigned n_in = 4;
    static constexpr unsigned n_out = 2;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned n_zeros = 0;
    static constexpr unsigned n_nonzeros = 8;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef b5_t bias_t;
    typedef w5_t weight_t;
    typedef layer5_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct relu_config7 : nnet::activ_config {
    static constexpr unsigned n_in = 2;
    static constexpr unsigned table_size = 1024;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 1;
    typedef relu2_table_t table_t;
};


#endif
