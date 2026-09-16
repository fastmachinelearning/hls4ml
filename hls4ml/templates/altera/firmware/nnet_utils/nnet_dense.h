#ifndef NNET_DENSE_LARGE_H_
#define NNET_DENSE_LARGE_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include <cstdint>

namespace nnet {

struct dense_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    static const unsigned reuse_factor = 1;
    static constexpr unsigned num_banks = 1;
    static const unsigned block_factor = 1;      // DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    static const unsigned multiplier_limit = 1;  // DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor)
    static const unsigned multiplier_factor = 1; // min n_in, rf
    static const unsigned multiplier_scale = 1;  // M_LIMIT/CONFIG_T::n_out;
    static const unsigned reciprocal = 1;        // 2^35 / 25
    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?

    // Default multiplication
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void dense_rf_gt(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
                 const typename CONFIG_T::bias_t &biases) {

    assert((CONFIG_T::multiplier_limit % CONFIG_T::n_out == 0 || CONFIG_T::reuse_factor >= CONFIG_T::n_in) &&
           "The current Reuse Factor is not allowed");

    assert((CONFIG_T::reuse_factor > CONFIG_T::n_in) && "This function is correct only for RF > N_IN");

    //#pragma ii CONFIG_T::reuse_factor
    [[intel::fpga_register]] typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
Load:
    #pragma unroll
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }
    [[intel::fpga_register]] int out_index[CONFIG_T::reuse_factor][CONFIG_T::block_factor];
    [[intel::fpga_register]] int d_index[CONFIG_T::reuse_factor][CONFIG_T::block_factor];

    #pragma unroll
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        #pragma unroll
        for (int im = 0; im < CONFIG_T::block_factor; im++) {
            uint32_t w_index = ir + CONFIG_T::reuse_factor * im;
            out_index[ir][im] = (w_index / CONFIG_T::multiplier_factor);
            d_index[ir][im] =
                (w_index >= CONFIG_T::n_in) ? (w_index - CONFIG_T::n_in) : w_index; // FPGA does not like modulo
        }
    }
Product1:
    [[intel::nofusion, intel::speculated_iterations(0)]] for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        [[intel::fpga_register]] typename CONFIG_T::accum_t tmp_acc[CONFIG_T::block_factor];
    Product2:
        #pragma unroll
        for (int im = 0; im < CONFIG_T::block_factor; im++) {
            uint32_t w_index = ir + (CONFIG_T::reuse_factor_rounded)*im;
            if (w_index >= CONFIG_T::reuse_factor_rounded * CONFIG_T::block_factor_rounded)
                continue;
            int data_index = d_index[ir][im];
            // Modified this
            tmp_acc[im] =
                CONFIG_T::template product<typename data_T::value_type, typename CONFIG_T::weight_t::value_type>::product(
                    data[data_index], weights[w_index]);
        }
        [[intel::fpga_register]] typename CONFIG_T::accum_t mult[CONFIG_T::multiplier_limit];
    ResetMult:
        #pragma unroll
        for (int imult = 0; imult < CONFIG_T::multiplier_limit; imult++) {
            mult[imult] = 0;
        }
    AccumLoop1:
        #pragma unroll
        for (int im = 0; im < CONFIG_T::block_factor; im++) {
            int o_index = out_index[ir][im];
            if (o_index >= CONFIG_T::n_out)
                continue; // check out of bounds
            mult[o_index] += tmp_acc[im];
        }
    AccumLoop2:
        #pragma unroll
        for (int im = 0; im < CONFIG_T::multiplier_limit; im++) {
            acc[im] += mult[im];
        }
    }
Store:
    #pragma unroll
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        res[ires] = cast<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(acc[ires]); // acc[jj];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_rf_lt(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
                 const typename CONFIG_T::bias_t &biases, unsigned head_offset = 0) {

    assert((CONFIG_T::multiplier_limit % CONFIG_T::n_out == 0 || CONFIG_T::reuse_factor >= CONFIG_T::n_in) &&
           "The current Reuse Factor is not allowed");

    assert((CONFIG_T::multiplier_limit == CONFIG_T::block_factor) && "This function is correct only for RF <= N_IN");

    constexpr unsigned N_LANES = CONFIG_T::num_lanes;                   // ceil(n_in/reuse_factor)
    constexpr unsigned N_PASSES = DIV_ROUNDUP(CONFIG_T::n_in, N_LANES); // <= reuse_factor (prevents dropping values)
    constexpr unsigned LAST_LANE_ADDR = N_LANES * (N_PASSES - 1);
    constexpr unsigned OVERFLOW_ADDRS = CONFIG_T::n_in - LAST_LANE_ADDR;

    if constexpr (CONFIG_T::argmax) {

        static_assert(std::tuple_size<res_T>{} == 1, "argmax must return a size 1 array");
        typename CONFIG_T::accum_t maxval = minval<typename CONFIG_T::accum_t>();
        unsigned idx = 0;

        [[intel::fpga_memory]] typename CONFIG_T::accum_t acc[(N_PASSES > 1) ? CONFIG_T::n_out : 1];
        Op_add<typename CONFIG_T::accum_t> op_add;
        unsigned w_base = CONFIG_T::n_in * head_offset;

        [[intel::nofusion, intel::speculated_iterations(0)]] // each reuse loop is seperate
        for (unsigned reuse_unit = 0; reuse_unit < N_PASSES; reuse_unit++) {

            unsigned data_offset = N_LANES * reuse_unit;
            unsigned w_offset = w_base + data_offset;
            bool last = (reuse_unit == N_PASSES - 1);

            for (unsigned el = 0; el < CONFIG_T::n_out; el++) {

                [[intel::fpga_register]] typename CONFIG_T::accum_t prod[N_LANES];

                #pragma unroll
                for (unsigned i = 0; i < N_LANES; i++) {

                    bool calculate = (i < OVERFLOW_ADDRS) || (!last);

                    prod[i] = calculate ? CONFIG_T::template product<
                                              typename data_T::value_type,
                                              typename CONFIG_T::weight_t::value_type>::product(data[data_offset + i],
                                                                                                weights[w_offset + i])
                                        : 0;
                }

                auto partial = reduce<typename CONFIG_T::accum_t, N_LANES, Op_add<typename CONFIG_T::accum_t>>(prod, op_add);

                typename CONFIG_T::accum_t total;
                if constexpr (N_PASSES == 1) {
                    total = (typename CONFIG_T::accum_t)(partial + biases[el + head_offset]);
                } else {
                    total = (reuse_unit == 0) ? (typename CONFIG_T::accum_t)(partial + biases[el + head_offset])
                                              : (typename CONFIG_T::accum_t)(partial + acc[el]);
                    if (!last)
                        acc[el] = total;
                }

                if (last && total > maxval) {
                    maxval = total;
                    idx = el;
                }
                w_offset += CONFIG_T::n_in;
            }
        }
        res[0] = static_cast<typename res_T::value_type>(idx);
    } else {

        constexpr unsigned N_OUT = std::tuple_size<res_T>{};
        [[intel::fpga_memory]] typename CONFIG_T::accum_t acc[(N_PASSES > 1) ? N_OUT : 1];
        Op_add<typename CONFIG_T::accum_t> op_add;
        unsigned w_base = CONFIG_T::n_in * head_offset;

        [[intel::nofusion, intel::speculated_iterations(0)]] // each reuse loop is seperate
        for (unsigned reuse_unit = 0; reuse_unit < N_PASSES; reuse_unit++) {

            unsigned data_offset = N_LANES * reuse_unit;
            unsigned w_offset = w_base + data_offset;
            bool last = (reuse_unit == N_PASSES - 1);

            for (unsigned el = 0; el < N_OUT; el++) {

                [[intel::fpga_register]] typename CONFIG_T::accum_t prod[N_LANES];

                #pragma unroll
                for (unsigned i = 0; i < N_LANES; i++) {

                    bool calculate = (i < OVERFLOW_ADDRS) || (!last);

                    prod[i] = calculate ? CONFIG_T::template product<
                                              typename data_T::value_type,
                                              typename CONFIG_T::weight_t::value_type>::product(data[data_offset + i],
                                                                                                weights[w_offset + i])
                                        : 0;
                }

                auto partial = reduce<typename CONFIG_T::accum_t, N_LANES, Op_add<typename CONFIG_T::accum_t>>(prod, op_add);

                typename CONFIG_T::accum_t total;
                if constexpr (N_PASSES == 1) {
                    total = (typename CONFIG_T::accum_t)(partial + biases[el + head_offset]);
                } else {
                    total = (reuse_unit == 0) ? (typename CONFIG_T::accum_t)(partial + biases[el + head_offset])
                                              : (typename CONFIG_T::accum_t)(partial + acc[el]);
                    if (!last)
                        acc[el] = total;
                }

                if (last)
                    res[el] = cast<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(total);
                w_offset += CONFIG_T::n_in;
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource(const data_T &data, res_T &res, unsigned kernel_offset = 0) {
    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        dense_rf_lt<data_T, res_T, CONFIG_T>(data, res, CONFIG_T::weights, CONFIG_T::biases, kernel_offset);
    } else {
        dense_rf_gt<data_T, res_T, CONFIG_T>(data, res, CONFIG_T::weights, CONFIG_T::biases);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
                    const typename CONFIG_T::bias_t &biases, unsigned kernel_offset = 0) {
    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        dense_rf_lt<data_T, res_T, CONFIG_T>(data, res, weights, biases, kernel_offset);
    } else {
        dense_rf_gt<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}
} // namespace nnet
#endif
