#ifndef NNET_EINSUM_DENSE_H_
#define NNET_EINSUM_DENSE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"

namespace nnet {

struct einsum_dense_config {
    // Internal data type definitions
    typedef void tpose_inp_conf;
    typedef void tpose_out_conf;
    typedef void dense_conf;

    // Layer Sizes
    static const unsigned n_free_data = 1;
    static const unsigned n_free_kernel = 1;
    static const unsigned n_contract = 1;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1000;

    // Product function to use
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void einsum_dense(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
                  const typename CONFIG_T::bias_t &biases) {
    [[intel::fpga_register]] data_T inp_tpose;
    [[intel::fpga_register]] res_T out_tpose;

    nnet::transpose<data_T, data_T, typename CONFIG_T::tpose_inp_conf>(data, inp_tpose);

    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    using Dense_in_T = nnet::array<typename data_T::value_type, C>;
    using Dense_out_T = nnet::array<typename res_T::value_type, L1>;
    using Dense_weights_T = nnet::array<typename CONFIG_T::weight_t::value_type, L1 * C>;
    using Dense_biases_T = nnet::array<typename CONFIG_T::bias_t::value_type, L1>;

    #pragma unroll CONFIG_T::parallelization_factor
    for (unsigned l0 = 0; l0 < L0; l0++) {
        #pragma unroll
        for (unsigned i = 0; i < I; i++) {
            [[intel::fpga_register]] Dense_in_T dense_in;
            [[intel::fpga_register]] Dense_out_T dense_out;
            [[intel::fpga_register]] Dense_weights_T dense_weights;
            [[intel::fpga_register]] Dense_biases_T dense_biases;

            #pragma unroll
            for (unsigned c_idx = 0; c_idx < C; c_idx++) {
                dense_in[c_idx] = inp_tpose[(i * L0 + l0) * C + c_idx];
            }

            // Reorder weights from column-major (source) to row-major (destination) during copy
            const unsigned weights_offset = i * L1 * C;
            #pragma unroll
            for (unsigned j = 0; j < L1; j++) {
                #pragma unroll
                for (unsigned k = 0; k < C; k++) {
                    dense_weights[j * C + k] = weights[weights_offset + (k * L1 + j)];
                }
            }

            #pragma unroll
            for (unsigned b_idx = 0; b_idx < L1; b_idx++) {
                dense_biases[b_idx] = biases[((i * L0 + l0) * L1) + b_idx];
            }

            // Create a temporary config to ensure the types of the local buffers
            // match what dense_resource expects for its weight_t and bias_t.
            struct dense_slice_config : CONFIG_T::dense_conf {
                using weight_t = Dense_weights_T;
                using bias_t = Dense_biases_T;
            };

            // Call the dense_resource function with the reordered weights
            nnet::dense_resource<Dense_in_T, Dense_out_T, dense_slice_config>(dense_in, dense_out, dense_weights,
                                                                              dense_biases);

            #pragma unroll
            for (unsigned j = 0; j < L1; j++) {
                out_tpose[((i * L0 + l0) * L1) + j] = dense_out[j];
            }
        }
    }

    nnet::transpose<res_T, res_T, typename CONFIG_T::tpose_out_conf>(out_tpose, res);
}

} // namespace nnet

#endif
