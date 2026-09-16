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

// weights are already transposed during compile-time in the config
template <class data_pipe, class res_pipe, typename CONFIG_T> void einsum_dense_stream() {

    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned HEAD_DIM_IN = static_cast<unsigned>(C / CONFIG_T::n_head);
    constexpr unsigned HEAD_DIM_OUT = static_cast<unsigned>(L1 / CONFIG_T::n_head);

    using Dense_in_T = typename ExtractPipeType<data_pipe>::value_type;
    using Dense_out_T = typename ExtractPipeType<res_pipe>::value_type;
    using Dense_in_data_T = typename Dense_in_T::value_type;
    using Dense_concat_T = nnet::array<Dense_in_data_T, C>;

    [[intel::fpga_register]] Dense_in_T dense_in;
    [[intel::fpga_register]] Dense_out_T dense_out;
    [[intel::fpga_register]] Dense_concat_T dense_in_concat;
    //[[intel::fpga_register]] Dense_heads_T dense_out_head;

    for (unsigned l0 = 0; l0 < L0; l0++) {

        #pragma unroll 4
        for (unsigned i = 0; i < I; i++) {

            if constexpr (!CONFIG_T::opt_dense) {
                dense_in = data_pipe::read(); // 1xC read
            }

            // The reason why we collect heads is due to structured agreement between the
            // causal einsum module and dense layers where we stream data-first to the pipes
            // so each write by causal_einsum is like data0_head0, data0_head1,...
            if constexpr (CONFIG_T::opt_dense) {
                for (unsigned h = 0; h < CONFIG_T::n_head; h++) {
                    dense_in = data_pipe::read();
                    #pragma unroll
                    for (unsigned c = 0; c < HEAD_DIM_IN; c++) {
                        dense_in_concat[HEAD_DIM_IN * h + c] = dense_in[c];
                    }
                }
            }

            // Call the dense_resource function with the reordered weights
            if constexpr (!CONFIG_T::opt_dense) {
                for (unsigned h = 0; h < CONFIG_T::n_head; h++) {
                    nnet::dense_resource<Dense_in_T, Dense_out_T, typename CONFIG_T::dense_conf>(dense_in, dense_out,
                                                                                                 HEAD_DIM_OUT * h);
                    res_pipe::write(dense_out);
                }
            } else {
                nnet::dense_resource<Dense_concat_T, Dense_out_T, typename CONFIG_T::dense_conf>(dense_in_concat, dense_out);
                res_pipe::write(dense_out);
            }
        }
    }
}

} // namespace nnet

#endif
