#ifndef NNET_EINSUM_H_
#define NNET_EINSUM_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"

namespace nnet {

struct config_einsum {
    typedef void tpose_inp0_config;
    typedef void tpose_inp1_config;
    typedef void tpose_out_conf;

    // Layer Sizes
    static const unsigned n_free0;
    static const unsigned n_free1;
    static const unsigned n_contract;
    static const unsigned n_inplace;

    // Resource reuse info
    static const unsigned io_type;
    static const unsigned reuse_factor;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void einsum(const data0_T &data0, const data1_T &data1, res_T &res) {
    [[intel::fpga_register]] data0_T tpose_i0;
    [[intel::fpga_register]] data1_T tpose_i1;
    [[intel::fpga_register]] res_T tpose_o;

    nnet::transpose<data0_T, data0_T, typename CONFIG_T::tpose_inp0_config>(data0, tpose_i0);
    nnet::transpose<data1_T, data1_T, typename CONFIG_T::tpose_inp1_config>(data1, tpose_i1);

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    #pragma unroll
    for (unsigned i = 0; i < I; i++) {
        #pragma unroll
        for (unsigned l0 = 0; l0 < L0; l0++) {
            #pragma unroll
            for (unsigned l1 = 0; l1 < L1; l1++) {
                [[intel::fpga_register]] typename CONFIG_T::accum_t accum_buf = 0;
                #pragma unroll
                for (unsigned c = 0; c < C; c++) {
                    typename data0_T::value_type a = tpose_i0[(i * L0 + l0) * C + c];
                    typename data1_T::value_type b = tpose_i1[i * L1 * C + l1 * C + c];
                    accum_buf +=
                        CONFIG_T::template product<typename data0_T::value_type, typename data1_T::value_type>::product(a,
                                                                                                                        b);
                }
                tpose_o[(i * L0 + l0) * L1 + l1] = accum_buf;
            }
        }
    }

    nnet::transpose<res_T, res_T, typename CONFIG_T::tpose_out_conf>(tpose_o, res);
}
} // namespace nnet

#endif
