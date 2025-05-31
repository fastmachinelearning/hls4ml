#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "nnet_common.h"
#include "nnet_types.h"

namespace nnet {

// *************************************************
//       Linear Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> [[intel::use_stall_enable_clusters]] void linear_stream() {
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;
LinearActLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            auto in_data = data_pipe::read();

        LinearPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                out_data.data[j] = in_data.data[j];
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       ReLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> [[intel::use_stall_enable_clusters]] void relu_stream() {
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;
ReLUActLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            [[intel::fpga_register]] auto in_data = data_pipe::read();
        ReLUPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                if (in_data.data[j] > 0)
                    out_data.data[j] = in_data.data[j];
                else
                    out_data.data[j] = 0;
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void leaky_relu_stream(typename CONFIG_T::param_t alpha) {
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(std::tuple_size<DataT>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<DataT>{} / multiplier_limit;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;
LeakyReLUActLoop:
    [[intel::initiation_interval(pipeline)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            auto in_data = data_pipe::read();

        LeakyReLUPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                if (in_data.data[j] > 0)
                    out_data.data[j] = in_data.data[j];
                else
                    out_data.data[j] = alpha * in_data.data[j];
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void thresholded_relu_stream(typename CONFIG_T::param_t theta) {
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

ThresholdedReLUActLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            auto in_data = data_pipe::read();

        ThresholdedReLUPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                if (in_data.data[j] > theta)
                    out_data.data[j] = in_data.data[j];
                else
                    out_data.data[j] = 0;
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       ELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void elu_stream(typename CONFIG_T::param_t alpha) {
#include "activation_tables/elu_table.tb"
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(std::tuple_size<DataT>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<DataT>{} / multiplier_limit;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;
EluActLoop:
    [[intel::initiation_interval(pipeline)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            auto in_data = data_pipe::read();

        EluPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                [[intel::fpga_register]] auto datareg = in_data.data[j];
                if (datareg >= 0) {
                    out_data.data[j] = datareg;
                } else {
                    int index = (datareg * CONFIG_T::table_size / -8).to_int();
                    if (index > CONFIG_T::table_size - 1)
                        index = CONFIG_T::table_size - 1;
                    out_data.data[j] = alpha * elu_table[index];
                }
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       SeLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> [[intel::use_stall_enable_clusters]] void selu_stream() {
#include "activation_tables/selu_table.tb"
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

    constexpr ac_fixed<16, 1, false, AC_RND> scale = 1.0507009873554804934193349852946;

SeluActLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            auto in_data = data_pipe::read();

        SeluPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                [[intel::fpga_register]] auto datareg = in_data.data[j];

                if (datareg >= 0) {
                    out_data.data[j] = scale * datareg;
                } else {
                    int index = (datareg * CONFIG_T::table_size / -8).to_int();
                    if (index > CONFIG_T::table_size - 1)
                        index = CONFIG_T::table_size - 1;
                    out_data.data[j] = selu_table[index];
                }
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void prelu_stream(typename CONFIG_T::param_t alpha) {
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(std::tuple_size<DataT>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<DataT>{} / multiplier_limit;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;
PReLUActLoop:
    [[intel::initiation_interval(pipeline)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            auto in_data = data_pipe::read();

        PReLUPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                if (in_data.data[j] > 0)
                    out_data.data[j] = in_data.data[j];
                else
                    out_data.data[j] = alpha[i * std::tuple_size<ResT>{} + j] * in_data.data[j];
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> [[intel::use_stall_enable_clusters]] void softplus_stream() {
#include "activation_tables/softplus_table.tb"

    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

SoftplusActLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            auto in_data = data_pipe::read();

        SoftplusPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                [[intel::fpga_register]] int data_round = (in_data.data[j] * CONFIG_T::table_size / 16).to_int();
                [[intel::fpga_register]] int index = data_round + 8 * CONFIG_T::table_size / 16;
                if (index < 0)
                    index = 0;
                else if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data.data[j] = softplus_table[index];
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> [[intel::use_stall_enable_clusters]] void softsign_stream() {
#include "activation_tables/softsign_table.tb"
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    constexpr int MAX_VALUE = 8;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;
SoftsignActLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            auto in_data = data_pipe::read();

        SoftsignPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                [[intel::fpga_register]] typename DataT::value_type absValue;
                ;
                if (in_data.data[j] < 0) {
                    absValue = -in_data.data[j];
                } else {
                    absValue = in_data.data[j];
                }
                ac_int<16> index = (absValue * CONFIG_T::table_size / MAX_VALUE).to_int();
                if (absValue > MAX_VALUE)
                    index = CONFIG_T::table_size - 1;
                if (in_data.data[j] < 0) {
                    out_data.data[j] = -softsign_table[index];
                } else {
                    out_data.data[j] = softsign_table[index];
                }
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void softmax_stable_stream() {
#include "activation_tables/exp_table.tb"
#include "activation_tables/invert_table.tb"
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(std::tuple_size<DataT>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<DataT>{} / multiplier_limit;

    [[intel::fpga_register]] typename DataT::value_type data_array[std::tuple_size<DataT>{}];

    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

SoftmaxArrayLoop:
    [[intel::initiation_interval(pipeline)]] while (keep_going) {
        for (unsigned i = 0; i < CONFIG_T::n_in / std::tuple_size<DataT>{}; i++) {
            auto in_data = data_pipe::read();

        SoftmaxArrayPackLoop:
            #pragma unroll
            for (unsigned j = 0; j < std::tuple_size<DataT>{}; j++) {
                data_array[j] = in_data.data[j];
            }

            // Find the max and compute all delta(x_i, x_max)
            Op_max<typename DataT::value_type> op_max;
            [[intel::fpga_register]] auto x_max =
                reduce<typename DataT::value_type, std::tuple_size<DataT>{}, Op_max<typename DataT::value_type>>(data_array,
                                                                                                                 op_max);

            // For the diffs, use the same type as the input but force rounding and saturation
            [[intel::fpga_register]] ac_fixed<ExtractPipeType<data_pipe>::value_type::value_type::width,
                                              ExtractPipeType<data_pipe>::value_type::value_type::i_width, true, AC_RND,
                                              AC_SAT>
                d_xi_xmax[std::tuple_size<DataT>{}];
            #pragma unroll
            for (unsigned j = 0; j < std::tuple_size<DataT>{}; j++) {
                d_xi_xmax[j] = data_array[j] - x_max;
            }

            // Calculate all the e^x's
            [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_res[std::tuple_size<DataT>{}];
            #pragma unroll
            for (unsigned j = 0; j < std::tuple_size<DataT>{}; j++) {
                exp_res[j] = exp_table[softmax_stable_idx_from_real_val<DataT::value_type, CONFIG_T>(d_xi_xmax[j])];
            }

            // Explicitly sum the results with an adder tree.
            // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
            Op_add<typename CONFIG_T::exp_table_t> op_add;
            [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_sum =
                reduce<typename CONFIG_T::exp_table_t, std::tuple_size<DataT>{}, Op_add<typename CONFIG_T::exp_table_t>>(
                    exp_res, op_add);

            [[intel::fpga_register]] typename CONFIG_T::inv_table_t inv_exp_sum =
                invert_table[softmax_stable_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];

        SoftmaxInvPackLoop:
            #pragma unroll
            for (unsigned j = 0; j < std::tuple_size<ResT>{}; j++) {

                // TODO - Find Quartus-equivalent pragma
                // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

                out_data.data[j] = exp_res[j] * inv_exp_sum;
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void softmax_latency_stream() {
#include "activation_tables/exp_table_latency.tb"
#include "activation_tables/invert_table_latency.tb"
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(std::tuple_size<DataT>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<DataT>{} / multiplier_limit;

    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

    // Calculate all the e^x's
    [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_res[std::tuple_size<DataT>{}];

SoftmaxExpLoop:
    [[intel::initiation_interval(pipeline)]] while (keep_going) {
        for (unsigned i = 0; i < CONFIG_T::n_in / std::tuple_size<DataT>{}; i++) {
            auto in_data = data_pipe::read();

        SoftmaxExpPackLoop:
            #pragma unroll
            for (unsigned j = 0; j < std::tuple_size<DataT>{}; j++) {
                exp_res[j] =
                    exp_table_latency[softmax_latency_idx_from_real_val<DataT::value_type, CONFIG_T>(in_data.data[j])];
            }

            // Explicitly sum the results with an adder tree.
            // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
            Op_add<typename CONFIG_T::exp_table_t> op_add;
            [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_sum =
                reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res,
                                                                                                               op_add);

            // Multiply previously calculated exponetials with the reciprocal of the sum
            [[intel::fpga_register]] typename CONFIG_T::inv_table_t inv_exp_sum =
                invert_table_latency[softmax_latency_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];

        SoftmaxInvPackLoop:
            #pragma unroll
            for (unsigned j = 0; j < std::tuple_size<ResT>{}; j++) {
                // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
                out_data.data[j] = exp_res[j] * inv_exp_sum;
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void softmax_legacy_stream() {
#include "activation_tables/exp_table_legacy.tb"
#include "activation_tables/invert_table_legacy.tb"
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

    // Index into the lookup table based on data for exponentials
    [[intel::fpga_register]] typename CONFIG_T::table_t exp_res[std::tuple_size<DataT>{}];
    [[intel::fpga_register]] typename CONFIG_T::table_t exp_diff_res;
    [[intel::fpga_register]] typename DataT::value_type data_cache[std::tuple_size<DataT>{}];

SoftmaxInitLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (unsigned s = 0; s < CONFIG_T::n_in / std::tuple_size<DataT>{}; s++) {
            auto in_data = data_pipe::read();

        SoftmaxInitPackLoop:
            #pragma unroll
            for (unsigned j = 0; j < std::tuple_size<DataT>{}; j++) {
                data_cache[j] = in_data.data[j];
                exp_res[j] = 0;
            }

        SoftmaxExpLoop:
            #pragma unroll
            for (int i = 0; i < std::tuple_size<DataT>{}; i++) {
            SoftmaxExpInner:
                #pragma unroll
                for (int j = 0; j < std::tuple_size<DataT>{}; j++) {
                    if (i == j) {
                        exp_diff_res = 1;
                    } else {
                        int data_round = ((data_cache[j] - data_cache[i]) * CONFIG_T::table_size / 16).to_int();
                        int index = data_round + 8 * CONFIG_T::table_size / 16;
                        if (index < 0)
                            index = 0;
                        if (index > CONFIG_T::table_size - 1)
                            index = CONFIG_T::table_size - 1;
                        exp_diff_res = exp_table_legacy[index];
                    }
                    exp_res[i] += exp_diff_res;
                }
            }

        SoftmaxInvPackLoop:
            #pragma unroll
            for (unsigned j = 0; j < std::tuple_size<ResT>{}; j++) {
                int exp_res_index = (exp_res[j] * CONFIG_T::table_size / 64).to_int();
                if (exp_res_index < 0)
                    exp_res_index = 0;
                if (exp_res_index > CONFIG_T::table_size - 1)
                    exp_res_index = CONFIG_T::table_size - 1;
                out_data.data[j] = static_cast<typename ResT::value_type>(invert_table_legacy[exp_res_index]);
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void softmax_argmax_stream() {
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            auto in_data = data_pipe::read();

            #pragma unroll
            for (int i = 0; i < std::tuple_size<ResT>{}; i++) {
                out_data.data[i] = 0;
            }

            [[intel::fpga_register]] auto maximum = in_data.data[0];
            [[intel::fpga_register]] int idx = 0;

            [[intel::initiation_interval(1)]] for (int i = 1; i < std::tuple_size<ResT>{}; i++) {
                if (in_data.data[i] > maximum) {
                    maximum = in_data.data[i];
                    idx = i;
                }
            }

            out_data.data[idx] = 1;

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> [[intel::use_stall_enable_clusters]] void softmax_stream() {
    switch (CONFIG_T::implementation) {
    case softmax_implementation::latency:
        softmax_latency_stream<data_pipe, res_pipe, CONFIG_T>();
        break;
    case softmax_implementation::stable:
        softmax_stable_stream<data_pipe, res_pipe, CONFIG_T>();
        break;
    case softmax_implementation::legacy:
        softmax_legacy_stream<data_pipe, res_pipe, CONFIG_T>();
        break;
    case softmax_implementation::argmax:
        softmax_argmax_stream<data_pipe, res_pipe, CONFIG_T>();
        break;
    default:
        softmax_stable_stream<data_pipe, res_pipe, CONFIG_T>();
        break;
    }
}

// *************************************************
//       TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> [[intel::use_stall_enable_clusters]] void dense_tanh_stream() {
#include "activation_tables/tanh_table.tb"
    constexpr int MAX_VALUE = 4;
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(std::tuple_size<DataT>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<DataT>{} / multiplier_limit;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

TanHActLoop:
    [[intel::initiation_interval(pipeline)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {

            auto in_data = data_pipe::read();

        TanHPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                [[intel::fpga_register]] typename DataT::value_type absoluteValue;

                if (in_data.data[j] < 0)
                    absoluteValue = (-1) * in_data.data[j];
                else
                    absoluteValue = in_data.data[j];

                [[intel::fpga_register]] int index;
                if (absoluteValue <= MAX_VALUE)
                    index = (absoluteValue * (CONFIG_T::table_size / MAX_VALUE)).to_int();
                else
                    index = CONFIG_T::table_size - 1;

                if (in_data.data[j] > 0)
                    out_data.data[j] = tanh_table[index];
                else
                    out_data.data[j] = -tanh_table[index];
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> [[intel::use_stall_enable_clusters]] void sigmoid_stream() {
#include "activation_tables/sigmoid_table.tb"
    constexpr int MAX_VALUE = 8;
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(std::tuple_size<DataT>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<DataT>{} / multiplier_limit;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

SigmoidActLoop:
    [[intel::initiation_interval(pipeline)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {
            auto in_data = data_pipe::read();

        SigmoidPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                [[intel::fpga_register]] typename DataT::value_type absoluteValue;

                if (in_data.data[j] < 0)
                    absoluteValue = (-1) * in_data.data[j];
                else
                    absoluteValue = in_data.data[j];

                [[intel::fpga_register]] int index;
                if (absoluteValue <= MAX_VALUE)
                    index = (absoluteValue * (CONFIG_T::table_size / MAX_VALUE)).to_int();
                else
                    index = CONFIG_T::table_size - 1;

                if (in_data.data[j] > 0)
                    out_data.data[j] = sigmoid_table[index];
                else
                    out_data.data[j] = 1 - sigmoid_table[index];
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************
// Note - Theano and Tensorflow might have different definitions for hard sigmoid; could provide two implementations
template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void hard_sigmoid_stream() {
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(std::tuple_size<DataT>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<DataT>{} / multiplier_limit;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

HardSigmoidActLoop:
    [[intel::initiation_interval(pipeline)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {

            auto in_data = data_pipe::read();

        HardSigmoidPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                [[intel::fpga_register]] auto datareg = CONFIG_T::slope * in_data.data[j] + CONFIG_T::shift;
                if (datareg > 1)
                    datareg = 1;
                else if (datareg < 0)
                    datareg = 0;
                out_data.data[j] = datareg;
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> [[intel::use_stall_enable_clusters]] void hard_tanh_stream() {
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(std::tuple_size<DataT>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<DataT>{} / multiplier_limit;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

HardSigmoidActLoop:
    [[intel::initiation_interval(pipeline)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {

            auto in_data = data_pipe::read();

        HardSigmoidPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                auto sigmoid = CONFIG_T::slope * in_data.data[j] + CONFIG_T::shift;
                if (sigmoid > 1)
                    sigmoid = 1;
                else if (sigmoid < 0)
                    sigmoid = 0;
                out_data.data[j] = 2 * sigmoid - 1;
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void binary_tanh_stream() {
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

BinaryTanHActLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {

            [[intel::fpga_register]] auto in_data = data_pipe::read();

        BinaryTanHPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                if (in_data.data[j] > 0)
                    out_data.data[j] = 1;
                else
                    out_data.data[j] = -1;
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void ternary_tanh_stream() {
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

TernaryTanHActLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<ResT>{}; i++) {

            [[intel::fpga_register]] auto in_data = data_pipe::read();

        TernaryTanHPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
                if (in_data.data[j] > 1)
                    out_data.data[j] = 1;
                else if (in_data.data[j] <= -1)
                    out_data.data[j] = -1;
                else
                    out_data.data[j] = 0;
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

} // namespace nnet

#endif
