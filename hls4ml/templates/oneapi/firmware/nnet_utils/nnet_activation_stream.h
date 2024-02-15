#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "nnet_common.h"
#include "nnet_types.h"

namespace nnet {

// *************************************************
//       Linear Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void linear() {
LinearActLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    LinearPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            out_data[j] = in_data[j];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       ReLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void relu() {
ReLUActLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    ReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
void leaky_relu(const typename data_pipe::value_type::value_type alpha) {
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_pipe::value_type::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_pipe::value_type::size / multiplier_limit;

LeakyReLUActLoop:
    [[intel::initiation_interval(pipeline)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    LeakyReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha * in_data[j];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
void thresholded_relu(const typename data_pipe::value_type::value_type theta) {
ThresholdedReLUActLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    ThresholdedReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            if (in_data[j] > theta)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       ELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
void elu(const typename data_pipe::value_type::value_type alpha) {
#include "activation_tables/elu_table.tb"

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_pipe::value_type::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_pipe::value_type::size / multiplier_limit;

EluActLoop:
    [[intel::initiation_interval(pipeline)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    EluPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            [[intel::fpga_register]] typename data_pipe::value_type::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = datareg;
            } else {
                int index = (datareg * CONFIG_T::table_size / -8).to_int();
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data[j] = alpha * elu_table[index];
            }
        }

        res_pipe::write(out_data);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void elu() { elu<data_pipe, res_pipe, CONFIG_T>(1.0); }

// *************************************************
//       SeLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void selu() {
#include "activation_tables/selu_table.tb"

SeluActLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    SeluPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            [[intel::fpga_register]] typename data_pipe::value_type::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = typename data_pipe::value_type::value_type(1.0507009873554804934193349852946) * datareg;
            } else {
                int index = (datareg * CONFIG_T::table_size / -8).to_int();
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data[j] = selu_table[index];
            }
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
void prelu(const typename data_pipe::value_type::value_type alpha[CONFIG_T::n_in]) {
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_pipe::value_type::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_pipe::value_type::size / multiplier_limit;

PReLUActLoop:
    [[intel::initiation_interval(pipeline)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    PReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha[i * res_pipe::value_type::size + j] * in_data[j];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void softplus() {
#include "activation_tables/softplus_table.tb"

SoftplusActLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    SoftplusPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            [[intel::fpga_register]] int data_round = (in_data[j] * CONFIG_T::table_size / 16).to_int();
            [[intel::fpga_register]] int index = data_round + 8 * CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            out_data[j] = softplus_table[index];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void softsign() {
#include "activation_tables/softsign_table.tb"

    static const int MAX_VALUE = 8;

SoftsignActLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    SoftsignPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            [[intel::fpga_register]] typename data_pipe::value_type::value_type absValue;
            ;
            if (in_data[j] < 0) {
                absValue = -in_data[j];
            } else {
                absValue = in_data[j];
            }
            ac_int<16> index = (absValue * CONFIG_T::table_size / MAX_VALUE).to_int();
            if (absValue > MAX_VALUE)
                index = CONFIG_T::table_size - 1;
            if (in_data[j] < 0) {
                out_data[j] = static_cast<typename res_pipe::value_type::value_type>(-softsign_table[index]);
            } else {
                out_data[j] = static_cast<typename res_pipe::value_type::value_type>(softsign_table[index]);
            }
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_stable() {
#include "activation_tables/exp_table.tb"
#include "activation_tables/invert_table.tb"

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_pipe::value_type::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_pipe::value_type::size / multiplier_limit;

    [[intel::fpga_register]] typename data_pipe::value_type::value_type data_array[data_pipe::value_type::size];

SoftmaxArrayLoop:
    [[intel::initiation_interval(pipeline)]] for (unsigned i = 0; i < CONFIG_T::n_in / data_pipe::value_type::size; i++) {
        auto in_pack = data_pipe::read();

    SoftmaxArrayPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < data_pipe::value_type::size; j++) {
            data_array[j] = in_pack[j];
        }

        // Find the max and compute all delta(x_i, x_max)
        Op_max<typename data_pipe::value_type::value_type> op_max;
        [[intel::fpga_register]] typename data_pipe::value_type::value_type x_max =
            reduce<typename data_pipe::value_type::value_type, data_pipe::value_type::size,
                   Op_max<typename data_pipe::value_type::value_type>>(data_array, op_max);

        // For the diffs, use the same type as the input but force rounding and saturation
        [[intel::fpga_register]] ac_fixed<data_pipe::value_type::value_type::width,
                                          data_pipe::value_type::value_type::i_width, true, AC_RND, AC_SAT>
            d_xi_xmax[data_pipe::value_type::size];
        #pragma unroll
        for (unsigned j = 0; j < data_pipe::value_type::size; j++) {
            d_xi_xmax[j] = data_array[j] - x_max;
        }

        // Calculate all the e^x's
        [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_res[data_pipe::value_type::size];
        #pragma unroll
        for (unsigned j = 0; j < data_pipe::value_type::size; j++) {
            exp_res[j] = exp_table[softmax_stable_idx_from_real_val<typename data_pipe::value_type::value_type, CONFIG_T>(
                d_xi_xmax[j])];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_sum =
            reduce<typename CONFIG_T::exp_table_t, data_pipe::value_type::size, Op_add<typename CONFIG_T::exp_table_t>>(
                exp_res, op_add);

        [[intel::fpga_register]] typename CONFIG_T::inv_table_t inv_exp_sum =
            invert_table[softmax_stable_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
        typename res_pipe::value_type out_pack;

    SoftmaxInvPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < res_pipe::value_type::size; j++) {

            // TODO - Find Quartus-equivalent pragma
            // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

            out_pack[j] = exp_res[j] * inv_exp_sum;
        }

        res_pipe::write(out_pack);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_latency() {
#include "activation_tables/exp_table_latency.tb"
#include "activation_tables/invert_table_latency.tb"

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_pipe::value_type::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_pipe::value_type::size / multiplier_limit;

    // Calculate all the e^x's
    [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_res[data_pipe::value_type::size];

SoftmaxExpLoop:
    [[intel::initiation_interval(pipeline)]] for (unsigned i = 0; i < CONFIG_T::n_in / data_pipe::value_type::size; i++) {
        auto in_pack = data_pipe::read();

    SoftmaxExpPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < data_pipe::value_type::size; j++) {
            exp_res[j] =
                exp_table_latency[softmax_latency_idx_from_real_val<typename data_pipe::value_type::value_type, CONFIG_T>(
                    in_pack[j])];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_sum =
            reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        // Multiply previously calculated exponetials with the reciprocal of the sum
        [[intel::fpga_register]] typename CONFIG_T::inv_table_t inv_exp_sum =
            invert_table_latency[softmax_latency_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];

        typename res_pipe::value_type out_pack;
    SoftmaxInvPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < res_pipe::value_type::size; j++) {
            // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }

        res_pipe::write(out_pack);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_legacy() {
#include "activation_tables/exp_table_legacy.tb"
#include "activation_tables/invert_table_legacy.tb"

    // Index into the lookup table based on data for exponentials
    [[intel::fpga_register]] typename CONFIG_T::table_t exp_res[data_pipe::value_type::size];
    [[intel::fpga_register]] typename CONFIG_T::table_t exp_diff_res;
    [[intel::fpga_register]] typename data_pipe::value_type::value_type data_cache[data_pipe::value_type::size];

SoftmaxInitLoop:
    [[intel::initiation_interval(1)]] for (unsigned s = 0; s < CONFIG_T::n_in / data_pipe::value_type::size; s++) {
        auto in_pack = data_pipe::read();

    SoftmaxInitPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < data_pipe::value_type::size; j++) {
            data_cache[j] = in_pack[j];
            exp_res[j] = 0;
        }

    SoftmaxExpLoop:
        #pragma unroll
        for (int i = 0; i < data_pipe::value_type::size; i++) {
        SoftmaxExpInner:
            #pragma unroll
            for (int j = 0; j < data_pipe::value_type::size; j++) {
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

        typename res_pipe::value_type out_pack;
    SoftmaxInvPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < res_pipe::value_type::size; j++) {
            int exp_res_index = (exp_res[j] * CONFIG_T::table_size / 64).to_int();
            if (exp_res_index < 0)
                exp_res_index = 0;
            if (exp_res_index > CONFIG_T::table_size - 1)
                exp_res_index = CONFIG_T::table_size - 1;
            out_pack[j] = static_cast<typename res_pipe::value_type::value_type>(invert_table_legacy[exp_res_index]);
        }

        res_pipe::write(out_pack);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_argmax() {
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

        #pragma unroll
        for (int i = 0; i < res_pipe::value_type::size; i++) {
            out_data[i] = static_cast<typename res_pipe::value_type::value_type>(0);
        }

        [[intel::fpga_register]] typename data_pipe::value_type::value_type maximum = in_data[0];
        [[intel::fpga_register]] int idx = 0;

        [[intel::initiation_interval(1)]] for (int i = 1; i < res_pipe::value_type::size; i++) {
            if (in_data[i] > maximum) {
                maximum = in_data[i];
                idx = i;
            }
        }

        out_data[idx] = static_cast<typename res_pipe::value_type::value_type>(1);
        res_pipe::write(out_data);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax() {
    switch (CONFIG_T::implementation) {
    case softmax_implementation::latency:
        softmax_latency<data_pipe, res_pipe, CONFIG_T>();
        break;
    case softmax_implementation::stable:
        softmax_stable<data_pipe, res_pipe, CONFIG_T>();
        break;
    case softmax_implementation::legacy:
        softmax_legacy<data_pipe, res_pipe, CONFIG_T>();
        break;
    case softmax_implementation::argmax:
        softmax_argmax<data_pipe, res_pipe, CONFIG_T>();
        break;
    default:
        softmax_stable<data_pipe, res_pipe, CONFIG_T>();
        break;
    }
}

// *************************************************
//       TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void dense_tanh() {
#include "activation_tables/tanh_table.tb"
    static const int MAX_VALUE = 4;

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_pipe::value_type::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_pipe::value_type::size / multiplier_limit;

TanHActLoop:
    [[intel::initiation_interval(pipeline)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {

        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    TanHPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            [[intel::fpga_register]] typename data_pipe::value_type::value_type absoluteValue;

            if (in_data[j] < 0)
                absoluteValue = (-1) * in_data[j];
            else
                absoluteValue = in_data[j];

            [[intel::fpga_register]] int index;
            if (absoluteValue <= MAX_VALUE)
                index = (absoluteValue * (CONFIG_T::table_size / MAX_VALUE)).to_int();
            else
                index = CONFIG_T::table_size - 1;

            if (in_data[j] > 0)
                out_data[j] = tanh_table[index];
            else
                out_data[j] = -tanh_table[index];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void sigmoid() {
#include "activation_tables/sigmoid_table.tb"
    static const int MAX_VALUE = 8;

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_pipe::value_type::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_pipe::value_type::size / multiplier_limit;

SigmoidActLoop:
    [[intel::initiation_interval(pipeline)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {
        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    SigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            [[intel::fpga_register]] typename data_pipe::value_type::value_type absoluteValue;

            if (in_data[j] < 0)
                absoluteValue = (-1) * in_data[j];
            else
                absoluteValue = in_data[j];

            [[intel::fpga_register]] int index;
            if (absoluteValue <= MAX_VALUE)
                index = (absoluteValue * (CONFIG_T::table_size / MAX_VALUE)).to_int();
            else
                index = CONFIG_T::table_size - 1;

            if (in_data[j] > 0)
                out_data[j] = sigmoid_table[index];
            else
                out_data[j] = 1 - sigmoid_table[index];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************
// Note - Theano and Tensorflow might have different definitions for hard sigmoid; could provide two implementations
template <class data_pipe, class res_pipe, typename CONFIG_T> void hard_sigmoid() {

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_pipe::value_type::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_pipe::value_type::size / multiplier_limit;

HardSigmoidActLoop:
    [[intel::initiation_interval(pipeline)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {

        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    HardSigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            [[intel::fpga_register]] auto datareg = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (datareg > 1)
                datareg = 1;
            else if (datareg < 0)
                datareg = 0;
            out_data[j] = datareg;
        }

        res_pipe::write(out_data);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void hard_tanh() {

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_pipe::value_type::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_pipe::value_type::size / multiplier_limit;

HardSigmoidActLoop:
    [[intel::initiation_interval(pipeline)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {

        auto in_data = data_pipe::read();
        typename res_pipe::value_type out_data;

    HardSigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            auto sigmoid = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (sigmoid > 1)
                sigmoid = 1;
            else if (sigmoid < 0)
                sigmoid = 0;
            out_data[j] = 2 * sigmoid - 1;
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void binary_tanh() {
BinaryTanHActLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {

        [[intel::fpga_register]] auto in_data = data_pipe::read();
        [[intel::fpga_register]] typename res_pipe::value_type out_data;

    BinaryTanHPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            if (in_data[j] > 0)
                out_data[j] = static_cast<typename res_pipe::value_type::value_type>(1);
            else
                out_data[j] = static_cast<typename res_pipe::value_type::value_type>(-1);
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void ternary_tanh() {
TernaryTanHActLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / res_pipe::value_type::size; i++) {

        [[intel::fpga_register]] auto in_data = data_pipe::read();
        [[intel::fpga_register]] typename res_pipe::value_type out_data;

    TernaryTanHPackLoop:
        #pragma unroll
        for (int j = 0; j < res_pipe::value_type::size; j++) {
            if (in_data[j] > 1)
                out_data[j] = static_cast<typename res_pipe::value_type::value_type>(1);
            else if (in_data[j] <= -1)
                out_data[j] = static_cast<typename res_pipe::value_type::value_type>(-1);
            else
                out_data[j] = static_cast<typename res_pipe::value_type::value_type>(0);
        }

        res_pipe::write(out_data);
    }
}

} // namespace nnet

#endif
