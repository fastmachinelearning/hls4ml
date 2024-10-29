#ifndef NNET_ACTIVATION_H_
#define NNET_ACTIVATION_H_

#include "nnet_common.h"

namespace nnet {

struct activ_config {
    // IO size
    static constexpr unsigned n_in = 10;

    // Internal info
    static constexpr unsigned table_size = 512;

    // Resource reuse info
    static constexpr unsigned io_type = io_parallel;
    static constexpr unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef ac_fixed<16, 8> table_t;
};

// *************************************************
//       LINEAR Activation -- See Issue 53
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void linear(const data_T &data, res_T &res) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = data[ii];
        res[ii] = datareg;
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void relu(const data_T &data, res_T &res) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = 0;
    }
}

template <class data_T, class res_T, int MAX_INT, typename CONFIG_T> void relu_max(const data_T &data, res_T &res) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = data[ii];
        if (datareg < 0)
            res[ii] = 0;
        else if (datareg > MAX_INT)
            res[ii] = MAX_INT;
        else
            res[ii] = datareg;
    }
}

template <class data_T, class res_T, typename CONFIG_T> void relu6(const data_T &data, res_T &res) {
    relu_max<data_T, res_T, 6, CONFIG_T>(data, res);
}

template <class data_T, class res_T, typename CONFIG_T> void relu1(const data_T &data, res_T &res) {
    relu_max<data_T, res_T, 1, CONFIG_T>(data, res);
}

// *************************************************
//       Sigmoid Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void sigmoid(const data_T &data, res_T &res) {
    static constexpr int MAX_VALUE = 8;
#include "activation_tables/sigmoid_table.tb"
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        [[intel::fpga_register]] typename data_T::value_type absoluteValue;
        [[intel::fpga_register]] typename res_T::value_type temp2;
        if (data[ii] < 0) {
            absoluteValue = -data[ii];
        } else {
            absoluteValue = data[ii];
        }
        int index = (absoluteValue * (CONFIG_T::table_size / MAX_VALUE)).to_int();
        if (absoluteValue > MAX_VALUE)
            index = CONFIG_T::table_size - 1;
        temp2 = static_cast<typename res_T::value_type>(sigmoid_table[index]);
        if (data[ii] < 0) {
            res[ii] = 1 - temp2;
        } else {
            res[ii] = temp2;
        }
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

enum class softmax_implementation { latency = 0, legacy = 1, stable = 2, argmax = 3 };

template <class data_T, typename CONFIG_T> inline unsigned softmax_stable_idx_from_real_val(const data_T x) {
    // Number of address bits for table
    static constexpr int N = ceillog2<CONFIG_T::table_size>::val;

    // Slice the top N bits of the input
    [[intel::fpga_register]] ac_int<N, false> y = x.template slc<N>(x.width - N - 1);
    // If x is the most negative value, the slice will be 0, so we need to set the 0-th bit to ensure correctness
    if (x != 0 && y == 0)
        y[0] = 1;
    return y.to_uint();
}

template <class data_T, typename CONFIG_T> inline unsigned softmax_latency_idx_from_real_val(const data_T x) {
    // Number of address bits for table
    static constexpr int N = ceillog2<CONFIG_T::table_size>::val;

    // Slice the top N bits of the input
    [[intel::fpga_register]] ac_int<N, false> y = x.template slc<N>(x.width - N);
    return y.to_uint();
}

template <class data_T, class res_T, typename CONFIG_T> void softmax_stable(const data_T &data, res_T &res) {
// Look-up tables
#include "activation_tables/exp_table.tb"
#include "activation_tables/invert_table.tb"

    // Find maximum
    Op_max<typename data_T::value_type> op_max;
    [[intel::fpga_register]] auto x_max =
        reduce<typename data_T::value_type, CONFIG_T::n_in, Op_max<typename data_T::value_type>>(data.data(), op_max);

    // For the diffs, use the same type as the input but force rounding and saturation
    [[intel::fpga_register]] ac_fixed<data_T::value_type::width, data_T::value_type::i_width, true, AC_RND, AC_SAT>
        d_xi_xmax[CONFIG_T::n_in];
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        d_xi_xmax[i] = data[i] - x_max;
    }

    // Calculate all the e^x's
    [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        exp_res[i] = exp_table[softmax_stable_idx_from_real_val<typename data_T::value_type, CONFIG_T>(d_xi_xmax[i])];
    }

    // Explicitly sum previously calculated exponentials with an adder tree
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_sum =
        reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    // Multiply previously calculated exponetials with the reciprocal of the sum
    [[intel::fpga_register]] typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table[softmax_stable_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

// TODO - Improve accuracy
template <class data_T, class res_T, typename CONFIG_T> void softmax_latency(const data_T &data, res_T &res) {
#include "activation_tables/exp_table_latency.tb"
#include "activation_tables/invert_table_latency.tb"

    // Calculate all the e^x's
    [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        exp_res[i] = exp_table_latency[softmax_latency_idx_from_real_val<typename data_T::value_type, CONFIG_T>(data[i])];
    }

    // Explicitly sum the results with an adder tree.
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_sum =
        reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    // Multiply previously calculated exponetials with the reciprocal of the sum
    [[intel::fpga_register]] typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table_latency[softmax_latency_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

template <class data_T, class res_T, typename CONFIG_T> void softmax_legacy(const data_T &data, res_T &res) {
#include "activation_tables/exp_table_legacy.tb"
#include "activation_tables/invert_table_legacy.tb"

    [[intel::fpga_register]] int data_round[CONFIG_T::n_in];
New_loop:
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round[ii] = (data[ii] * CONFIG_T::table_size / 16).to_int();
    }
NN_Outer:
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        typename CONFIG_T::exp_table_t exp_res_temp = 0;
    NN_Inner:
        #pragma unroll
        for (int jj = 0; jj < CONFIG_T::n_in; jj++) {
            if (ii == jj) {
                exp_res_temp += 1;
            } else {
                int _data_cache = (data_round[jj] - data_round[ii]);
                int index = _data_cache + 8 * CONFIG_T::table_size / 16;

                if (index < 0)
                    index = 0;
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;

                typename CONFIG_T::exp_table_t temp_exp = exp_table_legacy[index];
                exp_res_temp += temp_exp;
            }
        }
        int exp_res_index = (exp_res_temp * CONFIG_T::table_size / 64).to_int();
        if (exp_res_index < 0)
            exp_res_index = 0;
        if (exp_res_index > CONFIG_T::table_size - 1)
            exp_res_index = CONFIG_T::table_size - 1;
        res[ii] = invert_table_legacy[exp_res_index];
    }
}

template <class data_T, class res_T, typename CONFIG_T> void softmax_argmax(const data_T &data, res_T &res) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        res[i] = static_cast<typename res_T::value_type>(0);
    }

    [[intel::fpga_register]] auto maximum = data[0];
    [[intel::fpga_register]] int idx = 0;

    [[intel::initiation_interval(1)]] for (int i = 1; i < CONFIG_T::n_in; i++) {
        if (data[i] > maximum) {
            maximum = data[i];
            idx = i;
        }
    }

    res[idx] = static_cast<typename res_T::value_type>(1);
}

template <class data_T, class res_T, typename CONFIG_T> inline void softmax(const data_T &data, res_T &res) {
    switch (CONFIG_T::implementation) {
    case softmax_implementation::stable:
        softmax_stable<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::latency:
        softmax_latency<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::legacy:
        softmax_legacy<data_T, res_T, CONFIG_T>(data, res);
        break;
    default:
        softmax_stable<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::argmax:
        softmax_argmax<data_T, res_T, CONFIG_T>(data, res);
        break;
    }
}

// *************************************************
//       TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void dense_tanh(const data_T &data, res_T &res) {
    static constexpr int MAX_VALUE = 4;
// Initialize the lookup table
#include "activation_tables/tanh_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        [[intel::fpga_register]] typename data_T::value_type temp;
        [[intel::fpga_register]] typename res_T::value_type temp2;
        if (data[ii] < 0) {
            temp = -data[ii];
        } else {
            temp = data[ii];
        }
        ac_int<16> index = (temp * (CONFIG_T::table_size / MAX_VALUE)).to_int();
        if (temp > MAX_VALUE)
            index = CONFIG_T::table_size - 1;
        temp2 = static_cast<typename res_T::value_type>(tanh_table[index]);
        if (data[ii] < 0) {
            res[ii] = -temp2;
        } else {
            res[ii] = temp2;
        }
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void hard_sigmoid(const data_T &data, res_T &res) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = CONFIG_T::slope * data[ii] + CONFIG_T::shift;
        if (datareg > 1)
            datareg = 1;
        else if (datareg < 0)
            datareg = 0;
        res[ii] = datareg;
    }
}

template <class data_T, class res_T, typename CONFIG_T> void hard_tanh(const data_T &data, res_T &res) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto sigmoid = CONFIG_T::slope * data[ii] + CONFIG_T::shift;
        if (sigmoid > 1)
            sigmoid = 1;
        else if (sigmoid < 0)
            sigmoid = 0;
        res[ii] = 2 * sigmoid - 1;
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void leaky_relu(const data_T &data, const typename CONFIG_T::param_t alpha, res_T &res) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = alpha * datareg;
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void thresholded_relu(const data_T &data, const typename CONFIG_T::param_t theta, res_T &res) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = data[ii];
        if (datareg > theta)
            res[ii] = datareg;
        else
            res[ii] = 0;
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void softplus(const data_T &data, res_T &res) {
// Initialize the lookup table
#include "activation_tables/softplus_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        ac_int<16> data_round = (data[ii] * CONFIG_T::table_size / 16).to_int();
        ac_int<16> index = data_round + 8 * CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = static_cast<typename res_T::value_type>(softplus_table[index]);
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void softsign(const data_T &data, res_T &res) {
    static constexpr int MAX_VALUE = 8;
// Initialize the lookup table
#include "activation_tables/softsign_table.tb"

    // Index into the lookup table based on data
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        [[intel::fpga_register]] typename data_T::value_type temp;
        [[intel::fpga_register]] typename res_T::value_type temp2;
        if (data[ii] < 0) {
            temp = -data[ii];
        } else {
            temp = data[ii];
        }
        ac_int<16> index = (temp * CONFIG_T::table_size / MAX_VALUE).to_int();
        if (temp > MAX_VALUE)
            index = CONFIG_T::table_size - 1;
        temp2 = static_cast<typename res_T::value_type>(softsign_table[index]);
        if (data[ii] < 0) {
            res[ii] = -temp2;
        } else {
            res[ii] = temp2;
        }
    }
}

// *************************************************
//       ELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void elu(const data_T &data, const typename CONFIG_T::param_t alpha, res_T &res) {
// Initialize the lookup table
#include "activation_tables/elu_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = datareg;
        } else {
            ac_int<16> index = (datareg * CONFIG_T::table_size / -8).to_int();
            if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            res[ii] = alpha * elu_table[index];
        }
    }
}

// *************************************************
//       SELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void selu(const data_T &data, res_T &res) {
// Initialize the lookup table
#include "activation_tables/selu_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = static_cast<typename res_T::value_type>(1.0507009873554804934193349852946) * datareg;
        } else {
            ac_int<16> index = (datareg * CONFIG_T::table_size / -8).to_int();
            if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            res[ii] = selu_table[index];
        }
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void prelu(const data_T &data, const typename CONFIG_T::param_t &alpha, res_T &res) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = alpha[ii] * datareg;
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void binary_tanh(const data_T &data, res_T &res) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = data[ii];
        typename res_T::value_type cache;
        if (datareg > 0)
            cache = 1;
        else
            cache = -1;

        res[ii] = cache;
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void ternary_tanh(const data_T &data, res_T &res) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = 2 * data[ii];
        typename res_T::value_type cache;
        if (datareg > 1)
            cache = 1;
        else if (datareg > -1 && datareg <= 1)
            cache = 0;
        else
            cache = -1;

        res[ii] = cache;
    }
}

} // namespace nnet

#endif
