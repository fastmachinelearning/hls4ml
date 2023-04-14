#ifndef NNET_ACTIVATION_H_
#define NNET_ACTIVATION_H_

#include "nnet_common.h"

namespace nnet {

struct activ_config {
    // IO size
    static const unsigned n_in = 10;

    // Internal info
    static const unsigned table_size = 512;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef ac_fixed<16, 8> table_t;
};

// *************************************************
//       LINEAR Activation -- See Issue 53
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void linear(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        res[ii] = datareg;
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void relu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = 0;
    }
}

template <class data_T, class res_T, int MAX_INT, typename CONFIG_T>
void relu_max(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        if (datareg < 0)
            res[ii] = 0;
        else if (datareg > MAX_INT)
            res[ii] = MAX_INT;
        else
            res[ii] = datareg;
    }
}

template <class data_T, class res_T, typename CONFIG_T> void relu6(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    relu_max<data_T, res_T, 6, CONFIG_T>(data, res);
}

template <class data_T, class res_T, typename CONFIG_T> void relu1(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    relu_max<data_T, res_T, 1, CONFIG_T>(data, res);
}

// *************************************************
//       Sigmoid Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    static const int MAX_VALUE = 8;
#include "activation_tables/sigmoid_table.tb"
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T absoluteValue hls_register;
        res_T temp2 hls_register;
        if (data[ii] < 0) {
            absoluteValue = -data[ii];
        } else {
            absoluteValue = data[ii];
        }
        int index = (absoluteValue * (CONFIG_T::table_size / MAX_VALUE)).to_int();
        if (absoluteValue > MAX_VALUE)
            index = CONFIG_T::table_size - 1;
        temp2 = (res_T)sigmoid_table[index];
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
    static constexpr int N = ceillog2(CONFIG_T::table_size);

    // Slice the top N bits of the input
    hls_register ac_int<N, false> y = x.template slc<N>(x.width - N - 1);
    // If x is the most negative value, the slice will be 0, so we need to set the 0-th bit to ensure correctness
    if (x != 0 && y == 0)
        y[0] = 1;
    return y.to_uint();
}

template <class data_T, typename CONFIG_T> inline unsigned softmax_latency_idx_from_real_val(const data_T x) {
    // Number of address bits for table
    static constexpr int N = ceillog2(CONFIG_T::table_size);

    // Slice the top N bits of the input
    hls_register ac_int<N, false> y = x.template slc<N>(x.width - N);
    return y.to_uint();
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
// Look-up tables
#include "activation_tables/exp_table.tb"
#include "activation_tables/invert_table.tb"

    // Find maximum
    Op_max<data_T> op_max;
    hls_register data_T x_max = reduce<data_T, CONFIG_T::n_in, Op_max<data_T>>(data, op_max);

    // For the diffs, use the same type as the input but force rounding and saturation
    hls_register ac_fixed<data_T::width, data_T::i_width, true, AC_RND, AC_SAT> d_xi_xmax[CONFIG_T::n_in];
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        d_xi_xmax[i] = data[i] - x_max;
    }

    // Calculate all the e^x's
    hls_register typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        exp_res[i] = exp_table[softmax_stable_idx_from_real_val<data_T, CONFIG_T>(d_xi_xmax[i])];
    }

    // Explicitly sum previously calculated exponentials with an adder tree
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    hls_register typename CONFIG_T::exp_table_t exp_sum =
        reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    // Multiply previously calculated exponetials with the reciprocal of the sum
    hls_register typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table[softmax_stable_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

// TODO - Improve accuracy
template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
#include "activation_tables/exp_table_latency.tb"
#include "activation_tables/invert_table_latency.tb"

    // Calculate all the e^x's
    hls_register typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        exp_res[i] = exp_table_latency[softmax_latency_idx_from_real_val<data_T, CONFIG_T>(data[i])];
    }

    // Explicitly sum the results with an adder tree.
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    hls_register typename CONFIG_T::exp_table_t exp_sum =
        reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    // Multiply previously calculated exponetials with the reciprocal of the sum
    hls_register typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table_latency[softmax_latency_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
#include "activation_tables/exp_table_legacy.tb"
#include "activation_tables/invert_table_legacy.tb"

    hls_register int data_round[CONFIG_T::n_in];
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

template <class data_T, class res_T, typename CONFIG_T>
void softmax_argmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        res[i] = (res_T)0;
    }

    hls_register data_T maximum = data[0];
    hls_register int idx = 0;

    #pragma ii 1
    for (int i = 1; i < CONFIG_T::n_in; i++) {
        if (data[i] > maximum) {
            maximum = data[i];
            idx = i;
        }
    }

    res[idx] = (res_T)1;
}

template <class data_T, class res_T, typename CONFIG_T>
inline void softmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
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
template <class data_T, class res_T, typename CONFIG_T>
void dense_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    static const int MAX_VALUE = 4;
// Initialize the lookup table
#include "activation_tables/tanh_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T temp hls_register;
        res_T temp2 hls_register;
        if (data[ii] < 0) {
            temp = -data[ii];
        } else {
            temp = data[ii];
        }
        ac_int<16> index = (temp * (CONFIG_T::table_size / MAX_VALUE)).to_int();
        if (temp > MAX_VALUE)
            index = CONFIG_T::table_size - 1;
        temp2 = (res_T)tanh_table[index];
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
template <class data_T, class res_T, typename CONFIG_T>
void hard_sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
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

template <class data_T, class res_T, typename CONFIG_T>
void hard_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
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
void leaky_relu(data_T data[CONFIG_T::n_in], data_T alpha, res_T res[CONFIG_T::n_in]) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
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
void thresholded_relu(data_T data[CONFIG_T::n_in], data_T theta, res_T res[CONFIG_T::n_in]) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        if (datareg > theta)
            res[ii] = datareg;
        else
            res[ii] = 0;
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void softplus(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
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
        res[ii] = (res_T)softplus_table[index];
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void softsign(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    static const int MAX_VALUE = 8;
// Initialize the lookup table
#include "activation_tables/softsign_table.tb"

    // Index into the lookup table based on data
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T temp hls_register;
        res_T temp2 hls_register;
        if (data[ii] < 0) {
            temp = -data[ii];
        } else {
            temp = data[ii];
        }
        ac_int<16> index = (temp * CONFIG_T::table_size / MAX_VALUE).to_int();
        if (temp > MAX_VALUE)
            index = CONFIG_T::table_size - 1;
        temp2 = (res_T)softsign_table[index];
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
void elu(data_T data[CONFIG_T::n_in], const res_T alpha, res_T res[CONFIG_T::n_in]) {
// Initialize the lookup table
#include "activation_tables/elu_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
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

template <class data_T, class res_T, typename CONFIG_T> void elu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    elu<data_T, res_T, CONFIG_T>(data, 1.0, res);
}

// *************************************************
//       SELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void selu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
// Initialize the lookup table
#include "activation_tables/selu_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = res_T(1.0507009873554804934193349852946) * datareg;
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
void prelu(data_T data[CONFIG_T::n_in], const data_T alpha[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = alpha[ii] * datareg;
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void binary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        res_T cache;
        if (datareg > 0)
            cache = 1;
        else
            cache = -1;

        res[ii] = (res_T)cache;
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void ternary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = 2 * data[ii];
        res_T cache;
        if (datareg > 1)
            cache = 1;
        else if (datareg > -1 && datareg <= 1)
            cache = 0;
        else
            cache = -1;

        res[ii] = (res_T)cache;
    }
}

} // namespace nnet

#endif
