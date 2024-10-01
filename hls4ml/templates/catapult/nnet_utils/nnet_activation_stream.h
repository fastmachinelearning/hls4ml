
// Change History:
//   2022-06-30  dgburnette - Cleaned up code to separate AC Math from LUT code.
//                            Activation functions not implemented in AC Math will assert.
//   2022-06-28  dgburnette - Replaced AP Types with AC Datatypes.

#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "ac_channel.h"
#include "ac_fixed.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_stream.h"
#include "nnet_types.h"
#include <ac_math/ac_elu_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_relu.h>
#include <ac_math/ac_selu_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math/ac_softmax_pwl.h>
#include <ac_math/ac_softplus_pwl.h>
#include <ac_math/ac_softsign_pwl.h>
#include <ac_math/ac_tanh_pwl.h>
#include <ac_std_float.h>
#include <cmath>

namespace nnet {

// *************************************************
//       LINEAR Activation
// *************************************************
// Adding this to work around problem with Catapult and SR model where the output channel appears to be inout
template <class data_T, class res_T, typename CONFIG_T> void linear(ac_channel<data_T> &data, ac_channel<res_T> &res) {
LinearActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    LinearPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            out_data[j] = in_data[j];
        }

        res.write(out_data);
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void relu(ac_channel<data_T> &data, ac_channel<res_T> &res) {
ReLUActLoop:
    for (unsigned int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    ReLUPackLoop:
        for (unsigned int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
#ifndef USE_AC_MATH
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
#else
            ac_math::ac_relu(in_data[j], out_data[j]);
#endif
        }

        res.write(out_data);
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************
#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(sigmoid_table);
        initialized = true;
    }

SigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    SigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            int data_round = in_data[j].to_double() * (int)CONFIG_T::table_size / 16;
            int index = data_round + 8 * (int)CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = (int)CONFIG_T::table_size - 1;
            out_data[j] = sigmoid_table[index];
        }

        res.write(out_data);
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) {
SigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
    SigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            // ac_math::ac_sigmoid_pwl(in_data[j], out_data[j]);
            ac_sigmoid_pwl_wrapper(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       Softmax Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<typename data_T::value_type, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned ii = data_T::size / multiplier_limit;
    (void)ii;

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[data_T::size];
    //#pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);

SoftmaxExpLoop:
    for (unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        //#pragma HLS PIPELINE II=ii

        data_T in_pack = data.read();
    SoftmaxExpPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            //#pragma HLS UNROLL
            unsigned x = softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T>(in_pack[j]);
            exp_res[j] = exp_table[x];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        exp_sum =
            reduce<typename CONFIG_T::exp_table_t, data_T::size, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        typename CONFIG_T::inv_table_t inv_exp_sum =
            invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];

        res_T out_pack;
    //#pragma HLS DATA_PACK variable=out_pack
    SoftmaxInvPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<typename data_T::value_type, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned ii = data_T::size / multiplier_limit;
    (void)ii;

    typename data_T::value_type data_array[data_T::size];
    //#pragma HLS ARRAY_PARTITION variable=data_array complete

    if constexpr (ii == 1) {
    }
    if constexpr (ii != 1) {
        // future enhancement for Catapult
    }
SoftmaxArrayLoop:
    for (unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        //#pragma HLS PIPELINE II=ii

        data_T in_pack = data.read();
    SoftmaxArrayPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            //#pragma HLS UNROLL
            data_array[j] = in_pack[j];
        }

        // Find the max and compute all delta(x_i, x_max)
        Op_max<typename data_T::value_type> op_max;
        typename data_T::value_type x_max =
            reduce<typename data_T::value_type, data_T::size, Op_max<typename data_T::value_type>>(data_array, op_max);

        // For the diffs, use the same type as the input but force rounding and saturation
        ac_fixed<data_T::value_type::width, data_T::value_type::i_width, true, AC_RND, AC_SAT> d_xi_xmax[data_T::size];
        for (unsigned j = 0; j < data_T::size; j++) {
            //#pragma HLS UNROLL
            d_xi_xmax[j] = data_array[j] - x_max;
        }

        // Calculate all the e^x's
        typename CONFIG_T::exp_table_t exp_res[data_T::size];
        //#pragma HLS ARRAY_PARTITION variable=exp_res complete
        typename CONFIG_T::exp_table_t exp_sum(0);
        for (unsigned j = 0; j < data_T::size; j++) {
            //#pragma HLS UNROLL
            unsigned x = softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T>(d_xi_xmax[j]);
            exp_res[j] = exp_table[x];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        exp_sum =
            reduce<typename CONFIG_T::exp_table_t, data_T::size, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        typename CONFIG_T::inv_table_t inv_exp_sum =
            invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];

        res_T out_pack;
    //#pragma HLS DATA_PACK variable=out_pack
    SoftmaxInvPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_invert_table_legacy<CONFIG_T, CONFIG_T::table_size>(invert_table);
        initialized = true;
    }

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[data_T::size];
    typename CONFIG_T::table_t exp_diff_res;
    typename data_T::value_type data_cache[data_T::size];

SoftmaxInitLoop:
    for (unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        //#pragma HLS PIPELINE
        data_T in_pack = data.read();
    SoftmaxInitPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            //#pragma HLS UNROLL
            data_cache[j] = in_pack[j];
            exp_res[j] = 0;
        }

    SoftmaxExpLoop:
        for (int i = 0; i < data_T::size; i++) {
        //#pragma HLS UNROLL
        SoftmaxExpInner:
            for (int j = 0; j < data_T::size; j++) {
                //#pragma HLS UNROLL

                if (i == j) {
                    exp_diff_res = 1;
                } else {
                    int data_round =
                        (data_cache[j].to_double() - data_cache[i].to_double()) * (int)CONFIG_T::table_size / 16;
                    int index = data_round + 8 * (int)CONFIG_T::table_size / 16;
                    if (index < 0)
                        index = 0;
                    if (index > CONFIG_T::table_size - 1)
                        index = (int)CONFIG_T::table_size - 1;
                    exp_diff_res = exp_table[index];
                }

                exp_res[i] += exp_diff_res;
            }
        }

        res_T out_pack;
    //#pragma HLS DATA_PACK variable=out_pack
    SoftmaxInvPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL

            int exp_res_index = exp_res[j].to_double() * (int)CONFIG_T::table_size / 64;
            if (exp_res_index < 0)
                exp_res_index = 0;
            if (exp_res_index > CONFIG_T::table_size - 1)
                exp_res_index = (int)CONFIG_T::table_size - 1;

            out_pack[j] = (typename res_T::value_type)invert_table[exp_res_index];
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T> void softmax(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    assert(CONFIG_T::axis == -1);

    switch (CONFIG_T::implementation) {
    case softmax_implementation::latency:
        softmax_latency<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::stable:
        softmax_stable<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::legacy:
        softmax_legacy<data_T, res_T, CONFIG_T>(data, res);
        break;
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void softmax(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    typename data_T::value_type data_cache[data_T::size];
    typename res_T::value_type res_cache[res_T::size];
SoftmaxInitLoop:
    for (unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        data_T in_pack = data.read();

    SoftmaxInitPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            data_cache[j] = in_pack[j];
        }

        res_T out_pack;
        // ac_math::ac_softmax_pwl(data_cache,res_cache);
        ac_softmax_pwl_wrapper(data_cache, res_cache);

    SoftmaxResPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            out_pack[j] = res_cache[j];
        }

        res.write(out_pack);
    }
}

#endif

// *************************************************
//       TanH Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>(tanh_table);
        initialized = true;
    }

TanHActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    TanHPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            int data_round = in_data[j].to_double() * (int)CONFIG_T::table_size / 8;
            int index = data_round + 4 * (int)CONFIG_T::table_size / 8;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = (int)CONFIG_T::table_size - 1;
            out_data[j] = tanh_table[index];
        }

        res.write(out_data);
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
TanHActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;
    TanHPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            // int data_round = in_data[j]*CONFIG_T::table_size/8;
            ac_math::ac_tanh_pwl(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       Hard sigmoid Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void hard_sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    typename data_T::value_type slope = (typename data_T::value_type)0.2;
    typename data_T::value_type shift = (typename data_T::value_type)0.5;

HardSigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    HardSigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            typename data_T::value_type datareg = slope * in_data[j] + shift;
            if (datareg > 1)
                datareg = 1;
            else if (datareg < 0)
                datareg = 0;
            out_data[j] = datareg;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Hard TanH Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void hard_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // typename data_T::value_type slope = (typename data_T::value_type) 0.2;
    // typename data_T::value_type shift = (typename data_T::value_type) 0.5;

HardTanhActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        // PRAGMA_DATA_PACK(out_data)

    HardTanhPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            auto sigmoid = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (sigmoid > 1)
                sigmoid = 1;
            else if (sigmoid < 0)
                sigmoid = 0;
            out_data[j] = 2 * sigmoid - 1;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void leaky_relu(ac_channel<data_T> &data, param_T alpha, ac_channel<res_T> &res) {
LeakyReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    LeakyReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha * in_data[j];
        }
        res.write(out_data);
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void thresholded_relu(ac_channel<data_T> &data, param_T theta, ac_channel<res_T> &res) {
ThresholdedReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    ThresholdedReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if (in_data[j] > theta)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Softplus Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void softplus(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t softplus_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t softplus_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_softplus_table<CONFIG_T, CONFIG_T::table_size>(softplus_table);
        initialized = true;
    }

SoftplusActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    SoftplusPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            int data_round = in_data[j].to_double() * (int)CONFIG_T::table_size / 16;
            int index = data_round + 8 * (int)CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = (int)CONFIG_T::table_size - 1;
            out_data[j] = softplus_table[index];
        }
        res.write(out_data);
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void softplus(ac_channel<data_T> &data, ac_channel<res_T> &res) {
SoftplusActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
    SoftplusPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            ac_softplus_pwl_wrapper(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       Softsign Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void softsign(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t softsign_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t softsign_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_softsign_table<CONFIG_T, CONFIG_T::table_size>(softsign_table);
        initialized = true;
    }

SoftsignActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    SoftsignPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            int data_round = in_data[j].to_double() * (int)CONFIG_T::table_size / 16;
            int index = data_round + 8 * (int)CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = (int)CONFIG_T::table_size - 1;
            out_data[j] = softsign_table[index];
        }
        res.write(out_data);
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void softsign(ac_channel<data_T> &data, ac_channel<res_T> &res) {
SoftsignActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
    SoftsignPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            ac_math::ac_softsign_pwl(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       ELU Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void elu(ac_channel<data_T> &data, param_T alpha, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t elu_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t elu_table[CONFIG_T::table_size];
#endif

    if (!initialized) {
        init_elu_table<CONFIG_T, CONFIG_T::table_size>(elu_table);
        initialized = true;
    }

EluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    EluPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL

            typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = datareg;
            } else {
                int index = (int)datareg.to_double() * (int)CONFIG_T::table_size / -8;
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data[j] = alpha * elu_table[index];
            }
        }
        res.write(out_data);
    }
}

#else
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void elu(ac_channel<data_T> &data, param_T alpha, ac_channel<res_T> &res) {
EluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
    EluPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            ac_math::ac_elu_pwl(in_data[j], out_data[j], alpha);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       SELU Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void selu(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t selu_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t selu_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_selu_table<CONFIG_T, CONFIG_T::table_size>(selu_table);
        initialized = true;
    }

SeluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    SeluPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL

            typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = (typename data_T::value_type)1.0507009873554804934193349852946 * datareg;
            } else {
                int index = (int)datareg.to_double() * (int)CONFIG_T::table_size / -8;
                if (index > CONFIG_T::table_size - 1)
                    index = (int)CONFIG_T::table_size - 1;
                out_data[j] = selu_table[index];
            }
        }
        res.write(out_data);
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void selu(ac_channel<data_T> &data, ac_channel<res_T> &res) {
SeluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
    SeluPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            ac_math::ac_selu_pwl(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       PReLU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void prelu(ac_channel<data_T> &data, const param_T alpha[CONFIG_T::n_in], ac_channel<res_T> &res) {
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha[i * res_T::size + j] * in_data[j];
        }
        res.write(out_data);
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void binary_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if (in_data[j] > 0)
                out_data[j] = (typename res_T::value_type)1;
            else
                out_data[j] = (typename res_T::value_type) - 1;
        }
        res.write(out_data);
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void ternary_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if (in_data[j] > 1)
                out_data[j] = (typename res_T::value_type)1;
            else if (in_data[j] <= -1)
                out_data[j] = (typename res_T::value_type) - 1;
            else
                out_data[j] = (typename res_T::value_type)0;
        }
        res.write(out_data);
    }
}

} // namespace nnet

#endif
