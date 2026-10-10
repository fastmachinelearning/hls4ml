#ifndef NNET_DENSE_FUSED_H_
#define NNET_DENSE_FUSED_H_

#include "hls_stream.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_mult.h"

// Kernels of the fused strategy: a chain of Dense layers computed in one DATAFLOW region. A Dense layer
// needs every input before it can produce any output, so it streams on one side only.
//
//   dense_fused_dot   array in, stream out   weights w[j * n_in + i], transposed by the fusion pass
//   dense_fused_axpy  stream in, array out   weights w[i * n_out + j], as hls4ml stores them
//   dense_fused       array in, array out    weights w[i * n_out + j], first layer of an odd-length chain
//
// A dot layer and the axpy layer after it run at the same time. The activation that followed the layer
// is computed here, with the same tables and arithmetic as the separate layer, so the results are equal.

namespace nnet {

enum FusedActivation {
    FUSED_LINEAR = 0,
    FUSED_RELU,
    FUSED_SIGMOID,
    FUSED_TANH,
    FUSED_SOFTPLUS,
    FUSED_SOFTSIGN,
    FUSED_SELU,
    FUSED_ELU,
    FUSED_LEAKY_RELU,
    FUSED_THRESHOLDED_RELU,
    FUSED_HARD_SIGMOID,
    FUSED_HARD_TANH,
    FUSED_BINARY_TANH
};

struct dense_fused_config {
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;
    typedef ap_fixed<18, 8> table_t;

    typedef ap_fixed<16, 6> param_t;
    typedef ap_ufixed<16, 0> slope_t;
    typedef ap_ufixed<2, 0> shift_t;
    typedef ap_fixed<16, 6> preact_t;

    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = 1;

    static const unsigned activation = FUSED_LINEAR;
    static const unsigned table_size = 1024;

    static const unsigned pad_cycles = 0;

    static const param_t activation_param;
    static const slope_t slope;
    static const shift_t shift;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

// Fill the lookup table the same way the separate activation layer does. An activation without a table
// leaves it empty, and the tool removes it.
template <typename CONFIG_T> void fused_init_table(typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    if (CONFIG_T::activation == FUSED_SIGMOID) {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(table);
    } else if (CONFIG_T::activation == FUSED_TANH) {
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>(table);
    } else if (CONFIG_T::activation == FUSED_SOFTPLUS) {
        init_softplus_table<CONFIG_T, CONFIG_T::table_size>(table);
    } else if (CONFIG_T::activation == FUSED_SOFTSIGN) {
        init_softsign_table<CONFIG_T, CONFIG_T::table_size>(table);
    } else if (CONFIG_T::activation == FUSED_SELU) {
        init_selu_table<CONFIG_T, CONFIG_T::table_size>(table);
    } else if (CONFIG_T::activation == FUSED_ELU) {
        init_elu_table<CONFIG_T, CONFIG_T::table_size>(table);
    }
}

// One value through the folded activation. Each branch is the matching function of nnet_activation.h.
template <class in_T, class res_T, typename CONFIG_T>
inline res_T fused_activate(in_T value, const typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    #pragma HLS INLINE

    if (CONFIG_T::activation == FUSED_RELU) {
        return value > 0 ? (res_T)value : (res_T)0;

    } else if (CONFIG_T::activation == FUSED_SIGMOID || CONFIG_T::activation == FUSED_SOFTPLUS ||
               CONFIG_T::activation == FUSED_SOFTSIGN) {
        int index = (int)(value * (int)CONFIG_T::table_size / 16) + 8 * (int)CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > (int)CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        return (res_T)table[index];

    } else if (CONFIG_T::activation == FUSED_TANH) {
        int index = (int)(value * (int)CONFIG_T::table_size / 8) + 4 * (int)CONFIG_T::table_size / 8;
        if (index < 0)
            index = 0;
        if (index > (int)CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        return (res_T)table[index];

    } else if (CONFIG_T::activation == FUSED_SELU) {
        typedef ap_ufixed<16, 1> selu_const_t;
        const selu_const_t lambda = 1.0507009873554805;
        if (value >= 0) {
            return (res_T)(lambda * value);
        }
        int index = (int)(value * (int)CONFIG_T::table_size / -8);
        if (index < 0)
            index = 0;
        if (index > (int)CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        return (res_T)table[index];

    } else if (CONFIG_T::activation == FUSED_ELU) {
        if (value >= 0) {
            return (res_T)value;
        }
        int index = (int)(value * (int)CONFIG_T::table_size / -8);
        if (index > (int)CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        return (res_T)(CONFIG_T::activation_param * table[index]);

    } else if (CONFIG_T::activation == FUSED_LEAKY_RELU) {
        return value > 0 ? (res_T)value : (res_T)(CONFIG_T::activation_param * value);

    } else if (CONFIG_T::activation == FUSED_THRESHOLDED_RELU) {
        return value > CONFIG_T::activation_param ? (res_T)value : (res_T)0;

    } else if (CONFIG_T::activation == FUSED_HARD_SIGMOID) {
        auto scaled = CONFIG_T::slope * value + CONFIG_T::shift;
        if (scaled > 1)
            scaled = 1;
        else if (scaled < 0)
            scaled = 0;
        return (res_T)scaled;

    } else if (CONFIG_T::activation == FUSED_HARD_TANH) {
        auto scaled = CONFIG_T::slope * value + CONFIG_T::shift;
        if (scaled > 1)
            scaled = 1;
        else if (scaled < 0)
            scaled = 0;
        return (res_T)(2 * scaled - 1);

    } else if (CONFIG_T::activation == FUSED_BINARY_TANH) {
        ap_int<2> sign = value >= 0 ? 1 : -1;
        return binary_cast<ap_int<2>, res_T>(sign);
    }

    return (res_T)value; // FUSED_LINEAR
}

// Array in, array out: the leading layer of a chain of odd length.
template <class data_T, class res_T, typename CONFIG_T>
void dense_fused(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                 typename CONFIG_T::weight_t weights[CONFIG_T::n_out * CONFIG_T::n_in],
                 typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    const unsigned PAR = CONFIG_T::multiplier_limit;
    #pragma HLS ARRAY_RESHAPE variable=weights cyclic factor=PAR dim=1

    typename CONFIG_T::table_t table[CONFIG_T::table_size];
    fused_init_table<CONFIG_T>(table);

    // Static so that no loop is needed to clear the sums before each run: they start at zero, and the
    // result loop adds the bias and sets them back to zero for the next run.
    static typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc cyclic factor=PAR

FusedAccum:
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
    FusedLanes:
        for (unsigned jb = 0; jb < CONFIG_T::n_out; jb += PAR) {
            #pragma HLS PIPELINE II=1
            for (unsigned p = 0; p < PAR; p++) {
                #pragma HLS UNROLL
                unsigned j = jb + p;
                if (j < CONFIG_T::n_out) {
                    acc[j] += CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(
                        data[i], weights[i * CONFIG_T::n_out + j]);
                }
            }
        }
    }

FusedResult:
    for (unsigned j = 0; j < CONFIG_T::n_out; j++) {
        #pragma HLS UNROLL factor=PAR
        typename CONFIG_T::accum_t sum = acc[j] + (typename CONFIG_T::accum_t)biases[j];
        acc[j] = 0;
        typename CONFIG_T::preact_t value = cast<data_T, typename CONFIG_T::preact_t, CONFIG_T>(sum);
        res[j] = fused_activate<typename CONFIG_T::preact_t, res_T, CONFIG_T>(value, table);
    }

    FUSED_PAD(CONFIG_T::pad_cycles)
}

// Array in, stream out: each output is written as soon as it is finished, so the next layer can start.
template <class data_T, class res_T, typename CONFIG_T>
void dense_fused_dot(data_T data[CONFIG_T::n_in], hls::stream<res_T> &res,
                     typename CONFIG_T::weight_t weights[CONFIG_T::n_out * CONFIG_T::n_in],
                     typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    const unsigned PAR = CONFIG_T::multiplier_limit;
    #pragma HLS ARRAY_RESHAPE variable=weights cyclic factor=PAR dim=1
    #pragma HLS ARRAY_RESHAPE variable=data cyclic factor=PAR dim=1

    typename CONFIG_T::table_t table[CONFIG_T::table_size];
    fused_init_table<CONFIG_T>(table);

FusedDotOut:
    for (unsigned j = 0; j < CONFIG_T::n_out; j++) {
        typename CONFIG_T::accum_t part[PAR];
        #pragma HLS ARRAY_PARTITION variable=part complete

    FusedDotAccum:
        for (unsigned i = 0; i < CONFIG_T::n_in; i += PAR) {
            #pragma HLS PIPELINE II=1
            for (unsigned p = 0; p < PAR; p++) {
                #pragma HLS UNROLL
                if (i + p < CONFIG_T::n_in) {
                    typename CONFIG_T::accum_t term =
                        CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(
                            data[i + p], weights[j * CONFIG_T::n_in + i + p]);
                    part[p] = (i == 0) ? term : (typename CONFIG_T::accum_t)(part[p] + term);
                } else if (i == 0) {
                    part[p] = (typename CONFIG_T::accum_t)0;
                }
            }
        }

        typename CONFIG_T::accum_t acc = (typename CONFIG_T::accum_t)biases[j];
    FusedDotDrain:
        for (unsigned p = 0; p < PAR; p++) {
            #pragma HLS UNROLL
            acc += part[p];
        }
        typename CONFIG_T::preact_t value = cast<data_T, typename CONFIG_T::preact_t, CONFIG_T>(acc);
        res.write(fused_activate<typename CONFIG_T::preact_t, res_T, CONFIG_T>(value, table));
    }

    FUSED_PAD(CONFIG_T::pad_cycles)
}

// Stream in, array out. Each value is used as soon as it arrives. Weights w[i * n_out + j].
template <class data_T, class res_T, typename CONFIG_T>
void dense_fused_axpy(hls::stream<data_T> &data, res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    const unsigned PAR = CONFIG_T::multiplier_limit;

    // Static, and set back to zero by the result loop
    static typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc cyclic factor=PAR
    #pragma HLS ARRAY_RESHAPE variable=weights cyclic factor=PAR dim=1

    typename CONFIG_T::table_t table[CONFIG_T::table_size];
    fused_init_table<CONFIG_T>(table);

FusedAxpyAccum:
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        data_T cache = data.read();
    FusedAxpyLanes:
        for (unsigned jb = 0; jb < CONFIG_T::n_out; jb += PAR) {
            #pragma HLS PIPELINE II=1
            for (unsigned p = 0; p < PAR; p++) {
                #pragma HLS UNROLL
                unsigned j = jb + p;
                if (j < CONFIG_T::n_out) {
                    acc[j] += CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(
                        cache, weights[i * CONFIG_T::n_out + j]);
                }
            }
        }
    }

// Adding the bias and applying the activation during the last input instead would save n_out / PAR cycles,
// but it lengthens the critical path of that loop and timing then fails.
FusedAxpyResult:
    for (unsigned j = 0; j < CONFIG_T::n_out; j++) {
        #pragma HLS UNROLL factor=PAR
        typename CONFIG_T::accum_t sum = acc[j] + (typename CONFIG_T::accum_t)biases[j];
        acc[j] = 0;
        typename CONFIG_T::preact_t value = cast<data_T, typename CONFIG_T::preact_t, CONFIG_T>(sum);
        res[j] = fused_activate<typename CONFIG_T::preact_t, res_T, CONFIG_T>(value, table);
    }

    FUSED_PAD(CONFIG_T::pad_cycles)
}

} // namespace nnet

#endif
