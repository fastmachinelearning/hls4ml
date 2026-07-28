#ifndef NNET_ACTIVATION_H_
#define NNET_ACTIVATION_H_

#include "ap_fixed.h"
#include "gcem/include/gcem.hpp"
#include "nnet_common.h"
#include <array>
#include <cmath>
#include <limits>

namespace nnet {

struct activ_config {
    // IO size
    static const unsigned n_in = 10;

    // Internal info
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef ap_fixed<18, 8> table_t;
};

// *************************************************
//       LINEAR Activation -- See Issue 53
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void linear(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        res[ii] = data[ii];
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void relu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = 0;
    }
}

template <class data_T, class res_T, int MAX_INT, typename CONFIG_T>
void relu_max(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
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

constexpr inline float exp_with_clamp_fcn_float(float input) {
    // Keep constexpr table generation finite for large-magnitude inputs.
    // Without this clamp, exp() can overflow to Inf and break ac_fixed conversion.
    constexpr float max_exp_input = 80.0f;
    constexpr float min_exp_input = -80.0f;
    const float clamped_input = (input > max_exp_input) ? max_exp_input : ((input < min_exp_input) ? min_exp_input : input);
    using gcem::exp;
    return exp(clamped_input);
}

// *************************************************
//       Sigmoid Activation
// *************************************************
constexpr inline float sigmoid_fcn_float(float input) { return 1.0 / (1 + exp_with_clamp_fcn_float(-input)); }
#ifdef OLD_SIGMOID
template <typename CONFIG_T, int N_TABLE> void init_sigmoid_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default logistic sigmoid function:
    //   result = 1/(1+e^(-x))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = sigmoid_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}
#else
template <typename CONFIG_T, std::size_t N_TABLE>
constexpr typename CONFIG_T::table_t compute_sigmoid_fcn_float_index(size_t ii) {
    // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
    float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
    // Next, compute lookup table function
    typename CONFIG_T::table_t real_val = sigmoid_fcn_float(in_val);
    return real_val;
}

template <typename CONFIG_T, std::size_t N, std::size_t... I>
constexpr static std::array<typename CONFIG_T::table_t, sizeof...(I)> init_sigmoid_table(std::index_sequence<I...>) {
    return std::array<typename CONFIG_T::table_t, sizeof...(I)>{compute_sigmoid_fcn_float_index<CONFIG_T, N>(I)...};
}

template <typename CONFIG_T, std::size_t N> constexpr static std::array<typename CONFIG_T::table_t, N> init_sigmoid_table() {
    return init_sigmoid_table<CONFIG_T, N>(std::make_index_sequence<N>{});
}
#endif

template <class data_T, class res_T, typename CONFIG_T>
void sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef OLD_SIGMOID
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
#else
    static constexpr const ::std::array<typename CONFIG_T::table_t, CONFIG_T::table_size> sigmoid_table =
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>();
#endif
    //#pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii] * CONFIG_T::table_size / 16;
        index = data_round + 8 * CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = (res_T)sigmoid_table[index];
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

enum class softmax_implementation { latency = 0, legacy = 1, stable = 2, argmax = 3 };

constexpr inline float exp_fcn_float(float input) { return exp_with_clamp_fcn_float(input); }

template <class data_T, unsigned table_size> constexpr inline float softmax_real_val_from_idx(unsigned i) {
    // Treat the index as the top N bits
    constexpr int N = ceillog2(table_size); // number of address bits for table
    data_T x(0);
    x(x.width - 1, x.width - N) = i;
    return (float)x;
}

template <class data_T, unsigned table_size> constexpr inline unsigned softmax_idx_from_real_val(data_T x) {
    // Slice the top N bits to get an index into the table
    constexpr int N = ceillog2(table_size);     // number of address bits for table
    ap_uint<N> y = x(x.width - 1, x.width - N); // slice the top N bits of input
    return (unsigned)y(N - 1, 0);
}

#ifdef OLD_EXP
template <class data_T, typename CONFIG_T>
void init_exp_table(typename CONFIG_T::exp_table_t table_out[CONFIG_T::exp_table_size], bool negative = false) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < CONFIG_T::exp_table_size; i++) {
        // Slicing bits for address is going to round towards 0, so take the central value
        float x = softmax_real_val_from_idx<data_T, CONFIG_T::exp_table_size>(i) * CONFIG_T::exp_scale;
        if (negative) {
            // for normalized inputs, we keep the normalization values positive (x_bar = x_max - x)
            // so we need to negate the input (exp(-x_bar) = exp(x - x_max))
            x = -x;
        }
        typename CONFIG_T::exp_table_t exp_x = exp_fcn_float(x);
        table_out[i] = exp_x;
    }
}
#else
template <class data_T, typename CONFIG_T, bool negative>
constexpr typename CONFIG_T::exp_table_t compute_exp_index(size_t i) {
    float x = softmax_real_val_from_idx<data_T, CONFIG_T::exp_table_size>(i) * CONFIG_T::exp_scale;
    if (negative) {
        // for normalized inputs, we keep the normalization values positive (x_bar = x_max - x)
        // so we need to negate the input (exp(-x_bar) = exp(x - x_max))
        x = -x;
    }
    typename CONFIG_T::exp_table_t exp_x = exp_fcn_float(x);
    return exp_x;
}

template <class data_T, bool negative, typename CONFIG_T, std::size_t... I>
constexpr static std::array<typename CONFIG_T::exp_table_t, sizeof...(I)> init_exp_table(std::index_sequence<I...>) {
    return std::array<typename CONFIG_T::exp_table_t, sizeof...(I)>{compute_exp_index<data_T, CONFIG_T, negative>(I)...};
}

template <class data_T, typename CONFIG_T, bool negative>
constexpr static std::array<typename CONFIG_T::exp_table_t, CONFIG_T::exp_table_size> init_exp_table() {
    return init_exp_table<data_T, negative, CONFIG_T>(std::make_index_sequence<CONFIG_T::exp_table_size>{});
}

#endif

#ifdef OLD_INVERT
template <class data_T, typename CONFIG_T>
void init_invert_table(typename CONFIG_T::inv_table_t table_out[CONFIG_T::inv_table_size]) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < CONFIG_T::inv_table_size; i++) {
        float x = softmax_real_val_from_idx<data_T, CONFIG_T::inv_table_size>(i);
        typename CONFIG_T::inv_table_t inv_x = 1 / x;
        table_out[i] = inv_x;
    }
}
#else
template <class data_T, typename CONFIG_T> constexpr typename CONFIG_T::inv_table_t compute_inv_index(size_t i) {
    float x = softmax_real_val_from_idx<data_T, CONFIG_T::inv_table_size>(i);
    float safe_x = (x == 0.0f) ? std::numeric_limits<float>::min() : x;
    typename CONFIG_T::inv_table_t inv_x = 1 / safe_x;
    return inv_x;
}

template <class data_T, typename CONFIG_T, std::size_t... I>
constexpr static std::array<typename CONFIG_T::inv_table_t, sizeof...(I)> init_inv_table(std::index_sequence<I...>) {
    return std::array<typename CONFIG_T::inv_table_t, sizeof...(I)>{compute_inv_index<data_T, CONFIG_T>(I)...};
}

template <class data_T, typename CONFIG_T>
constexpr static std::array<typename CONFIG_T::inv_table_t, CONFIG_T::inv_table_size> init_inv_table() {
    return init_inv_table<data_T, CONFIG_T>(std::make_index_sequence<CONFIG_T::inv_table_size>{});
}

#endif

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(data_T data[CONFIG_T::n_slice], res_T res[CONFIG_T::n_slice]) {
    //#pragma HLS pipeline
    // Initialize the lookup tables
#ifdef OLD_EXP
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<data_T, CONFIG_T>(exp_table);
        initialized = true;
    }
#else
    static constexpr const ::std::array<typename CONFIG_T::exp_table_t, CONFIG_T::exp_table_size> exp_table =
        init_exp_table<data_T, CONFIG_T, false>();
#endif
#ifdef OLD_INVERT
#ifdef __HLS_SYN__
    bool initializedinv = false;
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];
#else
    static bool initializedinv = false;
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];

#endif
    if (!initializedinv) {
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::inv_table_t, CONFIG_T>(invert_table);
        initializedinv = true;
    }
#else
    static constexpr const ::std::array<typename CONFIG_T::inv_table_t, CONFIG_T::inv_table_size> invert_table =
        init_inv_table<typename CONFIG_T::inv_inp_t, CONFIG_T>();
#endif
    // Calculate all the e^x's
    typename CONFIG_T::accum_t exp_res[CONFIG_T::n_slice];
    #pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::inv_inp_t exp_sum(0);
    #pragma clang loop unroll(full)
    for (unsigned i = 0; i < CONFIG_T::n_slice; i++) {
        unsigned x = softmax_idx_from_real_val<data_T, CONFIG_T::exp_table_size>(data[i]);
        exp_res[i] = exp_table[x];
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::accum_t> op_add;
    exp_sum = reduce<typename CONFIG_T::accum_t, CONFIG_T::n_slice, Op_add<typename CONFIG_T::accum_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table[softmax_idx_from_real_val<typename CONFIG_T::inv_inp_t, CONFIG_T::inv_table_size>(exp_sum)];
    #pragma clang loop unroll(full)
    for (unsigned i = 0; i < CONFIG_T::n_slice; i++) {
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(data_T data[CONFIG_T::n_slice], res_T res[CONFIG_T::n_slice]) {
    //#pragma HLS pipeline
    // Initialize the lookup tables
#ifdef OLD_EXP
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<typename CONFIG_T::inp_norm_t, CONFIG_T>(exp_table, true);
        initialized = true;
    }
#else
    static constexpr const ::std::array<typename CONFIG_T::exp_table_t, CONFIG_T::exp_table_size> exp_table =
        init_exp_table<typename CONFIG_T::inp_norm_t, CONFIG_T, true>();
#endif
#ifdef OLD_INVERT
#ifdef __HLS_SYN__
    bool initializedinv = false;
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];
#else
    static bool initializedinv = false;
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];

#endif
    if (!initializedinv) {
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::inv_inp_t, CONFIG_T>(invert_table);
        initializedinv = true;
    }
#else
    static constexpr const ::std::array<typename CONFIG_T::inv_table_t, CONFIG_T::inv_table_size> invert_table =
        init_inv_table<typename CONFIG_T::inv_inp_t, CONFIG_T>();
#endif

    // Find the max and compute all delta(x_i, x_max)
    Op_max<data_T> op_max;
    data_T x_max = reduce<data_T, CONFIG_T::n_slice, Op_max<data_T>>(data, op_max);

    typename CONFIG_T::inp_norm_t d_xi_xmax[CONFIG_T::n_slice];
    #pragma clang loop unroll(full)
    for (unsigned i = 0; i < CONFIG_T::n_slice; i++) {
        d_xi_xmax[i] = x_max - data[i];
    }

    // Calculate all the e^x's
    typename CONFIG_T::accum_t exp_res[CONFIG_T::n_slice];
    #pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::inv_inp_t exp_sum(0);
    #pragma clang loop unroll(full)
    for (unsigned i = 0; i < CONFIG_T::n_slice; i++) {
        unsigned x = softmax_idx_from_real_val<typename CONFIG_T::inp_norm_t, CONFIG_T::exp_table_size>(d_xi_xmax[i]);
        exp_res[i] = exp_table[x];
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::accum_t> op_add;
    exp_sum = reduce<typename CONFIG_T::accum_t, CONFIG_T::n_slice, Op_add<typename CONFIG_T::accum_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table[softmax_idx_from_real_val<typename CONFIG_T::inv_inp_t, CONFIG_T::inv_table_size>(exp_sum)];
    #pragma clang loop unroll(full)
    for (unsigned i = 0; i < CONFIG_T::n_slice; i++) {
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

// Compile-time constexpr helpers for softmax_legacy (default path)
#ifndef OLD_SOFTMAX_LEGACY
template <typename CONFIG_T, std::size_t N_TABLE>
constexpr typename CONFIG_T::table_t compute_exp_fcn_float_index_legacy(size_t ii) {
    float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
    typename CONFIG_T::table_t real_val = exp_fcn_float(in_val);
    return real_val;
}

template <typename CONFIG_T, std::size_t N, std::size_t... I>
constexpr static std::array<typename CONFIG_T::table_t, sizeof...(I)> init_exp_table_legacy(std::index_sequence<I...>) {
    return std::array<typename CONFIG_T::table_t, sizeof...(I)>{compute_exp_fcn_float_index_legacy<CONFIG_T, N>(I)...};
}

template <typename CONFIG_T, std::size_t N>
constexpr static std::array<typename CONFIG_T::table_t, N> init_exp_table_legacy() {
    return init_exp_table_legacy<CONFIG_T, N>(std::make_index_sequence<N>{});
}

template <typename CONFIG_T, std::size_t N_TABLE>
constexpr typename CONFIG_T::table_t compute_invert_fcn_float_index_legacy(size_t ii) {
    float in_val = 64.0 * ii / float(N_TABLE);
    typename CONFIG_T::table_t real_val = (in_val > 0.0) ? (1.0 / in_val) : 0.0;
    return real_val;
}

template <typename CONFIG_T, std::size_t N, std::size_t... I>
constexpr static std::array<typename CONFIG_T::table_t, sizeof...(I)> init_invert_table_legacy(std::index_sequence<I...>) {
    return std::array<typename CONFIG_T::table_t, sizeof...(I)>{compute_invert_fcn_float_index_legacy<CONFIG_T, N>(I)...};
}

template <typename CONFIG_T, std::size_t N>
constexpr static std::array<typename CONFIG_T::table_t, N> init_invert_table_legacy() {
    return init_invert_table_legacy<CONFIG_T, N>(std::make_index_sequence<N>{});
}
#endif

// Runtime init functions (for backward compatibility with OLD_SOFTMAX_LEGACY)
#ifdef OLD_SOFTMAX_LEGACY
template <typename CONFIG_T, int N_TABLE> void init_exp_table_legacy(typename CONFIG_T::table_t table_out[N_TABLE]) {
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = exp_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <typename CONFIG_T, int N_TABLE> void init_invert_table_legacy(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Inversion function:
    //   result = 1/x
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +64)
        float in_val = 64.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        if (in_val > 0.0)
            table_out[ii] = 1.0 / in_val;
        else
            table_out[ii] = 0.0;
    }
}
#endif

template <class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(data_T data[CONFIG_T::n_slice], res_T res[CONFIG_T::n_slice]) {
    // Initialize the lookup tables
#ifdef OLD_SOFTMAX_LEGACY
    // Runtime initialization for backward compatibility
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t exp_table[CONFIG_T::exp_table_size];
    typename CONFIG_T::table_t invert_table[CONFIG_T::inv_table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t exp_table[CONFIG_T::exp_table_size];
    static typename CONFIG_T::table_t invert_table[CONFIG_T::inv_table_size];
#endif
    if (!initialized) {
        init_exp_table_legacy<CONFIG_T, CONFIG_T::exp_table_size>(exp_table);
        init_invert_table_legacy<CONFIG_T, CONFIG_T::inv_table_size>(invert_table);
        initialized = true;
    }
#else
    // Compile-time initialization (default)
    static constexpr const ::std::array<typename CONFIG_T::table_t, CONFIG_T::exp_table_size> exp_table =
        init_exp_table_legacy<CONFIG_T, CONFIG_T::exp_table_size>();
    static constexpr const ::std::array<typename CONFIG_T::table_t, CONFIG_T::inv_table_size> invert_table =
        init_invert_table_legacy<CONFIG_T, CONFIG_T::inv_table_size>();
#endif

    //#pragma HLS PIPELINE

    // [rest of softmax_legacy implementation remains the same]
    typename CONFIG_T::table_t exp_res[CONFIG_T::n_slice];
    typename CONFIG_T::table_t exp_diff_res;
    data_T data_cache[CONFIG_T::n_slice];
    int data_round;
    int index;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_slice; ii++) {
        data_cache[ii] = data[ii];
        exp_res[ii] = 0;
    }

    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_slice; ii++) {
        #pragma clang loop unroll(full)
        for (int jj = 0; jj < CONFIG_T::n_slice; jj++) {
            if (ii == jj)
                exp_diff_res = 1;
            else {
                data_round = (data_cache[jj] - data_cache[ii]) * CONFIG_T::exp_table_size / 16;
                index = data_round + 8 * CONFIG_T::exp_table_size / 16;
                if (index < 0)
                    index = 0;
                if (index > CONFIG_T::exp_table_size - 1)
                    index = CONFIG_T::exp_table_size - 1;
                exp_diff_res = exp_table[index];
            }
            exp_res[ii] += exp_diff_res;
        }
    }

    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_slice; ii++) {
        int exp_res_index = exp_res[ii] * CONFIG_T::inv_table_size / 64;
        if (exp_res_index < 0)
            exp_res_index = 0;
        if (exp_res_index > CONFIG_T::inv_table_size - 1)
            exp_res_index = CONFIG_T::inv_table_size - 1;
        res[ii] = (res_T)invert_table[exp_res_index];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_argmax(data_T data[CONFIG_T::n_slice], res_T res[CONFIG_T::n_slice]) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < CONFIG_T::n_slice; i++) {
        res[i] = (res_T)0;
    }

    data_T maximum = data[0];
    int idx = 0;

    for (int i = 1; i < CONFIG_T::n_slice; i++) {
        //#pragma HLS PIPELINE
        if (data[i] > maximum) {
            maximum = data[i];
            idx = i;
        }
    }

    res[idx] = (res_T)1;
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax(data_T data[CONFIG_T::n_slice], res_T res[CONFIG_T::n_slice]) {
    #pragma HLS inline
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
    case softmax_implementation::argmax:
        softmax_argmax<data_T, res_T, CONFIG_T>(data, res);
        break;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_multidim(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS inline
    //#pragma HLS allocation instances = softmax<CONFIG_T> limit = CONFIG_T::parallelization_factor function
    data_T buffer_in[CONFIG_T::n_slice];
    res_T buffer_out[CONFIG_T::n_slice];
    #pragma clang loop unroll(full)
    for (signed i = 0; i < CONFIG_T::n_outer; i++) {
        //#pragma HLS UNROLL
        #pragma clang loop unroll(full)
        for (signed k = 0; k < CONFIG_T::n_inner; k++) {
            //#pragma HLS UNROLL
            #pragma clang loop unroll(full)
            for (signed j = 0; j < CONFIG_T::n_slice; j++) {
                //#pragma HLS UNROLL
                buffer_in[j] = data[i * CONFIG_T::n_slice * CONFIG_T::n_inner + j * CONFIG_T::n_inner + k];
            }
            softmax<data_T, res_T, CONFIG_T>(buffer_in, buffer_out);
            #pragma clang loop unroll(full)
            for (signed j = 0; j < CONFIG_T::n_slice; j++) {
                //#pragma HLS UNROLL
                res[i * CONFIG_T::n_slice * CONFIG_T::n_inner + j * CONFIG_T::n_inner + k] = buffer_out[j];
            }
        }
    }
}

// *************************************************
//       TanH Activation
// *************************************************

constexpr inline float tanh_fcn_float(float input) {
    using gcem::tanh;
    return tanh(input);
}
template <typename CONFIG_T, int N_TABLE> void init_tanh_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Implement tanh lookup
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
        float in_val = 2 * 4.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = tanh(in_val);
        table_out[ii] = real_val;
    }
}

template <typename CONFIG_T, std::size_t N_TABLE>
constexpr typename CONFIG_T::table_t compute_tanh_fcn_float_index(size_t ii) {
    float in_val = 2 * 4.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
    // Compute lookup table function
    typename CONFIG_T::table_t real_val = tanh_fcn_float(in_val);
    return real_val;
}

template <typename CONFIG_T, std::size_t N, std::size_t... I>
constexpr static std::array<typename CONFIG_T::table_t, sizeof...(I)> init_tanh_table(std::index_sequence<I...>) {
    return std::array<typename CONFIG_T::table_t, sizeof...(I)>{compute_tanh_fcn_float_index<CONFIG_T, N>(I)...};
}

template <typename CONFIG_T, std::size_t N> constexpr static std::array<typename CONFIG_T::table_t, N> init_tanh_table() {
    return init_tanh_table<CONFIG_T, N>(std::make_index_sequence<N>{});
}

template <class data_T, class res_T, typename CONFIG_T> void tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table at compile time
#ifdef OLD_TANH
    // Keep old runtime initialization for backwards compatibility
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
#else
    // Compile-time initialization
    static constexpr const ::std::array<typename CONFIG_T::table_t, CONFIG_T::table_size> tanh_table =
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>();
#endif

    int data_round;
    int index;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii] * CONFIG_T::table_size / 8;
        index = data_round + 4 * CONFIG_T::table_size / 8;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = (res_T)tanh_table[index];
    }
}

// *************************************************
//       UnaryLUT Activation
// *************************************************
template <int table_size, class data_T> inline unsigned get_index_unary_lut(data_T x) {
    // Slice the top N bits to get an index into the table
    static constexpr int N = ceillog2(table_size);
    return (unsigned)(x(x.width - 1, 0));
}

template <class data_T, class res_T, typename CONFIG_T>
void unary_lut(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in],
               typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    //#pragma HLS function_instantiate variable=table
    #pragma HLS ARRAY_PARTITION variable=table

    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        unsigned index = get_index_unary_lut<CONFIG_T::table_size>(data[ii]);
        res[ii] = (res_T)table[index];
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void hard_sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    #pragma clang loop unroll(full)
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
    if (CONFIG_T::io_type == io_parallel) {
        //#pragma HLS PIPELINE
        /// TO BE RECONSIDERED FF
    }

    #pragma clang loop unroll(full)
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
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void leaky_relu(data_T data[CONFIG_T::n_in], param_T alpha, res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = alpha * datareg;
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void thresholded_relu(data_T data[CONFIG_T::n_in], param_T theta, res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > theta)
            res[ii] = datareg;
        else
            res[ii] = 0;
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
constexpr inline float softplus_fcn_float(float input) {
    using gcem::log;
    return log(exp_with_clamp_fcn_float(input) + 1.);
}

#ifdef OLD_SOFTPLUS

template <typename CONFIG_T, int N_TABLE> void init_softplus_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default softplus function:
    //   result = log(exp(x) + 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = softplus_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}
#else
template <typename CONFIG_T, std::size_t N_TABLE>
constexpr typename CONFIG_T::table_t compute_softplus_fcn_float_index(std::size_t ii) {
    // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
    float in_val = 2 * 8.0f * (static_cast<float>(ii) - float(N_TABLE) / 2.0f) / float(N_TABLE);
    // Next, compute lookup table function
    typename CONFIG_T::table_t real_val = softplus_fcn_float(in_val);
    return real_val;
}

template <typename CONFIG_T, std::size_t N, std::size_t... I>
constexpr static std::array<typename CONFIG_T::table_t, sizeof...(I)> init_softplus_table(std::index_sequence<I...>) {
    return std::array<typename CONFIG_T::table_t, sizeof...(I)>{compute_softplus_fcn_float_index<CONFIG_T, N>(I)...};
}

template <typename CONFIG_T, std::size_t N>
constexpr static std::array<typename CONFIG_T::table_t, N> init_softplus_table() {
    return init_softplus_table<CONFIG_T, N>(std::make_index_sequence<N>{});
}
#endif
template <class data_T, class res_T, typename CONFIG_T>
void softplus(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef OLD_SOFTPLUS
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
#else
    static const ::std::array<typename CONFIG_T::table_t, CONFIG_T::table_size> softplus_table =
        init_softplus_table<CONFIG_T, CONFIG_T::table_size>();
#endif
    //#pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii] * CONFIG_T::table_size / 16;
        index = data_round + 8 * CONFIG_T::table_size / 16;
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
constexpr inline float softsign_fcn_float(float input) {
    using gcem::abs;
    return input / (abs(input) + 1.0f);
}
#ifdef OLD_SOFTSIGN
template <typename CONFIG_T, int N_TABLE> void init_softsign_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default softsign function:
    //   result = x / (abs(x) + 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0f * (ii - float(N_TABLE) / 2.0f) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = softsign_fcn_float(in_val);
        table_out[ii] = real_val;
    }
}
#else
template <typename CONFIG_T, std::size_t N_TABLE>
constexpr typename CONFIG_T::table_t compute_softsign_fcn_float_index(std::size_t ii) {
    // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
    float in_val = 2 * 8.0f * (static_cast<float>(ii) - float(N_TABLE) / 2.0f) / float(N_TABLE);
    // Next, compute lookup table function
    typename CONFIG_T::table_t real_val = softsign_fcn_float(in_val);
    return real_val;
}

template <typename CONFIG_T, std::size_t N, std::size_t... I>
constexpr static std::array<typename CONFIG_T::table_t, sizeof...(I)> init_softsign_table(std::index_sequence<I...>) {
    return std::array<typename CONFIG_T::table_t, sizeof...(I)>{compute_softsign_fcn_float_index<CONFIG_T, N>(I)...};
}

template <typename CONFIG_T, std::size_t N>
constexpr static std::array<typename CONFIG_T::table_t, N> init_softsign_table() {
    return init_softsign_table<CONFIG_T, N>(std::make_index_sequence<N>{});
}
#endif // OLD_SOFTSIGN

template <class data_T, class res_T, typename CONFIG_T>
void softsign(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef OLD_SOFTSIGN
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
#else
    static const ::std::array<typename CONFIG_T::table_t, CONFIG_T::table_size> softsign_table =
        init_softsign_table<CONFIG_T, CONFIG_T::table_size>();
#endif

    // Index into the lookup table based on data
    int data_round;
    int index;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii] * CONFIG_T::table_size / 16;
        index = data_round + 8 * CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = (res_T)softsign_table[index];
    }
}

// *************************************************
//       ELU Activation
// *************************************************
constexpr inline float elu_fcn_float(float input) { return exp_with_clamp_fcn_float(input) - 1.; }

#ifdef OLD_ELU
template <typename CONFIG_T, int N_TABLE> void init_elu_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default ELU function:
    //   result = alpha * (e^(x) - 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to 0)
        float in_val = -8.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = elu_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}
#else
template <typename CONFIG_T, std::size_t N_TABLE>
constexpr typename CONFIG_T::table_t compute_elu_fcn_float_index(size_t ii) {
    // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
    float in_val = -8.0 * ii / float(N_TABLE);
    // Next, compute lookup table function
    typename CONFIG_T::table_t real_val = elu_fcn_float(in_val);
    return real_val;
}

template <typename CONFIG_T, std::size_t N, std::size_t... I>
constexpr static std::array<typename CONFIG_T::table_t, sizeof...(I)> init_elu_table(std::index_sequence<I...>) {
    return std::array<typename CONFIG_T::table_t, sizeof...(I)>{compute_elu_fcn_float_index<CONFIG_T, N>(I)...};
}

template <typename CONFIG_T, std::size_t N> constexpr static std::array<typename CONFIG_T::table_t, N> init_elu_table() {
    return init_elu_table<CONFIG_T, N>(std::make_index_sequence<N>{});
}
#endif

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void elu(data_T data[CONFIG_T::n_in], const param_T alpha, res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef OLD_ELU
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
#else
    static constexpr const ::std::array<typename CONFIG_T::table_t, CONFIG_T::table_size> elu_table =
        init_elu_table<CONFIG_T, CONFIG_T::table_size>();
#endif
    //#pragma HLS PIPELINE

    data_T datareg;
    // Index into the lookup table based on data
    int index;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = datareg;
        } else {
            index = datareg * CONFIG_T::table_size / -8;
            if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            res[ii] = alpha * elu_table[index];
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T> void elu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    elu<data_T, ap_uint<1>, res_T, CONFIG_T>(data, 1.0, res);
}

// *************************************************
//       SELU Activation
// *************************************************
constexpr inline float selu_fcn_float(float input) {
    return 1.0507009873554804934193349852946 * (1.6732632423543772848170429916717 * (exp_with_clamp_fcn_float(input) - 1.));
}

#ifdef OLD_SELU
template <typename CONFIG_T, int N_TABLE> void init_selu_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default SELU function:
    //   result = 1.05 * (1.673 * (e^(x) - 1))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to 0)
        float in_val = -8.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = selu_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}
#else
template <typename CONFIG_T, std::size_t N_TABLE>
constexpr typename CONFIG_T::table_t compute_selu_fcn_float_index(size_t ii) {
    // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
    float in_val = -8.0 * ii / float(N_TABLE);
    // Next, compute lookup table function
    typename CONFIG_T::table_t real_val = selu_fcn_float(in_val);
    return real_val;
}

template <typename CONFIG_T, std::size_t N, std::size_t... I>
constexpr static std::array<typename CONFIG_T::table_t, sizeof...(I)> init_selu_table(std::index_sequence<I...>) {
    return std::array<typename CONFIG_T::table_t, sizeof...(I)>{compute_selu_fcn_float_index<CONFIG_T, N>(I)...};
}

template <typename CONFIG_T, std::size_t N> constexpr static std::array<typename CONFIG_T::table_t, N> init_selu_table() {
    return init_selu_table<CONFIG_T, N>(std::make_index_sequence<N>{});
}
#endif

template <class data_T, class res_T, typename CONFIG_T> void selu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef OLD_SELU
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
#else
    static constexpr const ::std::array<typename CONFIG_T::table_t, CONFIG_T::table_size> selu_table =
        init_selu_table<CONFIG_T, CONFIG_T::table_size>();
#endif

    //#pragma HLS PIPELINE

    typedef ap_ufixed<16, 1> selu_const_t;
    constexpr const selu_const_t lambda = 1.0507009873554805;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];

        if (datareg >= 0) {
            // Positive branch  y = λ · x
            res[ii] = lambda * datareg;
        } else {
            // Negative branch  y = table(x)
            int index = datareg * CONFIG_T::table_size / -8;

            // clamp index to [0, table_size-1]
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1) {
                index = CONFIG_T::table_size - 1;
            }

            res[ii] = selu_table[index];
        }
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void prelu(data_T data[CONFIG_T::n_in], param_T alpha[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = alpha[ii] * datareg;
    }
}

template <class data_T, class res_T>
inline typename std::enable_if<(!std::is_same<res_T, ap_uint<1>>::value), res_T>::type binary_cast(data_T data) {
    return static_cast<res_T>(data);
}

// should choose this via function overloading
template <class data_T, class res_T>
inline typename std::enable_if<(std::is_same<res_T, ap_uint<1>>::value), res_T>::type binary_cast(data_T data) {
    return (data > 0) ? static_cast<res_T>(data) : static_cast<res_T>(0);
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void binary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE
    using cache_T = ap_int<2>;
    data_T datareg;
    cache_T cache;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg >= 0)
            cache = 1;
        else
            cache = -1;

        res[ii] = binary_cast<cache_T, res_T>(cache);
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void ternary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    res_T cache;
    #pragma clang loop unroll(full)
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = 2 * data[ii];
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
