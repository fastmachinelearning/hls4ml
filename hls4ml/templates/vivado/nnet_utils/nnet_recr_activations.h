#ifndef NNET_RECR_ACTIVATION_H_
#define NNET_RECR_ACTIVATION_H_

#include "hls_stream.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_helpers.h"
#include <math.h>

namespace nnet {

namespace activation {

// Each activation computes either an array, with activation(), or one value, with single(), for cells that update one
// unit after another. single() gives the same result as activation() and reads a table filled by init_table().

template <class data_T, class res_T, typename CONFIG_T> class Activation {
  public:
    // *************************************************
    //       Blank Activation
    // *************************************************
    static void activation(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {} // Nothing to do here
    static void init_table(typename CONFIG_T::table_t table[CONFIG_T::table_size]) {}
};

template <class data_T, class res_T, typename CONFIG_T> class linear : public Activation<data_T, res_T, CONFIG_T> {
  public:
    // *************************************************
    //       Linear Activation
    // *************************************************
    static void activation(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
        nnet::linear<data_T, res_T, CONFIG_T>(data, res);
    }
    static res_T single(data_T data, typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
        #pragma HLS INLINE
        return data;
    }
};

template <class data_T, class res_T, typename CONFIG_T> class relu : public Activation<data_T, res_T, CONFIG_T> {
  public:
    // *************************************************
    //       Relu Activation
    // *************************************************
    static void activation(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
        nnet::relu<data_T, res_T, CONFIG_T>(data, res);
    }
    static res_T single(data_T data, typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
        #pragma HLS INLINE
        if (data > 0)
            return data;
        else
            return 0;
    }
};

template <class data_T, class res_T, typename CONFIG_T> class sigmoid : public Activation<data_T, res_T, CONFIG_T> {
  public:
    // *************************************************
    //       Sigmoid Activation
    // *************************************************
    static void activation(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
        nnet::sigmoid<data_T, res_T, CONFIG_T>(data, res);
    }
    static void init_table(typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(table);
    }
    static res_T single(data_T data, typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
        #pragma HLS INLINE
        int data_round = data * CONFIG_T::table_size / 16;
        int index = data_round + 8 * CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        return (res_T)table[index];
    }
};

template <class data_T, class res_T, typename CONFIG_T> class tanh : public Activation<data_T, res_T, CONFIG_T> {
  public:
    // *************************************************
    //       TanH Activation
    // *************************************************
    static void activation(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
        nnet::tanh<data_T, res_T, CONFIG_T>(data, res);
    }
    static void init_table(typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>(table);
    }
    static res_T single(data_T data, typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
        #pragma HLS INLINE
        int data_round = data * CONFIG_T::table_size / 8;
        int index = data_round + 4 * CONFIG_T::table_size / 8;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        return (res_T)table[index];
    }
};

} // namespace activation

} // namespace nnet

#endif
