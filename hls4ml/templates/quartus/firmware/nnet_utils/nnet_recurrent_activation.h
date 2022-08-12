#ifndef NNET_RECR_ACTIVATION_H_
#define NNET_RECR_ACTIVATION_H_

#include "nnet_common.h"
#include "nnet_activation.h"

namespace nnet {

namespace activation {

template<class data_T, class res_T, typename CONFIG_T>
class Activation {
    public:
    // *************************************************
    //       Blank Activation
    // *************************************************
    static void activation(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {}
};

template<class data_T, class res_T, typename CONFIG_T>
class relu : public Activation<data_T, res_T, CONFIG_T> {
    public:
    // *************************************************
    //       Relu Activation
    // *************************************************
    static void activation(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
        nnet::relu<data_T, res_T, CONFIG_T>(data, res);   
    }
};

template<class data_T, class res_T, typename CONFIG_T>
class sigmoid : public Activation<data_T, res_T, CONFIG_T>{
    public:
    // *************************************************
    //       Sigmoid Activation
    // *************************************************
    static void activation(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
        nnet::sigmoid<data_T, res_T, CONFIG_T>(data, res);
    }
};

template<class data_T, class res_T, typename CONFIG_T>
class tanh : public Activation<data_T, res_T, CONFIG_T>{
    public:
    // *************************************************
    //       TanH Activation
    // *************************************************
    static void activation(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
        nnet::dense_tanh<data_T, res_T, CONFIG_T>(data, res);
    }
};

}

}

#endif