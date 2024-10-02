#ifndef NNET_SEPARABLE_CONV2D_H_
#define NNET_SEPARABLE_CONV2D_H_

#include "nnet_common.h"
#include "nnet_conv2d.h"
#include "nnet_sepconv2d_latency.h"


namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_cl(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
                const typename CONFIG_T::bias_t &biases) {

    depthwise_conv_2d_latency_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);


}
/*
template <class data_T, class dw_res_T, class res_T, typename CONFIG_T>
void separable_conv_2d_cl(const data_t &data, res_T &res, const typename CONFIG_T::weight_t &d_weights,
                const typename CONFIG_T::bias_t &d_biases, const typename CONFIG_T::weight_t &p_weights,
                const typename CONFIG_T::bias_t &p_biases) {

    CONFIG::dw_res_T depthwise_res;

                  depthwise_conv_2d_cl<data_T, dw_res_T, CONFIG_T::depthwise_config>(data, depthwise_res, d_weights, d_biases);
                  pointwise_conv_2d_cl<dw_res_T, res_T, CONFIG_T::pointwise_config>(depthwise_res, res, p_weights, p_biases); 



}

*/

} // namespace nnet


#endif
