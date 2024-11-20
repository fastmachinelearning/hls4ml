#ifndef NNET_DEPTH_CONV2D_H_
#define NNET_DEPTH_CONV2D_H_

#include "nnet_common.h"
#include "nnet_conv2d.h"
#include "nnet_depthconv2d_resource.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_cl(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
                          const typename CONFIG_T::bias_t &biases) {

    depthwise_conv_2d_resource_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

} // namespace nnet

#endif
