#ifndef NNET_DEPTH_CONV1D_H_
#define NNET_DEPTH_CONV1D_H_

#include "nnet_common.h"
#include "nnet_conv1d.h"
#include "nnet_depthconv1d_resource.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_1d_cl(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
                          const typename CONFIG_T::bias_t &biases) {

    depthwise_conv_1d_resource_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

} // namespace nnet

#endif
