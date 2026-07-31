#ifndef MODEL_WRAPPER_HPP_
#define MODEL_WRAPPER_HPP_

#include "ap_axi_sdata.h"
#include "hls_stream.h"

#define COYOTE_AXI_STREAM_BITS 512
typedef ap_axiu<COYOTE_AXI_STREAM_BITS, 0, 0, 0> axi_s;

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_axi_utils.h"
#include "firmware/nnet_utils/nnet_axi_utils_stream.h"

void model_wrapper(hls::stream<axi_s> &data_in, hls::stream<axi_s> &data_out);

#endif
