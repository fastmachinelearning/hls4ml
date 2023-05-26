#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include "nnet_helpers.h"

namespace nnet {

template <class data_T, typename CONFIG_T> class FillConv1DBuffer {
  public:
    static void fill_buffer(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                            data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
                            const unsigned partition) {
        // To be implemented in subclasses
    }
};

template <class data_T, typename CONFIG_T> class FillConv2DBuffer {
  public:
    static void
    fill_buffer(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
                const unsigned partition) {
        // To be implemented in subclasses
    }
};

template <class data_T, class res_T, typename CONFIG_T> class DenseResourceUnrolled {
  public:
    static void dense_unrolled(
      data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]  
    ) {
        // To be implemented in subclasses
    }  
};

// hls4ml insert code

} // namespace nnet

#endif
