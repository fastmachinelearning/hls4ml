#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include <iostream>
#include "nnet_helpers.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
class FillLineBuffer1D{
    public:
    static void fill_line(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T line[CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned pixel_idx
    ) {
        // To be implemented in subclasses
    }
};

template<class data_T, typename CONFIG_T>
class FillLineBuffer2D{
    public:
    static void fill_line(
        data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T line[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned pixel_idx
    ) {
        // To be implemented in subclasses
    }
};

//hls4ml insert instructions

}

#endif