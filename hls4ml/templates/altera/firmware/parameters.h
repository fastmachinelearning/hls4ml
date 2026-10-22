#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"

// Compile-time transpose
template <class arr_T, unsigned orig_rows, unsigned orig_cols> constexpr auto tpose(const arr_T &arr) {
    arr_T result;

    for (unsigned r = 0; r < orig_rows; r++) {
        for (unsigned c = 0; c < orig_cols; c++) {
            result[c * orig_rows + r] = arr[r * orig_cols + c];
        }
    }
    return result;
}

// hls-fpga-machine-learning insert includes

// hls-fpga-machine-learning insert softmax tables

// hls-fpga-machine-learning insert layer-config

#endif
