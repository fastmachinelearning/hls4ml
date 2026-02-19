#include <hls_stream.h>
#include <iostream>
#include <stdint.h>
#include "MY_PROJECT_DM_INC.h"

#define STREAM_BUF_IN_SZ VAL
#define STREAM_BUF_OUT_SZ VAL

template <typename ATOMIC_TYPE, typename INPUT_LAYER_ARR>
void load_input(ATOMIC_TYPE *in, hls::stream<INPUT_LAYER_ARR> &inStream, int batch_size, const int TENSOR_SIZE) {
mem_rd:
    int baseQuery = 0;
    for (int q = 0; q < batch_size; q++) {
        for (int i = 0; i < TENSOR_SIZE / INPUT_LAYER_ARR::size; i++) {
            INPUT_LAYER_ARR tmp;
            for (int j = 0; j < INPUT_LAYER_ARR::size; j++) {
                tmp[j] = in[baseQuery];
                baseQuery++;
            }
            inStream.write(tmp);
        }
    }
}

template <typename ATOMIC_TYPE, typename OUT_LAYER_ARR>
void store_result(ATOMIC_TYPE *out, hls::stream<OUT_LAYER_ARR> &out_stream, int batch_size, const int TENSOR_SIZE) {
mem_wr:
    int baseQuery = 0;
    for (int q = 0; q < batch_size; q++) {
        for (int i = 0; i < TENSOR_SIZE / OUT_LAYER_ARR::size; i++) {
            OUT_LAYER_ARR tmp = out_stream.read();
            for (int j = 0; j < OUT_LAYER_ARR::size; j++) {
                out[baseQuery] = tmp[j];
                baseQuery++;
            }
        }
    }
}

// vitis-unified-wrapper-compute-func
void compute(// vitis-unified-wrapper-compute-signature) {
    for (int q = 0; q < batch_size; q++) {
        // vitis-unified-wrapper-compute-body
    }
}

void MY_PROJECT_TOP_FUNC(
    // vitis-unified-wrapper-io
    , int batch_size

) {

    // vitis-unified-wrapper-interface

    // vitis-unified-wrapper-stream-dec


    // vitis-unified-wrapper-stream-config

    #pragma HLS dataflow

    // vitis-unified-wrapper-load

    compute(// vitis-unified-wrapper-compute-call-args);

    // vitis-unified-wrapper-store
}
