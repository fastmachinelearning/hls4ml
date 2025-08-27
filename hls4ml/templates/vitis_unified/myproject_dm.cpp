#include <stdint.h>
#include <hls_stream.h>
#include <iostream>
//#include "ap_axi_sdata.h"
#include "MY_PROJECT_DM_INC.h"

#define STREAM_BUF_IN_SZ VAL
#define STREAM_BUF_OUT_SZ VAL


template<typename ATOMIC_TYPE, typename INPUT_LAYER_ARR>
void load_input(ATOMIC_TYPE* in, hls::stream<INPUT_LAYER_ARR>& inStream, int size) {
mem_rd:

    for (int i = 0; i < size; i = i + INPUT_LAYER_ARR::size) {
#pragma HLS PIPELINE II=1
        INPUT_LAYER_ARR tmp;
        for (int j = 0; j < INPUT_LAYER_ARR::size; j++){
            tmp[j] = in[i+j];
        }
        inStream.write(tmp);
    }
}

template<typename ATOMIC_TYPE, typename OUT_LAYER_ARR>
void store_result(ATOMIC_TYPE* out, hls::stream<OUT_LAYER_ARR>& out_stream, int size) {
mem_wr:
    for (int i = 0; i < size; i = i + OUT_LAYER_ARR::size){
#pragma HLS PIPELINE II=1
        OUT_LAYER_ARR tmp = out_stream.read();
        for (int j = 0; j < OUT_LAYER_ARR::size; j++){
            out[i+j] = tmp[j];
        }
    }
}


void MY_PROJECT_TOP_FUNC(
// vitis-unified-wrapper-io

){

// vitis-unified-wrapper-interface
#pragma HLS INTERFACE s_axilite port=return bundle=control


// vitis-unified-wrapper-stream-dec


// vitis-unified-wrapper-stream-config


#pragma HLS dataflow

// vitis-unified-wrapper-load

// vitis-unified-wrapper-compute

// // vitis-unified-wrapper-store


}