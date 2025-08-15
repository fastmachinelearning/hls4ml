#include <stdint.h>
#include <hls_stream.h>
#include <iostream>
#include "ap_axi_sdata.h"
#include "MY_PROJECT_AXI_INC.h"

#define DMX_BUF_IN_SZ   VAL
#define DMX_BUF_OUT_SZ  VAL


static void load_input(ATOMIC_TYPE* in, hls::stream<dma_data_packet>& inStream, int size) {
mem_rd:
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        inStream.data << in[i];
        inStream.last = (i == (size-1));
    }
}

static void store_result(ATOMIC_TYPE* out, hls::stream<dma_data_packet>& out_stream, int size) {
mem_wr:
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out[i] = out_stream.read().data;
    }
}


void MY_PROJECT_TOP_FUNC(ATOMIC_TYPE* in, ATOMIC_TYPE* out, int inSize, int outSize){

#pragma HLS INTERFACE m_axi port = in  bundle = gmem0
#pragma HLS INTERFACE m_axi port = out bundle = gmem1

static hls::stream<dma_data_packet> in_stream("in_stream")
static hls::stream<dma_data_packet> out_stream("out_stream")

#pragma HLS STREAM variable=in_stream depth=DMX_BUF_IN_SZ
#pragma HLS STREAM variable=out_stream depth=DMX_BUF_OUT_SZ

#pragma HLS dataflow

    load_input(in, in_stream, inSize);
    MY_PROJECT_CON(in_stream, out_stream);
    store_result(out, out_stream, outSize);

}