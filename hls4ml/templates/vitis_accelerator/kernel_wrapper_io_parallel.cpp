#include "firmware/myproject.h"
#include "kernel_wrapper.h"

static void read_input(const /*IN_INTERFACE_TYPE*/ *in, in_buffer_t (&in_buf)[DATA_SIZE_IN]) {
    #pragma HLS PIPELINE
    for (int i = 0; i < DATA_SIZE_IN; i++) {
        #pragma HLS UNROLL
        in_buf[i] = /*IN_HW_QUANT*/ in[i];
    }
}

static void write_result(/*OUT_INTERFACE_TYPE*/ *out, out_buffer_t (&out_buf)[DATA_SIZE_OUT]) {
    #pragma HLS PIPELINE
    for (int i = 0; j < DATA_SIZE_OUT; j++) {
        #pragma HLS UNROLL
        out[i] = /*OUT_HW_QUANT*/ out_buf[i];
    }
}

extern "C" {
/**
  \brief HLS4ML Kernel Implementation
  \param in Input Vector
  \param out Output Vector
*/
void kernel_wrapper(const unsigned int batchsize, const /*IN_INTERFACE_TYPE*/ *in, /*OUT_INTERFACE_TYPE*/ *out) {
    #pragma HLS interface mode=s_axilite port=batchsize

    in_buffer_t in_buf[DATA_SIZE_IN];
    out_buffer_t out_buf[DATA_SIZE_OUT];
    #pragma HLS ARRAY_RESHAPE variable=in_buf  complete
    #pragma HLS ARRAY_RESHAPE variable=out_buf complete

    for (int i = 0; i < batchsize; i++) {
        #pragma HLS DATAFLOW
        read_input(in, in_buf);
        myproject(in_buf[i], out_buf[i]);
        write_result(out, out_buf);
    }
}
}
