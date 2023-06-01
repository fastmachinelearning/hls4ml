#include <iostream>

#include "firmware/myproject.cpp"
#include "firmware/myproject.h"
#include "firmware/parameters.h"

constexpr int c_size = 1024;

static void load_input(input_t *in, hls::stream<input_t> &inStream, int size) {
mem_rd:
    for (int i = 0; i < size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        inStream << in[i];
    }
}
// static void store_result(result_t* out, hls::stream<result_t>& out_stream, int size) {
static void store_result(result_t *out, hls::stream<result_t> &out_stream, int size) {
mem_wr:
    for (int i = 0; i < size; i++) {
        #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        result_t temp = out_stream.read();
        out[i] = temp;
    }
}

void myproject_kernel(
    // hls-fpga-machine-learning insert header
) {
    #pragma HLS INTERFACE m_axi port = project_input bundle = gmem0
    #pragma HLS INTERFACE m_axi port = project_output bundle = gmem1
    static hls::stream<input_t> project_input_stream("project_input_stream");
    static hls::stream<result_t> project_output_stream("project_output_stream");
    #pragma HLS dataflow
    load_input(project_input, project_input_stream, size);
    // hls-fpga-machine-learning insert project top
    store_result(project_output, project_output_stream, size);
}
