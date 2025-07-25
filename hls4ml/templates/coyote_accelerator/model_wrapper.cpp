#include "model_wrapper.hpp"

// TODO: Remove interfaces in myproject.cpp by moving the function to CoyoteAcceleratorWriter...
void model_wrapper (
    hls::stream<axi_s> &data_in,
    hls::stream<axi_s> &data_out
) {
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS INTERFACE axis register port=data_in name=data_in
    #pragma HLS INTERFACE axis register port=data_out name=data_out

    // hls-fpga-machine-learning insert data

    // hls-fpga-machine-learning insert top-level function

}
