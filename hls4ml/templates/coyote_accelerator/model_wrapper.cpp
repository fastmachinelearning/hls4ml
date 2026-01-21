#include "model_wrapper.hpp"

/**
 * @brief A wrapper for an hls4ml model deployed with Coyote.
 *
 * In Coyote, data is passed through 512-bit AXI streams; the data can originate 
 * from host or card memory, or the network (from other nodes). The model wrapper
 * encapsulates the hls4ml model and converter functions that convert beats from
 * 512-bit AXI streams to the model's input format (depends whether io_parallel or io_stream)
 * and vice-versa for the output. Important, when running the Coyote accelerator backend and
 * moving data from/to the host, it is packed as float32 to the 512-bit AXI stream. That is
 * each AXI beat (.tvalid asserted) contains 16 float32 values. The reason for this is two-fold:
 * (1) the predict function inherently works with float32 data, and (2) when moving data between
 * the host and the accelerator, one must specify the size of the buffer moved. While it's perfectly
 * possible to "emulate" ap_fixed on the host and convert the float32 to ap_fixed, it is unclear
 * what the exact size/alignment etc. of the buffer will be on the host (e.g, ap_fixed<1> cannot 
 * possibly be 1 bit in a "convential" OS, so some padding would almost certainly be added; this
 * padding will then have to be removed by the model_wrapper, which could be error-prone).
 */
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
