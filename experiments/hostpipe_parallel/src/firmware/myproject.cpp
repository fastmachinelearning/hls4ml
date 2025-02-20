#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w5.h"
#include "weights/wr5.h"
#include "weights/b5.h"
#include "weights/br5.h"

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes

using sycl::ext::intel::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in
    auto conv1d_input = Conv1DInputPipe::read();

    // hls-fpga-machine-learning declare task sequences

    // hls-fpga-machine-learning insert layers

    [[intel::fpga_register]] layer2_t layer2_out;
    nnet::conv_1d_cl<input_t, layer2_t, config2>(conv1d_input, layer2_out, w2, b2);
    [[intel::fpga_register]] layer4_t layer4_out;
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out);
    [[intel::fpga_register]] result_t layer5_out;
    nnet::gru<layer4_t, result_t, config5>(layer4_out, layer5_out, w5, wr5, b5, br5);

    // hls-fpga-machine-learning return
    Layer5OutPipe::write(layer5_out);
}
