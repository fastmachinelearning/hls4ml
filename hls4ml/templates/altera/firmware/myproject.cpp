#include "myproject.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes

using hls4ml_sycl_ext::experimental::task_sequence;

void MyProject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences

    // hls-fpga-machine-learning insert layers

    // hls-fpga-machine-learning return
}
