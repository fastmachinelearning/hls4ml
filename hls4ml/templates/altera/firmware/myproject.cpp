#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/altera/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights

using sycl::ext::altera::experimental::task_sequence;

// hls-fpga-machine-learning lib stamp

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes

// hls-fpga-machine-learning namespace end

void MyProject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences

    // hls-fpga-machine-learning insert layers

    // hls-fpga-machine-learning return
}
