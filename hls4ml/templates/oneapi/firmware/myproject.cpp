#include "myproject.h"
#include "parameters.h"
#ifdef AHLS
#include <sycl/ext/altera/experimental/task_sequence.hpp>
#else
#include <sycl/ext/intel/experimental/task_sequence.hpp>
#endif

// hls-fpga-machine-learning insert weights

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes

#ifdef AHLS
using sycl::ext::altera::experimental::task_sequence;
#else
using sycl::ext::intel::experimental::task_sequence;
#endif

void MyProject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences

    // hls-fpga-machine-learning insert layers

    // hls-fpga-machine-learning return
}
