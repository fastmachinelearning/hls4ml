#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w4.h"
#include "weights/b4.h"

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes
class Layer2OutPipeID;
using Layer2OutPipe = sycl::ext::intel::experimental::pipe<Layer2OutPipeID, fc1_result_t, 1>;

using sycl::ext::intel::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences
    task_sequence<nnet::dense_resource_stream<Fc1InputPipe, Layer2OutPipe, config2>> fc1;
    task_sequence<nnet::dense_resource_stream<Layer2OutPipe, Layer4OutPipe, config4>> fc2;

    // hls-fpga-machine-learning insert layers

    fc1.async(w2, b2);
    fc2.async(w4, b4);

    // hls-fpga-machine-learning return
}
