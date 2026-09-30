#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/altera/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights

// The inter-task pipes need to be declared at namespace scope
namespace myproject_mystamp {
// hls-fpga-machine-learning insert inter-task pipes
} // namespace myproject_mystamp

using sycl::ext::altera::experimental::task_sequence;

void MyProject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences

    // hls-fpga-machine-learning insert layers

    // hls-fpga-machine-learning return
}
