#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

// This file defines the interface to the kernel

// currently this is fixed

using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(sycl::ext::altera::experimental::ready_latency<0>));

// Pipe IDs are registered process-wide by name, so the declarations are namespaced
// to avoid collisions between libraries loaded into the same process
namespace mynamespace {

// Need to declare the input and output pipes

// hls-fpga-machine-learning insert inputs
// hls-fpga-machine-learning insert outputs

class MyProjectID;

struct MyProject {

    // kernel property method to config invocation interface
    auto get(sycl::ext::oneapi::experimental::properties_tag) {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::altera::experimental::streaming_interface<>,
                                                           sycl::ext::altera::experimental::pipelined<>};
    }

    SYCL_EXTERNAL void operator()() const;
};

} // namespace mynamespace

using namespace mynamespace;

#endif
