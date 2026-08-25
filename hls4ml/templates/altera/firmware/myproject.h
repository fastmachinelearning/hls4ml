#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

// This file defines the interface to the kernel

// currently this is fixed

using PipeProps = decltype(
    sycl::ext::oneapi::experimental::properties(hls4ml_sycl_ext::experimental::ready_latency<0>));

// Need to declare the input and output pipes

// hls-fpga-machine-learning insert inputs
// hls-fpga-machine-learning insert outputs

class MyProjectID;

struct MyProject {

    // kernel property method to config invocation interface
    auto get(sycl::ext::oneapi::experimental::properties_tag) {
        return sycl::ext::oneapi::experimental::properties{
            hls4ml_sycl_ext::experimental::streaming_interface<>, hls4ml_sycl_ext::experimental::pipelined<>};
    }

    SYCL_EXTERNAL void operator()() const;
};

#endif
