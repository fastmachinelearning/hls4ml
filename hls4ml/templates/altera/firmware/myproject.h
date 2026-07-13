#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

// This file defines the interface to the kernel

// currently this is fixed

using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(
#ifdef AHLS
    sycl::ext::altera::experimental::ready_latency<0>
#else
    sycl::ext::intel::experimental::ready_latency<0>
#endif
    ));

// Need to declare the input and output pipes

// hls-fpga-machine-learning insert inputs
// hls-fpga-machine-learning insert outputs

class MyProjectID;

struct MyProject {

    // kernel property method to config invocation interface
    auto get(sycl::ext::oneapi::experimental::properties_tag) {
        return sycl::ext::oneapi::experimental::properties{
#ifdef AHLS
            sycl::ext::altera::experimental::streaming_interface<>, sycl::ext::altera::experimental::pipelined<>
#else
            sycl::ext::intel::experimental::streaming_interface<>, sycl::ext::intel::experimental::pipelined<>
#endif
        };
    }

    SYCL_EXTERNAL void operator()() const;
};

#endif
