#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

// This file defines the interface to the kernel

// Pipe properties for host pipes. Host pipes connect to the data source DMA and sink DMA.
// They are connected to the first and the last layer to stream data into and out from the kernel.
using HostPipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>, sycl::ext::intel::experimental::bits_per_symbol<16>,
    sycl::ext::intel::experimental::uses_valid<true>, sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>,
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready));

// Need to declare the input and output pipes

// hls-fpga-machine-learning insert inputs
// hls-fpga-machine-learning insert outputs

class MyProjectID;

struct MyProject {
#ifndef IS_BSP
    // kernel property method to config invocation interface
    auto get(sycl::ext::oneapi::experimental::properties_tag) {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::streaming_interface<>,
                                                           sycl::ext::intel::experimental::pipelined<>};
    }
#else
    // kernel properties and pipelining is not supported in BSP (accelerator style).
#endif

    SYCL_EXTERNAL void operator()() const;
};

#endif
