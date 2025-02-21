#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

// This file defines the interface to the kernel

// currently this is fixed
using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(sycl::ext::intel::experimental::ready_latency<0>));

// Pipe properties for host pipes. Host pipes connect to the data source DMA and sink DMA.
// They are connected to the first and the last layer to stream data into and out from the kernel.
using HostPipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::bits_per_symbol<8>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>,
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready
));

// Data wrapper type used in the host pipes.
// first argument: datatype carried over this Avalon streaming interface's data signal.
// second argument: enable startofpacket and endofpacket signals for synchronization.
// third argument: to enable the empty signal.
using InputBeatT = sycl::ext::intel::experimental::StreamingBeat<
    input_t,    // input_t should match the input type of the first layer.
    true,
    true>;
using OutputBeatT = sycl::ext::intel::experimental::StreamingBeat<
    result_t,    // result_t should match the output type of the last layer.
    true,
    true>;

namespace nnet {

#if !defined(IS_BSP)
// Definition for buffer locations for Avalon MM host.
inline constexpr unsigned kInputBufferLocation = 0;
inline constexpr unsigned kOutputBufferLocation = 1;
#endif

// Name for DMAs.
class IDInputDMA;
class IDOutputDMA;

// Implementation of a direct memory access kernel. Move data from source, convert, 
// and send to the sink. Adaptive to SYCL HLS and hardware acceleration flow.
template <class srcType, class dest_pipe, size_t SIZE> 
struct DMA_convert_data {
#if !defined(IS_BSP)
    // When targeting a device family, we instantiate an Avalon Memory Mapped Host for 
    // data transaction between host and the DMA kernel during emulation and simulation.
    sycl::ext::oneapi::experimental::annotated_arg<srcType *, 
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::latency<0>,
          sycl::ext::intel::experimental::dwidth<8>,
          sycl::ext::intel::experimental::buffer_location<kInputBufferLocation>,
          sycl::ext::intel::experimental::read_write_mode_read,
          sycl::ext::intel::experimental::wait_request_requested})>
#else
    // When targeting oneAPI BSP, we can use USM pointer to access host memory.
    srcType *const
#endif
        src;

    [[intel::kernel_args_restrict]]
    void operator()() const {
        
#if defined(IS_BSP)
        // Access data using host pointer.
        sycl::ext::intel::host_ptr<srcType> src_ptr(src);
#else
        // Host allocation is not supported when targeting an FPGA family or part number.
        srcType *src_ptr(src);
#endif
        // First, extract the PipeDataT from the pipe
        using PipeDataType = typename nnet::ExtractPipeType<dest_pipe>::value_type;
        // Then, extract the DataT from StreamingBeat
        using DstDataType = typename nnet::ExtractDataType<PipeDataType>::value_type;
        constexpr auto dstTypeSize = std::tuple_size<DstDataType>{};

        [[intel::fpga_register]]
        typename nnet::ExtractPipeType<dest_pipe>::value_type ctype;

        for (size_t i = 0; i < SIZE / dstTypeSize; i++) {
            #pragma unroll
            for (size_t j = 0; j < dstTypeSize; j++) {
                ctype.data[j] = src_ptr[i * dstTypeSize + j];
            }
            ctype.sop = (i == 0);
            ctype.eop = (i == (SIZE / dstTypeSize - 1));
            dest_pipe::write(ctype);
        }
    }
};

// Symmetrical to the DMA_convert_data above.
template <class src_pipe, class dstType, size_t SIZE> 
struct DMA_convert_data_back {
#if !defined(IS_BSP)
    sycl::ext::oneapi::experimental::annotated_arg<dstType *, 
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::latency<0>,
          sycl::ext::intel::experimental::dwidth<8>,
          sycl::ext::intel::experimental::buffer_location<kOutputBufferLocation>,
          sycl::ext::intel::experimental::read_write_mode_write,
          sycl::ext::intel::experimental::wait_request_requested})>
#else
    dstType *const
#endif
        dst;

    [[intel::kernel_args_restrict]]
    void operator()() const {
#if defined(IS_BSP)
        sycl::ext::intel::host_ptr<dstType> dst_ptr(dst);
#else
        dstType *dst_ptr(dst);
#endif
        constexpr auto srcTypeSize = std::tuple_size<typename nnet::ExtractPipeType<src_pipe>::value_type>{};

        [[intel::fpga_register]] 
        typename nnet::ExtractPipeType<src_pipe>::value_type ctype;

        for (size_t i = 0; i < SIZE / srcTypeSize; i++) {
            ctype = src_pipe::read();
            #pragma unroll
            for (size_t j = 0; j < srcTypeSize; j++) {
                dst_ptr[i * srcTypeSize + j] = ctype[j].to_double();
            }
        }
    }
};

}   // namespace nnet

// Need to declare the input and output pipes

// hls-fpga-machine-learning insert inputs
// hls-fpga-machine-learning insert outputs

class MyProjectID;

struct MyProject {

    // kernel property method to config invocation interface
    auto get(sycl::ext::oneapi::experimental::properties_tag) {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::streaming_interface<>,
                                                           sycl::ext::intel::experimental::pipelined<>};
    }

    SYCL_EXTERNAL void operator()() const;
};

#endif
