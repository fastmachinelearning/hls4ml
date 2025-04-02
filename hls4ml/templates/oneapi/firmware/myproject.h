#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

// This file defines the interface to the kernel

// currently this is fixed
using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(sycl::ext::intel::experimental::ready_latency<0>));

// Pipe properties for host pipes. Host pipes connect to the data source DMA and sink DMA.
// They are connected to the first and the last layer to stream data into and out from the kernel.
using HostPipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>, sycl::ext::intel::experimental::bits_per_symbol<16>,
    sycl::ext::intel::experimental::uses_valid<true>, sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>,
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready));

namespace nnet {

#if !defined(IS_BSP)
// Definition for buffer locations for Avalon MM host.
inline constexpr unsigned kInputBufferLocation = 0;
inline constexpr unsigned kOutputBufferLocation = 1;
#endif

// Implementation of a direct memory access kernel. Move data from source, convert,
// and send to the sink. Adaptive to SYCL HLS and hardware acceleration flow.
template <class src_T, class dest_pipe> struct DMA_convert_data {
#if !defined(IS_BSP)
    // When targeting a device family, we instantiate an Avalon Memory Mapped Host for
    // data transaction between host and the DMA kernel during emulation and simulation.
    sycl::ext::oneapi::experimental::annotated_arg<
        src_T *,
        decltype(sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::latency<0>, sycl::ext::intel::experimental::dwidth<16>,
            sycl::ext::intel::experimental::buffer_location<kInputBufferLocation>,
            sycl::ext::intel::experimental::read_write_mode_read, sycl::ext::intel::experimental::wait_request_requested})>
#else
    // When targeting oneAPI BSP, we can use USM pointer to access host memory.
    src_T *const
#endif
        src;
    size_t num_iteration;

    [[intel::kernel_args_restrict]] void operator()() const {

#if defined(IS_BSP)
        // Access data using host pointer.
        sycl::ext::intel::host_ptr<src_T> src_ptr(src);
#else
        // Host allocation is not supported when targeting an FPGA family or part number.
        src_T *src_ptr(src);
#endif
        // First, extract the PipeDataT from the pipe
        using PipeDataType = typename nnet::ExtractPipeType<dest_pipe>::value_type;
        // Then, extract the DataT from StreamingBeat
        using DstDataType = typename nnet::ExtractDataType<PipeDataType>::value_type;
        constexpr auto dstTypeSize = std::tuple_size<DstDataType>{};

        [[intel::fpga_register]] typename nnet::ExtractPipeType<dest_pipe>::value_type packet;

        // Keep sending data to the input layer and keep the kernels running.
        for (size_t i = 0; i < num_iteration; i++) {
            #pragma unroll
            for (size_t j = 0; j < dstTypeSize; j++) {
                packet.data[j] = src_ptr[i * dstTypeSize + j];
            }
            packet.sop = (i == 0);
            // Assert end-of-packet signal after the last iteration.
            // All down-stream kernels will stop seeing eop.
            packet.eop = (i == (num_iteration - 1));
            dest_pipe::write(packet);
        }
    }
};

// Symmetrical to the DMA_convert_data above, this DMA drains the output pipe and
// writes result to memory.
template <class src_pipe, class dst_T> struct DMA_convert_data_back {
#if !defined(IS_BSP)
    // Without BSP, instantiate an Avalon Memory Mapped Host to write to host.
    sycl::ext::oneapi::experimental::annotated_arg<
        dst_T *,
        decltype(sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::latency<0>, sycl::ext::intel::experimental::dwidth<16>,
            sycl::ext::intel::experimental::buffer_location<kOutputBufferLocation>,
            sycl::ext::intel::experimental::read_write_mode_write, sycl::ext::intel::experimental::wait_request_requested})>
#else
    // USM pointer, otherwise.
    dst_T *const
#endif
        dst;
    size_t num_iteration;

    [[intel::kernel_args_restrict]] void operator()() const {
#if defined(IS_BSP)
        sycl::ext::intel::host_ptr<dst_T> dst_ptr(dst);
#else
        dst_T *dst_ptr(dst);
#endif
        // First, extract the PipeDataT from the pipe
        using PipeDataType = typename nnet::ExtractPipeType<src_pipe>::value_type;
        // Then, extract the DataT from StreamingBeat
        using SrcDataType = typename nnet::ExtractDataType<PipeDataType>::value_type;
        constexpr auto srcTypeSize = std::tuple_size<SrcDataType>{};

        [[intel::fpga_register]] typename nnet::ExtractPipeType<src_pipe>::value_type packet;

        // Drain the output pipe and write result to memory.
        for (size_t i = 0; i < num_iteration; i++) {
            packet = src_pipe::read();
            #pragma unroll
            for (size_t j = 0; j < srcTypeSize; j++) {
                dst_ptr[i * srcTypeSize + j] = static_cast<dst_T>(packet.data[j].to_double());
            }
        }
    }
};

} // namespace nnet

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
    // kernel properties and pipelining is not supported in BSP.
#endif

    SYCL_EXTERNAL void operator()() const;
};

#endif
