#ifndef NNET_AXI_UTILS_STREAM_H
#define NNET_AXI_UTILS_STREAM_H

#include "ap_axi_sdata.h"

namespace nnet {

// Converts an stream of data (fixed-point numbers) into 512-bit AXI stream packets; see model_wrapper.hpp for usage
template <class array_T, class axi_T, unsigned int SIZE, unsigned int AXI_BITS, unsigned int PRECISION> 
void data_to_axi_stream(hls::stream<array_T> &data_in, hls::stream<ap_axiu<AXI_BITS, 0, 0, 0>> &axi_out) {
    #pragma HLS INLINE OFF
    #pragma HLS PIPELINE

    constexpr const unsigned int ELEMENTS_PER_AXI = (SIZE <= (AXI_BITS / PRECISION)) ? SIZE : (AXI_BITS / PRECISION);
    constexpr const unsigned int NUM_BEATS = SIZE / ELEMENTS_PER_AXI + (SIZE % ELEMENTS_PER_AXI != 0);

    unsigned int index = 0;
    ap_axiu<AXI_BITS, 0, 0, 0> axi_packet;

    for (int i = 0; i < SIZE / array_T::size; i++) {
        array_T in_data = data_in.read();

        for (int j = 0; j < array_T::size; j++) {
            #pragma HLS UNROLL    
            axi_T axi_tmp = axi_T (in_data[j]);
            ap_uint<PRECISION> axi_bits = *reinterpret_cast<ap_uint<PRECISION>*>(&axi_tmp);
            axi_packet.data.range((index + 1) * PRECISION - 1, index * PRECISION) = axi_bits;
            index++;
            if (index == ELEMENTS_PER_AXI) {
                axi_packet.last = 0;
                axi_out.write(axi_packet);
                index = 0;
            }
        }
    }

    if (index != ELEMENTS_PER_AXI && index != 0) {
        axi_packet.last = 1;
        axi_out.write(axi_packet);
    }

}

// Unpacks beats of 512-bit AXI beats into an stream of data (fixed-point numbers) see model_wrapper.hpp for usage
template <class array_T, class axi_T, unsigned int SIZE, unsigned int AXI_BITS, unsigned int PRECISION> 
void axi_stream_to_data(hls::stream<ap_axiu<AXI_BITS, 0, 0, 0>> &axi_in, hls::stream<array_T> &data_out) {
    #pragma HLS INLINE OFF
    #pragma HLS PIPELINE

    constexpr const unsigned int ELEMENTS_PER_AXI = (SIZE <= (AXI_BITS / PRECISION)) ? SIZE : (AXI_BITS / PRECISION);
    constexpr const unsigned int NUM_BEATS = SIZE / ELEMENTS_PER_AXI + (SIZE % ELEMENTS_PER_AXI != 0);

    array_T tmp;
    unsigned int index = 0;
    ap_axiu<AXI_BITS, 0, 0, 0> axi_packet;

    for (int i = 0; i < NUM_BEATS; i++) {
        ap_axiu<AXI_BITS, 0, 0, 0> axi_packet = axi_in.read();
    
        for (int j = 0; j < ELEMENTS_PER_AXI; j++) {
            #pragma HLS UNROLL
            ap_uint<PRECISION> axi_bits = axi_packet.data.range((j + 1) * PRECISION - 1, j * PRECISION);
            axi_T axi_tmp = *reinterpret_cast<axi_T*>(&axi_bits);
            tmp[index] = typename array_T::value_type(axi_tmp);
            index++;
            if (index == array_T::size) {
                index = 0;
                data_out.write(tmp);
    
            }
        }
    }
}

}

#endif