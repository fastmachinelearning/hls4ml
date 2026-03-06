#ifndef NNET_AXI_UTILS_H_
#define NNET_AXI_UTILS_H_

#include "ap_axi_sdata.h"

namespace nnet {

// Converts an array of data (fixed-point numbers) into 512-bit AXI stream packets; see model_wrapper.hpp for usage
template <class array_T, class axi_T, unsigned int SIZE, unsigned int AXI_BITS, unsigned int PRECISION> 
void data_to_axi_stream(array_T data_in[SIZE], hls::stream<ap_axiu<AXI_BITS, 0, 0, 0>> &axi_out) {
    #pragma HLS INLINE OFF
    #pragma HLS PIPELINE

    constexpr const unsigned int ELEMENTS_PER_AXI = AXI_BITS / PRECISION;
    constexpr const unsigned int NUM_BEATS = (SIZE + ELEMENTS_PER_AXI - 1) / ELEMENTS_PER_AXI;

    for (unsigned int i = 0; i < NUM_BEATS; i++) {
        if (i == NUM_BEATS - 1) {
            ap_axiu<AXI_BITS, 0, 0, 0> axi_packet;
            unsigned int index = i * ELEMENTS_PER_AXI;

            for (unsigned int j = 0; j < SIZE - index; j++) {
                #pragma HLS UNROLL
                
                axi_T axi_tmp = axi_T(data_in[index + j]);
                ap_uint<PRECISION> axi_bits = *reinterpret_cast<ap_uint<PRECISION>*>(&axi_tmp);
                axi_packet.data.range((j + 1) * PRECISION - 1, j * PRECISION) = axi_bits;
            }

            axi_packet.last = 1;
            axi_out.write(axi_packet);

        } else {
            ap_axiu<AXI_BITS, 0, 0, 0> axi_packet;
            unsigned int index = i * ELEMENTS_PER_AXI;
            
            for (unsigned int j = 0; j < ELEMENTS_PER_AXI; j++) {
                #pragma HLS UNROLL
                
                axi_T axi_tmp = axi_T(data_in[index + j]);
                ap_uint<PRECISION> axi_bits = *reinterpret_cast<ap_uint<PRECISION>*>(&axi_tmp);
                axi_packet.data.range((j + 1) * PRECISION - 1, j * PRECISION) = axi_bits;
            }

            axi_packet.last = 0;
            axi_out.write(axi_packet);
        }
    }
}

// Unpacks beats of 512-bit AXI beats into an array of data (fixed-point numbers) see model_wrapper.hpp for usage
template <class array_T, class axi_T, unsigned int SIZE, unsigned int AXI_BITS, unsigned int PRECISION> 
void axi_stream_to_data(hls::stream<ap_axiu<AXI_BITS, 0, 0, 0>> &axi_in, array_T data_out[SIZE]) {
    #pragma HLS INLINE OFF
    #pragma HLS PIPELINE

    constexpr const unsigned int ELEMENTS_PER_AXI = AXI_BITS / PRECISION;
    constexpr const unsigned int NUM_BEATS = (SIZE + ELEMENTS_PER_AXI - 1) / ELEMENTS_PER_AXI;

    for (unsigned int i = 0; i < NUM_BEATS; i++) {
        if (i == NUM_BEATS - 1) {
            unsigned int index = i * ELEMENTS_PER_AXI;
            ap_axiu<AXI_BITS, 0, 0, 0> axi_packet = axi_in.read();

            for (unsigned int j = 0; j < SIZE - index; j++) {
                #pragma HLS UNROLL
                    
                ap_uint<PRECISION> axi_bits = axi_packet.data.range((j + 1) * PRECISION - 1, j * PRECISION);
                axi_T axi_tmp = *reinterpret_cast<axi_T*>(&axi_bits);
                data_out[index + j] = array_T(axi_tmp);
            }

        } else {
            unsigned int index = i * ELEMENTS_PER_AXI;
            ap_axiu<AXI_BITS, 0, 0, 0> axi_packet = axi_in.read();

            for (unsigned int j = 0; j < ELEMENTS_PER_AXI; j++) {
                #pragma HLS UNROLL
                    
                ap_uint<PRECISION> axi_bits = axi_packet.data.range((j + 1) * PRECISION - 1, j * PRECISION);
                axi_T axi_tmp = *reinterpret_cast<axi_T*>(&axi_bits);
                data_out[index + j] = array_T(axi_tmp);
            }
        }
    }
}

}

#endif