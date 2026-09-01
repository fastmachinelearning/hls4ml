#ifndef NNET_HELPERS_AXI_H
#define NNET_HELPERS_AXI_H

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include <iostream>
#include <vector>

namespace nnet {

#ifndef __SYNTHESIS__

template <class srcType, typename dstType, size_t SIZE>
void convert_data_axis(srcType *src, hls::stream<hls::axis<float, 0, 0, 0>> &dst) {
    for (size_t i = 0; i < SIZE; i++) {
        hls::axis<float, 0, 0, 0> ctype;
        ctype.data = dstType(src[i]);
        dst.write(ctype);
    }
}

template <class srcType, typename dstType, size_t SIZE>
void convert_data_axis(std::vector<srcType> &src, hls::stream<hls::axis<float, 0, 0, 0>> &dst) {
    for (auto i = 0; i < SIZE; i++) {
        hls::axis<float, 0, 0, 0> pack;
        pack.data = src[i];
        if (i == SIZE - 1) {
            pack.last = 1;
        } else {
            pack.last = 0;
        }
        dst.write(pack);
    }
}

template <typename srcType, class dstType, size_t SIZE>
void convert_data_axis(hls::stream<hls::axis<float, 0, 0, 0>> &src, dstType *dst) {
    for (size_t i = 0; i < SIZE; i++) {
        hls::axis<float, 0, 0, 0> ctype = src.read();
        dst[i] = dstType(ctype.data);
    }
}

#endif

template <class res_T, size_t SIZE>
void print_result_axis(hls::stream<res_T> &result, std::ostream &out, bool keep = false) {
    for (int i = 0; i < SIZE; i++) {
        res_T res_pack = result.read();
        out << res_pack.data << " ";
        if (keep)
            result.write(res_pack);
    }
    out << std::endl;
}

template <class data_T, size_t SIZE> void fill_zero_axi(hls::stream<data_T> &data, bool reqLast) {
    for (int i = 0; i < SIZE; i++) {
        data_T data_pack;
        data_pack.data = 0;
        data_pack.last = reqLast && (i == (SIZE - 1)) ? 1 : 0;
        data.write(data_pack);
    }
}

} // namespace nnet

#endif
