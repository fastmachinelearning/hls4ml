
#ifndef NNET_HELPERS_H
#define NNET_HELPERS_H

#include "ac_channel.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

extern const char *get_weights_dir();

namespace nnet {

#ifndef __SYNTHESIS__

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR get_weights_dir()
#endif

template <class T, size_t SIZE> void load_weights_from_txt(T *w, const char *fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        size_t i = 0;
        while (std::getline(iss, token, ',')) {
            // CATAPULT_PORT
            // std::istringstream(token) >> w[i];
            double tmp;
            std::istringstream(token) >> tmp;
            w[i] = tmp;
            i++;
        }

        if (SIZE != i) {
            std::cerr << "ERROR: Expected " << SIZE << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}

template <class T, size_t SIZE> void load_compressed_weights_from_txt(T *w, const char *fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;
        std::string extra_chars = "} ";

        size_t i = 0;
        while (std::getline(iss, token, '{')) {
            if (token.length() == 0) {
                continue;
            }
            for (char c : extra_chars) {
                token.erase(std::remove(token.begin(), token.end(), c), token.end());
            }
            if (token.back() == ',') {
                token.erase(token.end() - 1);
            }

            std::replace(token.begin(), token.end(), ',', ' ');
            std::istringstream structss(token);

            if (!(structss >> w[i].row_index >> w[i].col_index >> w[i].weight)) {
                std::cerr << "ERROR: Unable to parse file " << std::string(fname);
                exit(1);
            }
            i++;
        }

        if (SIZE != i) {
            std::cerr << "ERROR: Expected " << SIZE << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}

template <class T, size_t SIZE> void load_exponent_weights_from_txt(T *w, const char *fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;
        std::string extra_chars = "} ";

        size_t i = 0;
        while (std::getline(iss, token, '{')) {
            if (token.length() == 0) {
                continue;
            }
            for (char c : extra_chars) {
                token.erase(std::remove(token.begin(), token.end(), c), token.end());
            }
            if (token.back() == ',') {
                token.erase(token.end() - 1);
            }

            std::replace(token.begin(), token.end(), ',', ' ');
            std::istringstream structss(token);

            double sign;
            double weight;
            if (!(structss >> sign >> weight)) {
                std::cerr << "ERROR: Unable to parse file " << std::string(fname);
                exit(1);
            }
            w[i].sign = sign;
            w[i].weight = weight;
            i++;
        }

        if (SIZE != i) {
            std::cerr << "ERROR: Expected " << SIZE << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}

template <int Twidth, int Ibits, bool Signed, ac_q_mode Qmode, ac_o_mode Omode>
void convert_single_data(ac_fixed<Twidth, Ibits, Signed, Qmode, Omode> &src, double &dst) {
    dst = src.to_double();
}
template <int Twidth, int Ibits, bool Signed, ac_q_mode Qmode, ac_o_mode Omode>
void convert_single_data(ac_fixed<Twidth, Ibits, Signed, Qmode, Omode> &src, float &dst) {
    dst = src.to_double();
}
template <class srcType, class dstType> void convert_single_data(srcType &src, dstType &dst) { dst = dstType(src); }
template <class srcType, class dstType, size_t SIZE> void convert_data(srcType *src, dstType *dst) {
    for (size_t i = 0; i < SIZE; i++) {
        convert_single_data(src[i], dst[i]);
    }
}

template <class srcType, class dstType, size_t SIZE> void convert_data(srcType *src, ac_channel<dstType> &dst) {
    for (size_t i = 0; i < SIZE / dstType::size; i++) {
        dstType ctype;
        for (size_t j = 0; j < dstType::size; j++) {
            ctype[j] = typename dstType::value_type(src[i * dstType::size + j]);
        }
        dst.write(ctype);
    }
}

template <class srcType, class dstType, size_t SIZE> void convert_data(ac_channel<srcType> &src, dstType *dst) {
    for (size_t i = 0; i < SIZE / srcType::size; i++) {
        srcType ctype = src.read();
        for (size_t j = 0; j < srcType::size; j++) {
            dst[i * srcType::size + j] = dstType(ctype[j].to_double()); // this may only work for ac_fixed
        }
    }
}

extern bool trace_enabled;
extern std::map<std::string, void *> *trace_outputs;
extern size_t trace_type_size;

template <class data_T, class save_T> void save_output_array(data_T *data, save_T *ptr, size_t layer_size) {
    for (int i = 0; i < layer_size; i++) {
        ptr[i] = static_cast<save_T>(data[i].to_double());
    }
}

template <class data_T, class save_T> void save_output_array(ac_channel<data_T> &data, save_T *ptr, size_t layer_size) {
    for (size_t i = 0; i < layer_size / data_T::size; i++) {
        data_T ctype = data.read();
        for (size_t j = 0; j < data_T::size; j++) {
            ptr[i * data_T::size + j] = save_T(ctype[j]);
        }
        data.write(ctype);
    }
}

template <class data_T> void save_output_array(ac_channel<data_T> &data, float *ptr, size_t layer_size) {
    for (size_t i = 0; i < layer_size / data_T::size; i++) {
        data_T ctype = data.read();
        for (size_t j = 0; j < data_T::size; j++) {
            ptr[i * data_T::size + j] = ctype[j].to_double();
        }
        data.write(ctype);
    }
}

template <class data_T> void save_output_array(ac_channel<data_T> &data, double *ptr, size_t layer_size) {
    for (size_t i = 0; i < layer_size / data_T::size; i++) {
        data_T ctype = data.read();
        for (size_t j = 0; j < data_T::size; j++) {
            ptr[i * data_T::size + j] = ctype[j].to_double();
        }
        data.write(ctype);
    }
}

// We don't want to include save_T in this function because it will be inserted into myproject.cpp
// so a workaround with element size is used
template <class data_T> void save_layer_output(data_T *data, const char *layer_name, size_t layer_size) {
    if (!trace_enabled)
        return;

    if (trace_outputs) {
        if (trace_outputs->count(layer_name) > 0) {
            if (trace_type_size == 4) {
                save_output_array<data_T>(data, (float *)(*trace_outputs)[layer_name], layer_size);
            } else if (trace_type_size == 8) {
                save_output_array<data_T>(data, (double *)(*trace_outputs)[layer_name], layer_size);
            } else {
                std::cout << "Unknown trace type!" << std::endl;
            }
        } else {
            std::cout << "Layer name: " << layer_name << " not found in debug storage!" << std::endl;
        }
    } else {
        std::ostringstream filename;
        filename << "./tb_data/" << layer_name << "_output.log"; // TODO if run as a shared lib, path should be ../tb_data
        std::fstream out;
        out.open(filename.str(), std::ios::app);
        assert(out.is_open());
        for (int i = 0; i < layer_size; i++) {
            out << data[i] << " "; // We don't care about precision in text files
        }
        out << std::endl;
        out.close();
    }
}

template <class data_T> void save_layer_output(ac_channel<data_T> &data, const char *layer_name, size_t layer_size) {
    if (!trace_enabled)
        return;

    if (trace_outputs) {
        if (trace_outputs->count(layer_name) > 0) {
            if (trace_type_size == 4) {
                save_output_array<data_T>(data, (float *)(*trace_outputs)[layer_name], layer_size);
            } else if (trace_type_size == 8) {
                save_output_array<data_T>(data, (double *)(*trace_outputs)[layer_name], layer_size);
            } else {
                std::cout << "Unknown trace type!" << std::endl;
            }
        } else {
            std::cout << "Layer name: " << layer_name << " not found in debug storage!" << std::endl;
        }
    } else {
        std::ostringstream filename;
        filename << "./tb_data/" << layer_name << "_output.log"; // TODO if run as a shared lib, path should be ../tb_data
        std::fstream out;
        out.open(filename.str(), std::ios::app);
        assert(out.is_open());
        for (size_t i = 0; i < layer_size / data_T::size; i++) {
            data_T ctype = data.read();
            for (size_t j = 0; j < data_T::size; j++) {
                out << ctype[j].to_double();
                out << " "; // We don't care about precision in text files
            }
            data.write(ctype);
        }
        out << std::endl;
        out.close();
    }
}

#endif

template <class src_T, class dst_T, size_t OFFSET, size_t SIZE> void copy_data(std::vector<src_T> src, dst_T dst[SIZE]) {
    typename std::vector<src_T>::const_iterator in_begin = src.cbegin() + OFFSET;
    typename std::vector<src_T>::const_iterator in_end = in_begin + SIZE;
    std::copy(in_begin, in_end, dst);
}

template <class src_T, class dst_T, size_t OFFSET, size_t SIZE>
void copy_data(std::vector<src_T> src, ac_channel<dst_T> &dst) {
    typename std::vector<src_T>::const_iterator in_begin = src.cbegin() + OFFSET;
    typename std::vector<src_T>::const_iterator in_end = in_begin + SIZE;

    size_t i_pack = 0;
    dst_T dst_pack;
    for (typename std::vector<src_T>::const_iterator i = in_begin; i != in_end; ++i) {
        dst_pack[i_pack++] = typename dst_T::value_type(*i);
        if (i_pack == dst_T::size) {
            i_pack = 0;
            dst.write(dst_pack);
        }
    }
}

template <class src_T, class dst_T, size_t OFFSET, size_t SIZE> void copy_data_axi(std::vector<src_T> src, dst_T dst[SIZE]) {
    for (auto i = 0; i < SIZE; i++)
        if (i == SIZE - 1) {
            dst[i].data = src[i];
            dst[i].last = 1;
        } else {
            dst[i].data = src[i];
            dst[i].last = 0;
        }
}

template <class res_T, size_t SIZE> void print_result(res_T result[SIZE], std::ostream &out, bool keep = false) {
    for (unsigned i = 0; i < SIZE; i++) {
        out << result[i] << " ";
    }
    out << std::endl;
}

template <class res_T, size_t SIZE> void print_result(ac_channel<res_T> &result, std::ostream &out, bool keep = false) {
    if (!keep) {
        while (result.available(1)) {
            res_T res_pack = result.read();
            for (unsigned int j = 0; j < res_T::size; j++) {
                out << res_pack[j] << " ";
            }
        }
        out << std::endl;
    } else {
        if (result.debug_size() >= SIZE / res_T::size) {
            for (unsigned int i = 0; i < SIZE / res_T::size; i++) {
                res_T res_pack = result[i]; // peek
                for (unsigned int j = 0; j < res_T::size; j++) {
                    out << res_pack[j] << " ";
                }
            }
            out << std::endl;
        }
    }
}

template <class data_T, size_t SIZE> void fill_zero(data_T data[SIZE]) { std::fill_n(data, SIZE, 0.); }

template <class data_T, size_t SIZE> void fill_zero(ac_channel<data_T> &data) {
    for (unsigned int i = 0; i < SIZE / data_T::size; i++) {
        data_T data_pack;
        for (unsigned int j = 0; j < data_T::size; j++) {
            data_pack[j] = 0.;
        }
        data.write(data_pack);
    }
}

// Fix for CAT-36531
template <class data_T, size_t SIZE> void fill_random(data_T data[SIZE]) {
    // std::cout << "Fill_Random SIZE:"<< SIZE << std::endl;
    data_T MAX_VALUE;
    for (unsigned int i = 0; i < SIZE; i++) {
        // Generate a random value (for example, between 0 and 1)
        data_T random_value = (data_T)rand() / MAX_VALUE.template set_val<AC_VAL_MIN>();
        data[i] = random_value;
    }
}

template <class data_T, size_t SIZE> void fill_random(ac_channel<data_T> &data) {
    typedef typename data_T::value_type base_T;
    base_T MAX_VALUE;
    for (unsigned int i = 0; i < SIZE / data_T::size; i++) {
        data_T data_pack;
        for (unsigned int j = 0; j < data_T::size; j++) {
            // Generate a random value (for example, between 0 and 1)
            base_T random_value = (base_T)rand() / MAX_VALUE.template set_val<AC_VAL_MIN>();
            data_pack[j] = random_value;
        }
        data.write(data_pack);
    }
    // std::cout << "Fill_Random AC_CHANNEL" << std::endl;
}

template <class dataType, unsigned int nrows> int read_file_1D(const char *filename, dataType data[nrows]) {
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == 0) {
        return -1;
    }
    // Read data from file
    float newval;
    for (int ii = 0; ii < nrows; ii++) {
        if (fscanf(fp, "%f\n", &newval) != 0) {
            data[ii] = newval;
        } else {
            return -2;
        }
    }
    fclose(fp);
    return 0;
}

template <class dataType, unsigned int nrows, unsigned int ncols>
int read_file_2D(const char *filename, dataType data[nrows][ncols]) {
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == 0) {
        return -1;
    }
    // Read data from file
    float newval;
    for (int ii = 0; ii < nrows; ii++) {
        for (int jj = 0; jj < ncols; jj++) {
            if (fscanf(fp, "%f\n", &newval) != 0) {
                data[ii][jj] = newval;
            } else {
                return -2;
            }
        }
    }
    fclose(fp);
    return 0;
}

template <class in_T, class out_T, int N_IN> void change_type(ac_channel<in_T> &in, ac_channel<out_T> &out) {
    in_T datareg;
    ac_channel<out_T> input_trunc;
    for (int ii = 0; ii < N_IN; ii++) {
        out << (out_T)in.read();
    }
}

template <class data_T, int N_IN> void hls_stream_debug(ac_channel<data_T> &data, ac_channel<data_T> &res) {
    data_T datareg;
    for (int ii = 0; ii < N_IN; ii++) {
        datareg = data.read();
        std::cout << "[" << ii << "]: " << datareg << std::endl;
        res << datareg;
    }
}

constexpr int ceillog2(int x) { return (x <= 2) ? 1 : 1 + ceillog2((x + 1) / 2); }

constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }

constexpr int pow2(int x) { return x == 0 ? 1 : 2 * pow2(x - 1); }

} // namespace nnet

#endif
