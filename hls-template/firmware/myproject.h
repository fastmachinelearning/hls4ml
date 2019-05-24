//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "parameters.h"


#ifndef __SYNTHESIS__

#include <fstream>
#define xstr(a) str(a)
#define str(a) #a
template<class T, size_t SIZE>
void load_txt_file(T *w, const char* fname) {

    std::string full_path = std::string(xstr(WEIGHTS_DIR)) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname);
        std::cerr << " does not exist" << std::endl;
        exit(1);
    }

    size_t i = 0;
    size_t size;
    std::string line;

    // The first line of the input file contains the total number of values.
    if (std::getline(infile, line)) {
         std::istringstream iss(line);
         iss >> size;
         if (size != SIZE) {
            std::cerr << "ERROR: file " << std::string(fname);
            std::cerr << " contains an unexpected number of elements (";
            std::cerr << size << " rather than  "<< SIZE << ")" << std::endl;
            exit(1);
        }
    };

    // The second line of the input file contains all of the values.
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        for (size_t i = 0; i < size; i++) {
            double fdata;
            iss >> fdata;
            w[i] = T(fdata);
        }
    }
}

#endif


// Prototype of top level function for C-synthesis
void myproject(
    //hls-fpga-machine-learning insert header
);

#endif
