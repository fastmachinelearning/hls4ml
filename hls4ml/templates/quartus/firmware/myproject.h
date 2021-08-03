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

#ifndef __INTELFPGA_COMPILER__
#include "ac_int.h"
#include "ac_fixed.h"
#define hls_register
#else
#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif

#include "parameters.h"

struct input_data {
  //hls-fpga-machine-learning insert inputs
};

struct output_data {
  //hls-fpga-machine-learning insert outputs
};


#ifndef __INTELFPGA_COMPILER__
output_data myproject(
  input_data inputs
);
#else
//hls-fpga-machine-learning insert cpragmas
component output_data myproject(
  input_data inputs
);
#endif

#endif
