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
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/parameters.h"
#include "firmware/myproject.h"
#include "nnet_helpers.h"


int main(int argc, char **argv)
{
    
    //hls-fpga-machine-learning insert data
    //input_t  data_str[Y_INPUTS][N_CHAN];
    //for(int i=0; i<Y_INPUTS; i++){
    //    for(int j=0; j<N_CHAN; j++){
    //	    data_str[i][j]=1;
    //	}
    //}
    input_t  data_str[Y_INPUTS][N_CHAN] = {0.0001808515, 0.01076681, 0.01013856, 0.00657948, 0.009275574, 0.008928878, 0.005065897, 0.007488683, 0.01081025, 0.006140966, 0.004045101, 0.006944317, 0.004757375, 0.003552423, 0.006213647, 0.002537128, 0.003306744, 0.004084532, 0.007571941, 0.005118448, 0.005407117, 0.005892268, 0.006221276, 0.007249871, 0.006895028, 0.009790774, 0.00461026, 0.007661138, 0.007331333, 0.00518936, 0.005077667, 0.006823079};


    result_t res_str[N_OUTPUTS];
    for(int i=0; i<N_OUTPUTS; i++){
	    res_str[i]=0;
    }

    unsigned short size_in, size_out;
    myproject(data_str, res_str, size_in, size_out);

    result_t res_expected[N_OUTPUTS] = {0.05420331,  0.03950224,  0.01763429,  0.28226044,  0.3246591,   0.28174062};
    
    for(int i=0; i<N_OUTPUTS; i++){
	std::cout << res_str[i] << " (expected " << res_expected[i] << ", " << 100.0*((float)res_str[i]-(float)res_expected[i])/(float)res_expected[i] << " percent difference)" << std::endl;
    }
    //std::cout << std::endl;
    
    return 0;
}
