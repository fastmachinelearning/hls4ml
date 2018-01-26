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
  input_t  data_str[Y_INPUTS][N_CHAN] = {0.2794435, 0.04060272, -0.05572316, -0.1889425, 0.2780532, -0.1243143, -0.05448467, 0.1114198, -0.07794067, -0.09206132, -0.1932618, 0.1010775, 0.009639716, -0.06205285, 0.01033651, 0.2041055, 0.200241, -0.07361811, 0.3155678, 0.1373228, 0.3606532, -0.3027224, -0.07007817, 0.2620515};

    result_t res_str[N_OUTPUTS];
    for(int i=0; i<N_OUTPUTS; i++){
	    res_str[i]=0;
    }

    unsigned short size_in, size_out;
    myproject(data_str, res_str, size_in, size_out);

    result_t res_expected[N_OUTPUTS] = {0.528625502721, 0.352447456382, 0.11757160967, 0.000254924257721, 8.45225898784e-05, 0.00101598437936};
    
    for(int i=0; i<N_OUTPUTS; i++){
	std::cout << res_str[i] << " (expected " << res_expected[i] << ", " << 100.0*((float)res_str[i]-(float)res_expected[i])/(float)res_expected[i] << " percent difference)" << std::endl;
    }
    //std::cout << std::endl;
    
    return 0;
}
