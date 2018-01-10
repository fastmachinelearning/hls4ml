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
    input_t  data_str[Y_INPUTS][N_CHAN] = {0.0001808515, -0.1232167, 1.012817, 0.01076681, 0.01013856, -0.1268756, 1.022833, 0.00657948, 0.009275574, -0.1240037, 1.022028, 0.008928878, 0.005065897, -0.1249279, 1.017877, 0.007488683, 0.01081025, -0.1257667, 1.02368, 0.006140966, 0.004045101, -0.124462, 1.016974, 0.006944317, 0.004757375, -0.1273606, 1.017746, 0.003552423, 0.006213647, -0.1278912, 1.019263, 0.002537128, 0.003306744, -0.1258682, 1.016417, 0.004084532, 0.007571941, -0.1243682, 1.020745, 0.005118448, 0.005407117, -0.1231382, 1.018643, 0.005892268, 0.006221276, -0.1213345, 1.019521, 0.007249871, 0.006895028, -0.1183578, 1.02026, 0.009790774, 0.00461026, -0.120062, 1.018041, 0.007661138, 0.007331333, -0.1221186, 1.020829, 0.00518936, 0.005077667, -0.12008, 1.018644, 0.006823079, 0.005762556, -0.1209017, 1.019398, 0.005606931, 0.006692748, -0.1213949, 1.020399, 0.004729647, 0.005443238, -0.1215677, 1.019222, 0.004183207, 0.008240952, -0.1246812, 1.022093, 0.0007066339, 0.006506451, -0.1254896, 1.020433, -0.0004544368, 0.006531523, -0.1249345, 1.020534, -0.0002410674, 0.007422441, -0.1249063, 1.021503, -0.0005438688, 0.005771769, -0.1249926, 1.019931, -0.0009499844, 0.006240187, -0.1251552, 1.02048, -0.001421207, 0.004622982, -0.1247985, 1.018945, -0.001361999, 0.004832962, -0.1254793, 1.019238, -0.00232884, 0.005498746, -0.1268068, 1.019989, -0.00393101, 0.004341186, -0.1272888, 1.018917, -0.004676219, 0.005098018, -0.123713, 1.019762, -0.00135205, 0.004268742, -0.1192631, 1.019021, 0.002857771, 0.003045187, -0.1226967, 1.017887, -0.0008045508};


    result_t res_str[N_OUTPUTS];
    for(int i=0; i<N_OUTPUTS; i++){
	    res_str[i]=0;
    }

    unsigned short size_in, size_out;
    myproject(data_str, res_str, size_in, size_out);

    result_t res_expected[N_OUTPUTS] = {  1.87593365e-02,   5.61190830e-04,   6.72846166e-05,   1.25997232e-01,
					  8.54614956e-01,   8.57646913e-12};
    
    for(int i=0; i<N_OUTPUTS; i++){
	std::cout << res_str[i] << " (expected " << res_expected[i] << ", " << 100.0*((float)res_str[i]-(float)res_expected[i])/(float)res_expected[i] << " percent difference)" << std::endl;
    }
    //std::cout << std::endl;
    
    return 0;
}
