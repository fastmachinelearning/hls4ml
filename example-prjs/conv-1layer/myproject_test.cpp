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
  input_t  data_str[Y_INPUTS][N_CHAN] = {0.2794435, 0.04060272, -0.1661478, -0.05572316, -0.1889425, -0.2392852, 0.2780532, 0.9463616, -0.1243143, -0.06952011, 0.1452374, -0.2136291, 0.2173666, 0.2392023, -0.1315853, -0.3639361, 1.218793, -0.178556, -0.6426811, -0.4164166, -0.2609797, -0.03127378, -0.310254, -0.6907321, 0.5052217, 0.9687944, -0.2263131, 0.2636545, -0.05106907, -0.03182655, -0.2036392, -0.5492674, -0.3250218, 1.291949, 0.7964888, 0.009623515, -0.05448467, 0.1114198, 0.04741726, -0.07794067, -0.09206132, -0.1652722, -0.1932618, 0.9225838, 0.1010775, 0.5997802, -0.259151, -0.02667081, 0.4529535, -0.5766279, -0.5385103, -0.0262578, 1.450872, 0.03892736, 0.3081347, -0.1107511, -0.2609757, 0.04384806, -0.5914985, -0.3912915, -0.3458357, 1.040917, -0.1872596, 0.09905793, 0.1459248, -0.2183437, 0.03353712, 0.2807747, -0.1373495, 0.2192456, 1.0301, -0.1388814, 0.009639716, -0.06205285, -0.1586413, 0.01033651, 0.2041055, -0.3491349, 0.200241, 1.004057, -0.07361811, -0.2534722, -0.2555391, -0.2195919, -0.1729546, -1.371886, -0.5446452, -0.585857, 0.8165365, -0.1283241, 0.4001302, 0.2223, -0.1037138, -0.15401, -0.9962393, -0.06809918, 0.955946, 0.8334367, -0.004300367, 0.2350878, 0.1537797, 0.08465498, -0.2867855, -0.1118864, -0.1391232, -0.3144899, 0.7009632, 0.1942542, 0.3155678, 0.1373228, 0.1444861, 0.3606532, -0.3027224, -0.1568424, -0.07007817, 1.346409, 0.2620515, 0.3058458, -0.1755011, -0.3160568, 0.05408659, -0.6275515, -0.4691448, 0.05465026, 1.039109, -0.1956121, 0.0974554, 0.1191869, -0.232762, -0.02155651, 0.3255003, -0.1750441, 0.3312638, 0.9644814, -0.1130844, 0.004224028, 0.02650983, -0.2171846, 0.02620083, 0.4554119, -0.269497, 0.5259855, 1.011495, -0.09914717};

    result_t res_str[N_OUTPUTS];
    for(int i=0; i<N_OUTPUTS; i++){
	    res_str[i]=0;
    }

    unsigned short size_in, size_out;
    myproject(data_str, res_str, size_in, size_out);

    result_t res_expected[N_OUTPUTS] = {0.895853397443, 0.103402912154, 0.000741177860886, 9.78720788763e-08, 2.41452465432e-06, 1.45553795063e-10};
    
    for(int i=0; i<N_OUTPUTS; i++){
	std::cout << res_str[i] << " (expected " << res_expected[i] << ", " << 100.0*((float)res_str[i]-(float)res_expected[i])/(float)res_expected[i] << " percent difference)" << std::endl;
    }
    //std::cout << std::endl;
    
    return 0;
}
