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
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/parameters.h"
#include "firmware/myproject.h"
//#include "nnet_helpers.h"

#include "galapagos_stream.hpp"

#define DEST 0

void myproject_galapagos(galapagos_stream <float> * in, galapagos_stream <float> * out)
{

    input_t data_str[Y_INPUTS][N_CHAN];
    result_t res_str[N_OUTPUTS];

    unsigned short size_in, size_out;
    //galapagos_packet gp;
    galapagos_packet <float> gp;
    for(int i=0; i<Y_INPUTS; i++){
        for(int j=0; j<N_CHAN; j++){
            gp = in->read();
            data_str[i][j] = (input_t)gp.data;
         //   std::cout << "received[ " << i << "][" << j << "]" <<  data_str[i][j] << std::endl;
        }
    }

    myproject(data_str, res_str, size_in, size_out);


    for(int i=0; i<N_OUTPUTS; i++){
        gp.data = res_str[i];
        if(i == N_OUTPUTS-1)
            gp.last = 1;
        else
            gp.last = 0;
        gp.dest = DEST;
        out->write(gp);
    }
    
}
