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

  //2x2 example:
  input_t  X_str[N_NODES][N_FEATURES] = {0.03186538815498352, 0.4396468997001648, 0.027646400034427643, 0.07125214487314224, 0.41369643807411194, 0.06305709481239319, 0.0314718633890152, 0.7407453060150146, 0.01103190053254366, 0.0716928243637085, 0.7647077441215515, 0.028442300856113434};
  ap_uint<1> Ri_str[N_NODES][N_EDGES] =  {0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1};
  ap_uint<1> Ro_str[N_NODES][N_EDGES] =  {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0};

  //3x4 example:
  //input_t X_str[N_NODES][N_FEATURES] = {0.03206074237823486, 0.08260346949100494, -0.03793669864535332, 0.07236992567777634, 0.11904784291982651, -0.09039219468832016, 0.11681490391492844, 0.15196996927261353, -0.14816400408744812, 0.03199561685323715, 0.6990280747413635, -0.06354690343141556, 0.07234000414609909, 0.7281892895698547, -0.15507100522518158, 0.11638729274272919, 0.7579716444015503, -0.25495898723602295, 0.03210893273353577, -0.8778975009918213, 0.09253199398517609, 0.07271277159452438, -0.8508699536323547, 0.21608400344848633, 0.11491114646196365, -0.8255922198295593, 0.3443180024623871};
  //ap_uint<1> Ri_str[N_NODES][N_EDGES] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1};
  //ap_uint<1> Ro_str[N_NODES][N_EDGES] = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  
  result_t e_str[N_EDGES][1];
  for(int i=0; i<N_EDGES; i++){
    e_str[i][0]=0;
  }
  

  unsigned short size_in, size_out;
  myproject(X_str, Ri_str, Ro_str, e_str, size_in, size_out);
    
  std::cout << "e = " << std::endl;
  for(int i=0; i<N_EDGES; i++){
    std::cout << e_str[i][0] << " ";
  }
  std::cout << std::endl;
  
  return 0;
}

