#include "BDT.h"
#include "parameters.h"
#include "bdt_eg.h"
#include "bdt_config.h"

score_t bdt_eg(input_arr_t x){
	#pragma HLS pipeline II = 1
	#pragma HLS unroll factor = 1
	#pragma HLS array_partition variable=x

	return bdt.decision_function(x);
}