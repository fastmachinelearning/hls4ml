#include "BDT.h"
#include "parameters.h"
#include "../example-prjs/bdt/bdt_config.h"

score_t decision_function(input_arr_t x){
	#pragma HLS pipeline
	#pragma HLS array_partition variable=x

	return bdt.decision_function(x);
}
