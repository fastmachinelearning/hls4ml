dofile ./build_prj.tcl
go analyze
solution design set myproject -block
solution design set nnet::relu<nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,64U>,nnet::array<ac_fixed<7,1,true,AC_RND_CONV,AC_SAT>,64U>,relu_config4> -top
go libraries
go extract
go analyze
solution design set nnet::relu<nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,64U>,nnet::array<ac_fixed<7,1,true,AC_RND_CONV,AC_SAT>,64U>,relu_config4> -block
solution design set nnet::relu<nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,32U>,nnet::array<ac_fixed<7,1,true,AC_RND_CONV,AC_SAT>,32U>,relu_config7> -top
go libraries
go extract
go analyze
solution design set nnet::relu<nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,32U>,nnet::array<ac_fixed<7,1,true,AC_RND_CONV,AC_SAT>,32U>,relu_config7> -block
solution design set nnet::relu<nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,32U>,nnet::array<ac_fixed<7,1,true,AC_RND_CONV,AC_SAT>,32U>,relu_config10> -top
go libraries
go extract
go analyze
solution design set nnet::relu<nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,32U>,nnet::array<ac_fixed<7,1,true,AC_RND_CONV,AC_SAT>,32U>,relu_config10> -block
solution design set nnet::softmax<nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,5U>,nnet::array<ac_fixed<16,6,false,AC_TRN,AC_WRAP>,5U>,softmax_config13> -top
go libraries
go extract
go analyze
solution design set nnet::softmax<nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,5U>,nnet::array<ac_fixed<16,6,false,AC_TRN,AC_WRAP>,5U>,softmax_config13> -block
solution design set nnet::dense<nnet::array<ac_fixed<7,1,true,AC_RND_CONV,AC_SAT>,32U>,nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,5U>,config11> -top
go libraries
go assembly
go architect
go allocate
go extract
go analyze
solution design set nnet::dense<nnet::array<ac_fixed<7,1,true,AC_RND_CONV,AC_SAT>,32U>,nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,32U>,config8> -top
solution design set nnet::dense<nnet::array<ac_fixed<7,1,true,AC_RND_CONV,AC_SAT>,32U>,nnet::array<ac_fixed<16,6,true,AC_TRN,AC_WRAP>,5U>,config11> -block
go libraries
go assembly
go architect
go allocate
