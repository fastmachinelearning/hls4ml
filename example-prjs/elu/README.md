This is small relu 1 layer example. To run it:
1. Run the run_catapult.sh script.
2. Replace the elu_test.cpp in the my-Catapult-test with the elu_test.cpp a level up (if you would like the testbench to be self-checking).
3. Move tb_input_features.dat and tb_output_predictions.dat to my-Catapult-test/tb_data (if you want two pre-computed examples).
4. If testing the LUT version of elu (remove "#define USE_AC_MATH) within the define.h file change conv2d_elu_table_t accuracy to "typedef ac_fixed<32,2,true> conv2d_elu_table_t".

Note: You can create your own array and get the predictions by editing then running elu.py. 
