This is small conv2d 1 layer example using the binary_tanh activation function. To run it:
1. Run the run_catapult.sh script.
2. Replace the binary_tanh_test.cpp in the my-Catapult-test with the binary_tanh_test_test.cpp a level up (if you would like the testbench to be self-checking).
3. Move tb_input_features.dat and tb_output_predictions.dat to my-Catapult-test/tb_data (if you want two pre-computed examples).
4. If testing LUT version of binary_tanh (remove "#define USE_AC_MATH) within the define.h file change conv2d_tanh_table_t accuracy to "typedef ac_fixed<32,2,true> conv2d_tanh_table_t".
5. In firmware/ternary_tanh.cpp change the function being called to binary_tanh instead of tanh.

Note: You can create your own image matrix or filter and get the predictions by editing and running ternary_tanh.py. 
