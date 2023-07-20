This is small conv2d 1 layer example. To run it:
1. Run the run_catapult.sh script.
2. Replace the tanh_test.cpp in the my-Catapult-test with the tanh_test_test.cpp a level up (if you would like the testbench to be self-checking).
3. Move tb_input_features.dat and tb_output_predictions.dat to my-Catapult-test/tb_data (if you want two pre-computed examples).
4. Comment out "#define USE_AC_MATH" in nnet_activation.h to test lookup table implementation.
5. Within the define.h file change conv2d_sigmoid_table_t accuracy to "typedef ac_fixed<32,2,false> conv2d_tanh_table_t".

Note: You can create your own image matrix or filter and get the predictions by editing and running tanh.py. 
