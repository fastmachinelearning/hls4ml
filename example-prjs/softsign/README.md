This is small conv2d 1 layer with a softsign activation function example. To run it:
1. Run "python softsign.py" to generate the json and h5 files.
2. Run "python catapult.py" to run catapult with the config.yml file.
3. Run the run_catapult.sh script if you have made changes and would like a fresh my-Catapult-test directory.
4. Replace the softsign_test.cpp in the my-Catapult-test with the softsign_test.cpp a level up (if you would like the testbench to be self-checking).
5. Move tb_input_features.dat and tb_output_predictions.dat to my-Catapult-test/tb_data (if you want two pre-computed examples).
6. If testing LUT version of softsign (remove "#define USE_AC_MATH) within the define.h file change conv2d_softsign_table_t accuracy to "typedef ac_fixed<32,2,true> conv2d_softsign_table_t".

Note: You can create your own image matrix or filter and get the predictions by editing then running softsign.py. 
