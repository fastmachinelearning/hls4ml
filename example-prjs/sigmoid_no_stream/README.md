This is small conv2d 1 layer applying a sigmoid activation function example. To run it:
1. Run the run_catapult.sh script.
2. If testing the LUT version of sigmoid remove "#define USE_AC_MATH" in nnet_activation.h.
3. Within the define.h file change conv2d_sigmoid_table_t accuracy to "typedef ac_fixed<32,2,true> conv2d_sigmoid_table_t".
4. Replace the sigmoid_test.cpp in the my-Catapult-test with the sigmoid_test.cpp a level up (if you would like the testbench to be self-checking).
5. Move tb_input_features.dat and tb_output_predictions.dat to my-Catapult-test/tb_data (if you want two pre-computed examples).
6. Compile:
/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/bin/g++ -g -std=c++11 -DSC_INCLUDE_DYNAMIC_PROCESSES -Wl,-rpath=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/lib,-rpath=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/shared/lib ./sigmoid_test.cpp ./firmware/sigmoid.cpp -I/wv/scratch-baimar9c/venv/hls4ml/example-prjs/sigmoid_no_stream/my-Catapult-test -I/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/shared/include -L/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/shared/lib -Wl,-Bstatic -lsystemc -Wl,-Bdynamic -lpthread -o /wv/scratch-baimar9c/venv/hls4ml/example-prjs/sigmoid_no_stream/my-Catapult-test/sigmoid

Note: You can create your own array and get the predictions by editing then running sigmoid.py. 
