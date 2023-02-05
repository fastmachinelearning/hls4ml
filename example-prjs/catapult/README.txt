Use the script run_synth.sh to run the example design through HLS4ML + Catapult HLS + Vivado RTL
Use the script run_vivado.sh to run the example design through HLS4ML + Vivado HLS

Edit the python configuration in sample_config.py.
Things to change:

IOTypes: io_parallel, io_stream
config['IOTypes'] = 'io_stream'

Move weight arrays to the interface of the top function
#config['HLSConfig'['Model']['BramFactor']) = 0

The run_synth.sh script will automatically switch the Backend to 'Catapult' (and the output dir).
