# Example network that demonstrates HLS4ML with the Catapult HLS backend

- Use the script `create_env.sh` to create a python virtual environment suitable for running this example.
- Requires Python 3.7 to be installed. In case, you can follow this [instructions](https://tecadmin.net/install-python-3-7-on-centos).
- To run this example:
  ```
  bash
  cd $HOME/venv/hls4ml/example-prjs/catapult
  ./run_synth.sh
  ```
- Modify the options in `run_synth.sh` to turn off/on Vivado RTL synthesis after Catapult HLS.
- Modify `sample_config.py` to alter the HLS4ML configuration for the network.
