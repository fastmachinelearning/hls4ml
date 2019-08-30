# Quick start

For basic concepts to understand the tool, please visit the <a href="../CONCEPTS.html">Concepts</a> chapter.  
Here we give line-by-line instructions on how to build a first project and explain the various steps to producing an HLS4ML IP.

## Running the tool

These instructions are for simply running the tool out-of-the-box and getting a feel for the workflow.  

**1) To clone our latest version from github and install it from the main directory:** 
```
git clone https://github.com/hls-fpga-machine-learning/hls4ml.git
cd hls4ml/
pip install .
```
**NOTE FOR THIS STEP:**
- If you don't have privileges to install in the configured environment, you can also pass `--user` to the `pip` command as well. You can also add `-U` or `--upgrade` to update all packages to the newest available version. 
- To enable develop mode: `pip install -e .`
- In the future, we plan to support `hls4ml` as a package on PyPI. After that you can simply install the software with: `pip install hls4ml`
- We are also planning on supporting `conda` installation for `hls4ml`.


**2) Translate your model using configuration files (`.yml`):** 

The model files, along with other configuration parameters, are defined in the `.yml` files.
Further information about `.yml` files can be found in <a href="CONFIGURATION.html">Configuration Section</a>.
We provide some examples of configuration files in `example-models` directory.

In order to create an example HLS project:

- Go to `example-models/` from the main directory: 

```
cd example-models/
```

- And use this command to translate a Keras model:

```
hls4ml convert -c keras-config.yml
```

This will create a new HLS project directory with an implementation of a model from the `example-models/keras/` directory.
To build the HLS project, do:

```
hls4ml build -p my-hls-test -a
```

This will create a Vivado HLS project with your model implmentation!

**NOTE:** For the last step, you can alternatively do the following to build the HLS project:

```
cd my-hls-test
vivado_hls -f build.tcl
```

`vivado_hls` can be controlled with:

```
vivado_hls -f build.tcl "csim=1 synth=1 cosim=1 export=1"
```

Setting the additional parameters to `1` to `0` disables that step, but disabling `synth` also disables `cosim` and `export`.

**3) Further help:**

- For further information about how to use `hls4ml`, do: `hls4ml --help` or `hls4ml -h`

- If you need help for a particular `command`, `hls4ml command -h` will show help for the requested `command`

**To uninstall `hls4ml`:** 
```
pip uninstall hls4ml
```

## Existing examples

Other examples of various HLS projects with examples of different machine learning algorithm implementations is in the directory: `example-prjs`.
