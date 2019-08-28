# Quick start

For basic concepts to understand the tool, please visit the <a href="../CONCEPTS.html">Concepts</a> chapter.  
Here we give line-by-line instructions on how to build a first project and explain the various steps to producing an HLS4ML IP.

## Running the tool

These instructions are for simply running the tool out-of-the-box and getting a feel for the workflow.  

**1) First clone our latest version from github and install it from the main directory:** 
```
git clone https://github.com/hls-fpga-machine-learning/hls4ml.git
cd hls4ml/
pip install -U .
```
**NOTE FOR THIS STEP:**
- If you don't have privileges to install in the configured environment, you can also pass `--user` to the `pip` command as well. 
- In the future, we plan to support `hls4ml` as a package on PyPI. After that you can simply install the software with `pip install hls4ml`

**2) Translate your model using configuration files (`.yml`):** 

The model files, along with other configuration parameters, are defined in the `.yml` files.
We provide some examples of configuration files in `example-models` directory.

In order to create an example HLS project:

- Go to `example-models/` from the main directory: `cd example-models/`

- And use this command to translate a Keras model:
`hls4ml convert -c keras-config.yml`

This will create a new HLS project directory with an implementation of a model from the `example-models` directory.
To run the HLS project, do:

```
cd my-hls-test
vivado_hls -f build.tcl
```

This will create a Vivado HLS project with your model implmentation!

**3) For further help:**

For further information about how to use `hls4ml`, do: `hls4ml --help`

**To uninstall `hls4ml`:** `pip uninstall hls4ml`

## Existing examples

Other examples of various HLS projects with examples of different machine learning algorithm implementations is in the directory: `example-prjs`.
