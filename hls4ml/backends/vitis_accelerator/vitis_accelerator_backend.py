import os
import subprocess
import sys

import numpy as np

from hls4ml.backends import VitisBackend, VivadoBackend, VitisAcceleratorConfig
from hls4ml.model.flow import get_flow, register_flow
import ctypes

class VitisAcceleratorBackend(VitisBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name="VitisAccelerator")
        self._register_layer_attributes()
        self._register_flows()

    def create_initial_config(
        self,
        board="alveo-u55c",
        platform=None,
        part=None,
        clock_period=5,
        clock_uncertainty='27%',
        io_type="io_parallel",
        num_kernel=1,
        num_worker=1,
        batchsize=8192,
        hw_quant=False,
        vivado_directives=None,
        **_,
    ):
        """
        Create initial accelerator config with default parameters

        Args:
            board: one of the keys defined in supported_boards.json
            clock_period: clock period passed to hls project
            io_type: io_parallel or io_stream
            num_kernel: how many compute units to create on the fpga
            num_worker: how many threads the host cpu uses to drive each CU on the fpga
            batchsize: how many samples to process within a single buffer on the fpga
            vivado_directives: Directives passed down to Vivado that controls the hardware synthesis and implementation steps
        Returns:
            populated config
        """
        board = board if board is not None else "alveo-u55c"
        config = super().create_initial_config(part, clock_period, clock_uncertainty, io_type)
        config["AcceleratorConfig"] = {}
        config["AcceleratorConfig"]["Board"] = board
        config["AcceleratorConfig"]["Platform"] = platform
        config["AcceleratorConfig"]["Num_Kernel"] = num_kernel
        config["AcceleratorConfig"]["Num_Worker"] = num_worker
        config["AcceleratorConfig"]["Batchsize"] = batchsize
        config["AcceleratorConfig"]["HW_Quant"] = hw_quant
        config["AcceleratorConfig"]["Vivado_Directives"] = vivado_directives
        return config

    def build(
        self,
        model,
        reset=False,
        target="hw",
        debug=False,
        **kwargs,
    ):
        self._validate_target(target)

        if "linux" in sys.platform:

            curr_dir = os.getcwd()
            os.chdir(model.config.get_output_dir())

            command = f"TARGET={target} "

            if debug:
                command += "DEBUG=1 "

            command += " make all"

            # Cleaning
            if reset:
                os.system(f"TARGET={target} make clean")

            # Pre-loading libudev
            ldconfig_output = subprocess.check_output(["ldconfig", "-p"]).decode("utf-8")
            for line in ldconfig_output.split("\n"):
                if "libudev.so" in line and "x86" in line:
                    command = "LD_PRELOAD=" + line.split("=>")[1].strip() + " " + command
                    break
            os.system(command)

            os.chdir(curr_dir)
        else:
            raise Exception("Currently untested on non-Linux OS")

    def numpy_to_dat(self, model, x):
        if len(model.get_input_variables()) != 1:
            raise Exception("Currently unsupported for multi-input/output projects")

        # Verify numpy array of correct shape
        expected_shape = model.get_input_variables()[0].size()
        actual_shape = np.prod(x.shape[1:])
        if expected_shape != actual_shape:
            raise Exception(f"Input shape mismatch, got {x.shape}, expected (_, {expected_shape})")

        # Write to tb_data/tb_input_features.dat
        samples = x.reshape(x.shape[0], -1)
        input_dat = f"{model.config.get_output_dir()}/tb_data/tb_input_features.dat"
        np.savetxt(input_dat, samples, fmt="%.4e")

    def dat_to_numpy(self, model):
        expected_shape = model.get_output_variables()[0].size()
        output_file = f"{model.config.get_output_dir()}/tb_data/hw_results.dat"
        y = np.loadtxt(output_file, dtype=float).reshape(-1, expected_shape)
        return y

    def hardware_predict(self, model, x, target="hw", debug=False, profilingRepeat=-1, method="file"):
        if method == "file":
            """Run the hardware prediction using file-based communication."""
            command = ""

            if debug:
                command += "DEBUG=1 "
            if isinstance(profilingRepeat, int) and profilingRepeat > 0:
                command += "PROFILING_DATA_REPEAT_COUNT=" + profilingRepeat + " "
            self._validate_target(target)

            self.numpy_to_dat(model, x)

            currdir = os.getcwd()
            os.chdir(model.config.get_output_dir())
            command += "TARGET=" + target + " make run"
            os.system(command)
            os.chdir(currdir)

            return self.dat_to_numpy(model)
        
        elif method == "lib":
            """Run the hardware prediction using a shared library."""
            # Set array to contiguous memory layout
            X_test = np.ascontiguousarray(x, dtype=np.float64)

            # Create prediction array
            config = VitisAcceleratorConfig(model.config)
            batchsize = config.get_batchsize()
            originalSampleCount = X_test.shape[0]
            numBatches = int(np.ceil(originalSampleCount / batchsize))
            sampleOutputSIze = model.get_output_variables()[0].size()
            predictions_size = numBatches * batchsize * sampleOutputSIze
            predictions = np.zeros(predictions_size, dtype=np.float64)
            predictions_size = predictions.shape[0]
            predictions_ptr = predictions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


            # Flatten the input data
            X_test_flat = X_test.flatten()
            X_test_size = X_test_flat.shape[0]
            X_test_flat = X_test_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            
            # Print debug information
            print(f"Batch size: {batchsize}")
            print(f"Original sample count: {originalSampleCount}")
            print(f"Number of batches: {numBatches}")
            print(f"Sample output size: {sampleOutputSIze}")
            print(f"Predictions size: {predictions_size}")
            print(f"X_test_shape: {X_test.shape}")
            print(f"X_test_flat_shape: {X_test_size}")

            # Change working directory to the HLS project directory
            currdir = os.getcwd()
            os.chdir(model.config.get_output_dir())

            # Check if the shared library exists
            if not os.path.exists('./lib_host.so'):
                print("Shared library 'lib_host.so': host code not compiled as a shared library.")
                print("Compiling host code as a shared library...")
                os.system('make host_shared')
                print("Shared library 'lib_host.so' compiled successfully.")

            # Load the shared library
            lib = ctypes.cdll.LoadLibrary('./lib_host.so')

            # Call the predict function
            lib.predict.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
            lib.predict(X_test_flat, X_test_size, predictions_ptr, predictions_size)

            # Change back to the original directory
            os.chdir(currdir)


            # Reshape the predictions to match the expected output shape
            y_hls = predictions.reshape(-1, sampleOutputSIze)[:originalSampleCount, :]
            return y_hls

        else:
            raise Exception(f"Unsupported method {method} for hardware prediction")

    def _register_flows(self):
        validation_passes = [
            "vitisaccelerator:validate_conv_implementation",
            "vitisaccelerator:validate_strategy",
        ]
        validation_flow = register_flow(
            "validation",
            validation_passes,
            requires=["vivado:init_layers"],
            backend=self.name,
        )

        # Any potential templates registered specifically for Vitis backend
        template_flow = register_flow(
            "apply_templates",
            self._get_layer_templates,
            requires=["vivado:init_layers"],
            backend=self.name,
        )

        writer_passes = ["make_stamp", "vitisaccelerator:write_hls"]
        self._writer_flow = register_flow("write", writer_passes, requires=["vitis:ip"], backend=self.name)

        ip_flow_requirements = get_flow("vivado:ip").requires.copy()
        ip_flow_requirements.insert(ip_flow_requirements.index("vivado:init_layers"), validation_flow)
        ip_flow_requirements.insert(ip_flow_requirements.index("vivado:apply_templates"), template_flow)

        self._default_flow = register_flow("ip", None, requires=ip_flow_requirements, backend=self.name)

    def _validate_target(self, target):
        if target not in ["hw", "hw_emu", "sw_emu"]:
            raise Exception("Invalid target, must be one of 'hw', 'hw_emu' or 'sw_emu'")
