import os
import time
import ctypes
import logging
import numpy as np

class CoyoteOverlay:
    """
    CoyoteOverlay class, similar to NeuralNetworkOverlay for the VivadoAccelerator backend
    This class can be used to run model inference on the FPGA with the CoyoteAccelerator backend
    """
    def __init__(self, path: str, project_name: str = 'myproject'):
        """
        Default constructor

        Args:
            path (str): Path to the hls4ml folder, as specified in convert_model(...)
            project_name (str, optional): hls4ml model name, if different than myproject
        """

        self.path = path
        self.project_name = project_name

        # Set up dynamic C library
        self.coyote_lib = ctypes.cdll.LoadLibrary(
            f'{self.path}/build/{self.project_name}_cyt_sw/lib/libCoyoteInference.so'
        )

        self.coyote_lib.init_model_inference.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
        self.coyote_lib.init_model_inference.restype = ctypes.POINTER(ctypes.c_void_p)

        self.coyote_lib.flush.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.coyote_lib.predict.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

        self.coyote_lib.get_inference_predictions.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.coyote_lib.get_inference_predictions.restype = ctypes.POINTER(ctypes.c_float)

        self.coyote_lib.free_model_inference.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        
    def program_hacc_fpga(self):
        """
        Utility function for loading the Coyote-hls4ml bitstream and driver
        on the ETH Zurich Heteregenous Accelerate Compute Cluster (HACC)
        On other clusters, users would need to manually load the bitstream and driver
        Gudance on this is specified in Coyote docs.
        """
        os.system(
            f'cd {self.path}/Coyote/driver && '
            f'make && '  
            f'cd ../util && '
            f'bash program_hacc_local.sh ../../build/{self.project_name}_cyt_hw/bitstreams/cyt_top.bit ../driver/build/coyote_driver.ko'
        )

    def predict(self, X: np.array, y_shape: tuple, batch_size: int = 1):
        """
        Run model inference

        Args:
            X (np.array): Input data
            y_shape (tuple): Shape of the output; used for allocating sufficient memory for the output
            batch_size (int, optional): Inference batch size
        """
        if len(X.shape) == 1:
            X = np.array([X])
        if not (isinstance(X.dtype, float) or isinstance(X.dtype, np.float32)):
            logging.warning('CoyoteOverlay only supports (for now) floating-point inputs; casting input data to float')
            X = X.astype(np.float32)
        y = np.empty((len(X), *y_shape))
        np_pointer_nd = np.ctypeslib.ndpointer(dtype=np.float32, ndim=len(X[0].shape), flags='C')
        self.coyote_lib.set_inference_data.argtypes = [ctypes.POINTER(ctypes.c_void_p), np_pointer_nd, ctypes.c_uint]

        model = self.coyote_lib.init_model_inference(batch_size, int(np.prod(X[0].shape)), int(np.prod(y_shape)))
        
        cnt = 0
        avg_latency = 0
        avg_throughput = 0
        total_batches = 0
        for x in X:
            self.coyote_lib.set_inference_data(model, x, cnt)
            cnt += 1
            if cnt == batch_size:
                self.coyote_lib.flush(model)

                ts = time.time_ns()
                self.coyote_lib.predict(model)
                te = time.time_ns()

                time_taken = te - ts
                avg_latency += (time_taken / 1e3)
                avg_throughput += (batch_size / (time_taken * 1e-9))

                for j in range(batch_size):
                    tmp = self.coyote_lib.get_inference_predictions(model, j)
                    y[total_batches * batch_size + j] = np.ctypeslib.as_array(tmp, shape=y_shape)

                cnt = 0
                total_batches += 1

        self.coyote_lib.free_model_inference(model)
        print(f'Batch size: {batch_size}; batches processed: {total_batches}')
        print(f'Mean latency: {round(avg_latency / total_batches, 3)}us (inference only)')
        print(f'Mean throughput: {round(avg_throughput / total_batches, 1)} samples/s (inference only)')

        return y 