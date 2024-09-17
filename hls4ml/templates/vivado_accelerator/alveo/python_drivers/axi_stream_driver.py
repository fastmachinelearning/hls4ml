from datetime import datetime

import numpy as np
from pynq import Overlay, allocate


class NeuralNetworkOverlay(Overlay):
    def __init__(self, xclbin_name, dtbo=None, download=True, ignore_version=False, device=None):
        super().__init__(xclbin_name, dtbo=dtbo, download=download, ignore_version=ignore_version, device=device)
        self.input_buffer = None
        self.output_buffer = None

    def allocate_mem(self, X_shape, y_shape, dtype=np.float32, trg_in=None, trg_out=None):
        """Buffer allocation in the accelerator's memory.

        Args:
            X_shape (list): Input buffer shape.
            y_shape (list): Output buffer shape.
            dtype (dtype, optional): The data type of the elements of the input/output tensors. Must be an instance of
                numpy dtype. Defaults to np.float32.

                It should be set depending on the interface of the accelerator; if it uses 'float'
                data type for the 'data' AXI-Stream field, 'np.float32' dtype must be used. Instead if it uses
                'ap_fixed<A,B>', 'np.intA' is the correct dtype to use. Note that A cannot any integer value, but it can
                assume power of 2 values, i.e., {..., 8, 16, 32, ...}. Check `numpy` documentation for more information.
                In this case the encoding/decoding has to be computed by the host machine. For example for
                'ap_fixed<16,6>' type the following 2 functions are the correct one to use for encode/decode
                'float' -> 'ap_fixed<16,6>'::

                    def encode(xi):
                        return np.int16(round(xi * 2**10)) # note 2**10 = 2**(A-B)
                    def decode(yi):
                        return yi * 2**-10
                    encode_v = np.vectorize(encode) # to apply them element-wise
                    decode_v = np.vectorize(decode)

            trg_in (optional): Input buffer target memory. By default the v++ command set it to HBM[0] for
                alveo-u50. Defaults to None.
            trg_out (optional): Output buffer target memory. By default the v++ command set it to HBM[0] for
                alveo-u50. Defaults to None.
        """
        self.input_buffer = allocate(shape=X_shape, dtype=dtype, target=trg_in)
        self.output_buffer = allocate(shape=y_shape, dtype=dtype, target=trg_out)

    def predict(self, X, y_shape, dtype=np.float32, debug=False, profile=False, encode=None, decode=None):
        """Obtain the predictions of the NN implemented in the FPGA.

        Args:
            X (ndarray): The input tensor.
            y_shape (list): The shape of the output tensor, needed by the accelerator to set the TLAST bit properly.
            dtype (dtype, optional): The data type of the elements of the input/output tensors. Must be an instance of
                numpy dtype. Defaults to np.float32.
            debug (bool, optional): If set, the function will print information about the data transfers status.
                Defaults to False.
            profile (bool, optional): If set, the function will print the performance of the algorithm in terms of
                inference/s. Defaults to False.
            encode (Callable, optional): Function to transform the input tensor. Defaults to None.
            decode (Callable, optional): Function to transform the output tensor. Defaults to None.

        Returns:
            _type_: A ``np.ndarray`` with a shape equal of ``y_shape`` and ``dtype`` data type.
        """
        self.allocate_mem(X_shape=X.shape, y_shape=y_shape, dtype=dtype)
        if profile:
            timea = datetime.now()
        if encode is not None:
            X = encode(X)
        in_size = np.prod(X.shape)
        out_size = np.prod(y_shape)
        self.input_buffer[:] = X
        self.input_buffer.sync_to_device()
        if debug:
            print("Send OK")
        self.krnl_rtl_1.call(self.input_buffer, self.output_buffer, in_size, out_size)
        if debug:
            print("Kernel call OK")
        self.output_buffer.sync_from_device()
        if debug:
            print("Recieve OK")
        result = self.output_buffer.copy()
        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))
            self.input_buffer.flush()
            self.output_buffer.flush()
            self.free()
            return result, dts, rate
        self.input_buffer.flush()
        self.output_buffer.flush()
        return result

    def free_overlay(self):
        self.free()

    def _print_dt(self, timea, timeb, N):
        dt = timeb - timea
        dts = dt.seconds + dt.microseconds * 10**-6
        rate = N / dts
        print(f"Classified {N} samples in {dts} seconds ({rate} inferences / s)")
        print(f"Or {1 / rate * 1e6} us / inferences")
        return dts, rate
