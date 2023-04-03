from datetime import datetime

import numpy as np
from pynq import Overlay, allocate


class NeuralNetworkOverlay(Overlay):
    def __init__(self, xclbin_name, dtbo=None, download=True, ignore_version=False, device=None):

        super().__init__(xclbin_name, dtbo=dtbo, download=download, ignore_version=ignore_version, device=device)
        self.input_buffer = None
        self.output_buffer = None

    def allocate_mem(self, X_shape, y_shape, dtype=np.float32, trg_in=None, trg_out=None):
        """
        Buffer allocation in the card memory
        Parameters
        ----------
        X_shape : input buffer shape.
        y_shape : output buffer shape.
        dtype   : the data type of the elements of the input/output vectors.
                  Note: it should be set depending on the interface of the accelerator; if it uses 'float'
                  types for the 'data' AXI-Stream field, 'np.float32' dtype is the correct one to use.
                  Instead if it uses 'ap_fixed<A,B>', 'np.intA' is the correct one to use (note that A cannot
                  any integer value, but it can assume {..., 8, 16, 32, ...} values. Check `numpy`
                  doc for more info).
                  In this case the encoding/decoding has to be computed by the host machine. For example for
                  'ap_fixed<16,6>' type the following 2 functions are the correct one to use for encode/decode
                  'float' -> 'ap_fixed<16,6>':
                  ```
                    def encode(xi):
                        return np.int16(round(xi * 2**10)) # note 2**10 = 2**(A-B)
                    def decode(yi):
                        return yi * 2**-10
                    encode_v = np.vectorize(encode) # to apply them element-wise
                    decode_v = np.vectorize(decode)
                  ```
        trg_in  : input buffer target memory. By default the v++ command
                  set it to HBM[0] for alveo-u50.
        trg_out : output buffer target memory.By default the v++ command
                  set it to HBM[0] for alveo-u50.

        Assigns
        -------
        input_buffer : input PYNQ buffer, must be allocated first and just once.
        output_buffer : output PYNQ buffer, must be allocated first and just once.
        input_buffer, output_buffer : input and output PYNQ buffers

        """
        self.input_buffer = allocate(shape=X_shape, dtype=dtype, target=trg_in)
        self.output_buffer = allocate(shape=y_shape, dtype=dtype, target=trg_out)

    def predict(self, X, y_shape, dtype=np.float32, debug=None, profile=False, encode=None, decode=None):
        """
        Obtain the predictions of the NN implemented in the FPGA.
        Parameters:
        - X : the input vector. Should be numpy ndarray.
        - y_shape : the shape of the output vector. Needed to the accelerator to set the TLAST bit properly and
                    for sizing the output vector shape.
        - dtype   : the data type of the elements of the input/output vectors.
        - debug : boolean, if set the function will print information about the data transfers status.
        - profile : boolean. Set it to `True` to print the performance of the algorithm in term of `inference/s`.
        - encode/decode: function pointers. See `dtype` section for more information.
        - return: an output array based on `np.ndarray` with a shape equal to `y_shape` and a `dtype` equal to
                  the namesake parameter.
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
