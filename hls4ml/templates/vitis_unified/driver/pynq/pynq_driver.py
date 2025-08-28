# import the library
from pynq import Overlay     # import the overlay
from pynq import allocate    # import for CMA (contingeous memory allocation)
from pynq import DefaultIP   # import the ip connector library for extension
import numpy as np
import os
import subprocess
import re
import time


class MyDfxCtrl(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)

        self.REG_ADDR_AP_CTRL = 0x00
        self.REG_ADDR_AMT_QUERY = VAL

        self.INP_PORT_NAMEs = [
            #### hls-driver-input-dbg-name
        ]

        self.REG_ADDR_INP_PTRs = [
            #### hls-driver-input-ptr
        ]

        self.OUT_PORT_NAMEs = [
            #### hls-driver-output-dbg-name
        ]

        self.REG_ADDR_OUT_PTRs = [
            #### hls-driver-output-ptr
        ]

    bindto = ['xilinx.com:hls:<TOP_NAME>:1.0']

    ######## TODO interrupt


    def setSingleBit(self, addr, idx):
        self.write(addr, 1 << idx)

    def ctrlStart(self):
        self.write(0x00, 0x01)  # ap_start = 1

    def waitUntilDone(self):
        while (self.read(0x00) & 0x2) == 0:  # Wait for ap_done
            time.sleep(0.001)

    def setInput(self, idx, buffer):

        print(f"input {self.INP_PORT_NAMEs[idx]} will be set to addr: {buffer.physical_address} with elements: {buffer.size}")
        self.write(self.REG_ADDR_INP_PTRs[idx], buffer.physical_address)
        self.write(self.REG_ADDR_INP_PTRs[idx] + 4, 0)
        self.write(self.REG_ADDR_INP_SZs[idx], buffer.size)
        buffer.flush()

    def setOutput(self, idx, buffer):

        print(f"input {self.OUT_PORT_NAMEs[idx]} will be set to addr: {buffer.physical_address} with elements: {buffer.size}")
        self.write(self.REG_ADDR_OUT_PTRs[idx], buffer.physical_address)
        self.write(self.REG_ADDR_OUT_PTRs[idx] + 4, 0)
        self.write(self.REG_ADDR_OUT_SZs[idx], buffer.size)