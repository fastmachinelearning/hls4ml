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
        #####################
        self.REG_ADDR_GIE       = 0x04
        self.REG_ADDR_IER       = 0x08
        self.REG_ADDR_ISR       = 0x0C




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

    def enable_gie(self):
        print("global interrupt enable register")
        self.write(self.REG_ADDR_GIE, 0x01)
        print("enable gie successful")

    def disable_gie(self):
        print("global interrupt enable register")
        self.write(self.REG_ADDR_GIE, 0x01)
        print("disable gie successful")

    def enable_done_intr(self):
        print("ap_done interrupt enable register")
        self.write(self.REG_ADDR_IER, 0x01)
        print("enable ap_done interrupt successful")

    def clear_done_status(self):
        print("ap_done register clear")
        self.write(self.REG_ADDR_ISR, 0x01)
        print("clear ap_done interrupt successful")

    def prepare_intr(self):
        print("prepare your interrupt")
        self.enable_gie()
        self.enable_done_intr()
        self.clear_done_status()
        print("----------------------")


    def set_single_bit(self, addr, idx):
        self.write(addr, 1 << idx)

    def ctrl_start(self):
        self.write(0x00, 0x01)  # ap_start = 1

    def wait_until_done(self):
        while (self.read(0x00) & 0x2) == 0:  # Wait for ap_done
            time.sleep(0.001)

    def set_input(self, idx, buffer):

        print(f"input {self.INP_PORT_NAMEs[idx]} will be set to addr: {hex(buffer.physical_address)} with elements: {buffer.size}")
        self.write(self.REG_ADDR_INP_PTRs[idx], buffer.physical_address)
        self.write(self.REG_ADDR_INP_PTRs[idx] + 4, 0)
        buffer.flush()

    def set_output(self, idx, buffer):

        print(f"output {self.OUT_PORT_NAMEs[idx]} will be set to addr: {hex(buffer.physical_address)} with elements: {buffer.size}")
        self.write(self.REG_ADDR_OUT_PTRs[idx], buffer.physical_address)
        self.write(self.REG_ADDR_OUT_PTRs[idx] + 4, 0)

    def set_amt_query(self, val):
        print(f"amount of queries will be set to: {val} at address: {hex(self.REG_ADDR_AMT_QUERY)}")
        self.write(self.REG_ADDR_AMT_QUERY, val)