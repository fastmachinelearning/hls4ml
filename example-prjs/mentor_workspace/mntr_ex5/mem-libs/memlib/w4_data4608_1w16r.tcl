flow package require MemGen
flow run /MemGen/MemoryGenerator_BuildLib {
VENDOR           Xilinx
RTLTOOL          Vivado
TECHNOLOGY       KINTEX-u
LIBRARY          mnist_mlp_w4_RAMS
MODULE           w4_data4608_1w16r
OUTPUT_DIR       memlib 
FILES {
  { FILENAME /home/giuseppe/research/projects/fastml/hls4ml-mentor.git/example-prjs/mentor_workspace/mntr_ex5/memlibs/memlib/w4_data4608_1w16r.v          FILETYPE Verilog MODELTYPE synthesis PARSE 1 PATHTYPE copy STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME /home/giuseppe/research/projects/fastml/hls4ml-mentor.git/example-prjs/mentor_workspace/mntr_ex5/memlibs/scripts/memgen/virtex7/BRAM_512x32.v FILETYPE Verilog MODELTYPE generic   PARSE 1 PATHTYPE copy STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME /tools/Xilinx/Vivado/2019.1/data/verilog/src/retarget/RAMB16_S36_S36.v                                                                        FILETYPE Verilog MODELTYPE generic   PARSE 1 PATHTYPE copy STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME /tools/Xilinx/Vivado/2019.1/data/verilog/src/unisims/RAMB36E1.v                                                                               FILETYPE Verilog MODELTYPE generic   PARSE 1 PATHTYPE copy STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME /tools/Xilinx/Vivado/2019.1/data/verilog/src/glbl.v                                                                                           FILETYPE Verilog MODELTYPE generic   PARSE 1 PATHTYPE copy STATICFILE 1 VHDL_LIB_MAPS work }
}
VHDLARRAYPATH    {}
WRITEDELAY       0.1
INITDELAY        1
READDELAY        0.1
VERILOGARRAYPATH {}
INPUTDELAY       0.01
TIMEUNIT         1ns
WIDTH            4608
AREA             1152
RDWRRESOLUTION   RBW
WRITELATENCY     1
READLATENCY      1
DEPTH            16
PARAMETERS {
}
PORTS {
  { NAME port_0  MODE Write }
  { NAME port_1  MODE Read  }
  { NAME port_2  MODE Read  }
  { NAME port_3  MODE Read  }
  { NAME port_4  MODE Read  }
  { NAME port_5  MODE Read  }
  { NAME port_6  MODE Read  }
  { NAME port_7  MODE Read  }
  { NAME port_8  MODE Read  }
  { NAME port_9  MODE Read  }
  { NAME port_10 MODE Read  }
  { NAME port_11 MODE Read  }
  { NAME port_12 MODE Read  }
  { NAME port_13 MODE Read  }
  { NAME port_14 MODE Read  }
  { NAME port_15 MODE Read  }
  { NAME port_16 MODE Read  }
}
PINMAPS {
  { PHYPIN CLK  LOGPIN CLOCK        DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS {port_0 port_1 port_2 port_3 port_4 port_5 port_6 port_7 port_8 port_9 port_10 port_11 port_12 port_13 port_14 port_15 port_16} }
  { PHYPIN CE0  LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_0                                                                                                                          }
  { PHYPIN A0   LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_0                                                                                                                          }
  { PHYPIN D0   LOGPIN DATA_IN      DIRECTION in  WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_0                                                                                                                          }
  { PHYPIN WE0  LOGPIN WRITE_ENABLE DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_0                                                                                                                          }
  { PHYPIN WEM0 LOGPIN WRITE_MASK   DIRECTION in  WIDTH 4608.0 PHASE 1  DEFAULT {} PORTS port_0                                                                                                                          }
  { PHYPIN CE1  LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_1                                                                                                                          }
  { PHYPIN A1   LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_1                                                                                                                          }
  { PHYPIN Q1   LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_1                                                                                                                          }
  { PHYPIN CE2  LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_2                                                                                                                          }
  { PHYPIN A2   LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_2                                                                                                                          }
  { PHYPIN Q2   LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_2                                                                                                                          }
  { PHYPIN CE3  LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_3                                                                                                                          }
  { PHYPIN A3   LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_3                                                                                                                          }
  { PHYPIN Q3   LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_3                                                                                                                          }
  { PHYPIN CE4  LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_4                                                                                                                          }
  { PHYPIN A4   LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_4                                                                                                                          }
  { PHYPIN Q4   LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_4                                                                                                                          }
  { PHYPIN CE5  LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_5                                                                                                                          }
  { PHYPIN A5   LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_5                                                                                                                          }
  { PHYPIN Q5   LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_5                                                                                                                          }
  { PHYPIN CE6  LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_6                                                                                                                          }
  { PHYPIN A6   LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_6                                                                                                                          }
  { PHYPIN Q6   LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_6                                                                                                                          }
  { PHYPIN CE7  LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_7                                                                                                                          }
  { PHYPIN A7   LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_7                                                                                                                          }
  { PHYPIN Q7   LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_7                                                                                                                          }
  { PHYPIN CE8  LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_8                                                                                                                          }
  { PHYPIN A8   LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_8                                                                                                                          }
  { PHYPIN Q8   LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_8                                                                                                                          }
  { PHYPIN CE9  LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_9                                                                                                                          }
  { PHYPIN A9   LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_9                                                                                                                          }
  { PHYPIN Q9   LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_9                                                                                                                          }
  { PHYPIN CE10 LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_10                                                                                                                         }
  { PHYPIN A10  LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_10                                                                                                                         }
  { PHYPIN Q10  LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_10                                                                                                                         }
  { PHYPIN CE11 LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_11                                                                                                                         }
  { PHYPIN A11  LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_11                                                                                                                         }
  { PHYPIN Q11  LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_11                                                                                                                         }
  { PHYPIN CE12 LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_12                                                                                                                         }
  { PHYPIN A12  LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_12                                                                                                                         }
  { PHYPIN Q12  LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_12                                                                                                                         }
  { PHYPIN CE13 LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_13                                                                                                                         }
  { PHYPIN A13  LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_13                                                                                                                         }
  { PHYPIN Q13  LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_13                                                                                                                         }
  { PHYPIN CE14 LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_14                                                                                                                         }
  { PHYPIN A14  LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_14                                                                                                                         }
  { PHYPIN Q14  LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_14                                                                                                                         }
  { PHYPIN CE15 LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_15                                                                                                                         }
  { PHYPIN A15  LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_15                                                                                                                         }
  { PHYPIN Q15  LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_15                                                                                                                         }
  { PHYPIN CE16 LOGPIN PORT_ENABLE  DIRECTION in  WIDTH 1.0     PHASE 1  DEFAULT {} PORTS port_16                                                                                                                         }
  { PHYPIN A16  LOGPIN ADDRESS      DIRECTION in  WIDTH 4.0     PHASE {} DEFAULT {} PORTS port_16                                                                                                                         }
  { PHYPIN Q16  LOGPIN DATA_OUT     DIRECTION out WIDTH 4608.0 PHASE {} DEFAULT {} PORTS port_16                                                                                                                         }
}

}
