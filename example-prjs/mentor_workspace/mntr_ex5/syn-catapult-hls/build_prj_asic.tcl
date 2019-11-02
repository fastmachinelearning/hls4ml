array set opt {
    # The 'asic' flag specifies if you are targeting ASIC or FPGA:
    # = 0 -> Vivado (FPGA)
    # = 1 -> Mentor Design Compiler
    # = 2 -> Cadence Encounter RTL Compiler
    # = 3 -> Cadence Genus
    asic       2
    csim       0
    hsynth     1
    rtlsim     0
    validation 0
    lsynth     1
}

source ./build_prj.tcl
