# See 
# https://www.xilinx.com/html_docs/xilinx2019_1/SDK_Doc/xsct/intro/xsct_introduction.html

setws .
if { $::argc == 1 } {
    set myproject [lindex $::argv 0]
    createhw -name ${myproject}\_platform -hwspec ../hdf/${myproject}\_wrapper.hdf
    createapp -name ${myproject}\_standalone -app {Hello World} -proc psu_cortexa53_0 -hwproject ${myproject}\_platform -os standalone -arch 64
    configbsp -bsp ${myproject}\_standalone_bsp stdin psu_uart_1
    configbsp -bsp ${myproject}\_standalone_bsp stdout psu_uart_1
    updatemss -mss ${myproject}\_standalone_bsp/system.mss
    regenbsp -bsp ${myproject}\_standalone_bsp 
    configapp -app ${myproject}\_standalone build-config release
    configapp -app ${myproject}\_standalone -add linker-misc {-Wl,--defsym=_HEAP_SIZE=0x1000000}
    configapp -app ${myproject}\_standalone -add linker-misc {-Wl,--defsym=_STACK_SIZE=0x40000}
    projects -build
    #configapp -app ${myproject}\_standalone -add define-compiler-symbols {FLAG=VALUE}
}
