# See 
# https://www.xilinx.com/html_docs/xilinx2019_1/SDK_Doc/xsct/intro/xsct_introduction.html

setws .
if { $::argc == 1 } {
    set myproject [lindex $::argv 0]
    createhw -name ${myproject}\_platform -hwspec ../hdf/${myproject}\_wrapper.hdf
    createapp -name ${myproject}\_standalone -app {Hello World} -proc ps7_cortexa9_0 -hwproject ${myproject}\_platform -os standalone
    configapp -app ${myproject}\_standalone build-config release
    #createapp -name ${myproject}\_standalone -app {Hello World} -proc psu_cortexa53_0 -hwproject ${myproject}\_platform -os standalone -arch 64
    #configapp -app ${myproject}\_standalone -add define-compiler-symbols {FLAG=VALUE}
    #projects -build
}
