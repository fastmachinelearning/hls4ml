<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="myproject_prj" top="myproject_galapagos">
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
    <includePaths/>
    <libraryFlag/>
    <files>
        <file name="../../firmware/weights" sc="0" tb="1" cflags="  -Wno-unknown-pragmas"/>
        <file name="myproject_galapagos.cpp" sc="0" tb="false" cflags="-I/home/tarafdar/workDir/test/galapagos/hls4ml/nnet_utils -I/home/tarafdar/workDir/test/galapagos/middleware/CPP_lib/Galapagos_lib -I/home/tarafdar/workDir/test/galapagos/middleware/include"/>
        <file name="firmware/myproject.cpp" sc="0" tb="false" cflags="-I/home/tarafdar/workDir/test/galapagos/hls4ml/nnet_utils"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
</AutoPilot:project>

