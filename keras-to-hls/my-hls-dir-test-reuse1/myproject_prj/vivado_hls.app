<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="myproject_prj" top="myproject">
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
    <includePaths/>
    <libraryFlag/>
    <files>
        <file name="../../firmware/weights" sc="0" tb="1" cflags=" "/>
        <file name="../../myproject_test.cpp" sc="0" tb="1" cflags=" -I/home/ntran/HLS/ML/dev/HLS4ML/nnet_utils "/>
        <file name="firmware/myproject.cpp" sc="0" tb="false" cflags="-I/home/ntran/HLS/ML/dev/HLS4ML/nnet_utils"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
</AutoPilot:project>

