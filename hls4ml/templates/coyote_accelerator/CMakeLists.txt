cmake_minimum_required(VERSION 3.5)
set(CYT_DIR ${CMAKE_SOURCE_DIR}/Coyote/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)
find_package(CoyoteHW REQUIRED)
find_package(CoyoteSW REQUIRED)

set(FLOW "hw" CACHE STRING "Synthesize hardware (hw) or host software (sw)")

if(FLOW STREQUAL "hw")
    project(myproject)
    set(EN_STRM 1)
    set(N_STRM_AXI 1)
    set(N_REGIONS 1)

    validation_checks_hw()
    load_apps (
        VFPGA_C0_0 "src"
    )
    create_hw()
endif()

if(FLOW STREQUAL "sw")
    project(
        CoyoteInference
        VERSION 1.0.0
        DESCRIPTION "CoyoteInference library"
    )
    set(CYT_INCLUDE_PATH ${CYT_DIR}/sw/include)
    add_library(CoyoteInference SHARED "${CMAKE_SOURCE_DIR}/src/host_libs.cpp" "${CMAKE_SOURCE_DIR}/src/host_libs.hpp")
    target_include_directories(CoyoteInference PUBLIC ${CYT_INCLUDE_PATH})
    target_link_libraries(CoyoteInference PUBLIC Coyote)
    target_link_directories(CoyoteInference PUBLIC /usr/local/lib)

    project(myproject)
    set(EXEC test)
    set(TARGET_DIR "${CMAKE_SOURCE_DIR}/src/")
    add_executable(${EXEC} ${TARGET_DIR}/myproject_host.cpp)
    target_link_libraries(${EXEC} PUBLIC Coyote)
    target_link_libraries(${EXEC} PUBLIC CoyoteInference)
    target_link_directories(${EXEC} PUBLIC /usr/local/lib)
    target_include_directories(${EXEC} PUBLIC src/hls/model_wrapper/firmware/)
    target_include_directories(${EXEC} PUBLIC src/hls/model_wrapper/firmware/ap_types)

endif()