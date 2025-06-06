# Direct CMake to use icpx rather than the default C++ compiler/linker on Linux
# and icx-cl on Windows
if(UNIX)
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

cmake_minimum_required (VERSION 3.7.2)

project(myproject CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

###############################################################################
### Customize these build variables
###############################################################################
set(SOURCE_FILES src/firmware/myproject.cpp src/myproject_test.cpp)
set(LIBRARY_FILES src/firmware/myproject.cpp src/myproject_bridge.cpp)
set(LIB_STAMP mystamp)
set(TARGET_NAME myproject)
set(LIBRARY_NAME myproject-${LIB_STAMP})

# Use cmake -DFPGA_DEVICE=<board-support-package>:<board-variant> to choose a
# different device. Here are a few device examples (this list is not
# exhaustive):
#   intel_s10sx_pac:pac_s10
#   intel_s10sx_pac:pac_s10_usm
#   intel_a10gx_pac:pac_a10
# Note that depending on your installation, you may need to specify the full
# path to the board support package (BSP), this usually is in your install
# folder.
#
# You can also specify a device family (E.g. "Agilex7" or "Stratix10") or a
# specific part number (E.g. "10AS066N3F40E2SG") to generate a standalone IP.
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
endif()

# Use cmake -DUSER_FPGA_FLAGS=<flags> to set extra flags for FPGA backend
# compilation.
set(USER_FPGA_FLAGS -Wno-unused-label;${USER_FPGA_FLAGS})

# Use cmake -DUSER_FLAGS=<flags> to set extra flags for general compilation.
set(USER_FLAGS -Wno-unused-label -fconstexpr-steps=134217728 ${USER_FLAGS})

# Use cmake -DUSER_INCLUDE_PATHS=<paths> to set extra paths for general
# compilation.
set(USER_INCLUDE_PATHS src;src/firmware;${USER_INCLUDE_PATHS})

###############################################################################
### no changes after here
###############################################################################

# Print the device being used for the compiles
message(STATUS "Configuring the design to run on FPGA board ${FPGA_DEVICE}")

# Set the names of the makefile targets to be generated by cmake
set(EMULATOR_TARGET fpga_emu)
set(SIMULATOR_TARGET fpga_sim)
set(REPORT_TARGET report)
set(FPGA_TARGET fpga)
set(IP_EXPORT_TARGET fpga_ip_export)
set(LIBRARY_TARGET lib)

# Set the names of the generated files per makefile target
set(EMULATOR_OUTPUT_NAME ${TARGET_NAME}.${EMULATOR_TARGET})
set(SIMULATOR_OUTPUT_NAME ${TARGET_NAME}.${SIMULATOR_TARGET})
set(REPORT_OUTPUT_NAME ${TARGET_NAME}.${REPORT_TARGET})
set(FPGA_OUTPUT_NAME ${TARGET_NAME}.${FPGA_TARGET})
set(IP_EXPORT_OUTPUT_NAME ${TARGET_NAME}.${IP_EXPORT_TARGET})

message(STATUS "Additional USER_FPGA_FLAGS=${USER_FPGA_FLAGS}")
message(STATUS "Additional USER_FLAGS=${USER_FLAGS}")

include_directories(${USER_INCLUDE_PATHS})
message(STATUS "Additional USER_INCLUDE_PATHS=${USER_INCLUDE_PATHS}")

link_directories(${USER_LIB_PATHS})
message(STATUS "Additional USER_LIB_PATHS=${USER_LIB_PATHS}")

link_libraries(${USER_LIBS})
message(STATUS "Additional USER_LIBS=${USER_LIBS}")

if(WIN32)
    # add qactypes for Windows
    set(QACTYPES "-Qactypes")
    # This is a Windows-specific flag that enables exception handling in host code
    set(WIN_FLAG "/EHsc")
else()
    # add qactypes for Linux
    set(QACTYPES "-qactypes")
endif()

set(COMMON_COMPILE_FLAGS -fsycl -fintelfpga -Wall ${WIN_FLAG} ${QACTYPES} ${USER_FLAGS})
# for debugging need to do this. Not sure why
# set(COMMON_LINK_FLAGS -L/opt/intel/oneapi/compiler/2024.0/opt/oclfpga/host/linux64/lib -fsycl -fintelfpga ${QACTYPES} ${USER_FLAGS})
set(COMMON_LINK_FLAGS -fsycl -fintelfpga ${QACTYPES} ${USER_FLAGS})

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate
#    representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking. For
#    this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS -DFPGA_EMULATOR)
set(LIBRARY_COMPILE_FLAGS -DFPGA_EMULATOR)
set(EMULATOR_LINK_FLAGS )
set(LIBRARY_LINK_FLAGS -L$ENV{FPGA_VARS_DIR}/host/linux64/lib)
set(REPORT_COMPILE_FLAGS -DFPGA_HARDWARE)
set(REPORT_LINK_FLAGS -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_FPGA_FLAGS} -fsycl-link=early)
set(SIMULATOR_COMPILE_FLAGS -Xssimulation -DFPGA_SIMULATOR)
set(SIMULATOR_LINK_FLAGS -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_FPGA_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${SIMULATOR_OUTPUT_NAME})
set(FPGA_COMPILE_FLAGS -DFPGA_HARDWARE)
set(FPGA_LINK_FLAGS -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_FPGA_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_OUTPUT_NAME})
# get rid of this once host pipes work properly
set(IP_EXPORT_COMPILE_FLAGS -DFPGA_HARDWARE)
set(IP_EXPORT_LINK_FLAGS -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_FPGA_FLAGS} -fsycl-link=early -fsycl-device-code-split=per_kernel)

###############################################################################
### FPGA Emulator library
###############################################################################
add_library(${LIBRARY_TARGET} SHARED ${LIBRARY_FILES})
target_compile_options(${LIBRARY_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${LIBRARY_TARGET} PRIVATE ${LIBRARY_COMPILE_FLAGS})
target_link_libraries(${LIBRARY_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${LIBRARY_TARGET} ${LIBRARY_LINK_FLAGS})
set_target_properties(${LIBRARY_TARGET} PROPERTIES OUTPUT_NAME ${LIBRARY_NAME})

###############################################################################
### FPGA Emulator
###############################################################################
add_executable(${EMULATOR_TARGET} ${SOURCE_FILES})
target_compile_options(${EMULATOR_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${EMULATOR_TARGET} PRIVATE ${EMULATOR_COMPILE_FLAGS})
target_link_libraries(${EMULATOR_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${EMULATOR_TARGET} ${EMULATOR_LINK_FLAGS})
set_target_properties(${EMULATOR_TARGET} PROPERTIES OUTPUT_NAME ${EMULATOR_OUTPUT_NAME})

###############################################################################
### FPGA Simulator
###############################################################################
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILES})
target_compile_options(${SIMULATOR_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${SIMULATOR_TARGET} PRIVATE ${SIMULATOR_COMPILE_FLAGS})
target_link_libraries(${SIMULATOR_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${SIMULATOR_TARGET} ${SIMULATOR_LINK_FLAGS})
set_target_properties(${SIMULATOR_TARGET} PROPERTIES OUTPUT_NAME ${SIMULATOR_OUTPUT_NAME})

###############################################################################
### Generate Report
###############################################################################
add_executable(${REPORT_TARGET} ${SOURCE_FILES})
target_compile_options(${REPORT_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${REPORT_TARGET} PRIVATE ${REPORT_COMPILE_FLAGS})

# The report target does not need the QACTYPES flag at link stage
set(MODIFIED_COMMON_LINK_FLAGS_REPORT ${COMMON_LINK_FLAGS})
list(REMOVE_ITEM MODIFIED_COMMON_LINK_FLAGS_REPORT ${QACTYPES})

target_link_libraries(${REPORT_TARGET} ${MODIFIED_COMMON_LINK_FLAGS_REPORT})
target_link_libraries(${REPORT_TARGET} ${REPORT_LINK_FLAGS})
set_target_properties(${REPORT_TARGET} PROPERTIES OUTPUT_NAME ${REPORT_OUTPUT_NAME})

###############################################################################
### FPGA Hardware
###############################################################################
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILES})
target_compile_options(${FPGA_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${FPGA_TARGET} PRIVATE ${FPGA_COMPILE_FLAGS})
target_link_libraries(${FPGA_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${FPGA_TARGET} ${FPGA_LINK_FLAGS})
set_target_properties(${FPGA_TARGET} PROPERTIES OUTPUT_NAME ${FPGA_OUTPUT_NAME})

###############################################################################
### FPGA IP Export (only necessary until native host pipes)
###############################################################################
add_executable(${IP_EXPORT_TARGET} ${SOURCE_FILES})
target_compile_options(${IP_EXPORT_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${IP_EXPORT_TARGET} PRIVATE ${IP_EXPORT_COMPILE_FLAGS})

# The ip export target does not need the QACTYPES flag at link stage
set(MODIFIED_COMMON_LINK_FLAGS_EXPORT ${COMMON_LINK_FLAGS})
list(REMOVE_ITEM MODIFIED_COMMON_LINK_FLAGS_EXPORT ${QACTYPES})

target_link_libraries(${IP_EXPORT_TARGET} ${MODIFIED_COMMON_LINK_FLAGS_EXPORT})
target_link_libraries(${IP_EXPORT_TARGET} ${IP_EXPORT_LINK_FLAGS})
set_target_properties(${IP_EXPORT_TARGET} PROPERTIES OUTPUT_NAME ${IP_EXPORT_OUTPUT_NAME})

###############################################################################
### This part only manipulates cmake variables to print the commands to the user
###############################################################################

# set the correct object file extension depending on the target platform
if(WIN32)
    set(OBJ_EXTENSION "obj")
else()
    set(OBJ_EXTENSION "o")
endif()

# Set the source file names in a string
set(SOURCE_FILE_NAME "${SOURCE_FILES}")

function(getCompileCommands common_compile_flags special_compile_flags common_link_flags special_link_flags target output_name)

    set(file_names ${SOURCE_FILE_NAME})
    set(COMPILE_COMMAND )
    set(LINK_COMMAND )

    foreach(source ${file_names})
        # Get the relative path to the source and object files
        file(RELATIVE_PATH CURRENT_SOURCE_FILE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/${source})
        file(RELATIVE_PATH OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target}.dir/${source}.${OBJ_EXTENSION})

        # Creating a string that contains the compile command
        # Start by the compiler invocation
        set(COMPILE_COMMAND "${COMPILE_COMMAND}${CMAKE_CXX_COMPILER}")

        # Add all the potential includes
        foreach(INCLUDE ${USER_INCLUDE_PATHS})
            if(NOT IS_ABSOLUTE ${INCLUDE})
                file(RELATIVE_PATH INCLUDE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/${INCLUDE})
            endif()
            set(COMPILE_COMMAND "${COMPILE_COMMAND} -I${INCLUDE}")
        endforeach()

        # Add all the common compile flags
        foreach(FLAG ${common_compile_flags})
            set(COMPILE_COMMAND "${COMPILE_COMMAND} ${FLAG}")
        endforeach()

        # Add all the specific compile flags
        foreach(FLAG ${special_compile_flags})
            set(COMPILE_COMMAND "${COMPILE_COMMAND} ${FLAG}")
        endforeach()

        # Get the location of the object file
        file(RELATIVE_PATH OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target}.dir/${source}.${OBJ_EXTENSION})

        # Add the source file and the output file
        set(COMPILE_COMMAND "${COMPILE_COMMAND} -c ${CURRENT_SOURCE_FILE} -o ${OBJ_FILE}\n")
    endforeach()

    set(COMPILE_COMMAND "${COMPILE_COMMAND}" PARENT_SCOPE)

    # Creating a string that contains the link command
    # Start by the compiler invocation
    set(LINK_COMMAND "${LINK_COMMAND}${CMAKE_CXX_COMPILER}")

    # Add all the common link flags
    foreach(FLAG ${common_link_flags})
        set(LINK_COMMAND "${LINK_COMMAND} ${FLAG}")
    endforeach()

    # Add all the specific link flags
    foreach(FLAG ${special_link_flags})
        set(LINK_COMMAND "${LINK_COMMAND} ${FLAG}")
    endforeach()

    # Add the output file
    set(LINK_COMMAND "${LINK_COMMAND} -o ${output_name}")

    foreach(source ${file_names})
        # Get the relative path to the source and object files
        file(RELATIVE_PATH OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target}.dir/${source}.${OBJ_EXTENSION})

        # Add the source file and the output file
        set(LINK_COMMAND "${LINK_COMMAND} ${OBJ_FILE}")
    endforeach()

    # Add all the potential library paths
    foreach(LIB_PATH ${USER_LIB_PATHS})
        if(NOT IS_ABSOLUTE ${LIB_PATH})
            file(RELATIVE_PATH LIB_PATH ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/${LIB_PATH})
        endif()
        if(NOT WIN32)
            set(LINK_COMMAND "${LINK_COMMAND} -L${LIB_PATH}")
        else()
            set(LINK_COMMAND "${LINK_COMMAND} -L${LIB_PATH} -Wl,-rpath,${LIB_PATH}")
        endif()
    endforeach()

    # Add all the potential includes
    foreach(LIB ${USER_LIBS})
        set(LINK_COMMAND "${LINK_COMMAND} -l${LIB}")
    endforeach()

    set(LINK_COMMAND "${LINK_COMMAND}" PARENT_SCOPE)

endfunction()

# Windows executable is going to have the .exe extension
if(WIN32)
    set(EXECUTABLE_EXTENSION ".exe")
endif()

# Display the compile instructions in the emulation flow
getCompileCommands("${COMMON_COMPILE_FLAGS}" "${EMULATOR_COMPILE_FLAGS}" "${COMMON_LINK_FLAGS}" "${EMULATOR_LINK_FLAGS}" "${EMULATOR_TARGET}" "${EMULATOR_OUTPUT_NAME}${EXECUTABLE_EXTENSION}")

add_custom_target(  displayEmulationCompileCommands ALL
                    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
                    COMMENT "To compile manually:\n${COMPILE_COMMAND}\nTo link manually:\n${LINK_COMMAND}")
add_dependencies(${EMULATOR_TARGET} displayEmulationCompileCommands)

# Display the compile instructions in the simulation flow
getCompileCommands("${COMMON_COMPILE_FLAGS}" "${SIMULATOR_COMPILE_FLAGS}" "${COMMON_LINK_FLAGS}" "${SIMULATOR_LINK_FLAGS}" "${SIMULATOR_TARGET}" "${SIMULATOR_OUTPUT_NAME}${EXECUTABLE_EXTENSION}")

add_custom_target(  displaySimulationCompileCommands ALL
                    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
                    COMMENT "To compile manually:\n${COMPILE_COMMAND}\nTo link manually:\n${LINK_COMMAND}")
add_dependencies(${SIMULATOR_TARGET} displaySimulationCompileCommands)

# Display the compile instructions in the report flow
getCompileCommands("${COMMON_COMPILE_FLAGS}" "${REPORT_COMPILE_FLAGS}" "${MODIFIED_COMMON_LINK_FLAGS_REPORT}" "${REPORT_LINK_FLAGS}" "${REPORT_TARGET}" "${REPORT_OUTPUT_NAME}${EXECUTABLE_EXTENSION}")

add_custom_target(  displayReportCompileCommands ALL
                    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
                    COMMENT "To compile manually:\n${COMPILE_COMMAND}\nTo link manually:\n${LINK_COMMAND}")
add_dependencies(${REPORT_TARGET} displayReportCompileCommands)

# Display the compile instructions in the IP export flow (Remove after native host pipes work properly)
getCompileCommands("${COMMON_COMPILE_FLAGS}" "${IP_EXPORT_COMPILE_FLAGS}" "${MODIFIED_COMMON_LINK_FLAGS_EXPORT}" "${IP_EXPORT_LINK_FLAGS}" "${IP_EXPORT_TARGET}" "${IP_EXPORT_OUTPUT_NAME}${EXECUTABLE_EXTENSION}")

add_custom_target(  displayExportCompileCommands ALL
                    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
                    COMMENT "To compile manually:\n${COMPILE_COMMAND}\nTo link manually:\n${LINK_COMMAND}")
add_dependencies(${IP_EXPORT_TARGET} displayExportCompileCommands)

# Display the compile instructions in the fpga flow
getCompileCommands("${COMMON_COMPILE_FLAGS}" "${FPGA_COMPILE_FLAGS}" "${COMMON_LINK_FLAGS}" "${FPGA_LINK_FLAGS}" "${FPGA_TARGET}" "${FPGA_OUTPUT_NAME}${EXECUTABLE_EXTENSION}")

add_custom_target(  displayFPGACompileCommands ALL
                    ${CMAKE_COMMAND} -E cmake_echo_color --cyan ""
                    COMMENT "To compile manually:\n${COMPILE_COMMAND}\nTo link manually:\n${LINK_COMMAND}")
add_dependencies(${FPGA_TARGET} displayFPGACompileCommands)
