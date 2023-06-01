/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
#include "xcl2.hpp"
namespace xcl {
std::vector<cl::Device> get_devices(const std::string& vendor_name) {

    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    OCL_CHECK(err, err = cl::Platform::get(&platforms));
    cl::Platform platform;
    for (i  = 0 ; i < platforms.size(); i++){
        platform = platforms[i];
        OCL_CHECK(err, std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
        if (platformName == vendor_name){
            std::cout << "Found Platform" << std::endl;
            std::cout << "Platform Name: " << platformName.c_str() << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "Error: Failed to find Xilinx platform" << std::endl;
		exit(EXIT_FAILURE);
	}
   
    //Getting ACCELERATOR Devices and selecting 1st such device 
    std::vector<cl::Device> devices;
    OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
    return devices;
}
   
std::vector<cl::Device> get_xil_devices() {
	return get_devices("Xilinx");
}
cl::Program::Binaries import_binary_file(std::string xclbin_file_name) 
{
    std::cout << "INFO: Importing " << xclbin_file_name << std::endl;

	if(access(xclbin_file_name.c_str(), R_OK) != 0) {
		printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
		exit(EXIT_FAILURE);
	}
    //Loading XCL Bin into char buffer 
    std::cout << "Loading: '" << xclbin_file_name.c_str() << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
   
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    return bins;
}

std::string 
find_binary_file(const std::string& _device_name, const std::string& xclbin_name) 
{
    std::cout << "XCLBIN File Name: " << xclbin_name.c_str() << std::endl;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
	char *xcl_target = getenv("XCL_TARGET");
    std::string mode;

	/* Fall back mode if XCL_EMULATION_MODE is not set is "hw" */
	if(xcl_mode == NULL) {
		mode = "hw";
	} else {
		/* if xcl_mode is set then check if it's equal to true*/
		if(strcmp(xcl_mode,"true") == 0) {
			/* if it's true, then check if xcl_target is set */
			if(xcl_target == NULL) {
				/* default if emulation but not specified is software emulation */
				mode = "sw_emu";
			} else {
				/* otherwise, it's what ever is specified in XCL_TARGET */
				mode = xcl_target;
			}
		} else {
			/* if it's not equal to true then it should be whatever
			 * XCL_EMULATION_MODE is set to */
			mode = xcl_mode;
		}
    }
    char *xcl_bindir = getenv("XCL_BINDIR");

    // typical locations of directory containing xclbin files
    const char *dirs[] = {
        xcl_bindir, // $XCL_BINDIR-specified
        "xclbin",   // command line build
        "..",       // gui build + run
        ".",        // gui build, run in build directory
        NULL
    };
    const char **search_dirs = dirs;
    if (xcl_bindir == NULL) {
    	search_dirs++;
    }

    char *device_name = strdup(_device_name.c_str());
    if (device_name == NULL) {
        printf("Error: Out of Memory\n");
        exit(EXIT_FAILURE);
    }

    // fix up device name to avoid colons and dots.
    // xilinx:xil-accel-rd-ku115:4ddr-xpr:3.2 -> xilinx_xil-accel-rd-ku115_4ddr-xpr_3_2
    for (char *c = device_name; *c != 0; c++) {
        if (*c == ':' || *c == '.') {
            *c = '_';
        }
    }

    char *device_name_versionless = strdup(_device_name.c_str());
    if (device_name_versionless == NULL) {
        printf("Error: Out of Memory\n");
        exit(EXIT_FAILURE);
    }

    unsigned short colons = 0;
    bool colon_exist = false;
    for (char *c = device_name_versionless; *c != 0; c++) {
        if (*c == ':') {
            colons++;
            *c = '_';
            colon_exist = true;
        }
        /* Zero out version area */
        if (colons == 3) {
            *c = '\0';
        }
    }

    // versionless support if colon doesn't exist in device_name
    if(!colon_exist) {
        int len = strlen(device_name_versionless);
        device_name_versionless[len - 4] = '\0';
    }
    
    const char *aws_file_patterns[] = {
        "%1$s/%2$s.%3$s.%4$s.awsxclbin",     // <kernel>.<target>.<device>.awsxclbin
        "%1$s/%2$s.%3$s.%5$s.awsxclbin",     // <kernel>.<target>.<device_versionless>.awsxclbin
        "%1$s/binary_container_1.awsxclbin", // default for gui projects
        "%1$s/%2$s.awsxclbin",               // <kernel>.awsxclbin
        NULL
    };

    const char *file_patterns[] = {
        "%1$s/%2$s.%3$s.%4$s.xclbin",     // <kernel>.<target>.<device>.xclbin
        "%1$s/%2$s.%3$s.%5$s.xclbin",     // <kernel>.<target>.<device_versionless>.xclbin
        "%1$s/binary_container_1.xclbin", // default for gui projects
        "%1$s/%2$s.xclbin",               // <kernel>.xclbin
        NULL
    };
    char xclbin_file_name[PATH_MAX];
    memset(xclbin_file_name, 0, PATH_MAX);
    ino_t aws_ino = 0; // used to avoid errors if an xclbin found via multiple/repeated paths
    for (const char **dir = search_dirs; *dir != NULL; dir++) {
        struct stat sb;
        if (stat(*dir, &sb) == 0 && S_ISDIR(sb.st_mode)) {
            for (const char **pattern = aws_file_patterns; *pattern != NULL; pattern++) {
                char file_name[PATH_MAX];
                memset(file_name, 0, PATH_MAX);
                snprintf(file_name, PATH_MAX, *pattern, *dir, xclbin_name.c_str(), mode.c_str(), device_name, device_name_versionless);
                if (stat(file_name, &sb) == 0 && S_ISREG(sb.st_mode)) {
                    char* bindir = strdup(*dir);
                    if (bindir == NULL) {
                        printf("Error: Out of Memory\n");
                        exit(EXIT_FAILURE);
                    }
                    if (*xclbin_file_name && sb.st_ino != aws_ino) {
                    	printf("Error: multiple xclbin files discovered:\n %s\n %s\n", file_name, xclbin_file_name);
                    	exit(EXIT_FAILURE);
                    }
                    aws_ino = sb.st_ino;
                    strncpy(xclbin_file_name, file_name, PATH_MAX);
                }
            }
        }
    }
    ino_t ino = 0; // used to avoid errors if an xclbin found via multiple/repeated paths
    // if no awsxclbin found, check for xclbin
    if (*xclbin_file_name == '\0') {
        for (const char **dir = search_dirs; *dir != NULL; dir++) {
            struct stat sb;
            if (stat(*dir, &sb) == 0 && S_ISDIR(sb.st_mode)) {
                for (const char **pattern = file_patterns; *pattern != NULL; pattern++) {
                    char file_name[PATH_MAX];
                    memset(file_name, 0, PATH_MAX);
                    snprintf(file_name, PATH_MAX, *pattern, *dir, xclbin_name.c_str(), mode.c_str(), device_name, device_name_versionless);
                    if (stat(file_name, &sb) == 0 && S_ISREG(sb.st_mode)) {
                        char* bindir = strdup(*dir);
                        if (bindir == NULL) {
                            printf("Error: Out of Memory\n");
                            exit(EXIT_FAILURE);
                        }
                        if (*xclbin_file_name && sb.st_ino != ino) {
                    	    printf("Error: multiple xclbin files discovered:\n %s\n %s\n", file_name, xclbin_file_name);
                    	    exit(EXIT_FAILURE);
                        }
                        ino = sb.st_ino;
                        strncpy(xclbin_file_name, file_name, PATH_MAX);
                    }
                }
            }
        }
    }
    // if no xclbin found, preferred path for error message from xcl_import_binary_file()
    if (*xclbin_file_name == '\0') {
        snprintf(xclbin_file_name, PATH_MAX, file_patterns[0], *search_dirs, xclbin_name.c_str(), mode.c_str(), device_name);
    }
    free(device_name);
    return (xclbin_file_name);
}

bool is_emulation()
{
    bool ret =false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if (xcl_mode != NULL){
        ret = true;
    }
    return ret;
}

bool is_hw_emulation()
{
    bool ret =false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if ((xcl_mode != NULL) && !strcmp(xcl_mode, "hw_emu")){
        ret = true;
    }
    return ret;
}

bool is_xpr_device(const char *device_name) {
    const char *output = strstr(device_name,"xpr");

    if(output==NULL) {
        return false;
    }
    else {
        return true;
    }
}


};
