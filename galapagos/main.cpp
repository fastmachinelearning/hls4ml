
#include <cstddef>
#include <cstring>
#include <string>
#include <math.h>
#include <thread>
#include <chrono>

#include "firmware/parameters.h"
#include "firmware/myproject.h"
#include "nnet_helpers.h"

#include "myproject.h"
#include "test_galapagos.hpp"
#include "galapagos_kernel.hpp"
#include "galapagos_router.hpp"
#include "galapagos_node.hpp"



void myproject_galapagos(galapagos::stream * in, galapagos::stream * out);
int main(int argc, char *argv){
    
    int source = 0;
    int dest = 1;
    std::string my_address = "localhost";
    std::vector <std::string> kern_info;
    kern_info.push_back(my_address);
    kern_info.push_back(my_address);

    galapagos::node node(kern_info, my_address);
    node.add_kernel(source, myproject_galapagos);
    node.add_kernel(source, test_galapagos);
    node.start();
    node.end();


}
