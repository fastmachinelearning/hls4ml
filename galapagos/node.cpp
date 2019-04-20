
#include <string>
#include <math.h>
#include <thread>
#include <chrono>

#include "galapagos_node.hpp"
#include "argparse.hpp"
#include "kernel.h"
#include "myproject.h"


int main(int argc, const char** argv){

    int source = 0;
    int dest = 1;
    std::vector <std::string> kern_info;
   
    ArgumentParser parser;

    parser.addArgument("-s", "--source_ip", 1);
    parser.addArgument("-d", "--dest_ip", 1);
    parser.addArgument("-t", "--type", 1);

    parser.parse(argc, argv);

    std::string source_ip = parser.retrieve<std::string>("source_ip"); 
    std::string dest_ip = parser.retrieve<std::string>("dest_ip"); 
    std::string type = parser.retrieve<std::string>("type"); 



    std::vector<std::string> my_ip;
    std::vector<void (*)(galapagos::stream <float>  * , galapagos::stream <float>  *)> func;
    std::vector<int> my_kern_id;

    if(type == "s" || type == "a"){
        my_ip.push_back(source_ip);
        func.push_back(kern_send);
        my_kern_id.push_back(source);
        
    }
    if(type == "d" || type == "a"){
        my_ip.push_back(dest_ip);
        func.push_back(myproject_galapagos);
        my_kern_id.push_back(dest);
    }
    else{
        return -1;
    }
    
    kern_info.push_back(source_ip);
    kern_info.push_back(dest_ip);
        
    galapagos::node <float> node(kern_info, my_ip[0]);

    for(int i=0; i<func.size(); i++){
        node.add_kernel(my_kern_id[i], func[i]);
    }
    
    node.start();
    while(1);

}
