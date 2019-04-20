
SRC_LIB=$(GALAPAGOS_PATH)/middleware/CPP_lib/Galapagos_lib
CXXFLAGS = -DCPU -g -std=c++17 -pthread -I$(XILINX_VIVADO)/include -I$(GALAPAGOS_PATH)/middleware/include -I$(SRC_LIB) -I../nnet_utils

LDFLAGS = -lpthread

BOOST_LDFLAGS=-lboost_thread -lboost_system $(LDFLAGS) 



all: galapagos myproject.o myproject_galapagos.o kern_send.o hls 

galapagos:
	$(MAKE) -C $(SRC_LIB)

myproject.o: firmware/myproject.cpp
	$(CXX) $(CXXFLAGS) -c firmware/myproject.cpp -o $@ $(BOOST_LDFLAGS)

myproject_galapagos.o: myproject_galapagos.cpp
	$(CXX) $(CXXFLAGS) -c myproject_galapagos.cpp -o $@ $(BOOST_LDFLAGS)

kern_send.o: kern_send.cpp
	$(CXX) $(CXXFLAGS) -c kern_send.cpp -o $@ $(BOOST_LDFLAGS)

hls:
	vivado_hls build_prj.tcl

node.exe: $(SRC_LIB)/galapagos_node.o $(SRC_LIB)/galapagos_router.o $(SRC_LIB)/galapagos_router_node.o $(SRC_LIB)/galapagos_router_net.o $(SRC_LIB)/galapagos_net_tcp.o $(SRC_LIB)/galapagos_net_tcp_session.o $(SRC_LIB)/galapagos_net_tcp_server_send.o $(SRC_LIB)/galapagos_net_tcp_accept_server.o $(SRC_LIB)/galapagos_kernel.o  $(SRC_LIB)/galapagos_streaming_core.o $(SRC_LIB)/galapagos_stream.o kern_send.o myproject.o myproject_galapagos.o node.cpp 
	$(CXX) -I firmware $(CXXFLAGS) -I$(SRC_LIB) -I$(GALAPAGOS_PATH)/util/argparse $^ -o $@ $(BOOST_LDFLAGS)  

clean:
	rm -rf *.o myproject_prj
	rm -rf *.exe 
