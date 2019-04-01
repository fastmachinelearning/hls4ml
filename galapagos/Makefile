
SRC_LIB=$(GALAPAGOS_PATH)/middleware/CPP_lib/Galapagos_lib
CXXFLAGS = -DCPU -g -std=c++17 -pthread -I$(XILINX_VIVADO)/include -I$(GALAPAGOS_PATH)/middleware/include -I$(SRC_LIB) -I../nnet_utils

LDFLAGS = -lpthread

BOOST_LDFLAGS=-lboost_thread -lboost_system $(LDFLAGS) 

net_driver = tcp

UNIT_TEST=-I$(GALAPAGOS_PATH)/Catch2/single_include/catch2 


all:  galapagos_packet.o galapagos_kernel.o

$^:
	$(CXX) $(CXXFLAGS) -c $^ -o $@ $(BOOST_LDFLAGS)

GALAPAGOS_OBJECTS: $(SRC_LIB)/galapagos_node.o $(SRC_LIB)/galapagos_router.o $(SRC_LIB)/galapagos_kernel.o $(SRC_LIB)/galapagos_net.o $(SRC_LIB)/galapagos_streaming_core.o $(SRC_LIB)/galapagos_stream.o 

myproject.o: firmware/myproject.cpp
	$(CXX) $(CXXFLAGS) -c firmware/myproject.cpp -o $@ $(BOOST_LDFLAGS)

myproject_galapagos.o: myproject_galapagos.cpp
	$(CXX) $(CXXFLAGS) -c myproject_galapagos.cpp -o $@ $(BOOST_LDFLAGS)

test_galapagos.o: test_galapagos.cpp
	$(CXX) $(CXXFLAGS) -c test_galapagos.cpp -o $@ $(BOOST_LDFLAGS)

test.exe: $(GALAPAGOS_OBJECTS)  myproject.o myproject_galapagos.o test_galapagos.o main.cpp 
	$(CXX) $(CXXFLAGS) $(UNIT_TEST)  $(SRC_LIB)/galapagos_node.o $(SRC_LIB)/galapagos_router.o $(SRC_LIB)/galapagos_kernel.o $(SRC_LIB)/galapagos_net.o $(SRC_LIB)/galapagos_streaming_core.o $(SRC_LIB)/galapagos_stream.o $^ -o $@ $(BOOST_LDFLAGS)  



clean:
	rm -rf *.o
	rm -rf *.exe 
