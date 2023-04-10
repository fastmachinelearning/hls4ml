DEVICE   := Arria10
TARGETS  := myproject-fpga

CXX          		:= i++
CXXFLAGS     		:= -march=$(DEVICE)
RM           		:= rm -rf
DEBUG_FLAGS  		:= --time quartus-hlssynt.log
SOURCE_FILES 		:= myproject_test.cpp firmware/myproject.cpp
HEADER_FILES 		:= firmware/myproject.h
LOGGING_1			:=
LOGGING_2			:=
QUARTUS_COMPILE 	:=
CONT_IF_LARGE_AREA 	:=

.PHONY: test
test: $(TARGETS)
	@$(foreach t,$(TARGETS),echo ./$(t); ./$(t) | tee $(t)_run.log; echo "";)

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	-$(RM) $(foreach t,$(TARGETS),$(t).prj $(t) $(t)_time.log)

.PHONY: myproject-fpga
myproject-fpga: CXXFLAGS := $(CXXFLAGS)

$(TARGETS) : $(SOURCE_FILES) $(HEADER_FILES)
	$(CXX) $(LOGGING_1) $(LOGGING_2) $(CXXFLAGS) $(DEBUG_FLAGS) $(SOURCE_FILES) $(CONT_IF_LARGE_AREA) $(QUARTUS_COMPILE) -o $@
