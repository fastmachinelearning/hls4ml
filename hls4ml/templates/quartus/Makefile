DEVICE   := Arria10
TARGETS  := myproject-fpga

CXX          := i++
CXXFLAGS     := $(USERCXXFLAGS) -march=$(DEVICE)#--quartus-compile
RM           := rm -rf
DEBUG_FLAGS  = --time $@_time.log -v
SOURCE_FILES := myproject_test.cpp firmware/myproject.cpp
HEADER_FILES := firmware/myproject.h

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
myproject-fpga: CXXFLAGS := $(CXXFLAGS)
#force-dsp: CXXFLAGS := $(CXXFLAGS) --quartus-compile

$(TARGETS) : $(SOURCE_FILES) $(HEADER_FILES)
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $(SOURCE_FILES) -o $@
