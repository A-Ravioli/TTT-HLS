# Build the CPU-side artifacts of the TinyStories-on-Z2 accelerator:
#   make lib    -> build/libgemv_int8.{so,dylib}  (C++ datapath for the host runtime)
#   make test   -> build/test_gemv_ref            (golden-vector kernel test)
#   make check  -> run the kernel test against the exported golden vectors
#
# The FPGA bitstream is built separately (Vivado) via hdk/run_hls.tcl + build_bd.tcl.
CXX      ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wno-unknown-pragmas
KDIR      = kernel
BUILD     = build
GOLDEN   ?= golden/TinyStories-1M
TOL      ?= 1e-3

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LIBEXT = dylib
  SHARED = -dynamiclib
else
  LIBEXT = so
  SHARED = -shared -fPIC
endif

.PHONY: all lib test check clean
all: lib test

$(BUILD):
	mkdir -p $(BUILD)

lib: | $(BUILD)
	$(CXX) $(CXXFLAGS) -fPIC $(SHARED) $(KDIR)/gemv_int8_capi.cpp -I$(KDIR) \
		-o $(BUILD)/libgemv_int8.$(LIBEXT)

test: | $(BUILD)
	$(CXX) $(CXXFLAGS) $(KDIR)/test_gemv_ref.cpp -I$(KDIR) -o $(BUILD)/test_gemv_ref

check: test
	./$(BUILD)/test_gemv_ref $(GOLDEN) $(TOL)

clean:
	rm -f $(BUILD)/libgemv_int8.* $(BUILD)/test_gemv_ref $(BUILD)/*.o
