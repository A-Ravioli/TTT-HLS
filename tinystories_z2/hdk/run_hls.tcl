# run_hls.tcl -- synthesize the W8A8 GEMV datapath to an RTL IP for the Z7020.
#
#   vitis_hls -f hdk/run_hls.tcl
#
# Produces a packaged RTL IP under gemv_int8_prj/sol1/impl/ip that hdk/build_bd.tcl
# instantiates next to the Zynq7 PS and builds into a PYNQ .bit + .hwh overlay.
#
# Default part is the PYNQ Z2's XC7Z020-1; clock 7 ns (~142 MHz) is comfortable on
# the -1 speed grade. Override with FPGA_PART / CLOCK_NS env vars.

set PART     [expr {[info exists ::env(FPGA_PART)] ? $::env(FPGA_PART) : "xc7z020clg400-1"}]
set CLOCK_NS [expr {[info exists ::env(CLOCK_NS)]  ? $::env(CLOCK_NS)  : 7.0}]

open_project gemv_int8_prj -reset
set_top gemv_int8
add_files kernel/gemv_int8_hls.cpp -cflags "-I kernel"
add_files -tb kernel/test_gemv_ref.cpp -cflags "-I kernel"

open_solution sol1 -reset
set_part $PART
create_clock -period $CLOCK_NS -name default

csynth_design
export_design -rtl verilog -format ip_catalog
exit
