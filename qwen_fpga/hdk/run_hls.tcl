# run_hls.tcl -- synthesize the INT4 GEMV datapath to an RTL IP core.
#
#   vitis_hls -f hdk/run_hls.tcl
#
# Produces a packaged RTL IP under gemv_int4_prj/sol1/impl/ip that is then
# instantiated in the AWS HDK custom logic (see hdk/README.md) and built into a
# DCP -> AFI with aws_build_dcp_from_cl.py. We use HLS only to generate RTL; the
# AFI itself is produced by the HDK/Vivado flow (the supported F2 path).

set PART        [expr {[info exists ::env(FPGA_PART)] ? $::env(FPGA_PART) : "xcvu47p-fsvh2892-2-e"}]
set CLOCK_NS    [expr {[info exists ::env(CLOCK_NS)]  ? $::env(CLOCK_NS)  : 4.0}]

open_project gemv_int4_prj -reset
set_top gemv_int4
add_files kernel/gemv_int4_hls.cpp -cflags "-I kernel"
add_files -tb kernel/test_gemv_ref.cpp -cflags "-I kernel"

open_solution sol1 -reset
set_part $PART
create_clock -period $CLOCK_NS -name default

csynth_design
# export as Vivado IP for HDK integration (no Vitis AFI linking on F2)
export_design -rtl verilog -format ip_catalog
exit
