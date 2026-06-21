# build_bd.tcl -- Vivado block design: Zynq7 PS + gemv_int8 IP -> PYNQ .bit + .hwh
#
#   vivado -mode batch -source hdk/build_bd.tcl
#
# Wires the HLS W8A8 GEMV IP to the Zynq-7020 PS:
#   * s_axi_control (AXI-Lite)  <- PS M_AXI_GP0   (host MMIO / register kick)
#   * m_axi gmem_w, gmem_scale  -> PS S_AXI_HP0   (weights + scales from DDR)
#   * m_axi gmem_x,  gmem_y     -> PS S_AXI_HP1   (activation in, result out)
#
# apply_bd_automation inserts the AXI SmartConnects, clocks (FCLK_CLK0) and
# proc_sys_reset, so this stays robust across Vivado 2020.2 / 2022.1. Emits the
# .bit + .hwh PYNQ needs into ./build/.
#
# Assumptions: run from the repo's tinystories_z2/ dir, after run_hls.tcl created
# gemv_int8_prj/sol1/impl/ip. Set BOARD_PART (e.g. tul.com.tw:pynq-z2:part0:1.0)
# if the PYNQ-Z2 board files are installed; otherwise the raw part is used.

set PART      [expr {[info exists ::env(FPGA_PART)] ? $::env(FPGA_PART) : "xc7z020clg400-1"}]
set BOARD     [expr {[info exists ::env(BOARD_PART)] ? $::env(BOARD_PART) : ""}]
set IP_REPO   gemv_int8_prj/sol1/impl/ip
set PROJ      gemv_z2
set BD        design_1
set OUT       build

file mkdir $OUT
create_project $PROJ ./vivado_prj -part $PART -force
if {$BOARD ne ""} { set_property board_part $BOARD [current_project] }

set_property ip_repo_paths [list $IP_REPO] [current_project]
update_ip_catalog

create_bd_design $BD

# --- Zynq7 PS ---------------------------------------------------------------
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
if {$BOARD ne ""} {
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR" apply_board_preset "1" \
                 Master "Disable" Slave "Disable"} $ps
} else {
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR" apply_board_preset "0" \
                 Master "Disable" Slave "Disable"} $ps
}
# enable the two HP slave ports + GP0 master + one fabric clock
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP1 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {142} \
    CONFIG.PCW_EN_CLK0_PORT {1} ] $ps

# --- GEMV kernel IP ---------------------------------------------------------
set k [create_bd_cell -type ip -vlnv xilinx.com:hls:gemv_int8:1.0 gemv_int8_0]

# control slave <- M_AXI_GP0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
             Master {/ps7/M_AXI_GP0} Slave {/gemv_int8_0/s_axi_control} \
             intc_ip {New AXI Interconnect} master_apm {0}} \
    [get_bd_intf_pins gemv_int8_0/s_axi_control]

# weight + scale masters -> S_AXI_HP0
foreach port {m_axi_gmem_w m_axi_gmem_scale} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
                 Master /gemv_int8_0/$port Slave {/ps7/S_AXI_HP0} \
                 intc_ip {Auto} master_apm {0}" \
        [get_bd_intf_pins ps7/S_AXI_HP0]
}
# activation + output masters -> S_AXI_HP1
foreach port {m_axi_gmem_x m_axi_gmem_y} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
                 Master /gemv_int8_0/$port Slave {/ps7/S_AXI_HP1} \
                 intc_ip {Auto} master_apm {0}" \
        [get_bd_intf_pins ps7/S_AXI_HP1]
}

assign_bd_address
validate_bd_design
save_bd_design

# --- synth / impl / bitstream ----------------------------------------------
make_wrapper -files [get_files ./vivado_prj/$PROJ.srcs/sources_1/bd/$BD/$BD.bd] -top
add_files -norecurse ./vivado_prj/$PROJ.gen/sources_1/bd/$BD/hdl/${BD}_wrapper.v
set_property top ${BD}_wrapper [current_fileset]

launch_runs impl_1 -to_step write_bitstream -jobs [expr {[info exists ::env(JOBS)] ? $::env(JOBS) : 4}]
wait_on_run impl_1

# --- collect the PYNQ overlay (.bit + .hwh) --------------------------------
set bit [glob -nocomplain ./vivado_prj/$PROJ.runs/impl_1/${BD}_wrapper.bit]
set hwh [glob -nocomplain ./vivado_prj/$PROJ.gen/sources_1/bd/$BD/hw_handoff/$BD.hwh]
if {$bit ne ""} { file copy -force $bit $OUT/gemv_int8.bit }
if {$hwh ne ""} { file copy -force $hwh $OUT/gemv_int8.hwh }
puts "=== wrote $OUT/gemv_int8.bit and $OUT/gemv_int8.hwh ==="
exit
