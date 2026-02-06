# ModelSim/QuestaSim simulation script for BIST testbench
# Лабораторная работа №4: Проектирование встроенных схем самотестирования

# Create work library
if {[file exists work]} {
    vdel -lib work -all
}
vlib work

# Compile all Verilog files
vlog -work work mult.v
vlog -work work root3.v
vlog -work work morr_l2.v
vlog -work work hex_to_7seg.v
vlog -work work seg_display_ctrl.v
vlog -work work lfsr8.v
vlog -work work crc8.v
vlog -work work debounce.v
vlog -work work bist_ctrl.v
vlog -work work bist_wrapper.v
vlog -work work fec_saylinx_board_top.v
vlog -work work bist_tb.v

# Start simulation
vsim -t 1ps -L work work.bist_tb

# Add waves
add wave -divider "Clock & Reset"
add wave -position end sim:/bist_tb/clk
add wave -position end sim:/bist_tb/rst_n

add wave -divider "Buttons"
add wave -position end sim:/bist_tb/key2_n
add wave -position end sim:/bist_tb/key3_n
add wave -position end sim:/bist_tb/key4_n

add wave -divider "BIST Status"
add wave -position end sim:/bist_tb/test_mode
add wave -position end sim:/bist_tb/test_done
add wave -position end -radix unsigned sim:/bist_tb/iteration
add wave -position end -radix hexadecimal sim:/bist_tb/crc_value
add wave -position end -radix unsigned sim:/bist_tb/test_count

add wave -divider "DUT Interface"
add wave -position end -radix hexadecimal sim:/bist_tb/dut/a_to_dut
add wave -position end -radix hexadecimal sim:/bist_tb/dut/b_to_dut
add wave -position end sim:/bist_tb/dut/start_to_dut
add wave -position end -radix hexadecimal sim:/bist_tb/dut_result
add wave -position end sim:/bist_tb/dut_done_sig

add wave -divider "Display"
add wave -position end -radix hexadecimal sim:/bist_tb/dut/display_value
add wave -position end sim:/bist_tb/led

add wave -divider "LFSR Values"
add wave -position end -radix hexadecimal sim:/bist_tb/dut/bist_inst/lfsr1_data
add wave -position end -radix hexadecimal sim:/bist_tb/dut/bist_inst/lfsr2_data

add wave -divider "BIST FSM"
add wave -position end sim:/bist_tb/dut/bist_inst/bist_ctrl_inst/state

# Configure wave window
configure wave -namecolwidth 250
configure wave -valuecolwidth 100

# Run simulation
run 10ms

# Zoom to fit
wave zoom full

echo "Simulation complete! Check waveforms for BIST operation."
echo "Final CRC8 value should be visible in crc_value signal."



