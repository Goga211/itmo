# ============================================================
# ModelSim DO file for BIST Demo - Lab 4
# ============================================================

# Close previous simulation if any
catch {quit -sim}

# Delete old work library
if {[file exists work]} {
    vdel -lib work -all
}
vlib work

# ============================================================
# Compile all files
# ============================================================
puts "=== Compiling modules ==="

vlog -work work C:/Schema/lab3/mult.v
vlog -work work C:/Schema/lab3/root3.v
vlog -work work C:/Schema/lab3/morr_l2.v
vlog -work work C:/Schema/lab3/hex_to_7seg.v
vlog -work work C:/Schema/lab3/seg_display_ctrl.v
vlog -work work C:/Schema/lab3/lfsr8.v
vlog -work work C:/Schema/lab3/crc8.v
vlog -work work C:/Schema/lab3/debounce.v
vlog -work work C:/Schema/lab3/bist_ctrl.v
vlog -work work C:/Schema/lab3/bist_wrapper.v
vlog -work work C:/Schema/lab3/fec_saylinx_board_top.v
vlog -work work C:/Schema/lab3/bist_demo_tb.v

puts "=== Compilation complete ==="

# ============================================================
# Start simulation
# ============================================================
vsim -t 1ns -L work work.bist_demo_tb

# ============================================================
# Add waves
# ============================================================
add wave -divider "Clock & Reset"
add wave /bist_demo_tb/CLK
add wave /bist_demo_tb/RST_N

add wave -divider "Buttons"
add wave /bist_demo_tb/KEY2_N
add wave /bist_demo_tb/KEY3_N

add wave -divider "BIST Status"
add wave /bist_demo_tb/dut/test_mode
add wave /bist_demo_tb/dut/test_done
add wave -radix unsigned /bist_demo_tb/dut/test_count
add wave -radix unsigned /bist_demo_tb/dut/iteration

add wave -divider "LFSR Operands"
add wave -radix hexadecimal /bist_demo_tb/dut/a_to_dut
add wave -radix hexadecimal /bist_demo_tb/dut/b_to_dut

add wave -divider "DUT Result"
add wave -radix hexadecimal /bist_demo_tb/dut/dut_result
add wave /bist_demo_tb/dut/dut_done

add wave -divider "CRC8"
add wave -radix hexadecimal /bist_demo_tb/dut/crc_value

add wave -divider "LED"
add wave -radix binary /bist_demo_tb/LED

configure wave -namecolwidth 250
configure wave -valuecolwidth 100

# ============================================================
# Run simulation
# ============================================================
puts ""
puts "============================================================"
puts "  Starting BIST simulation..."
puts "  This will take ~2 minutes (256 iterations with debounce)"
puts "============================================================"
puts ""

run -all

wave zoom full

puts ""
puts "============================================================"
puts "  Simulation complete! Check:"
puts "  - iteration should reach 256"
puts "  - crc_value shows final CRC"
puts "============================================================"
