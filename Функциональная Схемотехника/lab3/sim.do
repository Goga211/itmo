# ============================================================
#   ModelSim DO file — Simulation of fec_saylinx_board_top_tb
# ============================================================

vlib work
vmap work work


# ------------------------------------------------------------
# Компиляция модулей (АБСОЛЮТНЫЕ ПУТИ)
# ------------------------------------------------------------
vlog "C:/lab3/fec_saylinx_board_top.v"
vlog "C:/lab3/seg_display_ctrl.v"
vlog "C:/lab3/hex_to_7seg.v"
vlog "C:/lab3/mult.v"
vlog "C:/lab3/func.v"
vlog "C:/lab3/fec_saylinx_board_top_tb.v"

# ------------------------------------------------------------
# Запуск симуляции TB
# ------------------------------------------------------------
vsim -voptargs="+acc" work.fec_saylinx_board_top_tb


# ------------------------------------------------------------
# WAVE signals
# ------------------------------------------------------------

# CLK + RESET + 3 BUTTONS
add wave -divider {CLOCK / RESET / BUTTONS}
add wave /fec_saylinx_board_top_tb/CLK
# add wave /fec_saylinx_board_top_tb/RST_N
add wave /fec_saylinx_board_top_tb/KEY2_N
add wave /fec_saylinx_board_top_tb/KEY3_N
add wave /fec_saylinx_board_top_tb/KEY4_N

# A and B (values fed to DUT)
add wave -divider {INPUT VALUES}
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/a_value
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/b_value

# 7-seg
add wave -divider {SEG DISPLAY}
add wave -radix hex /fec_saylinx_board_top_tb/SEG_DATA
add wave /fec_saylinx_board_top_tb/SEG_SEL

# Result register
add wave -divider {RESULT}
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/last_result

# Run simulation
run 200 ms
