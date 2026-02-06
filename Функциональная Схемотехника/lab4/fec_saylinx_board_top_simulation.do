# ============================================================
#   ModelSim DO file — Simulation of fec_saylinx_board_top_tb
# ============================================================

vlib work
vmap work work

# ------------------------------------------------------------
# Компиляция модулей (АБСОЛЮТНЫЕ ПУТИ)
# ------------------------------------------------------------
vlog "C:/lab3/root3.v"
vlog "C:/lab3/mult.v"
vlog "C:/lab3/morr_l2.v"
vlog "C:/lab3/hex_to_7seg.v"
vlog "C:/lab3/seg_display_ctrl.v"
vlog "C:/lab3/fec_saylinx_board_top.v"
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
add wave /fec_saylinx_board_top_tb/RST_N
add wave /fec_saylinx_board_top_tb/KEY2_N
add wave /fec_saylinx_board_top_tb/KEY3_N
add wave /fec_saylinx_board_top_tb/KEY4_N

# GPIO inputs
add wave -divider {GPIO INPUTS}
add wave -radix hex /fec_saylinx_board_top_tb/GPIO_0_input_pullup
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/a_input
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/b_input

# A and B (values fed to DUT)
add wave -divider {INPUT VALUES}
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/a_value
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/b_value

# LEDs and display mode
add wave -divider {LED INDICATORS}
add wave /fec_saylinx_board_top_tb/LED
add wave /fec_saylinx_board_top_tb/dut/display_mode

# 7-seg
add wave -divider {SEG DISPLAY}
add wave -radix hex /fec_saylinx_board_top_tb/SEG_DATA
add wave -radix binary /fec_saylinx_board_top_tb/SEG_SEL
add wave -radix hex /fec_saylinx_board_top_tb/dut/display_value

# Calculation module (morr_l2)
add wave -divider {CALCULATION MODULE}
add wave /fec_saylinx_board_top_tb/dut/calc_unit/state
add wave /fec_saylinx_board_top_tb/dut/calc_unit/next_state
add wave /fec_saylinx_board_top_tb/dut/start_calc
add wave /fec_saylinx_board_top_tb/dut/calc_done
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/calc_result

# Result register
add wave -divider {RESULT}
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/last_result
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/calc_done_counter

# Root3 module
add wave -divider {ROOT3 MODULE}
add wave /fec_saylinx_board_top_tb/dut/calc_unit/root3_inst/state
add wave /fec_saylinx_board_top_tb/dut/calc_unit/cub_en
add wave /fec_saylinx_board_top_tb/dut/calc_unit/cub_busy
add wave /fec_saylinx_board_top_tb/dut/calc_unit/cub_done
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/calc_unit/cubert_result

# Mult module
add wave -divider {MULT MODULE}
add wave /fec_saylinx_board_top_tb/dut/calc_unit/mult_inst/state
add wave /fec_saylinx_board_top_tb/dut/calc_unit/mult_en
add wave /fec_saylinx_board_top_tb/dut/calc_unit/mult_busy
add wave /fec_saylinx_board_top_tb/dut/calc_unit/mult_done
add wave -radix unsigned /fec_saylinx_board_top_tb/dut/calc_unit/mult_inst/y_bo

# Настройки отображения
view structure
view signals

# ------------------------------------------------------------
# Запуск симуляции
# ------------------------------------------------------------
# 3 теста: каждый ~300 ms (с дебаунсингом и вычислениями)
# Всего ~900 ms, даем запас до 1.2 секунды

echo "Starting simulation..."
echo "This may take 5-10 minutes (3 full tests)..."
run 1200ms

# Масштабирование Wave
wave zoom full

# Настройка временной шкалы в миллисекундах
configure wave -timelineunits ms

echo "============================================================"
echo "Simulation of fec_saylinx_board_top completed!"
echo "Check Transcript for test results [PASS/FAIL]"
echo "Check Wave for timing diagrams"
echo "Clock frequency: 50 MHz (period 20 ns)"
echo "Total tests: 3"
echo "============================================================"
