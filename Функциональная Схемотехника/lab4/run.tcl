# Скрипт для запуска симуляции в ModelSim
# Добавляем сигналы в Wave окно

# Верхний уровень тестбенча
add wave -divider "Тестбенч"
add wave /moore_tb1/clk
add wave /moore_tb1/rst
add wave /moore_tb1/start
add wave -radix unsigned /moore_tb1/a
add wave -radix unsigned /moore_tb1/b
add wave -radix unsigned /moore_tb1/result
add wave /moore_tb1/done

# Основной модуль morr_l2
add wave -divider "Модуль morr_l2"
add wave /moore_tb1/dut/state
add wave /moore_tb1/dut/next_state
add wave /moore_tb1/dut/cub_en
add wave /moore_tb1/dut/mult_en
add wave /moore_tb1/dut/cub_busy
add wave /moore_tb1/dut/cub_done
add wave /moore_tb1/dut/mult_busy
add wave /moore_tb1/dut/mult_done
add wave -radix unsigned /moore_tb1/dut/cubert_result

# Модуль root3
add wave -divider "Модуль root3"
add wave /moore_tb1/dut/root3_inst/state
add wave /moore_tb1/dut/root3_inst/phase
add wave /moore_tb1/dut/root3_inst/busy_o
add wave /moore_tb1/dut/root3_inst/done_o
add wave -radix unsigned /moore_tb1/dut/root3_inst/x
add wave -radix unsigned /moore_tb1/dut/root3_inst/y
add wave -radix unsigned /moore_tb1/dut/root3_inst/s

# Модуль mult
add wave -divider "Модуль mult"
add wave /moore_tb1/dut/mult_inst/state
add wave /moore_tb1/dut/mult_inst/busy_o
add wave /moore_tb1/dut/mult_inst/done_o
add wave -radix unsigned /moore_tb1/dut/mult_inst/a_reg
add wave -radix unsigned /moore_tb1/dut/mult_inst/b_reg
add wave -radix unsigned /moore_tb1/dut/mult_inst/y_bo

# Настройки отображения
view structure
view signals

# Запуск симуляции
run -all

# Масштабирование окна Wave для удобного просмотра
wave zoom full

