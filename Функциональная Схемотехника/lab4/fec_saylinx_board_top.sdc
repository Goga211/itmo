# Synopsys Design Constraints File для fec_saylinx_board_top
# Файл временных ограничений для Quartus Prime

# Определение тактового сигнала CLK
# Частота: 50 МГц (период 20 нс)
create_clock -name CLK -period 20.000 [get_ports {CLK}]

# Автоматическое определение производных тактовых сигналов
derive_pll_clocks

# Автоматическое определение неопределенности тактовых сигналов
derive_clock_uncertainty

# Задержки для входных сигналов (кнопки, GPIO)
set_input_delay -clock CLK -max 2.0 [get_ports {RST_N KEY2_N KEY3_N KEY4_N}]
set_input_delay -clock CLK -min 0.5 [get_ports {RST_N KEY2_N KEY3_N KEY4_N}]

set_input_delay -clock CLK -max 2.0 [get_ports {GPIO_0_input_pullup[*]}]
set_input_delay -clock CLK -min 0.5 [get_ports {GPIO_0_input_pullup[*]}]

set_input_delay -clock CLK -max 2.0 [get_ports {GPIO_1_input_pullup[*]}]
set_input_delay -clock CLK -min 0.5 [get_ports {GPIO_1_input_pullup[*]}]

# Задержки для выходных сигналов (LED, семисегментник, GPIO)
set_output_delay -clock CLK -max 2.0 [get_ports {LED[*]}]
set_output_delay -clock CLK -min 0.0 [get_ports {LED[*]}]

set_output_delay -clock CLK -max 2.0 [get_ports {SEG_DATA[*]}]
set_output_delay -clock CLK -min 0.0 [get_ports {SEG_DATA[*]}]

set_output_delay -clock CLK -max 2.0 [get_ports {SEG_SEL[*]}]
set_output_delay -clock CLK -min 0.0 [get_ports {SEG_SEL[*]}]

set_output_delay -clock CLK -max 2.0 [get_ports {GPIO_0_out_zero_value[*]}]
set_output_delay -clock CLK -min 0.0 [get_ports {GPIO_0_out_zero_value[*]}]

set_output_delay -clock CLK -max 2.0 [get_ports {GPIO_1_out_zero_value[*]}]
set_output_delay -clock CLK -min 0.0 [get_ports {GPIO_1_out_zero_value[*]}]

# False paths (асинхронные сигналы)
# Кнопки сброса не критичны по времени
set_false_path -from [get_ports {RST_N}] -to [all_registers]

# Multicycle paths для медленных вычислительных блоков
# root3 и mult могут выполняться несколько тактов
set_multicycle_path -setup -end 10 -from [get_registers {morr_l2:calc_unit|root3:root3_inst|*}]
set_multicycle_path -hold -end 9 -from [get_registers {morr_l2:calc_unit|root3:root3_inst|*}]

set_multicycle_path -setup -end 10 -from [get_registers {morr_l2:calc_unit|mult:mult_inst|*}]
set_multicycle_path -hold -end 9 -from [get_registers {morr_l2:calc_unit|mult:mult_inst|*}]

