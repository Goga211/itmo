# ============================================================
# Скрипт для добавления волн ПОСЛЕ загрузки симуляции Quartus
# Просто выполни: do C:/lab3/bist_waves.do
# ============================================================

# --- Группа: Тактирование и сброс ---
add wave -divider "=== Clock & Reset ==="
add wave -color yellow /bist_demo_tb/clk
add wave -color red /bist_demo_tb/rst_n

# --- Группа: Кнопки ---
add wave -divider "=== Кнопки (0=нажата) ==="
add wave -color cyan /bist_demo_tb/key2_n
add wave -color cyan /bist_demo_tb/key3_n
add wave -color cyan /bist_demo_tb/key4_n

# --- Группа: Состояние BIST ---
add wave -divider "=== BIST Состояние ==="
add wave -color green /bist_demo_tb/test_mode
add wave -color green /bist_demo_tb/test_done
add wave -radix unsigned -color orange /bist_demo_tb/test_count
add wave -radix unsigned -color orange /bist_demo_tb/iteration

# --- Группа: Операнды ---
add wave -divider "=== Операнды a, b ==="
add wave -radix hexadecimal -color magenta /bist_demo_tb/a_to_dut
add wave -radix hexadecimal -color magenta /bist_demo_tb/b_to_dut

# --- Группа: DUT ---
add wave -divider "=== DUT (a * cbrt(b)) ==="
add wave -radix hexadecimal /bist_demo_tb/dut_result
add wave /bist_demo_tb/dut_done

# --- Группа: CRC8 ---
add wave -divider "=== CRC8 ==="
add wave -radix hexadecimal -color gold /bist_demo_tb/crc_value

# --- Группа: LED ---
add wave -divider "=== LED ==="
add wave -radix binary /bist_demo_tb/led

# Настройка
configure wave -namecolwidth 250
configure wave -valuecolwidth 100

# Запуск
run -all
wave zoom full

puts "============================================================"
puts "  Симуляция завершена! Проверь:"
puts "  - iteration должен дойти до 256"
puts "  - crc_value - финальный CRC"
puts "============================================================"

