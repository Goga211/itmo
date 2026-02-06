`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// BIST Wrapper - обёртка для интеграции BIST с DUT
// 
// Объединяет:
// - BIST Control Logic
// - Два LFSR (для операндов a и b)
// - CRC8 (для проверки результатов)
// - Мультиплексоры для переключения режимов
//-----------------------------------------------------------------------------
module bist_wrapper #(
    // Полиномы LFSR (без x^8 и x^0)
    // LFSR1: x^8 + x^6 + x^4 + x^3 + 1 -> биты 6,4,3 = 8'b01011000
    parameter [7:0] LFSR1_POLY = 8'b01011000,
    // LFSR2: x^8 + x^7 + x^6 + x^3 + 1 -> биты 7,6,3 = 8'b11001000  
    parameter [7:0] LFSR2_POLY = 8'b11001000,
    // CRC8: x^8 + x^7 + x^3 + x + 1 -> биты 7,3,1,0 = 8'b10001011
    parameter [7:0] CRC8_POLY  = 8'b10001011,
    // Начальные значения
    parameter [7:0] LFSR1_SEED = 8'hA5,
    parameter [7:0] LFSR2_SEED = 8'h5A,
    parameter [7:0] CRC8_INIT  = 8'h00
)(
    input  wire        clk_i,
    input  wire        rst_i,
    
    // Кнопка управления режимом тестирования
    input  wire        btn_test_i,
    input  wire        slow_mode_i,    // Медленный режим теста
    
    // Внешние операнды (для нормального режима)
    input  wire [7:0]  a_ext_i,
    input  wire [31:0] b_ext_i,
    input  wire        start_ext_i,    // Внешний запуск вычисления
    
    // Интерфейс к DUT
    output wire [7:0]  a_dut_o,        // Операнд a для DUT
    output wire [31:0] b_dut_o,        // Операнд b для DUT
    output wire        start_dut_o,    // Запуск DUT
    input  wire [15:0] result_dut_i,   // Результат DUT
    input  wire        done_dut_i,     // DUT завершил вычисление
    
    // Выходы статуса
    output wire        test_mode_o,    // Режим тестирования активен
    output wire        test_done_o,    // Тестирование завершено
    output wire [7:0]  crc_o,          // Текущее значение CRC
    output wire [7:0]  test_count_o,   // Счётчик переходов в режим тестирования
    output wire [8:0]  iteration_o,    // Текущая итерация
    
    // Мультиплексированный выход (результат DUT или CRC)
    output wire [15:0] result_mux_o
);

    //-------------------------------------------------------------------------
    // Внутренние сигналы
    //-------------------------------------------------------------------------
    wire       lfsr_en, lfsr_load;
    wire       crc_en, crc_clear;
    wire       dut_start_bist;
    wire [7:0] lfsr1_data, lfsr2_data;
    wire [7:0] crc_value;
    wire       test_mode;
    
    //-------------------------------------------------------------------------
    // Антидребезг кнопки
    //-------------------------------------------------------------------------
    wire btn_debounced, btn_pulse;
    
    debounce #(
        .DEBOUNCE_CYCLES(1_000_000)  // ~20мс при 50МГц
    ) debounce_inst (
        .clk_i          (clk_i),
        .rst_i          (rst_i),
        .btn_i          (btn_test_i),
        .btn_o          (btn_debounced),
        .btn_posedge_o  (btn_pulse),
        .btn_negedge_o  ()
    );
    
    //-------------------------------------------------------------------------
    // BIST Control Logic
    //-------------------------------------------------------------------------
    bist_ctrl #(
        .TEST_ITERATIONS(256),
        .SLOW_DELAY(25_000_000)  // ~0.5 сек при 50МГц
    ) bist_ctrl_inst (
        .clk_i       (clk_i),
        .rst_i       (rst_i),
        .btn_test_i  (btn_pulse),
        .slow_mode_i (slow_mode_i),
        .dut_done_i  (done_dut_i),
        .test_mode_o (test_mode),
        .lfsr_en_o   (lfsr_en),
        .lfsr_load_o (lfsr_load),
        .crc_en_o    (crc_en),
        .crc_clear_o (crc_clear),
        .dut_start_o (dut_start_bist),
        .test_done_o (test_done_o),
        .test_count_o(test_count_o),
        .iteration_o (iteration_o)
    );
    
    //-------------------------------------------------------------------------
    // LFSR1 - генератор для операнда a
    //-------------------------------------------------------------------------
    lfsr8 #(
        .POLY(LFSR1_POLY),
        .SEED(LFSR1_SEED)
    ) lfsr1_inst (
        .clk_i  (clk_i),
        .rst_i  (rst_i),
        .en_i   (lfsr_en),
        .load_i (lfsr_load),
        .data_o (lfsr1_data)
    );
    
    //-------------------------------------------------------------------------
    // LFSR2 - генератор для операнда b
    //-------------------------------------------------------------------------
    lfsr8 #(
        .POLY(LFSR2_POLY),
        .SEED(LFSR2_SEED)
    ) lfsr2_inst (
        .clk_i  (clk_i),
        .rst_i  (rst_i),
        .en_i   (lfsr_en),
        .load_i (lfsr_load),
        .data_o (lfsr2_data)
    );
    
    //-------------------------------------------------------------------------
    // CRC8 - контрольная сумма результатов
    //-------------------------------------------------------------------------
    crc8 #(
        .POLY(CRC8_POLY),
        .INIT(CRC8_INIT)
    ) crc8_inst (
        .clk_i     (clk_i),
        .rst_i     (rst_i),
        .clear_i   (crc_clear),
        .en_i      (crc_en),
        .data_i    (result_dut_i[7:0]),  // Используем младшие 8 бит результата
        .byte_en_i (1'b1),
        .crc_o     (crc_value)
    );
    
    //-------------------------------------------------------------------------
    // Мультиплексоры
    //-------------------------------------------------------------------------
    
    // Операнд a: внешний или от LFSR1
    assign a_dut_o = test_mode ? lfsr1_data : a_ext_i;
    
    // Операнд b: внешний или от LFSR2 (расширяем до 32 бит)
    assign b_dut_o = test_mode ? {24'd0, lfsr2_data} : b_ext_i;
    
    // Запуск DUT: внешний или от BIST контроллера
    assign start_dut_o = test_mode ? dut_start_bist : start_ext_i;
    
    // Результат: DUT или CRC + счётчик (в режиме тестирования)
    assign result_mux_o = test_mode ? {test_count_o, crc_value} : result_dut_i;
    
    //-------------------------------------------------------------------------
    // Выходы статуса
    //-------------------------------------------------------------------------
    assign test_mode_o = test_mode;
    assign crc_o = crc_value;

endmodule

