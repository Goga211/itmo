`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// Top-level модуль с интегрированной схемой BIST
// 
// Лабораторная работа №4: Проектирование встроенных схем самотестирования
// 
// Функционал:
// - Нормальный режим: вычисление result = a * cbrt(b)
// - Режим тестирования (BIST): автоматический прогон 256 итераций
//   с псевдослучайными операндами и расчётом CRC8
//-----------------------------------------------------------------------------
module fec_saylinx_board_top
(
    input           CLK,
    input           RST_N,

    input           KEY2_N,
    input           KEY3_N,
    input           KEY4_N,

    output [3:0]    LED,

    output [7:0] SEG_DATA,
    output [7:0] SEG_SEL,

    output [16:0] GPIO_0_out_zero_value,
    input  [16:0] GPIO_0_input_pullup,
	 
    output [16:0] GPIO_1_out_zero_value,
    input  [16:0] GPIO_1_input_pullup
);

    //------------------------------------------------------------------------
    // Сигналы тактирования и сброса
    //------------------------------------------------------------------------
    wire clk;
    wire rst;

    assign clk = CLK;
    assign rst = ~RST_N;  // KEY1 для сброса

    //------------------------------------------------------------------------
    // Кнопки
    //------------------------------------------------------------------------
    wire key2, key3, key4;
    assign key2 = ~KEY2_N;  // Показать 'a' / Режим BIST
    assign key3 = ~KEY3_N;  // Запустить вычисление
    assign key4 = ~KEY4_N;  // Показать 'b'

    //------------------------------------------------------------------------
    // Дебаунсинг кнопок (оригинальная логика для KEY3, KEY4)
    //------------------------------------------------------------------------
    reg [19:0] debounce_cnt2, debounce_cnt3, debounce_cnt4;
    reg key2_stable, key2_prev;
    reg key3_stable, key3_prev;
    reg key4_stable, key4_prev;
    wire key2_pulse, key3_pulse, key4_pulse;
    
    parameter DOUBLE_CLICK_WINDOW = 25_000_000;
    
    reg [24:0] double_click_timer2, double_click_timer3, double_click_timer4;
    reg key2_pending, key3_pending, key4_pending;
    reg key2_double_click, key3_double_click, key4_double_click;

    // Дебаунсинг KEY2 (теперь также используется для BIST)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            debounce_cnt2 <= 0;
            key2_stable <= 0;
            key2_prev <= 0;
        end else begin
            if (key2 != key2_stable) begin
                debounce_cnt2 <= debounce_cnt2 + 1;
                if (debounce_cnt2 == 20'd1000000) begin
                    key2_stable <= key2;
                    debounce_cnt2 <= 0;
                end
            end else begin
                debounce_cnt2 <= 0;
            end
            key2_prev <= key2_stable;
        end
    end
    assign key2_pulse = key2_stable & ~key2_prev;

    // Дебаунсинг KEY3
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            debounce_cnt3 <= 0;
            key3_stable <= 0;
            key3_prev <= 0;
        end else begin
            if (key3 != key3_stable) begin
                debounce_cnt3 <= debounce_cnt3 + 1;
                if (debounce_cnt3 == 20'd1000000) begin
                    key3_stable <= key3;
                    debounce_cnt3 <= 0;
                end
            end else begin
                debounce_cnt3 <= 0;
            end
            key3_prev <= key3_stable;
        end
    end
    assign key3_pulse = key3_stable & ~key3_prev;

    // Дебаунсинг KEY4
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            debounce_cnt4 <= 0;
            key4_stable <= 0;
            key4_prev <= 0;
        end else begin
            if (key4 != key4_stable) begin
                debounce_cnt4 <= debounce_cnt4 + 1;
                if (debounce_cnt4 == 20'd1000000) begin
                    key4_stable <= key4;
                    debounce_cnt4 <= 0;
                end
            end else begin
                debounce_cnt4 <= 0;
            end
            key4_prev <= key4_stable;
        end
    end
    assign key4_pulse = key4_stable & ~key4_prev;

    // Логика двойного клика для KEY2
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            double_click_timer2 <= 0;
            key2_pending <= 0;
            key2_double_click <= 0;
        end else begin
            key2_double_click <= 0;
            
            if (key2_pulse) begin
                if (key2_pending) begin
                    key2_double_click <= 1;
                    key2_pending <= 0;
                    double_click_timer2 <= 0;
                end else begin
                    key2_pending <= 1;
                    double_click_timer2 <= 0;
                end
            end else if (key2_pending) begin
                if (double_click_timer2 >= DOUBLE_CLICK_WINDOW) begin
                    key2_pending <= 0;
                    double_click_timer2 <= 0;
                end else begin
                    double_click_timer2 <= double_click_timer2 + 1;
                end
            end
        end
    end

    // KEY3 - двойной клик для переключения формата
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            double_click_timer3 <= 0;
            key3_pending <= 0;
            key3_double_click <= 0;
        end else begin
            key3_double_click <= 0;
            
            if (key3_pulse) begin
                if (key3_pending) begin
                    key3_double_click <= 1;
                    key3_pending <= 0;
                    double_click_timer3 <= 0;
                end else begin
                    key3_pending <= 1;
                    double_click_timer3 <= 0;
                end
            end else if (key3_pending) begin
                if (double_click_timer3 >= DOUBLE_CLICK_WINDOW) begin
                    key3_pending <= 0;
                    double_click_timer3 <= 0;
                end else begin
                    double_click_timer3 <= double_click_timer3 + 1;
                end
            end
        end
    end

    // KEY4 - двойной клик
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            double_click_timer4 <= 0;
            key4_pending <= 0;
            key4_double_click <= 0;
        end else begin
            key4_double_click <= 0;
            
            if (key4_pulse) begin
                if (key4_pending) begin
                    key4_double_click <= 1;
                    key4_pending <= 0;
                    double_click_timer4 <= 0;
                end else begin
                    key4_pending <= 1;
                    double_click_timer4 <= 0;
                end
            end else if (key4_pending) begin
                if (double_click_timer4 >= DOUBLE_CLICK_WINDOW) begin
                    key4_pending <= 0;
                    double_click_timer4 <= 0;
                end else begin
                    double_click_timer4 <= double_click_timer4 + 1;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // GPIO входы
    //------------------------------------------------------------------------
    wire [7:0]  a_input;
    wire [7:0]  b_input;
    
    assign a_input = ~GPIO_0_input_pullup[7:0];
    assign b_input = ~GPIO_0_input_pullup[15:8];

    wire [7:0]  a_value;
    wire [31:0] b_value;
    assign a_value = a_input;
    assign b_value = {24'd0, b_input};
    
    //------------------------------------------------------------------------
    // Сигнал запуска вычисления (нормальный режим)
    //------------------------------------------------------------------------
    reg start_calc;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            start_calc <= 1'b0;
        end else begin
            start_calc <= 1'b0;
            
            // KEY3 - запустить вычисление (только если не в режиме BIST)
            if (key3_pulse && !test_mode) begin
                start_calc <= 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // BIST Wrapper
    //------------------------------------------------------------------------
    wire        test_mode;
    wire        test_done;
    wire [7:0]  crc_value;
    wire [7:0]  test_count;
    wire [8:0]  iteration;
    wire [15:0] result_mux;
    
    // Сигналы к DUT
    wire [7:0]  a_to_dut;
    wire [31:0] b_to_dut;
    wire        start_to_dut;
    wire [15:0] dut_result;
    wire        dut_done;

    bist_wrapper #(
        // LFSR1: x^8 + x^6 + x^4 + x^3 + 1
        .LFSR1_POLY(8'b01011000),
        // LFSR2: x^8 + x^7 + x^6 + x^3 + 1
        .LFSR2_POLY(8'b11001000),
        // CRC8: x^8 + x^7 + x^3 + x + 1
        .CRC8_POLY (8'b10001011),
        .LFSR1_SEED(8'hA5),
        .LFSR2_SEED(8'h5A),
        .CRC8_INIT (8'h00)
    ) bist_inst (
        .clk_i       (clk),
        .rst_i       (rst),
        .btn_test_i  (key2),           // KEY2 для входа/выхода из режима BIST
        .a_ext_i     (a_value),
        .b_ext_i     (b_value),
        .start_ext_i (start_calc),
        .a_dut_o     (a_to_dut),
        .b_dut_o     (b_to_dut),
        .start_dut_o (start_to_dut),
        .result_dut_i(dut_result),
        .done_dut_i  (dut_done),
        .test_mode_o (test_mode),
        .test_done_o (test_done),
        .crc_o       (crc_value),
        .test_count_o(test_count),
        .iteration_o (iteration),
        .result_mux_o(result_mux)
    );

    //------------------------------------------------------------------------
    // DUT (Design Under Test) - модуль из ЛР3
    //------------------------------------------------------------------------
    morr_l2 calc_unit (
        .clk_i   (clk),
        .rst_i   (rst),
        .start_i (start_to_dut),
        .a_bi    (a_to_dut),
        .b_bi    (b_to_dut),
        .result  (dut_result),
        .done    (dut_done)
    );

    //------------------------------------------------------------------------
    // Логика отображения
    //------------------------------------------------------------------------
    reg [31:0] display_value;
    reg [31:0] last_result;
    
    localparam [1:0]
        MODE_RESULT = 2'd0,
        MODE_SHOW_A = 2'd1,
        MODE_SHOW_B = 2'd2;
    
    reg [1:0] display_mode;
    reg display_format;  // 0 = HEX, 1 = DEC
    reg [7:0] calc_done_counter;
    
    // Функция для конвертации числа в BCD
    function [31:0] to_decimal_display;
        input [15:0] value;
        reg [15:0] temp0, temp1, temp2, temp3, temp4;
        reg [3:0] digit0, digit1, digit2, digit3, digit4;
        begin
            temp0 = value;
            digit0 = temp0 % 16'd10;
            
            temp1 = temp0 / 16'd10;
            digit1 = temp1 % 16'd10;
            
            temp2 = temp1 / 16'd10;
            digit2 = temp2 % 16'd10;
            
            temp3 = temp2 / 16'd10;
            digit3 = temp3 % 16'd10;
            
            temp4 = temp3 / 16'd10;
            digit4 = temp4 % 16'd10;

            to_decimal_display = {12'h000, digit4, digit3, digit2, digit1, digit0};
        end
    endfunction

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            display_value <= 32'h00000000;
            last_result <= 32'h00000000;
            display_mode <= MODE_RESULT;
            display_format <= 0;
            calc_done_counter <= 8'd0;
        end else begin
            // В режиме BIST отображаем CRC и счётчик
            if (test_mode) begin
                // Формат: [test_count][CRC8] или [iteration]
                if (test_done) begin
                    // После завершения: счётчик тестов + финальный CRC
                    display_value <= {16'h0000, test_count, crc_value};
                end else begin
                    // Во время тестирования: текущая итерация + CRC
                    display_value <= {15'h0000, iteration, crc_value};
                end
            end else begin
                // Нормальный режим
                if (dut_done && !test_mode) begin
                    last_result <= {16'h0000, dut_result};
                    display_mode <= MODE_RESULT;
                    calc_done_counter <= calc_done_counter + 1;
                end
                
                if (key2_double_click || key3_double_click || key4_double_click) begin
                    display_format <= ~display_format;
                end
                
                if (key2_pulse && !key2_pending && !test_mode) begin
                    display_mode <= MODE_SHOW_A;
                end else if (key4_pulse && !key4_pending) begin
                    display_mode <= MODE_SHOW_B;
                end else if (key3_pulse && !key3_pending && !test_mode) begin
                    display_mode <= MODE_RESULT;
                end
                
                case (display_mode)
                    MODE_SHOW_A: begin
                        if (display_format)
                            display_value <= to_decimal_display({8'h00, a_input});
                        else
                            display_value <= {24'h000000, a_input};
                    end
                    MODE_SHOW_B: begin
                        if (display_format)
                            display_value <= to_decimal_display({8'h00, b_input});
                        else
                            display_value <= {24'h000000, b_input};
                    end
                    MODE_RESULT: begin
                        if (display_format)
                            display_value <= to_decimal_display(last_result[15:0]);
                        else
                            display_value <= last_result;
                    end
                    default: begin
                        display_value <= last_result;
                    end
                endcase
            end
        end
    end

    //------------------------------------------------------------------------
    // 7-сегментный дисплей
    //------------------------------------------------------------------------
    wire blank_zeros;
    assign blank_zeros = !test_mode && ((display_mode == MODE_SHOW_A) || (display_mode == MODE_SHOW_B));
    
    seg_display_ctrl seg_ctrl (
        .clk                 (clk),
        .rst                 (rst),
        .value               (display_value),
        .blank_leading_zeros (blank_zeros),
        .seg_data            (SEG_DATA),
        .seg_sel             (SEG_SEL)
    );

    //------------------------------------------------------------------------
    // LED индикация
    //------------------------------------------------------------------------
    // LED[0] - Режим отображения 'a' / BIST активен
    // LED[1] - Режим отображения 'b'
    // LED[2] - Режим отображения результата / BIST завершён
    // LED[3] - Формат отображения / Тестирование в процессе
    
    assign LED[0] = test_mode ? 1'b1 : (display_mode == MODE_SHOW_A);
    assign LED[1] = test_mode ? test_done : (display_mode == MODE_SHOW_B);
    assign LED[2] = test_mode ? 1'b0 : (display_mode == MODE_RESULT);
    assign LED[3] = test_mode ? !test_done : display_format;

    //------------------------------------------------------------------------
    // GPIO - установка нулевых значений на выходы
    //------------------------------------------------------------------------
    assign GPIO_0_out_zero_value = {17{1'b0}};
    assign GPIO_1_out_zero_value = {17{1'b0}};

endmodule



