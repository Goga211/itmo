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
    assign rst = ~RST_N;  // Только KEY1 (RST_N) для сброса

    //------------------------------------------------------------------------
    // Кнопки
    //------------------------------------------------------------------------
    wire key2, key3, key4;
    assign key2 = ~KEY2_N;  // Показать 'a'
    assign key3 = ~KEY3_N;  // Запустить вычисление
    assign key4 = ~KEY4_N;  // Показать 'b'

    //------------------------------------------------------------------------
    // Дебаунсинг кнопок KEY2, KEY3, KEY4
    //------------------------------------------------------------------------
    reg [19:0] debounce_cnt2, debounce_cnt3, debounce_cnt4;
    reg key2_stable, key2_prev;
    reg key3_stable, key3_prev;
    reg key4_stable, key4_prev;
    wire key2_pulse, key3_pulse, key4_pulse;

    // Дебаунсинг KEY2
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

    //------------------------------------------------------------------------
    // Входные данные для вычислений из GPIO
    //------------------------------------------------------------------------
    // GPIO инвертированы для удобства (перемычка = 1, без перемычки = 0)
    // GPIO_0_input_pullup[7:0]  - значение 'a' (8 бит) - пины P1-T5
    // GPIO_0_input_pullup[15:8] - значение 'b' (8 бит) - пины T6-T13
    
    wire [7:0]  a_input;
    wire [7:0]  b_input;
    
    // Инвертируем GPIO (перемычка к GND = 1 в логике)
    assign a_input = ~GPIO_0_input_pullup[7:0];
    assign b_input = ~GPIO_0_input_pullup[15:8];

    // Постоянно обновляем значения из GPIO
    wire [7:0]  a_value;
    wire [7:0]  b_value;
    assign a_value = a_input;
    assign b_value = b_input;
    
    reg start_calc;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            start_calc <= 1'b0;
        end else begin
            start_calc <= 1'b0;
            
            // KEY3 - запустить вычисление с текущими a и b из GPIO
            if (key3_pulse) begin
                start_calc <= 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // Инстанс вашего модуля func
    //------------------------------------------------------------------------
    wire func_busy;
    wire func_valid;
    wire [23:0] func_y;

    func func_inst (
        .clk   (clk),
        .rst   (rst),
        .start (start_calc),
        .a     (a_value),
        .b     (b_value),
        .busy  (func_busy),
        .valid (func_valid),
        .y     (func_y)
    );

    //------------------------------------------------------------------------
    // Регистр для хранения результата и логика отображения
    //------------------------------------------------------------------------
    reg [31:0] display_value;
    reg [23:0] last_result;
    
    localparam [1:0]
        MODE_RESULT = 2'd0,
        MODE_SHOW_A = 2'd1,
        MODE_SHOW_B = 2'd2;
    
    reg [1:0] display_mode;
    reg [7:0] calc_done_counter; // Счетчик успешных вычислений

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            display_value <= 32'h00000000;
            last_result <= 24'h000000;
            display_mode <= MODE_RESULT;
            calc_done_counter <= 8'd0;
        end else begin
            // Сохраняем результат вычисления
            if (func_valid) begin
                last_result <= func_y;
                display_mode <= MODE_RESULT;
                calc_done_counter <= calc_done_counter + 1; // Считаем сколько раз посчитали
            end
            
            // Переключение режимов отображения по кнопкам
            if (key2_pulse) begin
                display_mode <= MODE_SHOW_A;
            end else if (key4_pulse) begin
                display_mode <= MODE_SHOW_B;
            end else if (key3_pulse) begin
                display_mode <= MODE_RESULT;
            end
            
            // Выбираем что отображать
            case (display_mode)
                MODE_SHOW_A: display_value <= {24'h000000, a_value};        // Показываем только a (8 бит)
                MODE_SHOW_B: display_value <= {24'h000000, b_value};        // Показываем только b (8 бит)
                MODE_RESULT: display_value <= {8'h00, last_result};         // Показываем результат (24 бита)
                default:     display_value <= {8'h00, last_result};
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Контроллер семисегментного дисплея
    //------------------------------------------------------------------------
    // Гасим ведущие нули когда показываем 'a' или 'b', но не результат
    wire blank_zeros;
    assign blank_zeros = (display_mode == MODE_SHOW_A) || (display_mode == MODE_SHOW_B);
    
    seg_display_ctrl seg_ctrl (
        .clk                 (clk),
        .rst                 (rst),
        .value               (display_value),
        .blank_leading_zeros (blank_zeros),
        .seg_data            (SEG_DATA),
        .seg_sel             (SEG_SEL)
    );

    //------------------------------------------------------------------------
    // Светодиоды - индикация состояния
    //------------------------------------------------------------------------
    assign LED[0] = (display_mode == MODE_SHOW_A); // Режим отображения 'a'
    assign LED[1] = (display_mode == MODE_SHOW_B); // Режим отображения 'b'
    assign LED[2] = func_busy;                     // Модуль занят вычислениями
    assign LED[3] = func_valid;                    // Валидный результат

    //------------------------------------------------------------------------
    // GPIO - установка нулевых значений на выходы
    //------------------------------------------------------------------------
    assign GPIO_0_out_zero_value = {17{1'b0}};
    assign GPIO_1_out_zero_value = {17{1'b0}};

endmodule