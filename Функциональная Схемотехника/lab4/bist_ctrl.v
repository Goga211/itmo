`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// Модуль управления BIST (Built-In Self-Test)
// 
// Функции:
// 1. Переключение режима работы (нормальный / тестирование) по нажатию кнопки
// 2. Управление LFSR и CRC модулями
// 3. Счёт итераций тестирования (256 итераций)
// 4. Хранение счётчика переходов в режим тестирования
//-----------------------------------------------------------------------------
module bist_ctrl #(
    parameter TEST_ITERATIONS = 256,     // Количество итераций тестирования
    parameter SLOW_DELAY = 25_000_000    // Задержка для медленного режима (~0.5 сек при 50МГц)
)(
    input  wire        clk_i,
    input  wire        rst_i,
    
    // Управление режимом
    input  wire        btn_test_i,      // Импульс нажатия кнопки для входа/выхода из теста
    input  wire        slow_mode_i,     // Медленный режим (1 = с задержкой)
    
    // Сигнал готовности DUT
    input  wire        dut_done_i,      // DUT завершил вычисление
    
    // Выходы управления
    output wire        test_mode_o,     // Режим тестирования активен
    output wire        lfsr_en_o,       // Разрешение сдвига LFSR
    output wire        lfsr_load_o,     // Загрузка начального значения LFSR
    output wire        crc_en_o,        // Разрешение обновления CRC
    output wire        crc_clear_o,     // Сброс CRC
    output wire        dut_start_o,     // Запуск DUT
    output wire        test_done_o,     // Тестирование завершено
    output wire [7:0]  test_count_o,    // Счётчик переходов в режим тестирования
    output wire [8:0]  iteration_o      // Текущая итерация (0-256)
);

    //-------------------------------------------------------------------------
    // Состояния FSM
    //-------------------------------------------------------------------------
    localparam [2:0]
        ST_IDLE       = 3'd0,  // Нормальный режим работы
        ST_TEST_INIT  = 3'd1,  // Инициализация тестирования
        ST_TEST_START = 3'd2,  // Запуск DUT
        ST_TEST_WAIT  = 3'd3,  // Ожидание результата DUT
        ST_TEST_CRC   = 3'd4,  // Обновление CRC
        ST_TEST_DELAY = 3'd5,  // Задержка для медленного режима
        ST_TEST_NEXT  = 3'd6,  // Переход к следующей итерации
        ST_TEST_DONE  = 3'd7;  // Тестирование завершено
    
    reg [2:0] state, next_state;
    
    //-------------------------------------------------------------------------
    // Регистры
    //-------------------------------------------------------------------------
    reg [8:0]  iteration_cnt;     // Счётчик итераций (0-256)
    reg [7:0]  test_count_reg;    // Счётчик переходов в режим тестирования
    reg        test_mode_reg;     // Флаг режима тестирования
    reg [24:0] delay_cnt;         // Счётчик задержки для медленного режима
    
    //-------------------------------------------------------------------------
    // Переход состояний
    //-------------------------------------------------------------------------
    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i)
            state <= ST_IDLE;
        else
            state <= next_state;
    end
    
    //-------------------------------------------------------------------------
    // Логика переходов
    //-------------------------------------------------------------------------
    always @(*) begin
        next_state = state;
        
        case (state)
            ST_IDLE: begin
                if (btn_test_i && !test_mode_reg)
                    next_state = ST_TEST_INIT;
            end
            
            ST_TEST_INIT: begin
                next_state = ST_TEST_START;
            end
            
            ST_TEST_START: begin
                next_state = ST_TEST_WAIT;
            end
            
            ST_TEST_WAIT: begin
                if (btn_test_i) begin
                    // Прерывание теста по нажатию кнопки
                    next_state = ST_IDLE;
                end else if (dut_done_i) begin
                    next_state = ST_TEST_CRC;
                end
            end
            
            ST_TEST_CRC: begin
                // После обновления CRC - задержка или сразу к следующей итерации
                if (slow_mode_i)
                    next_state = ST_TEST_DELAY;
                else
                    next_state = ST_TEST_NEXT;
            end
            
            ST_TEST_DELAY: begin
                // Ждём задержку в медленном режиме
                if (btn_test_i) begin
                    next_state = ST_IDLE;  // Прервать по кнопке
                end else if (delay_cnt >= SLOW_DELAY) begin
                    next_state = ST_TEST_NEXT;
                end
            end
            
            ST_TEST_NEXT: begin
                if (iteration_cnt >= TEST_ITERATIONS - 1) begin
                    next_state = ST_TEST_DONE;
                end else begin
                    next_state = ST_TEST_START;
                end
            end
            
            ST_TEST_DONE: begin
                if (btn_test_i)
                    next_state = ST_IDLE;
            end
            
            default: next_state = ST_IDLE;
        endcase
    end
    
    //-------------------------------------------------------------------------
    // Счётчик итераций
    //-------------------------------------------------------------------------
    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            iteration_cnt <= 9'd0;
        end else begin
            case (state)
                ST_TEST_INIT:
                    iteration_cnt <= 9'd0;
                ST_TEST_NEXT:
                    iteration_cnt <= iteration_cnt + 1;
                default: ;
            endcase
        end
    end
    
    //-------------------------------------------------------------------------
    // Счётчик задержки для медленного режима
    //-------------------------------------------------------------------------
    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            delay_cnt <= 25'd0;
        end else begin
            if (state == ST_TEST_DELAY)
                delay_cnt <= delay_cnt + 1;
            else
                delay_cnt <= 25'd0;
        end
    end
    
    //-------------------------------------------------------------------------
    // Счётчик переходов в режим тестирования
    //-------------------------------------------------------------------------
    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            test_count_reg <= 8'd0;
        end else if (state == ST_IDLE && next_state == ST_TEST_INIT) begin
            test_count_reg <= test_count_reg + 1;
        end
    end
    
    //-------------------------------------------------------------------------
    // Флаг режима тестирования
    //-------------------------------------------------------------------------
    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            test_mode_reg <= 1'b0;
        end else begin
            case (state)
                ST_IDLE:
                    test_mode_reg <= 1'b0;
                ST_TEST_INIT, ST_TEST_START, ST_TEST_WAIT, 
                ST_TEST_CRC, ST_TEST_DELAY, ST_TEST_NEXT, ST_TEST_DONE:
                    test_mode_reg <= 1'b1;
                default:
                    test_mode_reg <= 1'b0;
            endcase
        end
    end
    
    //-------------------------------------------------------------------------
    // Выходные сигналы
    //-------------------------------------------------------------------------
    assign test_mode_o  = test_mode_reg;
    assign test_count_o = test_count_reg;
    assign iteration_o  = iteration_cnt;
    
    // Управление LFSR
    assign lfsr_load_o = (state == ST_TEST_INIT);
    assign lfsr_en_o   = (state == ST_TEST_NEXT);
    
    // Управление CRC
    assign crc_clear_o = (state == ST_TEST_INIT);
    assign crc_en_o    = (state == ST_TEST_CRC);
    
    // Управление DUT
    assign dut_start_o = (state == ST_TEST_START);
    
    // Флаг завершения тестирования
    assign test_done_o = (state == ST_TEST_DONE);

endmodule

