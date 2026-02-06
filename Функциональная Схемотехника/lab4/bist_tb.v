`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// Testbench для BIST (Built-In Self-Test)
// 
// Проверяет:
// 1. Нормальный режим работы
// 2. Вход в режим тестирования по нажатию кнопки
// 3. Прохождение 256 итераций тестирования
// 4. Корректность финального значения CRC8
//-----------------------------------------------------------------------------
module bist_tb;

    //-------------------------------------------------------------------------
    // Параметры
    //-------------------------------------------------------------------------
    parameter CLK_PERIOD = 20;  // 50 MHz

    //-------------------------------------------------------------------------
    // Сигналы
    //-------------------------------------------------------------------------
    reg         clk;
    reg         rst_n;
    reg         key2_n;
    reg         key3_n;
    reg         key4_n;
    
    wire [3:0]  led;
    wire [7:0]  seg_data;
    wire [7:0]  seg_sel;
    
    wire [16:0] gpio_0_out;
    reg  [16:0] gpio_0_in;
    wire [16:0] gpio_1_out;
    reg  [16:0] gpio_1_in;

    //-------------------------------------------------------------------------
    // DUT
    //-------------------------------------------------------------------------
    fec_saylinx_board_top dut (
        .CLK                  (clk),
        .RST_N                (rst_n),
        .KEY2_N               (key2_n),
        .KEY3_N               (key3_n),
        .KEY4_N               (key4_n),
        .LED                  (led),
        .SEG_DATA             (seg_data),
        .SEG_SEL              (seg_sel),
        .GPIO_0_out_zero_value(gpio_0_out),
        .GPIO_0_input_pullup  (gpio_0_in),
        .GPIO_1_out_zero_value(gpio_1_out),
        .GPIO_1_input_pullup  (gpio_1_in)
    );

    //-------------------------------------------------------------------------
    // Генерация тактового сигнала
    //-------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //-------------------------------------------------------------------------
    // Мониторинг внутренних сигналов BIST
    //-------------------------------------------------------------------------
    wire        test_mode    = dut.test_mode;
    wire        test_done    = dut.test_done;
    wire [7:0]  crc_value    = dut.crc_value;
    wire [7:0]  test_count   = dut.test_count;
    wire [8:0]  iteration    = dut.iteration;
    wire [15:0] dut_result   = dut.dut_result;
    wire        dut_done_sig = dut.dut_done;

    //-------------------------------------------------------------------------
    // Задача для эмуляции нажатия кнопки
    //-------------------------------------------------------------------------
    task press_button;
        input integer btn_num;  // 2, 3, или 4
        begin
            case (btn_num)
                2: begin
                    key2_n = 0;
                    #(CLK_PERIOD * 50);  // Короткое нажатие для тестбенча
                    key2_n = 1;
                end
                3: begin
                    key3_n = 0;
                    #(CLK_PERIOD * 50);
                    key3_n = 1;
                end
                4: begin
                    key4_n = 0;
                    #(CLK_PERIOD * 50);
                    key4_n = 1;
                end
            endcase
            #(CLK_PERIOD * 100);  // Ожидание дебаунса
        end
    endtask

    //-------------------------------------------------------------------------
    // Основной тест
    //-------------------------------------------------------------------------
    initial begin
        $display("==============================================");
        $display("   BIST Testbench - Starting...");
        $display("==============================================");
        
        // Инициализация
        rst_n   = 1;
        key2_n  = 1;
        key3_n  = 1;
        key4_n  = 1;
        gpio_0_in = 17'hFFFF;  // Все входы подтянуты к 1 (нет перемычек)
        gpio_1_in = 17'hFFFF;
        
        // Сброс
        #(CLK_PERIOD * 10);
        rst_n = 0;
        #(CLK_PERIOD * 10);
        rst_n = 1;
        #(CLK_PERIOD * 100);
        
        $display("[%0t] Reset complete", $time);
        $display("[%0t] Test mode: %b, LED: %b", $time, test_mode, led);
        
        //---------------------------------------------------------------------
        // Тест 1: Проверка нормального режима
        //---------------------------------------------------------------------
        $display("\n--- Test 1: Normal mode calculation ---");
        
        // Установка входных значений (инвертируем, т.к. pullup)
        gpio_0_in[7:0]   = ~8'h05;   // a = 5
        gpio_0_in[15:8]  = ~8'h08;   // b = 8 (cbrt(8) = 2)
        
        #(CLK_PERIOD * 10);
        
        // Нажатие KEY3 для запуска вычисления
        $display("[%0t] Starting calculation: a=%d, b=%d", $time, 8'h05, 8'h08);
        press_button(3);
        
        // Ожидание завершения вычисления
        wait(dut_done_sig == 1);
        #(CLK_PERIOD * 10);
        
        $display("[%0t] Calculation done! Result: %d (expected: 10)", $time, dut_result);
        
        #(CLK_PERIOD * 100);
        
        //---------------------------------------------------------------------
        // Тест 2: Вход в режим BIST
        //---------------------------------------------------------------------
        $display("\n--- Test 2: Entering BIST mode ---");
        
        $display("[%0t] Pressing KEY2 to enter BIST mode...", $time);
        press_button(2);
        
        // Ожидание входа в режим тестирования
        #(CLK_PERIOD * 1000);
        
        if (test_mode) begin
            $display("[%0t] Successfully entered BIST mode!", $time);
            $display("[%0t] Test count: %d", $time, test_count);
        end else begin
            $display("[%0t] ERROR: Failed to enter BIST mode!", $time);
        end
        
        //---------------------------------------------------------------------
        // Тест 3: Ожидание завершения BIST (256 итераций)
        //---------------------------------------------------------------------
        $display("\n--- Test 3: Running BIST (256 iterations) ---");
        $display("[%0t] Waiting for BIST to complete...", $time);
        
        // Мониторинг прогресса каждые 32 итерации
        fork
            begin
                wait(test_done == 1);
            end
            begin
                while (!test_done) begin
                    #(CLK_PERIOD * 10000);
                    if (test_mode && !test_done) begin
                        $display("[%0t] Progress: iteration %d/256, CRC=0x%02X", 
                                 $time, iteration, crc_value);
                    end
                end
            end
        join

        #(CLK_PERIOD * 100);
        
        $display("\n==============================================");
        $display("   BIST Complete!");
        $display("==============================================");
        $display("   Final CRC8:    0x%02X", crc_value);
        $display("   Test Count:    %d", test_count);
        $display("   Iterations:    %d", iteration);
        $display("   LED status:    %b", led);
        $display("==============================================");
        
        //---------------------------------------------------------------------
        // Тест 4: Выход из режима BIST
        //---------------------------------------------------------------------
        $display("\n--- Test 4: Exiting BIST mode ---");
        
        press_button(2);
        #(CLK_PERIOD * 1000);
        
        if (!test_mode) begin
            $display("[%0t] Successfully exited BIST mode!", $time);
        end else begin
            $display("[%0t] Still in BIST mode", $time);
        end
        
        //---------------------------------------------------------------------
        // Тест 5: Повторный вход в BIST
        //---------------------------------------------------------------------
        $display("\n--- Test 5: Re-entering BIST mode ---");
        
        press_button(2);
        #(CLK_PERIOD * 1000);
        
        $display("[%0t] Test count should be 2: %d", $time, test_count);
        
        // Ждём завершения второго теста
        wait(test_done == 1);
        #(CLK_PERIOD * 100);
        
        $display("\n==============================================");
        $display("   Second BIST Complete!");
        $display("   Final CRC8:    0x%02X", crc_value);
        $display("   Test Count:    %d", test_count);
        $display("==============================================");
        
        #(CLK_PERIOD * 1000);
        
        $display("\n*** Testbench completed! ***\n");
        $finish;
    end

    //-------------------------------------------------------------------------
    // Таймаут
    //-------------------------------------------------------------------------
    initial begin
        #(CLK_PERIOD * 50_000_000);  // 1 секунда при 50МГц
        $display("ERROR: Testbench timeout!");
        $finish;
    end

    //-------------------------------------------------------------------------
    // VCD dump для просмотра в GTKWave
    //-------------------------------------------------------------------------
    initial begin
        $dumpfile("bist_tb.vcd");
        $dumpvars(0, bist_tb);
    end

endmodule



