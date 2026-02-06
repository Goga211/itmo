`timescale 1ns/1ps

module fec_saylinx_board_top_tb;

    // Тактовый сигнал и сброс
    reg         CLK;
    reg         RST_N;

    // Кнопки (активный ноль на плате)
    reg         KEY2_N;
    reg         KEY3_N;
    reg         KEY4_N;

    // GPIO
    reg  [16:0] GPIO_0_input_pullup;
    wire [16:0] GPIO_0_out_zero_value;

    reg  [16:0] GPIO_1_input_pullup;
    wire [16:0] GPIO_1_out_zero_value;

    // Выходы
    wire [3:0]  LED;
    wire [7:0]  SEG_DATA;
    wire [7:0]  SEG_SEL;

    //==================================================================
    // Инстанс тестируемого top-модуля
    //==================================================================
    fec_saylinx_board_top dut (
        .CLK                 (CLK),
        .RST_N               (RST_N),
        .KEY2_N              (KEY2_N),
        .KEY3_N              (KEY3_N),
        .KEY4_N              (KEY4_N),
        .LED                 (LED),
        .SEG_DATA            (SEG_DATA),
        .SEG_SEL             (SEG_SEL),
        .GPIO_0_out_zero_value (GPIO_0_out_zero_value),
        .GPIO_0_input_pullup   (GPIO_0_input_pullup),
        .GPIO_1_out_zero_value (GPIO_1_out_zero_value),
        .GPIO_1_input_pullup   (GPIO_1_input_pullup)
    );

    //==================================================================
    // Генератор тактового сигнала 50 МГц (период 20 нс)
    //==================================================================
    initial begin
        CLK = 1'b0;
        forever #10 CLK = ~CLK;  // 20 ns период
    end

    //==================================================================
    // Таски для "нажатия" кнопок (с учётом дебаунса 1_000_000 тиков)
    //==================================================================

    // Имитация нажатия KEY2_N (показ 'a')
    task press_key2;
    begin
        KEY2_N = 1'b0;                    // нажали (активный 0)
        repeat (1100000) @(posedge CLK);  // держим дольше, чем 1_000_000 тактов
        KEY2_N = 1'b1;                    // отпустили
        repeat (20) @(posedge CLK);       // пауза после
    end
    endtask

    // Имитация нажатия KEY3_N (старт вычисления)
    task press_key3;
    begin
        KEY3_N = 1'b0;
        repeat (1100000) @(posedge CLK);
        KEY3_N = 1'b1;
        repeat (20) @(posedge CLK);
    end
    endtask

    // Имитация нажатия KEY4_N (показ 'b')
    task press_key4;
    begin
        KEY4_N = 1'b0;
        repeat (1100000) @(posedge CLK);
        KEY4_N = 1'b1;
        repeat (20) @(posedge CLK);
    end
    endtask

    //==================================================================
    // "Функция" запуска тестового сценария
    //  a, b — те значения, которые должны быть поданы на входы
    //==================================================================
    task run_case(input [7:0] a, input [7:0] b);
    begin
        // Установка значений a и b:
        // Внутри top: a_input = ~GPIO_0_input_pullup[7:0];
        //            b_input = ~GPIO_0_input_pullup[15:8];
        // Поэтому подаём инверсию желаемых значений.
        GPIO_0_input_pullup[7:0]   = ~a;
        GPIO_0_input_pullup[15:8]  = ~b;
        GPIO_0_input_pullup[16]    = 1'b1; // не используется, просто 1

        // Показать 'a'
        press_key2;

        // Показать 'b'
        press_key4;

        // Запустить вычисление (как "вызов функции")
        press_key3;

        // Подождать, пока появится valid (LED[3] = func_valid)
        wait (LED[3] == 1'b1);
        // Небольшая пауза, чтобы результат стабилизировался на дисплее
        repeat (1000) @(posedge CLK);
    end
    endtask

    //==================================================================
    // Основная последовательность теста
    //==================================================================
    initial begin
        RST_N               = 1'b0;  
        KEY2_N              = 1'b1;
        KEY3_N              = 1'b1;
        KEY4_N              = 1'b1;
        GPIO_0_input_pullup = 17'h1FFFF; // все "отпущены" (1)
        GPIO_1_input_pullup = 17'h1FFFF;

        // Немного подержать сброс
        repeat (10) @(posedge CLK);
        RST_N = 1'b1;  // снимаем сброс

        // Подождать, пока всё инициализируется
        repeat (1000) @(posedge CLK);

        // ========= СЦЕНАРИЙ 1 =========
        // a = 8'd5, b = 8'd3
        run_case(8'd5, 8'd3);

        // Небольшой промежуток между сценариями
        repeat (500000) @(posedge CLK);

        // ========= СЦЕНАРИЙ 2 =========
        // a = 8'd10, b = 8'd7
        run_case(8'd10, 8'd7);

        // Ещё немного подождать и остановить симуляцию
        repeat (500000) @(posedge CLK);
        $stop;
    end

    //==================================================================
    // Опционально: текстовый монитор для отладки
    //==================================================================
    initial begin
        $display("Time(ns)\tLED\tSEG_DATA\tSEG_SEL\tGPIO0[15:0]");
        $monitor("%0t\t%b\t%h\t%h\t%h",
                 $time, LED, SEG_DATA, SEG_SEL, GPIO_0_input_pullup[15:0]);
    end

endmodule