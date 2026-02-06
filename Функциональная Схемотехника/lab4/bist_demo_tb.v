`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// Testbench for BIST Demo - Lab 4
//-----------------------------------------------------------------------------
module bist_demo_tb;

    // Clock and reset
    reg         CLK;
    reg         RST_N;

    // Buttons (active low)
    reg         KEY2_N;
    reg         KEY3_N;
    reg         KEY4_N;

    // GPIO
    reg  [16:0] GPIO_0_input_pullup;
    wire [16:0] GPIO_0_out_zero_value;
    reg  [16:0] GPIO_1_input_pullup;
    wire [16:0] GPIO_1_out_zero_value;

    // Outputs
    wire [3:0]  LED;
    wire [7:0]  SEG_DATA;
    wire [7:0]  SEG_SEL;

    //==================================================================
    // DUT instance
    //==================================================================
    fec_saylinx_board_top dut (
        .CLK                   (CLK),
        .RST_N                 (RST_N),
        .KEY2_N                (KEY2_N),
        .KEY3_N                (KEY3_N),
        .KEY4_N                (KEY4_N),
        .LED                   (LED),
        .SEG_DATA              (SEG_DATA),
        .SEG_SEL               (SEG_SEL),
        .GPIO_0_out_zero_value (GPIO_0_out_zero_value),
        .GPIO_0_input_pullup   (GPIO_0_input_pullup),
        .GPIO_1_out_zero_value (GPIO_1_out_zero_value),
        .GPIO_1_input_pullup   (GPIO_1_input_pullup)
    );

    //==================================================================
    // Clock generator 50 MHz (period 20 ns)
    //==================================================================
    initial begin
        CLK = 1'b0;
        forever #10 CLK = ~CLK;
    end

    //==================================================================
    // Button press tasks (with debounce ~22ms)
    //==================================================================
    task press_key2;
    begin
        $display("[%0t ns] KEY2 pressed (BIST toggle)", $time);
        KEY2_N = 1'b0;
        repeat (1100000) @(posedge CLK);  // Hold > debounce
        KEY2_N = 1'b1;
        repeat (100000) @(posedge CLK);   // Pause
    end
    endtask

    task press_key3;
    begin
        $display("[%0t ns] KEY3 pressed (Calculate/Slow mode)", $time);
        KEY3_N = 1'b0;
        repeat (1100000) @(posedge CLK);
        KEY3_N = 1'b1;
        repeat (100000) @(posedge CLK);
    end
    endtask

    //==================================================================
    // Main test sequence
    //==================================================================
    initial begin
        $display("\n====================================================");
        $display("  BIST TESTBENCH - Lab 4");
        $display("  Testing: Built-In Self-Test for y = a * cbrt(b)");
        $display("====================================================\n");
        
        // Init
        RST_N               = 1'b0;  
        KEY2_N              = 1'b1;
        KEY3_N              = 1'b1;
        KEY4_N              = 1'b1;
        GPIO_0_input_pullup = 17'h1FFFF;
        GPIO_1_input_pullup = 17'h1FFFF;

        // Reset
        repeat (10) @(posedge CLK);
        RST_N = 1'b1;
        $display("[%0t ns] Reset released", $time);
        repeat (1000) @(posedge CLK);

        // ========= TEST 1: Normal mode =========
        $display("\n--- TEST 1: Normal calculation ---");
        GPIO_0_input_pullup[7:0]  = ~8'd5;   // a = 5
        GPIO_0_input_pullup[15:8] = ~8'd8;   // b = 8
        $display("[%0t ns] GPIO: a=5, b=8, Expected: 5*cbrt(8) = 5*2 = 10", $time);
        
        repeat (1000) @(posedge CLK);
        press_key3;  // Start calculation
        
        // Wait for result
        repeat (5000000) @(posedge CLK);  // ~100 ms
        $display("[%0t ns] Result: %0d", $time, dut.dut_result);

        // ========= TEST 2: Enter BIST mode =========
        $display("\n--- TEST 2: BIST mode (256 iterations) ---");
        $display("[%0t ns] Pressing KEY2 to enter BIST...", $time);
        
        press_key2;  // Enter BIST
        
        $display("[%0t ns] test_mode = %b", $time, dut.test_mode);
        
        // Wait for BIST to complete
        $display("[%0t ns] Waiting for 256 iterations...", $time);
        
        // Monitor iterations
        wait(dut.test_done == 1);
        
        $display("\n====================================================");
        $display("  BIST COMPLETED!");
        $display("  Iterations: %0d", dut.iteration);
        $display("  Final CRC8: 0x%02h", dut.crc_value);
        $display("  Test count: %0d", dut.test_count);
        $display("====================================================\n");

        // ========= TEST 3: Exit and re-enter BIST =========
        $display("\n--- TEST 3: Exit and re-enter BIST ---");
        press_key2;  // Exit BIST
        repeat (100000) @(posedge CLK);
        
        press_key2;  // Re-enter BIST
        wait(dut.test_done == 1);
        
        $display("\n====================================================");
        $display("  SECOND BIST COMPLETED!");
        $display("  Final CRC8: 0x%02h (should be same)", dut.crc_value);
        $display("  Test count: %0d", dut.test_count);
        $display("====================================================\n");

        repeat (100000) @(posedge CLK);
        
        $display("\n  SIMULATION COMPLETED\n");
        $stop;
    end

    //==================================================================
    // Monitor BIST progress
    //==================================================================
    reg [8:0] last_iter;
    initial last_iter = 9'h1FF;
    
    always @(posedge CLK) begin
        if (dut.test_mode && !dut.test_done) begin
            if (dut.iteration != last_iter) begin
                last_iter <= dut.iteration;
                // Print every 32 iterations + first 3 + last 3
                if (dut.iteration <= 3 || dut.iteration >= 254 || (dut.iteration % 32 == 0)) begin
                    $display("  Iter %3d: a=0x%02h, b=0x%02h, CRC=0x%02h", 
                             dut.iteration, dut.a_to_dut, dut.b_to_dut[7:0], dut.crc_value);
                end
            end
        end
    end

endmodule
