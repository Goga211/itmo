// func_tb.v — тестбенч для проверки работы func
`timescale 1ns/1ps

module func_tb;
    reg clk, rst, start;
    reg [7:0] a, b;
    wire [15:0] y;
    wire done, busy;

    func #(8) dut (
        .clk(clk), .rst(rst), .start(start),
        .a(a), .b(b), .y(y), .done(done), .busy(busy)
    );

    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, func_tb);

        clk = 0;
        rst = 1;
        start = 0;
        a = 0; b = 0;
        #10 rst = 0;

        // Тест 1
        @(negedge clk);
        a = 3; b = 4;
        start = 1;
        @(negedge clk);
        start = 0;

        wait(done);
        #1 $display("a=%0d b=%0d y=%0d (ожидалось 39)", a, b, y);

        // Тест 2
        @(negedge clk);
        a = 5; b = 2;
        start = 1;
        @(negedge clk);
        start = 0;

        wait(done);
        #1 $display("a=%0d b=%0d y=%0d (ожидалось 135)", a, b, y);

        // Тест 3
        @(negedge clk);
        a = 10; b = 10;
        start = 1;
        @(negedge clk);
        start = 0;

        wait(done);
        #1 $display("a=%0d b=%0d y=%0d (ожидалось 1100)", a, b, y);

        // Дополнительные тесты
@(negedge clk); a = 0; b = 0; start = 1; @(negedge clk); start = 0; wait(done);
#1 $display("a=%0d b=%0d y=%0d (ожидалось 0)", a, b, y);

@(negedge clk); a = 1; b = 5; start = 1; @(negedge clk); start = 0; wait(done);
#1 $display("a=%0d b=%0d y=%0d (ожидалось 6)", a, b, y);

@(negedge clk); a = 2; b = 7; start = 1; @(negedge clk); start = 0; wait(done);
#1 $display("a=%0d b=%0d y=%0d (ожидалось 22)", a, b, y);

@(negedge clk); a = 4; b = 3; start = 1; @(negedge clk); start = 0; wait(done);
#1 $display("a=%0d b=%0d y=%0d (ожидалось 76)", a, b, y);

@(negedge clk); a = 8; b = 1; start = 1; @(negedge clk); start = 0; wait(done);
#1 $display("a=%0d b=%0d y=%0d (ожидалось 520)", a, b, y);

@(negedge clk); a = 10; b = 0; start = 1; @(negedge clk); start = 0; wait(done);
#1 #1$display("a=%0d b=%0d y=%0d (ожидалось 1000)", a, b, y);

@(negedge clk); a = 15; b = 15; start = 1; @(negedge clk); start = 0; wait(done);
#1 $display("a=%0d b=%0d y=%0d (ожидалось 3600)", a, b, y);

        #20 $finish;
    end

    always #5 clk = ~clk;
endmodule