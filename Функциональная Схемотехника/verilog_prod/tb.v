`timescale 1ns / 1ps

module decoder_3to8_tb;

    reg  [2:0] in_code;
    reg        EN;
    wire [7:0] out_lines;

    integer i;
    reg [7:0] expected_val;

    decoder_3to8 uut (
        .A(in_code[2]),
        .B(in_code[1]),
        .C(in_code[0]),
        .EN(EN),
        .Y0(out_lines[0]),
        .Y1(out_lines[1]),
        .Y2(out_lines[2]),
        .Y3(out_lines[3]),
        .Y4(out_lines[4]),
        .Y5(out_lines[5]),
        .Y6(out_lines[6]),
        .Y7(out_lines[7])
    );

    initial begin
        EN = 1;
        for (i = 0; i < 8; i = i + 1) begin
            in_code = i[2:0];
            expected_val = 8'b00000001 << i;

            #10;

            if (out_lines === expected_val)
                $display("PASS: input=%b  output=%b  expected=%b", in_code, out_lines, expected_val);
            else
                $display("FAIL: input=%b  output=%b  expected=%b", in_code, out_lines, expected_val);
        end
        #10 $stop;
    end
endmodule
