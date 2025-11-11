// mult.v — последовательный умножитель "в столбик" (16×8 бит)
module mult #(parameter W1 = 16, parameter W2 = 8)(
    input clk,
    input rst,
    input start,
    input [W1-1:0] a,
    input [W2-1:0] b,
    output reg [W1+W2-1:0] y,
    output reg done
);
    reg [W1-1:0] A;
    reg [W2-1:0] B;
    reg [W1+W2-1:0] acc;
    reg [3:0] count;
    reg busy;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            y <= 0;
            acc <= 0;
            A <= 0;
            B <= 0;
            count <= 0;
            done <= 0;
            busy <= 0;
        end else begin
            done <= 0;
            if (start && !busy) begin
                A <= a;
                B <= b;
                acc <= 0;
                count <= 0;
                busy <= 1;
            end else if (busy) begin
                if (B[0])
                    acc <= acc + A;
                A <= A << 1;
                B <= B >> 1;
                count <= count + 1;

                if (count == W2-1) begin
                    y <= acc;
                    done <= 1;
                    busy <= 0;
                end
            end
        end
    end
endmodule