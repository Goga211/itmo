// func.v — модуль вычисления функции y = a*b + a^3
module func #(parameter W = 8) (
    input clk,
    input rst,
    input start,
    input [W-1:0] a,
    input [W-1:0] b,
    output reg [2*W-1:0] y,
    output reg done,
    output reg busy
);
    // --- состояния автомата ---
    localparam IDLE = 3'd0;
    localparam MUL1 = 3'd1; // a*b
    localparam MUL2 = 3'd2; // a*a
    localparam MUL3 = 3'd3; // (a^2)*a
    localparam ADD  = 3'd4; // сложение
    localparam DONE = 3'd5;

    reg [2:0] state;

    // --- сигналы к умножителю ---
    reg        start_mult;
    reg [W-1:0] opA, opB;
    wire [23:0] mult_out;      // 16×8 = 24 бита
    wire        done_mult;

    // --- промежуточные результаты ---
    reg [15:0] ab_res;         // a*b
    reg [15:0] a2_res;         // a^2
    reg [23:0] a3_res;         // a^3 (шире!)

    // --- итерационный умножитель (16×8 бит) ---
    mult #(16, 8) u_mult(
        .clk(clk),
        .rst(rst),
        .start(start_mult),
        .a({8'b0, opA}),   // расширяем opA до 16 бит
        .b(opB),
        .y(mult_out),
        .done(done_mult)
    );

    // --- автомат управления ---
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state       <= IDLE;
            done        <= 0;
            busy        <= 0;
            y           <= 0;
            ab_res      <= 0;
            a2_res      <= 0;
            a3_res      <= 0;
            start_mult  <= 0;
            opA         <= 0;
            opB         <= 0;
        end else begin
            start_mult <= 0; // по умолчанию не дёргаем
            done       <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        busy       <= 1;
                        opA        <= a;
                        opB        <= b;
                        start_mult <= 1;   // запускаем a*b
                        state      <= MUL1;
                    end
                end

                MUL1: begin
                    if (done_mult) begin
                        ab_res     <= mult_out[15:0]; // сохраняем a*b
                        opA        <= a;
                        opB        <= a;
                        start_mult <= 1;              // запускаем a*a
                        state      <= MUL2;
                    end
                end

                MUL2: begin
                    if (done_mult) begin
                        a2_res     <= mult_out[15:0]; // сохраняем a^2 полностью
                        opA        <= mult_out[15:0]; // теперь не обрезаем!
                        opB        <= a;
                        start_mult <= 1;              // запускаем (a^2)*a
                        state      <= MUL3;
                    end
                end

                MUL3: begin
                    if (done_mult) begin
                        a3_res <= mult_out;  // сохраняем a^3 (24 бита)
                        state  <= ADD;
                    end
                end

                ADD: begin
                    y     <= ab_res + a3_res[15:0]; // складываем
                    state <= DONE;
                end

                DONE: begin
                    done  <= 1;
                    busy  <= 0;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule