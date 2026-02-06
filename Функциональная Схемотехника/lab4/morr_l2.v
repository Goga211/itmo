`timescale 1ns/1ps

module morr_l2 (
    input  wire        clk_i,
    input  wire        rst_i,
    input  wire        start_i,
    input  wire [7:0]  a_bi,
    input  wire [31:0] b_bi,
    output wire [15:0] result,
    output wire        done
);
    wire        cub_busy,  cub_done;
    wire        mult_busy, mult_done;
    wire [10:0] cubert_result;

    localparam [1:0]
        IDLE       = 2'b00,
        CALC_CUBRT = 2'b01,
        CALC_MULT  = 2'b10,
        DONE_STATE = 2'b11;

    reg [1:0] state, next_state;

    wire cub_en  = (state == CALC_CUBRT);
    wire mult_en = (state == CALC_MULT);

    assign done = (state == DONE_STATE);

    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) state <= IDLE;
        else       state <= next_state;
    end

    always @* begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start_i)
                    next_state = CALC_CUBRT;
            end
            CALC_CUBRT: begin
                if (cub_done)  // Достаточно проверить только done
                    next_state = CALC_MULT;
            end
            CALC_MULT: begin
                if (mult_done)  // Достаточно проверить только done
                    next_state = DONE_STATE;
            end
            DONE_STATE: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    root3 #(
        .IN_W (32),
        .OUT_W(11),
        .S_MAX(30),
        .LATENCY(6)          
    ) root3_inst (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .en_i    (cub_en),
        .x_bi    (b_bi),
        .busy_o  (cub_busy),
        .done_o  (cub_done),
        .y_bo    (cubert_result)
    );

    mult #(
        .A_W   (8),
        .B_W   (8),
        .Y_W   (16),
        .LATENCY(2)         
    ) mult_inst (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .en_i    (mult_en),
        .a_bi    (a_bi),
        .b_bi    (cubert_result[7:0]),
        .busy_o  (mult_busy),
        .done_o  (mult_done),
        .y_bo    (result)
    );

endmodule
