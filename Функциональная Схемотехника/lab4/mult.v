module mult #(
    parameter A_W = 8,
    parameter B_W = 8,
    parameter Y_W = 16,
    parameter LATENCY = 2
) (
    input wire clk_i,
    input wire rst_i,
    input wire en_i,
    input wire [A_W-1:0] a_bi,
    input wire [B_W-1:0] b_bi,
    
    output reg [Y_W-1:0] y_bo,
    output reg done_o,
    output reg busy_o
);

    localparam IDLE = 1'b0;
    localparam WORK = 1'b1;
    
    localparam N = (A_W > B_W) ? A_W : B_W;
    
    reg [$clog2(N):0] ctr;
    wire [$clog2(N):0] end_step;
    wire [A_W-1:0] part_sum;
    wire [Y_W-1:0] shifted_part_sum;
    reg [A_W-1:0] a_reg;
    reg [B_W-1:0] b_reg;
    reg [Y_W-1:0] part_res;
    reg state;

    assign part_sum = a_reg & {A_W{b_reg[0]}};
    assign shifted_part_sum = part_sum << ctr;
    assign end_step = (ctr == (N - 1));

    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            ctr <= 0;
            part_res <= 0;
            y_bo <= 0;
            state <= IDLE;
            done_o <= 0;
            busy_o <= 0;
            a_reg <= 0;
            b_reg <= 0;
        end else begin
            done_o <= 0;
            busy_o <= state;
            
            case (state)
                IDLE: begin
                    if (en_i) begin
                        state <= WORK;
                        a_reg <= a_bi;
                        b_reg <= b_bi;
                        ctr <= 0;
                        part_res <= 0;
                        done_o <= 0;
                    end
                end
                
                WORK: begin
                    if (end_step) begin
                        state <= IDLE;
                        y_bo <= part_res + shifted_part_sum;
                        done_o <= 1;
                    end else begin
                        part_res <= part_res + shifted_part_sum;
                        ctr <= ctr + 1;
                        b_reg <= b_reg >> 1;
                    end
                end
            endcase
        end
    end

endmodule