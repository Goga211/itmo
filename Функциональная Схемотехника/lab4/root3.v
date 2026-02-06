`timescale 1ns/1ps

module root3 #(
    parameter integer IN_W    = 32,
    parameter integer OUT_W   = 11,
    parameter integer S_MAX   = 30,
    parameter integer LATENCY = 0
)(
    input                   clk_i,
    input                   rst_i,

    input      [IN_W-1:0]   x_bi,
    input                   en_i,

    output                  busy_o,
    output reg              done_o,
    output reg [OUT_W-1:0]  y_bo
);
    localparam [1:0]
        ST_IDLE = 2'd0,
        ST_RUN  = 2'd1,
        ST_DONE = 2'd2;

    reg [1:0] state;

    reg [1:0] phase;

    reg [IN_W-1:0]   x;
    reg [OUT_W-1:0]  y;
    reg [5:0]        s;

    wire [OUT_W:0]   y_shift = {y,1'b0};
    wire [OUT_W:0]   y_plus1 = y_shift + {{OUT_W{1'b0}},1'b1};

    reg  [2*OUT_W:0]   prod_r;    
    reg  [2*OUT_W+2:0] prod3_r;  

    reg busy_r;
    assign busy_o = busy_r;

    // фронт en_i при неактивном busy
    reg en_d;
    wire start_pulse = (en_i & ~en_d) & ~busy_r;

    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            state   <= ST_IDLE;
            phase   <= 2'd0;

            en_d    <= 1'b0;
            busy_r  <= 1'b0;
            done_o  <= 1'b0;

            x       <= {IN_W{1'b0}};
            y       <= {OUT_W{1'b0}};
            s       <= 6'd0;

            y_bo    <= {OUT_W{1'b0}};
            prod_r  <= {2*OUT_W+1{1'b0}};
            prod3_r <= {2*OUT_W+3{1'b0}};
        end else begin
            en_d   <= en_i;
            done_o <= 1'b0;

            case (state)
                // --------- IDLE ---------
                ST_IDLE: begin
                    busy_r <= 1'b0;
                    phase  <= 2'd0;
                    if (start_pulse) begin
                        busy_r <= 1'b1;
                        x      <= x_bi;
                        y      <= {OUT_W{1'b0}};
                        s      <= S_MAX[5:0];
                        state  <= ST_RUN;
                    end
                end

                // --------- RUN (фазы 0/1/2) ---------
                ST_RUN: begin
                    case (phase)
                        // DSP1
                        2'd0: begin
                            prod_r <= y_shift * y_plus1;     // (2y)*(2y+1)
                            phase  <= 2'd1;
                        end
                        // DSP2
                        2'd1: begin
                            prod3_r <= prod_r * 3;           // 3*prod
                            phase   <= 2'd2;
                        end
                        2'd2: begin
                            // b = (3*prod + 1) << s  
                            if (x >= ( ((prod3_r + {{(2*OUT_W+2){1'b0}},1'b1}) << s) & {IN_W{1'b1}} )) begin
                                x <= x - ( ((prod3_r + {{(2*OUT_W+2){1'b0}},1'b1}) << s) & {IN_W{1'b1}} );
                                y <= y_plus1[OUT_W-1:0];     // (y<<1)+1
                            end else begin
                                y <= y_shift[OUT_W-1:0];     // (y<<1)
                            end

                            if (s >= 6'd3) begin
                                s     <= s - 6'd3;
                                phase <= 2'd0;             
                            end else begin
                   
                                state <= ST_DONE;
                            end
                        end
                        default: phase <= 2'd0;
                    endcase
                end

          
                ST_DONE: begin
                    y_bo   <= y;
                    busy_r <= 1'b0;
                    done_o <= 1'b1;
                    state  <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
