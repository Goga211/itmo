module mult16x8 #(
  parameter WA = 16, WB = 8
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 start,
  input  wire [WA-1:0]        A,
  input  wire [WB-1:0]        B,
  output reg                  busy,
  output reg                  done,
  output reg  [WA+WB-1:0]     Y
);
  localparam WOUT = WA + WB;

  reg [WOUT-1:0] a_ext;
  reg [WB-1:0]   b_reg;
  reg [WOUT-1:0] acc;
  reg [7:0]      cnt;

  wire [WOUT-1:0] add_term = b_reg[0] ? a_ext : {WOUT{1'b0}};
  wire [WOUT-1:0] acc_next = acc + add_term;

  localparam IDLE=1'b0, WORK=1'b1;
  reg state;

  always @(posedge clk) begin
    if (rst) begin
      state<=IDLE; busy<=0; done<=0; Y<=0;
      a_ext<=0; b_reg<=0; acc<=0; cnt<=0;
    end else begin
      done <= 1'b0;
      case (state)
        IDLE: begin
          if (start) begin
            busy  <= 1'b1;
            a_ext <= {{(WOUT-WA){1'b0}}, A}; 
            b_reg <= B;
            acc   <= {WOUT{1'b0}};
            cnt   <= 0;
            state <= WORK;
          end
        end
        WORK: begin
          acc   <= acc_next;
          a_ext <= a_ext << 1;  
          b_reg <= b_reg >> 1;
          cnt   <= cnt + 1;
          if (cnt == (WB-1)) begin
            Y    <= acc_next;
            busy <= 1'b0;
            done <= 1'b1;
            state<= IDLE;
          end
        end
      endcase
    end
  end
endmodule
